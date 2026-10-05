"""
Shared infrastructure for cross-corpus validation: train the DE-feature pipeline
on DEAP, evaluate zero-shot on another EEG emotion corpus (SEED-IV / DREAMER /
AMIGOS). This module contains everything that does NOT require the target
dataset's actual files, so it can be tested and reviewed independently of data
access.

Design note on why this is a separate "harmonize first" pipeline rather than
just reusing the DEAP model as-is:
DEAP's cached DE features (deap_features_ext_v1.npz) are extracted from all 32
DEAP channels. A target corpus will almost never share the exact same montage
(SEED-IV uses a 62-channel cap; DREAMER uses 14 Emotiv EPOC channels; AMIGOS
uses 14 Emotiv channels too). A model trained on 32-channel features cannot be
fed a 14-channel or a differently-ordered 62-channel feature vector. So a fair
zero-shot transfer requires:
  1. Intersecting channel *names* between DEAP and the target corpus.
  2. Re-extracting DE features for BOTH corpora using only that common channel
     subset, in the same fixed order.
  3. Retraining the DEAP-side model on the reduced/reordered feature space
     (this is a legitimate, cheap retrain -- it is not "peeking" at the target
     corpus, it only uses DEAP's own labels).
  4. Harmonizing labels: DEAP/DREAMER/AMIGOS use continuous 1-9 (or 1-5) SAM-style
     valence/arousal ratings; SEED-IV uses discrete film-clip-elicited categories
     (neutral/sad/fear/happy). Each corpus is binarized with its OWN documented
     convention (see per-adapter docstrings), not a shared numeric threshold,
     since the scales are not directly comparable.
"""
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, List, Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly

BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
BAND_ORDER = ["theta", "alpha", "beta", "gamma"]

DEAP_CHANNELS = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
]


@dataclass
class Trial:
    """One EEG trial/session segment, in a corpus's native sampling rate/channels."""
    subject_id: str
    channel_names: List[str]
    fs: float
    signal: np.ndarray          # (n_channels, n_samples)
    label: dict                 # e.g. {"valence": 1} or {"emotion": "positive"} -- adapter-specific,
                                 # turned into a binary int by `label_fn` below.


class CorpusAdapter:
    """Subclass per dataset (see seed_adapter.py). Must implement `iter_trials`."""
    name: str = "base"

    def iter_trials(self) -> Iterator[Trial]:
        raise NotImplementedError

    def binarize_label(self, trial: Trial, task: str) -> Optional[int]:
        """Return 0/1, or None to drop this trial for this task (e.g. SEED-IV's
        'neutral' class has no DEAP-comparable valence/arousal counterpart)."""
        raise NotImplementedError


def _bandpass_filters(fs):
    return {name: butter(4, band, btype="bandpass", fs=fs, output="sos") for name, band in BANDS.items()}


def resample_signal(signal, orig_fs, target_fs=128):
    if orig_fs == target_fs:
        return signal
    from math import gcd
    g = gcd(int(orig_fs), int(target_fs))
    up, down = int(target_fs) // g, int(orig_fs) // g
    return resample_poly(signal, up, down, axis=-1)


def compute_de_features_generic(signal, fs, window_sec=4):
    """signal: (n_channels, n_samples) at `fs` Hz -> (n_windows, n_channels*4 bands),
    channel-major band-minor flatten, matching the DEAP notebook's convention."""
    window_samples = int(window_sec * fs)
    filters = _bandpass_filters(fs)
    n_ch, n_samp = signal.shape
    n_win = n_samp // window_samples
    usable = n_win * window_samples
    de_per_band = []
    for band_name in BAND_ORDER:
        filtered = sosfiltfilt(filters[band_name], signal, axis=-1)
        windows = filtered[:, :usable].reshape(n_ch, n_win, window_samples)
        var = windows.var(axis=-1) + 1e-10
        de_per_band.append(0.5 * np.log(2 * np.pi * np.e * var))
    de_stack = np.stack(de_per_band, axis=-1)               # (n_ch, n_win, n_bands)
    de_stack = de_stack.transpose(1, 0, 2).reshape(n_win, n_ch * len(BAND_ORDER))
    return de_stack.astype(np.float32)


def intersect_channels(names_a: List[str], names_b: List[str]) -> List[str]:
    """Case/whitespace-insensitive name intersection, DEAP's channel order is the
    reference ordering for the common subset (so DEAP-trained-model feature
    columns and target-corpus feature columns line up 1:1)."""
    norm = lambda s: s.strip().upper()
    set_b = {norm(n) for n in names_b}
    common = [n for n in names_a if norm(n) in set_b]
    if len(common) < 4:
        raise ValueError(
            f"Only {len(common)} common channels found between the two montages "
            f"({names_a[:5]}... vs {names_b[:5]}...). Check that channel name "
            f"strings actually match (case/aliases like 'FP1' vs 'Fp1' are handled, "
            f"but e.g. 'T7' vs legacy '10-20' name 'T3' is NOT -- fix the adapter's "
            f"channel_names list if so)."
        )
    return common


def extract_common_channel_de(trial: Trial, common_channels: List[str], target_fs=128, window_sec=4):
    """Reindex `trial.signal` to `common_channels` order, resample to target_fs,
    and compute DE features restricted to that channel subset."""
    norm = lambda s: s.strip().upper()
    idx_map = {norm(n): i for i, n in enumerate(trial.channel_names)}
    try:
        idx = [idx_map[norm(c)] for c in common_channels]
    except KeyError as e:
        raise ValueError(f"Channel {e} not found in trial from subject {trial.subject_id}") from e
    sig = trial.signal[idx, :]
    sig = resample_signal(sig, trial.fs, target_fs)
    return compute_de_features_generic(sig, target_fs, window_sec)


def build_deap_common_channel_dataset(common_channels: List[str], subject_ids=None,
                                       data_dir="D:/EEG/deap-dataset/data_preprocessed_python",
                                       task="valence", label_threshold=5.0, window_sec=4):
    """Rebuild DEAP's DE features restricted to `common_channels` (same order),
    for retraining the DEAP-side model in a fair, channel-matched way."""
    import pickle
    from pathlib import Path

    subject_ids = subject_ids or list(range(1, 33))
    fs = 128
    baseline_samples = 3 * fs
    trial_samples = 60 * fs
    deap_idx = {n.upper(): i for i, n in enumerate(DEAP_CHANNELS)}
    col_idx = [deap_idx[c.upper()] for c in common_channels]
    label_col = {"valence": 0, "arousal": 1, "dominance": 2, "liking": 3}[task]

    X_list, y_list, groups_list = [], [], []
    for sid in subject_ids:
        fpath = Path(data_dir) / f"s{sid:02d}.dat"
        with open(fpath, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        data, labels = d["data"], d["labels"]
        for trial_idx in range(data.shape[0]):
            eeg = data[trial_idx][:32]
            baseline = eeg[:, :baseline_samples].mean(axis=-1, keepdims=True)
            sig = (eeg[:, baseline_samples:baseline_samples + trial_samples] - baseline).astype(np.float64)
            sig = sig[col_idx, :]
            de = compute_de_features_generic(sig, fs, window_sec)
            X_list.append(de)
            label_bin = int(labels[trial_idx, label_col] > label_threshold)
            y_list.append(np.full(de.shape[0], label_bin, dtype=np.int64))
            groups_list.append(np.full(de.shape[0], sid, dtype=np.int64))

    return (np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0), np.concatenate(groups_list, axis=0))


def evaluate_zero_shot(clf_pipeline, X_target, y_target, subject_ids_target):
    """Report overall + per-subject accuracy/F1/AUC for a model trained
    elsewhere and evaluated with zero further fitting on the target corpus."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    import pandas as pd

    y_pred = clf_pipeline.predict(X_target)
    y_prob = clf_pipeline.predict_proba(X_target)[:, 1]
    overall = {
        "accuracy": accuracy_score(y_target, y_pred),
        "f1_macro": f1_score(y_target, y_pred, average="macro"),
        "roc_auc": roc_auc_score(y_target, y_prob) if len(np.unique(y_target)) > 1 else np.nan,
        "n_windows": len(y_target),
        "n_subjects": len(np.unique(subject_ids_target)),
    }
    per_subject = []
    for sid in np.unique(subject_ids_target):
        mask = subject_ids_target == sid
        if len(np.unique(y_target[mask])) < 2:
            continue
        per_subject.append({
            "subject": sid,
            "accuracy": accuracy_score(y_target[mask], y_pred[mask]),
            "f1_macro": f1_score(y_target[mask], y_pred[mask], average="macro"),
            "roc_auc": roc_auc_score(y_target[mask], y_prob[mask]),
            "n_windows": mask.sum(),
        })
    return overall, pd.DataFrame(per_subject)
