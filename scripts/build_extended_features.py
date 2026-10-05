"""
Builds the extended Track-A feature cache (DE + Hjorth + frontal/hemispheric alpha
asymmetry + PLV/coherence connectivity) for all 32 DEAP subjects, mirroring the
existing notebook's preprocessing exactly (same FS, baseline correction, band
filters, window size, label threshold) so results are directly comparable to the
already-cached `deap_windows_4s_32subj_v2.npz`.

Run: python scripts/build_extended_features.py
"""
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

import rich_features as rf

DATA_DIR = Path(r"D:/EEG/deap-dataset/data_preprocessed_python")
CACHE_DIR = Path(r"D:/EEG/processed_cache")
OUT_PATH = CACHE_DIR / "deap_features_ext_v1.npz"

FS = 128
N_EEG_CHANNELS = 32
BASELINE_SEC = 3
BASELINE_SAMPLES = BASELINE_SEC * FS
TRIAL_SEC = 60
TRIAL_SAMPLES = TRIAL_SEC * FS
WINDOW_SEC = 4
WINDOW_SAMPLES = WINDOW_SEC * FS

BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
BAND_ORDER = ["theta", "alpha", "beta", "gamma"]
_SOS_FILTERS = {name: butter(4, band, btype="bandpass", fs=FS, output="sos") for name, band in BANDS.items()}

LABEL_THRESHOLD = 5.0
LABEL_COLUMNS = {"valence": 0, "arousal": 1, "dominance": 2, "liking": 3}
TASK_ORDER = list(LABEL_COLUMNS.keys())

ALL_SUBJECT_IDS = list(range(1, 33))


def load_subject(subject_id):
    fpath = DATA_DIR / f"s{subject_id:02d}.dat"
    with open(fpath, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    return d["data"], d["labels"]


def baseline_normalize(trial_all_channels):
    eeg = trial_all_channels[:N_EEG_CHANNELS]
    baseline = eeg[:, :BASELINE_SAMPLES]
    signal = eeg[:, BASELINE_SAMPLES:BASELINE_SAMPLES + TRIAL_SAMPLES]
    baseline_mean = baseline.mean(axis=-1, keepdims=True)
    return (signal - baseline_mean).astype(np.float64)


def compute_de_features(trial_norm):
    n_ch, n_samp = trial_norm.shape
    n_win = n_samp // WINDOW_SAMPLES
    usable = n_win * WINDOW_SAMPLES
    de_per_band = []
    for band_name in BAND_ORDER:
        filtered = sosfiltfilt(_SOS_FILTERS[band_name], trial_norm, axis=-1)
        windows = filtered[:, :usable].reshape(n_ch, n_win, WINDOW_SAMPLES)
        var = windows.var(axis=-1) + 1e-10
        de_per_band.append(0.5 * np.log(2 * np.pi * np.e * var))
    de_stack = np.stack(de_per_band, axis=-1)
    de_stack = de_stack.transpose(1, 0, 2).reshape(n_win, n_ch * len(BAND_ORDER))
    return de_stack.astype(np.float32)


def build(subject_ids):
    X_de_list, X_hjorth_list, X_faa_list, X_conn_list, groups_list = [], [], [], [], []
    y_lists = {task: [] for task in TASK_ORDER}

    for sid in tqdm(subject_ids, desc="Subjects"):
        data, labels = load_subject(sid)
        task_labels = {task: (labels[:, col] > LABEL_THRESHOLD).astype(np.int64)
                       for task, col in LABEL_COLUMNS.items()}
        for trial_idx in range(data.shape[0]):
            trial_norm = baseline_normalize(data[trial_idx])
            de_feats = compute_de_features(trial_norm)
            hjorth_feats = rf.compute_hjorth(trial_norm)
            faa_feats = rf.compute_alpha_asymmetry(de_feats)
            conn_feats = rf.compute_connectivity(trial_norm)
            n_win = de_feats.shape[0]

            X_de_list.append(de_feats)
            X_hjorth_list.append(hjorth_feats)
            X_faa_list.append(faa_feats)
            X_conn_list.append(conn_feats)
            for task in TASK_ORDER:
                y_lists[task].append(np.full(n_win, task_labels[task][trial_idx], dtype=np.int64))
            groups_list.append(np.full(n_win, sid, dtype=np.int64))

    out = dict(
        X_de=np.concatenate(X_de_list, axis=0),
        X_hjorth=np.concatenate(X_hjorth_list, axis=0),
        X_faa=np.concatenate(X_faa_list, axis=0),
        X_conn=np.concatenate(X_conn_list, axis=0),
        groups=np.concatenate(groups_list, axis=0),
        de_feature_names=np.array([f"de_{band}_{ch}" for ch in rf.DEAP_CHANNELS for band in BAND_ORDER]),
        hjorth_feature_names=np.array(rf.hjorth_feature_names()),
        faa_feature_names=np.array(rf.ASYM_PAIR_NAMES),
        conn_feature_names=np.array(rf.connectivity_feature_names()),
    )
    for task, y in y_lists.items():
        out[f"y_{task}"] = np.concatenate(y, axis=0)
    return out


if __name__ == "__main__":
    t0 = time.time()
    result = build(ALL_SUBJECT_IDS)
    np.savez(OUT_PATH, **result)
    print(f"Saved to {OUT_PATH} in {time.time() - t0:.1f}s")
    print("X_de:", result["X_de"].shape)
    print("X_hjorth:", result["X_hjorth"].shape)
    print("X_faa:", result["X_faa"].shape)
    print("X_conn:", result["X_conn"].shape)
    print("groups (unique subjects):", len(np.unique(result["groups"])))
    print("windows total:", len(result["groups"]))
