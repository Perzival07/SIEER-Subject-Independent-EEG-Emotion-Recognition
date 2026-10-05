"""
Builds a 50%-overlap variant of the Track B raw-window dataset (stride =
WINDOW_SAMPLES // 2 instead of WINDOW_SAMPLES), using the exact same baseline
normalization, per-window per-channel z-scoring, and label threshold as the
notebook's non-overlapping `deap_windows_4s_32subj_v2.npz` cache, so the two are
directly comparable.

IMPORTANT caveat (see notebook markdown / discussion for the full writeup):
adjacent overlapping windows share up to 50% of their raw samples, so they are
NOT independent observations. GroupKFold/GroupShuffleSplit by subject prevents
*subject* leakage, but does not make within-subject overlapping windows
independent of each other -- reported variance/CIs on the overlap dataset should
be read as optimistic relative to the non-overlap dataset even when subjects
never cross the train/test boundary. We report both variants side by side rather
than only the (larger-looking) overlap numbers.

Run: python scripts/build_overlap_windows.py
"""
import pickle
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

DATA_DIR = Path(r"D:/EEG/deap-dataset/data_preprocessed_python")
CACHE_DIR = Path(r"D:/EEG/processed_cache")
OUT_PATH = CACHE_DIR / "deap_windows_4s_overlap50_32subj_v1.npz"

FS = 128
N_EEG_CHANNELS = 32
BASELINE_SEC = 3
BASELINE_SAMPLES = BASELINE_SEC * FS
TRIAL_SEC = 60
TRIAL_SAMPLES = TRIAL_SEC * FS
WINDOW_SEC = 4
WINDOW_SAMPLES = WINDOW_SEC * FS          # 512
STRIDE_SAMPLES = WINDOW_SAMPLES // 2      # 256 -> 50% overlap

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


def make_overlap_windows(trial_norm, window_samples=WINDOW_SAMPLES, stride=STRIDE_SAMPLES):
    """(32, 7680) -> per-window per-channel standardized windows (n_windows, 32, window_samples),
    with 50% overlap between consecutive windows (stride < window_samples)."""
    n_ch, n_samp = trial_norm.shape
    n_win = (n_samp - window_samples) // stride + 1
    idx = np.arange(window_samples)[None, :] + stride * np.arange(n_win)[:, None]  # (n_win, window_samples)
    windows = trial_norm[:, idx]              # (32, n_win, window_samples)
    windows = windows.transpose(1, 0, 2)      # (n_win, 32, window_samples)
    mean = windows.mean(axis=-1, keepdims=True)
    std = windows.std(axis=-1, keepdims=True) + 1e-8
    return ((windows - mean) / std).astype(np.float32)


def build(subject_ids):
    X_raw_list, groups_list = [], []
    y_lists = {task: [] for task in TASK_ORDER}
    for sid in tqdm(subject_ids, desc="Subjects"):
        data, labels = load_subject(sid)
        task_labels = {task: (labels[:, col] > LABEL_THRESHOLD).astype(np.int64)
                       for task, col in LABEL_COLUMNS.items()}
        for trial_idx in range(data.shape[0]):
            trial_norm = baseline_normalize(data[trial_idx])
            raw_wins = make_overlap_windows(trial_norm)
            n_win = raw_wins.shape[0]
            X_raw_list.append(raw_wins)
            for task in TASK_ORDER:
                y_lists[task].append(np.full(n_win, task_labels[task][trial_idx], dtype=np.int64))
            groups_list.append(np.full(n_win, sid, dtype=np.int64))

    X_raw = np.concatenate(X_raw_list, axis=0)
    groups = np.concatenate(groups_list, axis=0)
    y_by_task = {task: np.concatenate(y_lists[task], axis=0) for task in TASK_ORDER}
    return X_raw, y_by_task, groups


if __name__ == "__main__":
    t0 = time.time()
    X_raw, y_by_task, groups = build(ALL_SUBJECT_IDS)
    np.savez(OUT_PATH, X_raw=X_raw, groups=groups, **{f"y_{t}": y for t, y in y_by_task.items()})
    print(f"Saved to {OUT_PATH} in {time.time()-t0:.1f}s")
    print("X_raw:", X_raw.shape, "(expect (32*40*29, 32, 512) = (37120, 32, 512))")
    print("windows per trial:", (TRIAL_SAMPLES - WINDOW_SAMPLES) // STRIDE_SAMPLES + 1)
    print("non-overlap equivalent windows per trial: 15  ->  overlap gives",
          (TRIAL_SAMPLES - WINDOW_SAMPLES) // STRIDE_SAMPLES + 1, "(~2x, as expected from 50% overlap)")
