"""
Builds a per-window array of raw continuous SAM ratings (valence, arousal,
dominance, liking), row-aligned 1:1 with the extended feature cache
(deap_features_ext_v1.npz) so label-cleanup experiments can reuse those
features without recomputing anything. Alignment is guaranteed by iterating
subjects/trials in the exact same order as build_extended_features.py and
verified against that cache's `groups` array before saving.

Run: python scripts/build_raw_ratings.py
"""
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

DATA_DIR = Path(r"D:/EEG/deap-dataset/data_preprocessed_python")
CACHE_DIR = Path(r"D:/EEG/processed_cache")
OUT_PATH = CACHE_DIR / "deap_raw_ratings_v1.npz"

FS = 128
TRIAL_SEC = 60
WINDOW_SEC = 4
N_WIN_PER_TRIAL = (TRIAL_SEC * FS) // (WINDOW_SEC * FS)  # 15
ALL_SUBJECT_IDS = list(range(1, 33))
TASK_ORDER = ["valence", "arousal", "dominance", "liking"]


def build(subject_ids):
    ratings_list, groups_list = [], []
    for sid in tqdm(subject_ids, desc="Subjects"):
        fpath = DATA_DIR / f"s{sid:02d}.dat"
        with open(fpath, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        labels = d["labels"]  # (40, 4) raw continuous ratings
        for trial_idx in range(labels.shape[0]):
            ratings_list.append(np.tile(labels[trial_idx], (N_WIN_PER_TRIAL, 1)))  # (15, 4)
            groups_list.append(np.full(N_WIN_PER_TRIAL, sid, dtype=np.int64))
    return np.concatenate(ratings_list, axis=0), np.concatenate(groups_list, axis=0)


if __name__ == "__main__":
    raw_ratings, groups = build(ALL_SUBJECT_IDS)
    print("raw_ratings:", raw_ratings.shape, "groups:", groups.shape)

    ext = np.load(CACHE_DIR / "deap_features_ext_v1.npz")
    assert np.array_equal(groups, ext["groups"]), "row alignment with deap_features_ext_v1.npz is broken!"
    print("Row alignment with deap_features_ext_v1.npz verified OK.")

    np.savez(OUT_PATH, raw_ratings=raw_ratings, groups=groups, columns=np.array(TASK_ORDER))
    print(f"Saved to {OUT_PATH}")
