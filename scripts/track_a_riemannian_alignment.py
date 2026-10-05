"""
Per-subject Euclidean Alignment (EA, He & Wu 2019) for Track A -- the
Riemannian/covariance-recentering sibling of CORAL that
D:/EEG/papers_EEG/literature_log.txt logged as "not yet implemented" after
2212.03176 (domain adaptation/generalization survey) and 2308.11635 (DS-AGC)
both named it alongside CORAL as a standard EEG cross-subject remedy.

Where this differs from track_a_domain_alignment.py's CORAL: CORAL aligns
already-extracted *feature* distributions per GroupKFold train/test fold
(a supervised-CV-fold-level operation). EA instead operates upstream, at the
raw-channel level, per SUBJECT, independent of any CV fold: each subject's own
windows are whitened by that subject's own mean spatial covariance before any
feature extraction happens, using no labels and no fold structure. This is the
standard EA recipe (He & Wu, "Transfer Learning for Brain-Computer Interfaces:
A Euclidean Space Data Alignment Approach", IEEE TBME 2019), implemented here
with plain numpy (eigendecomposition-based matrix square root, same
`_sym_matrix_power` pattern as track_a_domain_alignment.py -- no new
dependency):

  1. For each subject, compute every one of their own windows' raw spatial
     covariance C_i = X_i @ X_i.T / n_samples (32x32, no per-window mean
     subtraction, matching the original EA recipe), then average across all of
     that subject's windows to get the subject's reference covariance R_bar.
  2. Whiten every window of that subject by R_bar^(-1/2): X_i' = R_bar^(-1/2) @ X_i.
  3. Recompute Differential-Entropy (DE) band-power features from the
     whitened raw signal, using the same per-band sosfiltfilt + log-variance
     formula as scripts/build_extended_features.py::compute_de_features
     (vectorized here across all windows at once; not reimplemented from
     scratch -- same math, same channel-major/band-minor flatten order, so
     feature indices line up 1:1 with deap_features_ext_v1.npz's X_de).

Scope note: only the DE feature family (128-dim) is used for both conditions
here, not the Full (DE+Hjorth+FAA+Connectivity, 295-dim) set used by
track_a_ablation.py/track_a_domain_alignment.py. EA is fundamentally a
raw-channel-level linear transform; DE band-power is the feature family it
maps onto directly and unambiguously. Recomputing Hjorth/FAA/connectivity
consistently on EA-whitened signals is a separate, larger undertaking (FAA and
connectivity in particular depend on hemispheric channel identity and
cross-channel phase, which a full spatial whitening transform reshuffles) and
is out of scope for this pass -- this script isolates and tests EA's effect on
the one feature family it applies to most naturally.

Because R_bar is computed from a subject's OWN windows only (unsupervised,
label-free, using every window that subject has -- train or test fold alike),
applying it before GroupKFold splitting is not leakage: it never uses another
subject's data or any label, exactly like CORAL's use of test-fold *features*
(not labels) in track_a_domain_alignment.py.

Run: python scripts/track_a_riemannian_alignment.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
EA_EPS = 1e-6  # ridge added to covariance eigenvalues before inverting/rooting
TASK_ORDER = ["valence", "arousal", "dominance", "liking"]

CACHE_DIR = Path(r"D:/EEG/processed_cache")
RESULTS_DIR = Path(r"D:/EEG/results")
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS_PATH = CACHE_DIR / "deap_windows_4s_32subj_v2.npz"

FS = 128
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
BAND_ORDER = ["theta", "alpha", "beta", "gamma"]
_SOS_FILTERS = {name: butter(4, band, btype="bandpass", fs=FS, output="sos") for name, band in BANDS.items()}


def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                  tree_method="hist", eval_metric="logloss", random_state=SEED),
    }


def _sym_matrix_power(C, power, eps=EA_EPS):
    """C^power for symmetric PSD C, via eigendecomposition with a ridge for stability."""
    eigval, eigvec = np.linalg.eigh(C + eps * np.eye(C.shape[0]))
    eigval = np.clip(eigval, 1e-10, None)
    return (eigvec * (eigval ** power)) @ eigvec.T


def euclidean_align(X_raw, groups):
    """X_raw: (N, 32, 512) raw windows. Per-subject: R_bar = mean_i(X_i @ X_i.T / T),
    then X_i' = R_bar^(-1/2) @ X_i for every window of that subject. Returns array
    of the same shape, whitened in place per subject."""
    X_aligned = np.empty_like(X_raw)
    for sid in np.unique(groups):
        mask = groups == sid
        Xs = X_raw[mask]  # (n_i, 32, T)
        T = Xs.shape[-1]
        C_i = np.einsum("nij,nkj->nik", Xs, Xs) / T  # (n_i, 32, 32)
        R_bar = C_i.mean(axis=0)  # (32, 32)
        R_inv_sqrt = _sym_matrix_power(R_bar, -0.5)
        X_aligned[mask] = np.einsum("ij,njk->nik", R_inv_sqrt, Xs)
    return X_aligned


def compute_de_features_batch(X):
    """X: (N, 32, T) -> (N, 32*len(BAND_ORDER)) DE features, ch-major/band-minor
    flatten order, matching build_extended_features.py::compute_de_features."""
    de_per_band = []
    for band_name in BAND_ORDER:
        filtered = sosfiltfilt(_SOS_FILTERS[band_name], X, axis=-1)  # (N, 32, T)
        var = filtered.var(axis=-1) + 1e-10  # (N, 32)
        de_per_band.append(0.5 * np.log(2 * np.pi * np.e * var))
    de_stack = np.stack(de_per_band, axis=-1)  # (N, 32, n_bands)
    return de_stack.reshape(X.shape[0], -1).astype(np.float32)  # (N, 32*n_bands)


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
    }


def run():
    npz = np.load(WINDOWS_PATH)
    X_de_none = npz["X_de"]  # cached, unaligned DE features (matches deap_features_ext_v1.npz's X_de)
    groups_all = npz["groups"]
    y_by_task = {t: npz[f"y_{t}"] for t in TASK_ORDER}

    print("Computing per-subject Euclidean Alignment on raw windows...")
    t0 = time.time()
    X_raw_ea = euclidean_align(npz["X_raw"], groups_all)
    X_de_ea = compute_de_features_batch(X_raw_ea)
    print(f"  done in {time.time() - t0:.1f}s")

    conditions = {"none": X_de_none, "ea": X_de_ea}
    gkf = GroupKFold(n_splits=5)
    rows = []
    total = len(conditions) * 3 * len(TASK_ORDER)
    done = 0

    for task in TASK_ORDER:
        y = y_by_task[task]
        for model_name in get_models():
            fold_metrics = {c: [] for c in conditions}
            for train_idx, test_idx in gkf.split(X_de_none, y, groups_all):
                y_train, y_test = y[train_idx], y[test_idx]
                for cond, X in conditions.items():
                    scaler = StandardScaler().fit(X[train_idx])
                    X_train = scaler.transform(X[train_idx])
                    X_test = scaler.transform(X[test_idx])
                    model = get_models()[model_name]
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
                    fold_metrics[cond].append(compute_metrics(y_test, y_pred, y_prob))

            for cond in conditions:
                dfm = pd.DataFrame(fold_metrics[cond])
                rows.append({
                    "task": task, "model": model_name, "condition": cond,
                    "accuracy_mean": dfm.accuracy.mean(), "accuracy_std": dfm.accuracy.std(),
                    "f1_mean": dfm.f1_macro.mean(), "f1_std": dfm.f1_macro.std(),
                    "roc_auc_mean": dfm.roc_auc.mean(), "roc_auc_std": dfm.roc_auc.std(),
                })
                done += 1
                print(f"[{done}/{total}] {task:<10} {model_name:<18} {cond:<6} "
                      f"acc={dfm.accuracy.mean():.3f} f1={dfm.f1_macro.mean():.3f} "
                      f"auc={dfm.roc_auc.mean():.3f}+/-{dfm.roc_auc.std():.3f}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t0 = time.time()
    results = run()
    results.to_csv(TABLES_DIR / "track_a_riemannian_alignment.csv", index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. "
          f"Saved to {TABLES_DIR / 'track_a_riemannian_alignment.csv'}")

    piv = results.pivot_table(index=["task", "model"], columns="condition", values="roc_auc_mean")
    piv["delta_auc"] = piv["ea"] - piv["none"]
    print("\nEuclidean Alignment vs. no alignment, ROC-AUC delta (DE features only):")
    print(piv.round(4))
