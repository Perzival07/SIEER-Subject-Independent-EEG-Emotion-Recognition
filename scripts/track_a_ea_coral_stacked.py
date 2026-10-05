"""
Does CORAL stack on top of Euclidean Alignment? Follow-up to
track_a_riemannian_alignment.py (17.1, the strongest single alignment result
in the project: +0.044 mean AUC delta) and track_a_domain_alignment.py (16.2,
CORAL alone: +0.0059 mean AUC delta). Both scripts' own docstrings flagged
combining the two as a natural next step, not yet run.

Design: EA (He & Wu 2019) operates upstream, per-subject, at the raw-channel
level (whitens every window of a subject by that subject's own mean spatial
covariance, no fold structure, no labels). CORAL (Sun & Saenko 2016) operates
downstream, per-fold, at the feature level (whitens the training fold's
already-extracted features and recolors them to match the test fold's
covariance). These touch different parts of the pipeline, so stacking them
(EA first, on the raw signal -> extract DE features -> CORAL on the resulting
per-fold train/test split) is a coherent, non-redundant combination to test,
not a duplicate of either alone.

Four conditions, all on the DE-only feature set (128-dim) so this is an
apples-to-apples comparison across all four -- NOT the Full 295-dim set used
by track_a_ablation.py/track_a_domain_alignment.py's headline numbers, exactly
matching track_a_riemannian_alignment.py's own scope decision:
  - none:      raw DE features, StandardScaler only.
  - coral:     DE features, per-fold CORAL alignment (train whitened/recolored
               to test), no EA.
  - ea:        per-subject EA-whitened raw signal -> DE features, StandardScaler
               only, no CORAL. (Recomputed here identically to
               track_a_riemannian_alignment.py's "ea" condition, so this script
               is self-contained and its own "ea" row is a direct replication
               check against that script's results.)
  - ea_coral:  per-subject EA-whitened raw signal -> DE features, THEN per-fold
               CORAL alignment on top.

Same GroupKFold(5) / 3-model (LogisticRegression/RandomForest/XGBoost) setup
as every other Track A alignment script in this repo.

Run: python scripts/track_a_ea_coral_stacked.py
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
EA_EPS = 1e-6
CORAL_EPS = 1.0
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


def _sym_matrix_power(C, power, eps):
    eigval, eigvec = np.linalg.eigh(C + eps * np.eye(C.shape[0]))
    eigval = np.clip(eigval, 1e-10, None)
    return (eigvec * (eigval ** power)) @ eigvec.T


def euclidean_align(X_raw, groups):
    X_aligned = np.empty_like(X_raw)
    for sid in np.unique(groups):
        mask = groups == sid
        Xs = X_raw[mask]
        T = Xs.shape[-1]
        C_i = np.einsum("nij,nkj->nik", Xs, Xs) / T
        R_bar = C_i.mean(axis=0)
        R_inv_sqrt = _sym_matrix_power(R_bar, -0.5, EA_EPS)
        X_aligned[mask] = np.einsum("ij,njk->nik", R_inv_sqrt, Xs)
    return X_aligned


def compute_de_features_batch(X):
    de_per_band = []
    for band_name in BAND_ORDER:
        filtered = sosfiltfilt(_SOS_FILTERS[band_name], X, axis=-1)
        var = filtered.var(axis=-1) + 1e-10
        de_per_band.append(0.5 * np.log(2 * np.pi * np.e * var))
    de_stack = np.stack(de_per_band, axis=-1)
    return de_stack.reshape(X.shape[0], -1).astype(np.float32)


def coral_align(X_source, X_target):
    Cs = np.cov(X_source, rowvar=False)
    Ct = np.cov(X_target, rowvar=False)
    Cs_inv_sqrt = _sym_matrix_power(Cs, -0.5, CORAL_EPS)
    Ct_sqrt = _sym_matrix_power(Ct, 0.5, CORAL_EPS)
    Xs_mean = X_source.mean(axis=0)
    return (X_source - Xs_mean) @ Cs_inv_sqrt @ Ct_sqrt + Xs_mean


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
    X_de_none = npz["X_de"]
    groups_all = npz["groups"]
    y_by_task = {t: npz[f"y_{t}"] for t in TASK_ORDER}

    print("Computing per-subject Euclidean Alignment on raw windows...")
    t0 = time.time()
    X_raw_ea = euclidean_align(npz["X_raw"], groups_all)
    X_de_ea = compute_de_features_batch(X_raw_ea)
    print(f"  done in {time.time() - t0:.1f}s")

    conditions = ["none", "coral", "ea", "ea_coral"]
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

                scaler_none = StandardScaler().fit(X_de_none[train_idx])
                Xtr_none = scaler_none.transform(X_de_none[train_idx])
                Xte_none = scaler_none.transform(X_de_none[test_idx])

                scaler_ea = StandardScaler().fit(X_de_ea[train_idx])
                Xtr_ea = scaler_ea.transform(X_de_ea[train_idx])
                Xte_ea = scaler_ea.transform(X_de_ea[test_idx])

                feats = {
                    "none": (Xtr_none, Xte_none),
                    "coral": (coral_align(Xtr_none, Xte_none), Xte_none),
                    "ea": (Xtr_ea, Xte_ea),
                    "ea_coral": (coral_align(Xtr_ea, Xte_ea), Xte_ea),
                }
                for cond in conditions:
                    Xtr, Xte = feats[cond]
                    model = get_models()[model_name]
                    model.fit(Xtr, y_train)
                    y_pred = model.predict(Xte)
                    y_prob = model.predict_proba(Xte)[:, 1]
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
                print(f"[{done}/{total}] {task:<10} {model_name:<18} {cond:<9} "
                      f"acc={dfm.accuracy.mean():.3f} f1={dfm.f1_macro.mean():.3f} "
                      f"auc={dfm.roc_auc.mean():.3f}+/-{dfm.roc_auc.std():.3f}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t0 = time.time()
    results = run()
    results.to_csv(TABLES_DIR / "track_a_ea_coral_stacked.csv", index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. "
          f"Saved to {TABLES_DIR / 'track_a_ea_coral_stacked.csv'}")

    piv = results.pivot_table(index=["task", "model"], columns="condition", values="roc_auc_mean")
    piv["stack_vs_ea"] = piv["ea_coral"] - piv["ea"]
    piv["stack_vs_coral"] = piv["ea_coral"] - piv["coral"]
    piv["stack_vs_none"] = piv["ea_coral"] - piv["none"]
    print("\nAll four conditions, ROC-AUC, plus stacking deltas:")
    print(piv.round(4))

    print("\nMean AUC by condition (averaged across all 12 task/model combos):")
    print(results.groupby("condition")["roc_auc_mean"].mean().round(4).sort_values(ascending=False))
    print(f"\nea_coral beats ea alone in {(piv['stack_vs_ea'] > 0).sum()}/12 combos")
    print(f"ea_coral beats coral alone in {(piv['stack_vs_coral'] > 0).sum()}/12 combos")
