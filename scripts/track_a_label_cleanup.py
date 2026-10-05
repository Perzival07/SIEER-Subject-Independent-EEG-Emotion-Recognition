"""
Label-cleanup ablation: does the way raw SAM ratings get turned into binary
labels matter, independent of features/models? Three variants, same
GroupKFold(5), same Full (DE+Hjorth+FAA+Connectivity) feature set, same 3
representative models as track_a_ablation.py, so this is directly comparable
to that table:

  - raw_threshold  : current notebook convention, label = rating > 5.0 (fixed
                     cutoff on the raw 1-9 scale, same for every subject)
  - per_subject_z  : z-score each subject's own 40 ratings (per task) before
                     thresholding at 0 -- corrects for individual differences
                     in how people use a 1-9 self-report scale (anchoring
                     bias), a documented issue with SAM-style ratings
  - drop_middle    : keep only trials rated <=3 or >=6, label = rating>=6,
                     drop the ambiguous middle -- trades sample size for
                     (hopefully) less label noise

Run: python scripts/track_a_label_cleanup.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
CACHE_DIR = Path(r"D:/EEG/processed_cache")
RESULTS_DIR = Path(r"D:/EEG/results")
TABLES_DIR = RESULTS_DIR / "tables"

TASK_ORDER = ["valence", "arousal", "dominance", "liking"]
TASK_COL = {"valence": 0, "arousal": 1, "dominance": 2, "liking": 3}

ext = np.load(CACHE_DIR / "deap_features_ext_v1.npz")
X_full = np.concatenate([ext["X_de"], ext["X_hjorth"], ext["X_faa"], ext["X_conn"]], axis=1)
groups_all = ext["groups"]

ratings_npz = np.load(CACHE_DIR / "deap_raw_ratings_v1.npz")
raw_ratings = ratings_npz["raw_ratings"]  # (19200, 4)
assert np.array_equal(ratings_npz["groups"], groups_all), "feature/rating row alignment broken"


def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                  tree_method="hist", eval_metric="logloss", random_state=SEED),
    }


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
    }


def make_labels(variant, task):
    """Returns (mask, y) where mask selects which of the 19200 rows to use
    (all rows for raw_threshold/per_subject_z, a subset for drop_middle)."""
    col = TASK_COL[task]
    ratings = raw_ratings[:, col]
    mask = np.ones(len(ratings), dtype=bool)

    if variant == "raw_threshold":
        y = (ratings > 5.0).astype(np.int64)

    elif variant == "per_subject_z":
        y = np.zeros(len(ratings), dtype=np.int64)
        for sid in np.unique(groups_all):
            sub_mask = groups_all == sid
            sub_ratings = ratings[sub_mask]
            z = (sub_ratings - sub_ratings.mean()) / (sub_ratings.std() + 1e-8)
            y[sub_mask] = (z > 0).astype(np.int64)

    elif variant == "drop_middle":
        mask = (ratings <= 3.0) | (ratings >= 6.0)
        y = (ratings >= 6.0).astype(np.int64)

    else:
        raise ValueError(variant)
    return mask, y


def run():
    rows = []
    gkf = GroupKFold(n_splits=5)
    variants = ["raw_threshold", "per_subject_z", "drop_middle"]
    total = len(variants) * 3 * len(TASK_ORDER)
    done = 0

    for variant in variants:
        for task in TASK_ORDER:
            mask, y_full = make_labels(variant, task)
            X = X_full[mask]
            y = y_full[mask]
            groups = groups_all[mask]
            n_subjects = len(np.unique(groups))

            for model_name in get_models():
                fold_metrics = []
                for train_idx, test_idx in gkf.split(X, y, groups):
                    if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                        continue  # can happen for drop_middle on a small/imbalanced fold
                    model = get_models()[model_name]
                    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
                    pipe.fit(X[train_idx], y[train_idx])
                    y_pred = pipe.predict(X[test_idx])
                    y_prob = pipe.predict_proba(X[test_idx])[:, 1]
                    fold_metrics.append(compute_metrics(y[test_idx], y_pred, y_prob))

                dfm = pd.DataFrame(fold_metrics)
                rows.append({
                    "variant": variant, "task": task, "model": model_name,
                    "n_windows": len(y), "n_subjects": n_subjects,
                    "class_balance_high_frac": y.mean(),
                    "n_folds_valid": len(dfm),
                    "accuracy_mean": dfm.accuracy.mean(), "accuracy_std": dfm.accuracy.std(),
                    "f1_mean": dfm.f1_macro.mean(), "f1_std": dfm.f1_macro.std(),
                    "roc_auc_mean": dfm.roc_auc.mean(), "roc_auc_std": dfm.roc_auc.std(),
                })
                done += 1
                print(f"[{done}/{total}] {variant:<15} {task:<10} {model_name:<18} "
                      f"n={len(y):<6} acc={dfm.accuracy.mean():.3f} "
                      f"f1={dfm.f1_macro.mean():.3f} auc={dfm.roc_auc.mean():.3f}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t0 = time.time()
    results = run()
    results.to_csv(TABLES_DIR / "track_a_label_cleanup_results.csv", index=False)
    print(f"\nDone in {time.time()-t0:.1f}s")

    best = results.loc[results.groupby("task").roc_auc_mean.idxmax()]
    baseline = results[results.variant == "raw_threshold"].groupby("task").roc_auc_mean.max()
    print("\nBest AUC per task vs. raw_threshold's best AUC:")
    for task in TASK_ORDER:
        b = best[best.task == task].iloc[0]
        print(f"  {task:<10} best={b.variant}/{b.model} auc={b.roc_auc_mean:.3f} "
              f"vs raw_threshold best={baseline[task]:.3f} (delta={b.roc_auc_mean - baseline[task]:+.3f})")
