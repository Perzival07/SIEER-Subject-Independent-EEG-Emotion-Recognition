"""
Track A feature-set ablation: DE (baseline) vs. + Hjorth vs. + frontal/hemispheric
alpha asymmetry (FAA) vs. + PLV/coherence connectivity vs. full combined set,
evaluated with a single GroupKFold(5) pass (grouped by subject) across three
representative models (LogisticRegression as a fast linear baseline, RandomForest
and XGBoost as the strongest performers in the existing Track A results) and all
four DEAP label dimensions.

This intentionally does NOT repeat/nest the CV (that rigor is applied afterwards,
to the winning feature set only, in track_a_repeated_nested_cv.py) -- 6 feature
sets x 3 models x 4 tasks x 5 folds is already 360 fits and is meant to answer
"which features help" cheaply, not to produce the final headline numbers.

Run: python scripts/track_a_ablation.py
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
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TASK_ORDER = ["valence", "arousal", "dominance", "liking"]

npz = np.load(CACHE_DIR / "deap_features_ext_v1.npz")
X_de, X_hjorth, X_faa, X_conn = npz["X_de"], npz["X_hjorth"], npz["X_faa"], npz["X_conn"]
groups = npz["groups"]
y_by_task = {task: npz[f"y_{task}"] for task in TASK_ORDER}

FEATURE_SETS = {
    "DE_only": [X_de],
    "DE+Hjorth": [X_de, X_hjorth],
    "DE+FAA": [X_de, X_faa],
    "DE+Connectivity": [X_de, X_conn],
    "DE+Hjorth+FAA": [X_de, X_hjorth, X_faa],
    "Full (DE+Hjorth+FAA+Conn)": [X_de, X_hjorth, X_faa, X_conn],
}


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


def run_ablation():
    rows = []
    gkf = GroupKFold(n_splits=5)
    total = len(FEATURE_SETS) * 3 * len(TASK_ORDER)
    done = 0
    for feat_name, feat_arrays in FEATURE_SETS.items():
        X = np.concatenate(feat_arrays, axis=1)
        for task in TASK_ORDER:
            y = y_by_task[task]
            for model_name, model in get_models().items():
                fold_metrics = []
                fold_times = []
                for train_idx, test_idx in gkf.split(X, y, groups):
                    m = get_models()[model_name]
                    pipe = Pipeline([("scaler", StandardScaler()), ("clf", m)])
                    t0 = time.time()
                    pipe.fit(X[train_idx], y[train_idx])
                    fold_times.append(time.time() - t0)
                    y_pred = pipe.predict(X[test_idx])
                    y_prob = pipe.predict_proba(X[test_idx])[:, 1]
                    fold_metrics.append(compute_metrics(y[test_idx], y_pred, y_prob))
                dfm = pd.DataFrame(fold_metrics)
                rows.append({
                    "feature_set": feat_name, "n_features": X.shape[1],
                    "task": task, "model": model_name,
                    "accuracy_mean": dfm.accuracy.mean(), "accuracy_std": dfm.accuracy.std(),
                    "f1_mean": dfm.f1_macro.mean(), "f1_std": dfm.f1_macro.std(),
                    "roc_auc_mean": dfm.roc_auc.mean(), "roc_auc_std": dfm.roc_auc.std(),
                    "train_time_s": np.mean(fold_times),
                })
                done += 1
                print(f"[{done}/{total}] {feat_name:<28} {task:<10} {model_name:<18} "
                      f"acc={dfm.accuracy.mean():.3f}+/-{dfm.accuracy.std():.3f} "
                      f"f1={dfm.f1_macro.mean():.3f} auc={dfm.roc_auc.mean():.3f}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t0 = time.time()
    results = run_ablation()
    results.to_csv(TABLES_DIR / "track_a_ablation_results.csv", index=False)
    print(f"\nDone in {time.time()-t0:.1f}s. Saved to {TABLES_DIR / 'track_a_ablation_results.csv'}")

    best_per_task = (results.sort_values("accuracy_mean", ascending=False)
                      .groupby("task").first()[["feature_set", "model", "accuracy_mean", "f1_mean", "roc_auc_mean"]])
    print("\nBest feature_set/model per task (by accuracy):")
    print(best_per_task)
