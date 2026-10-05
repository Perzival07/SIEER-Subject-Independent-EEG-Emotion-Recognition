"""
Track A, upgraded: (a) repeated GroupKFold (multiple seeds) instead of a single
5-fold pass, so accuracy/F1/AUC come with a real confidence interval rather than
one point estimate; and (b) nested CV (inner RandomizedSearchCV, GroupKFold(3))
for hyperparameter tuning of the two strongest/most tunable models (XGBoost,
RandomForest), to check whether the fixed hardcoded hyperparameters in the
notebook were leaving accuracy on the table or, as the ablation study suggests,
whether the ceiling here is set by label/feature separability rather than tuning.

Feature set: the "Full" (DE+Hjorth+FAA+Connectivity) set from the ablation study.
The ablation showed no feature-set variant reliably beats DE_only on AUC (all
hover within noise of chance, ~0.45-0.53), so this is not "the winner" so much
as "the most complete configuration" -- reported honestly, not cherry-picked.

Repeated GroupKFold: sklearn has no RepeatedGroupKFold, so each repeat uses a
random relabeling of the 32 subject IDs before calling GroupKFold(5); since
GroupKFold's greedy fold assignment is sensitive to the sorted order of group
labels, this yields a different subject-to-fold partition per repeat while
still respecting group integrity every time.

Run: python scripts/track_a_repeated_nested_cv.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

SEED_BASE = 42
N_REPEATS = 5
N_SPLITS = 5
SVM_MAX_TRAIN_SAMPLES = 8000
TASK_ORDER = ["valence", "arousal", "dominance", "liking"]

CACHE_DIR = Path(r"D:/EEG/processed_cache")
RESULTS_DIR = Path(r"D:/EEG/results")
TABLES_DIR = RESULTS_DIR / "tables"

npz = np.load(CACHE_DIR / "deap_features_ext_v1.npz")
X = np.concatenate([npz["X_de"], npz["X_hjorth"], npz["X_faa"], npz["X_conn"]], axis=1)
groups_orig = npz["groups"]
y_by_task = {task: npz[f"y_{task}"] for task in TASK_ORDER}
print("Full feature matrix:", X.shape)


def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=SEED_BASE),
        "kNN": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED_BASE),
        "SVM-RBF": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=SEED_BASE),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                  tree_method="hist", eval_metric="logloss", random_state=SEED_BASE),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                              early_stopping=True, random_state=SEED_BASE),
    }


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
    }


def repeated_group_kfold_splits(n_samples, groups, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED_BASE):
    rng = np.random.RandomState(seed)
    unique_groups = np.unique(groups)
    for r in range(n_repeats):
        perm = rng.permutation(unique_groups)
        remap = {orig: new for new, orig in enumerate(perm)}
        remapped = np.array([remap[g] for g in groups])
        gkf = GroupKFold(n_splits=n_splits)
        for fold_i, (train_idx, test_idx) in enumerate(gkf.split(np.zeros(n_samples), groups=remapped)):
            yield r, fold_i, train_idx, test_idx


def mean_ci95(values):
    values = np.asarray(values)
    m = values.mean()
    se = values.std(ddof=1) / np.sqrt(len(values))
    h = se * stats.t.ppf(0.975, len(values) - 1)
    return m, values.std(ddof=1), m - h, m + h


def run_repeated_cv():
    rows = []
    for task in TASK_ORDER:
        y = y_by_task[task]
        for model_name in ["LogisticRegression", "kNN", "RandomForest", "SVM-RBF", "XGBoost", "MLP"]:
            fold_metrics = []
            t0 = time.time()
            for r, fold_i, train_idx, test_idx in repeated_group_kfold_splits(len(X), groups_orig):
                model = get_models()[model_name]
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                if model_name == "SVM-RBF" and len(X_train) > SVM_MAX_TRAIN_SAMPLES:
                    rng = np.random.RandomState(SEED_BASE + r * 10 + fold_i)
                    sub = rng.choice(len(X_train), SVM_MAX_TRAIN_SAMPLES, replace=False)
                    X_train, y_train = X_train[sub], y_train[sub]
                pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                y_prob = pipe.predict_proba(X_test)[:, 1]
                m = compute_metrics(y_test, y_pred, y_prob)
                m["repeat"] = r
                m["fold"] = fold_i
                fold_metrics.append(m)
            dfm = pd.DataFrame(fold_metrics)
            elapsed = time.time() - t0

            row = {"task": task, "model": model_name, "n_folds_total": len(dfm)}
            for metric in ["accuracy", "f1_macro", "roc_auc"]:
                m, sd, lo, hi = mean_ci95(dfm[metric].dropna())
                row[f"{metric}_mean"] = m
                row[f"{metric}_std"] = sd
                row[f"{metric}_ci95_lo"] = lo
                row[f"{metric}_ci95_hi"] = hi
            row["chance_in_auc_ci"] = row["roc_auc_ci95_lo"] <= 0.5 <= row["roc_auc_ci95_hi"]
            row["elapsed_s"] = elapsed
            rows.append(row)
            print(f"[{task:<10}] {model_name:<18} "
                  f"acc={row['accuracy_mean']:.3f} (95%CI {row['accuracy_ci95_lo']:.3f}-{row['accuracy_ci95_hi']:.3f}) "
                  f"auc={row['roc_auc_mean']:.3f} (95%CI {row['roc_auc_ci95_lo']:.3f}-{row['roc_auc_ci95_hi']:.3f}) "
                  f"chance_in_CI={row['chance_in_auc_ci']} [{elapsed:.0f}s/{len(dfm)} folds]")
    return pd.DataFrame(rows)


def run_nested_cv():
    """Nested CV for XGBoost + RandomForest: outer GroupKFold(5) (single pass),
    inner GroupKFold(3) via RandomizedSearchCV for hyperparameter tuning."""
    param_dists = {
        "XGBoost": {
            "clf__n_estimators": [100, 200, 300, 500],
            "clf__max_depth": [3, 4, 6, 8],
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "clf__subsample": [0.6, 0.8, 1.0],
            "clf__colsample_bytree": [0.6, 0.8, 1.0],
        },
        "RandomForest": {
            "clf__n_estimators": [100, 200, 300, 500],
            "clf__max_depth": [None, 8, 16, 24],
            "clf__min_samples_leaf": [1, 2, 5, 10],
            "clf__max_features": ["sqrt", "log2", 0.5],
        },
    }
    base_models = {
        # n_jobs=1 here deliberately: RandomizedSearchCV(n_jobs=-1) already parallelizes
        # across the outer param x inner-fold grid; parallelizing the estimator too would
        # oversubscribe the 12 cores and typically ends up *slower*, not faster.
        "XGBoost": XGBClassifier(tree_method="hist", eval_metric="logloss", random_state=SEED_BASE, n_jobs=1),
        "RandomForest": RandomForestClassifier(n_jobs=1, random_state=SEED_BASE),
    }

    rows = []
    outer_gkf = GroupKFold(n_splits=N_SPLITS)
    for task in TASK_ORDER:
        y = y_by_task[task]
        for model_name, base_model in base_models.items():
            fold_metrics = []
            best_params_per_fold = []
            t0 = time.time()
            for fold_i, (train_idx, test_idx) in enumerate(outer_gkf.split(X, y, groups_orig)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                inner_groups = groups_orig[train_idx]
                inner_gkf = GroupKFold(n_splits=3)

                pipe = Pipeline([("scaler", StandardScaler()), ("clf", base_model)])
                search = RandomizedSearchCV(
                    pipe, param_distributions=param_dists[model_name], n_iter=8,
                    cv=list(inner_gkf.split(X_train, y_train, inner_groups)),
                    scoring="roc_auc", n_jobs=-1, random_state=SEED_BASE + fold_i, refit=True,
                )
                search.fit(X_train, y_train)
                y_pred = search.predict(X_test)
                y_prob = search.predict_proba(X_test)[:, 1]
                m = compute_metrics(y_test, y_pred, y_prob)
                fold_metrics.append(m)
                best_params_per_fold.append(search.best_params_)

            dfm = pd.DataFrame(fold_metrics)
            elapsed = time.time() - t0
            row = {"task": task, "model": model_name, "n_folds": N_SPLITS, "elapsed_s": elapsed}
            for metric in ["accuracy", "f1_macro", "roc_auc"]:
                row[f"{metric}_mean"] = dfm[metric].mean()
                row[f"{metric}_std"] = dfm[metric].std()
            rows.append(row)
            print(f"[nested] [{task:<10}] {model_name:<15} "
                  f"acc={row['accuracy_mean']:.3f}+/-{row['accuracy_std']:.3f} "
                  f"auc={row['roc_auc_mean']:.3f}+/-{row['roc_auc_std']:.3f} [{elapsed:.0f}s]")
            print(f"           best params per fold: {best_params_per_fold}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 20, "REPEATED GROUPKFOLD (fixed hyperparameters)", "=" * 20)
    repeated_results = run_repeated_cv()
    repeated_results.to_csv(TABLES_DIR / "track_a_repeated_groupkfold.csv", index=False)

    print("\n" + "=" * 20, "NESTED CV (hyperparameter tuning)", "=" * 20)
    nested_results = run_nested_cv()
    nested_results.to_csv(TABLES_DIR / "track_a_nested_cv.csv", index=False)

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
