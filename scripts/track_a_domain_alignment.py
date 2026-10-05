"""
CORAL (CORrelation ALignment, Sun & Saenko 2016) feature alignment for Track A.

Motivation, drawn from the literature review in D:/EEG/papers_EEG/:
  - 2212.03176 (domain adaptation/generalization survey) and 2308.11635
    (DS-AGC, leave-one-subject-out on SEED) both report that a *linear,
    unsupervised* second-order-statistics alignment step -- CORAL -- closes a
    meaningful chunk of the subject-independent gap on EEG features before
    any adversarial/deep-learning machinery is needed (DS-AGC's own baseline
    table shows plain CORAL reaching ~65-70% under LOSO on SEED, vs. <60% for
    untransformed KNN/GFK/KPCA).
  - This project has not yet tried any per-fold feature-distribution alignment
    -- every experiment so far (ablation, nested CV, label cleanup,
    calibration) changes the features, model, or labels, but always trains and
    tests on the *same* (StandardScaler-only) feature distribution.

CORAL recipe (per GroupKFold fold, using the Full DE+Hjorth+FAA+Connectivity
feature set and the same 3 representative models as track_a_ablation.py):
  1. Fit StandardScaler on the training fold only (as elsewhere in this repo).
  2. Compute the training-fold ("source") and test-fold ("target") feature
     covariance matrices. Using the target's *features only* (no labels) to
     compute Ct is legitimate here: DEAP is a fixed, fully-observed dataset,
     so this is the standard transductive/unsupervised setting CORAL is
     designed for, not test-label leakage.
  3. Whiten the source features by Cs^(-1/2), then recolor by Ct^(1/2), so the
     aligned source distribution's second-order statistics match the target's.
  4. Fit the classifier on the aligned source features + real source labels;
     evaluate on the (unmodified, scaled) target features.

Matrix square roots are computed via eigendecomposition (covariance matrices
are symmetric PSD) with a small ridge (CORAL_EPS) added to the eigenvalues for
numerical stability, matching the original CORAL paper's regularization.

Run: python scripts/track_a_domain_alignment.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
CORAL_EPS = 1.0  # ridge added to covariance eigenvalues before inverting/rooting (per Sun & Saenko 2016)
TASK_ORDER = ["valence", "arousal", "dominance", "liking"]

CACHE_DIR = Path(r"D:/EEG/processed_cache")
RESULTS_DIR = Path(r"D:/EEG/results")
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

ext = np.load(CACHE_DIR / "deap_features_ext_v1.npz")
X_full = np.concatenate([ext["X_de"], ext["X_hjorth"], ext["X_faa"], ext["X_conn"]], axis=1)
groups_all = ext["groups"]
y_by_task = {t: ext[f"y_{t}"] for t in TASK_ORDER}


def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                  tree_method="hist", eval_metric="logloss", random_state=SEED),
    }


def _sym_matrix_power(C, power, eps=CORAL_EPS):
    """C^power for symmetric PSD C, via eigendecomposition with a ridge for stability."""
    eigval, eigvec = np.linalg.eigh(C + eps * np.eye(C.shape[0]))
    eigval = np.clip(eigval, 1e-8, None)
    return (eigvec * (eigval ** power)) @ eigvec.T


def coral_align(X_source, X_target):
    """Whiten X_source by its own covariance, recolor to X_target's covariance.
    Returns X_source_aligned (same shape as X_source); X_target is untouched."""
    Cs = np.cov(X_source, rowvar=False)
    Ct = np.cov(X_target, rowvar=False)
    Cs_inv_sqrt = _sym_matrix_power(Cs, -0.5)
    Ct_sqrt = _sym_matrix_power(Ct, 0.5)
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
    rows = []
    gkf = GroupKFold(n_splits=5)
    conditions = ["none", "coral"]
    total = len(conditions) * 3 * len(TASK_ORDER)
    done = 0

    for task in TASK_ORDER:
        y = y_by_task[task]
        for model_name in get_models():
            fold_metrics = {c: [] for c in conditions}
            for train_idx, test_idx in gkf.split(X_full, y, groups_all):
                scaler = StandardScaler().fit(X_full[train_idx])
                X_train = scaler.transform(X_full[train_idx])
                X_test = scaler.transform(X_full[test_idx])
                y_train, y_test = y[train_idx], y[test_idx]

                for cond in conditions:
                    X_train_use = coral_align(X_train, X_test) if cond == "coral" else X_train
                    model = get_models()[model_name]
                    model.fit(X_train_use, y_train)
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
    results.to_csv(TABLES_DIR / "track_a_domain_alignment.csv", index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. "
          f"Saved to {TABLES_DIR / 'track_a_domain_alignment.csv'}")

    piv = results.pivot_table(index=["task", "model"], columns="condition", values="roc_auc_mean")
    piv["delta_auc"] = piv["coral"] - piv["none"]
    print("\nCORAL vs. no alignment, ROC-AUC delta:")
    print(piv.round(4))
