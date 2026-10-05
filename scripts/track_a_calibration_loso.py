"""
Per-subject calibration (personalization) experiment: does giving the model a
small amount of the TARGET subject's own labeled data change the picture,
compared to pure zero-shot subject-independence?

True LOSO across all 32 subjects x 4 tasks. For each held-out subject:
  1. Train a base XGBoost model on the other 31 subjects (same as every other
     experiment in this repo -- zero information about the held-out subject).
  2. Split the held-out subject's 40 trials (not windows -- splitting by trial
     keeps a trial's 15 windows together, so calibration/eval windows are never
     from the same trial) into a small calibration set (20% of trials, 8/40)
     and the remaining eval set (32/40), with a fixed per-subject seed.
  3. "Zero-shot" score: evaluate the base model directly on the eval set.
  4. "Calibrated" score: continue boosting ~40 extra trees on top of the base
     model (XGBoost's `xgb_model=` warm-start), fit ONLY on the calibration
     set, then evaluate on the SAME eval set.
  5. Compare zero-shot vs. calibrated on identical held-out data -- isolates
     the effect of calibration from everything else.

Subjects/tasks where the 8-trial calibration set happens to be single-class
(possible given the label imbalance already documented) are skipped for the
calibrated arm (warm-starting on single-class data is meaningless) but still
reported for the zero-shot arm, and the skip count is reported explicitly
rather than silently dropped.

Run: python scripts/track_a_calibration_loso.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
CALIB_FRACTION = 0.2
N_CALIB_TREES = 40
TASK_ORDER = ["valence", "arousal", "dominance", "liking"]
TASK_COL = {"valence": 0, "arousal": 1, "dominance": 2, "liking": 3}
N_WIN_PER_TRIAL = 15
N_TRIALS_PER_SUBJECT = 40

CACHE_DIR = Path(r"D:/EEG/processed_cache")
RESULTS_DIR = Path(r"D:/EEG/results")
TABLES_DIR = RESULTS_DIR / "tables"

ext = np.load(CACHE_DIR / "deap_features_ext_v1.npz")
X_full = np.concatenate([ext["X_de"], ext["X_hjorth"], ext["X_faa"], ext["X_conn"]], axis=1)
groups_all = ext["groups"]
y_by_task = {t: ext[f"y_{t}"] for t in TASK_ORDER}
SUBJECTS = np.unique(groups_all)


def trial_ids_for_subject(subject_mask):
    """Reconstruct 0..39 trial index within a subject's contiguous block of
    600 rows (40 trials x 15 windows), matching build_extended_features.py's
    iteration order exactly."""
    n = subject_mask.sum()
    assert n == N_TRIALS_PER_SUBJECT * N_WIN_PER_TRIAL, f"unexpected subject block size {n}"
    return np.repeat(np.arange(N_TRIALS_PER_SUBJECT), N_WIN_PER_TRIAL)


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
    }


def run_task(task):
    y_all = y_by_task[task]
    rows = []
    n_calib_skipped = 0

    for held_out in SUBJECTS:
        test_subject_mask = groups_all == held_out
        train_mask = ~test_subject_mask

        scaler = StandardScaler().fit(X_full[train_mask])
        X_train = scaler.transform(X_full[train_mask])
        y_train = y_all[train_mask]

        base_model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                    tree_method="hist", eval_metric="logloss", random_state=SEED)
        base_model.fit(X_train, y_train)

        trial_ids = trial_ids_for_subject(test_subject_mask)
        rng = np.random.RandomState(SEED + int(held_out))
        calib_trials = rng.choice(N_TRIALS_PER_SUBJECT,
                                   size=int(N_TRIALS_PER_SUBJECT * CALIB_FRACTION), replace=False)
        is_calib_window = np.isin(trial_ids, calib_trials)

        X_subject = scaler.transform(X_full[test_subject_mask])
        y_subject = y_all[test_subject_mask]
        X_calib, y_calib = X_subject[is_calib_window], y_subject[is_calib_window]
        X_eval, y_eval = X_subject[~is_calib_window], y_subject[~is_calib_window]

        if len(np.unique(y_eval)) < 2:
            continue  # can't score AUC meaningfully; skip this subject/task entirely

        y_pred_zs = base_model.predict(X_eval)
        y_prob_zs = base_model.predict_proba(X_eval)[:, 1]
        m_zeroshot = compute_metrics(y_eval, y_pred_zs, y_prob_zs)

        m_calibrated = None
        if len(np.unique(y_calib)) >= 2:
            calibrated_model = XGBClassifier(n_estimators=N_CALIB_TREES, max_depth=6, learning_rate=0.1,
                                              tree_method="hist", eval_metric="logloss", random_state=SEED)
            calibrated_model.fit(X_calib, y_calib, xgb_model=base_model.get_booster())
            y_pred_cal = calibrated_model.predict(X_eval)
            y_prob_cal = calibrated_model.predict_proba(X_eval)[:, 1]
            m_calibrated = compute_metrics(y_eval, y_pred_cal, y_prob_cal)
        else:
            n_calib_skipped += 1

        rows.append({
            "task": task, "subject": held_out,
            "n_calib_windows": len(y_calib), "n_eval_windows": len(y_eval),
            "zeroshot_accuracy": m_zeroshot["accuracy"], "zeroshot_f1": m_zeroshot["f1_macro"],
            "zeroshot_auc": m_zeroshot["roc_auc"],
            "calibrated_accuracy": m_calibrated["accuracy"] if m_calibrated else np.nan,
            "calibrated_f1": m_calibrated["f1_macro"] if m_calibrated else np.nan,
            "calibrated_auc": m_calibrated["roc_auc"] if m_calibrated else np.nan,
            "calibration_skipped_single_class": m_calibrated is None,
        })

    df = pd.DataFrame(rows)
    print(f"[{task}] {len(df)} subjects scored, {n_calib_skipped} skipped calibration "
          f"(single-class calib set)")
    valid = df.dropna(subset=["calibrated_auc"])
    print(f"  zero-shot  AUC: {df.zeroshot_auc.mean():.3f} +/- {df.zeroshot_auc.std():.3f}")
    print(f"  calibrated AUC: {valid.calibrated_auc.mean():.3f} +/- {valid.calibrated_auc.std():.3f} "
          f"(n={len(valid)})")
    if len(valid):
        from scipy import stats
        paired_zs = valid.loc[valid.index, "zeroshot_auc"]
        t_stat, p_val = stats.ttest_rel(valid.calibrated_auc, paired_zs)
        print(f"  paired t-test (calibrated vs zero-shot on same subjects): t={t_stat:.3f} p={p_val:.4f}")
    return df


if __name__ == "__main__":
    t0 = time.time()
    all_rows = [run_task(task) for task in TASK_ORDER]
    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(TABLES_DIR / "track_a_calibration_loso.csv", index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Saved to "
          f"{TABLES_DIR / 'track_a_calibration_loso.csv'}")

    summary = results.groupby("task").agg(
        zeroshot_auc_mean=("zeroshot_auc", "mean"),
        calibrated_auc_mean=("calibrated_auc", "mean"),
        n_subjects=("subject", "count"),
        n_skipped=("calibration_skipped_single_class", "sum"),
    )
    summary["delta"] = summary.calibrated_auc_mean - summary.zeroshot_auc_mean
    print("\nSummary:")
    print(summary)
