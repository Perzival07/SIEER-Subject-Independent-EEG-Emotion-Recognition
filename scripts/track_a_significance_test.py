"""
Label-permutation significance test for the "AUC near chance" claim.

Sections 9-14 report ROC-AUC in the ~0.45-0.56 range under GroupKFold and call
it "near chance," but never formally tested whether the observed AUC differs
from a true empirical null. Khan et al. 2025 ("The Role of Review Process
Failures in Affective State Estimation," D:/EEG/papers_EEG/2508.02417v1.pdf)
show DEAP pipelines can look deceptively strong even on a label-randomized
"watermelon" control when evaluation is leaky -- the flip side, done properly
here, is to build a genuine null distribution *for this project's own already-
correct GroupKFold protocol* and see whether the observed AUC is actually
distinguishable from it.

Method: for each task, shuffle labels at the trial level, within each subject
(each subject keeps its own true set of 40 trial labels; only which of that
subject's trials gets which label is permuted), then propagate each shuffled
trial label to all 15 windows of that trial. This preserves (a) each
subject's real class balance, which matters because GroupKFold holds out
whole subjects, and (b) the within-trial correlation across a trial's 15
windows, so the null is built over the same effective units of independence
(trials nested in subjects) as the real evaluation. An earlier version of
this script shuffled all 19,200 window-level labels independently, which
treats windows as if they were exchangeable; they are not, since 15 windows
share one trial's label. That inflated the effective sample size behind the
null, made the null distribution far too narrow (std ~0.005), and inflated
every z-score, including dominance's. Rerun the exact same GroupKFold(5)
pipeline used elsewhere in this repo under each permutation and record the
mean fold AUC. Repeat N_PERM times to build an empirical null distribution,
then p = (1 + #{null_auc >= observed_auc}) / (N_PERM + 1) (Ojala & Garriga
2010's recommended +1 correction so p is never exactly 0).

Classifier: LogisticRegression only (not all 6 models). This is a documented,
pragmatic speed trade-off (same spirit as SVM_MAX_TRAIN_SAMPLES elsewhere in
this repo) -- section 9's ablation already showed inter-model AUC deltas on
the Full feature set are "well within one fold's standard deviation," so LR is
a fair, fast proxy for "is there any signal here at all," not a claim that LR
is the best model. N_PERM x 4 tasks x 5 folds x LogisticRegression fits are
parallelized across CPU cores with joblib.

Run: python scripts/track_a_significance_test.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_PERM = 300
TASK_ORDER = ["valence", "arousal", "dominance", "liking"]

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "processed_cache"
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

ext = np.load(CACHE_DIR / "deap_features_ext_v1.npz")
X_full = np.concatenate([ext["X_de"], ext["X_hjorth"], ext["X_faa"], ext["X_conn"]], axis=1)
groups_all = ext["groups"]
y_by_task = {t: ext[f"y_{t}"] for t in TASK_ORDER}


def make_trial_ids(groups, n_trials=40):
    """Reconstruct a per-subject trial id for each window.

    Not cached anywhere, but recoverable: build_extended_features.py loops
    `for trial_idx in range(40)` per subject and appends each trial's windows
    contiguously, so each subject's block of windows is exactly n_trials
    equal-sized, trial-ordered chunks.
    """
    trial_id = np.empty(len(groups), dtype=np.int64)
    for sid in np.unique(groups):
        idx = np.where(groups == sid)[0]
        assert len(idx) % n_trials == 0, (
            f"subject {sid}: {len(idx)} windows not divisible by {n_trials} trials"
        )
        win_per_trial = len(idx) // n_trials
        trial_id[idx] = np.repeat(np.arange(n_trials), win_per_trial)
    return trial_id


trial_ids_all = make_trial_ids(groups_all)


def permute_labels_trial_within_subject(y, groups, trial_ids, rng):
    """Shuffle trial-level labels within each subject, then broadcast each
    trial's (now-shuffled) label back out to all of that trial's windows.
    Preserves each subject's true class balance and the within-trial
    window-to-window label correlation; only breaks the trial-to-label
    assignment, which is the actual null hypothesis being tested.
    """
    y_perm = np.empty_like(y)
    for sid in np.unique(groups):
        mask = groups == sid
        sub_trials = trial_ids[mask]
        sub_y = y[mask]
        uniq_trials = np.unique(sub_trials)
        # one true label per trial, in a fixed trial-id order
        true_labels = np.array([sub_y[sub_trials == t][0] for t in uniq_trials])
        shuffled_labels = rng.permutation(true_labels)
        shuffled_map = dict(zip(uniq_trials, shuffled_labels))
        y_perm[mask] = np.array([shuffled_map[t] for t in sub_trials])
    return y_perm


def make_model():
    return LogisticRegression(max_iter=1000, random_state=SEED)


def mean_fold_auc(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    aucs = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", make_model())])
        pipe.fit(X[train_idx], y[train_idx])
        y_prob = pipe.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], y_prob))
    return float(np.mean(aucs)) if aucs else np.nan


def one_permutation(X, y, groups, trial_ids, perm_seed):
    rng = np.random.RandomState(perm_seed)
    y_shuffled = permute_labels_trial_within_subject(y, groups, trial_ids, rng)
    return mean_fold_auc(X, y_shuffled, groups)


def run_task(task):
    y = y_by_task[task]
    observed_auc = mean_fold_auc(X_full, y, groups_all)

    t0 = time.time()
    null_aucs = Parallel(n_jobs=-1)(
        delayed(one_permutation)(X_full, y, groups_all, trial_ids_all, SEED + i)
        for i in range(N_PERM)
    )
    null_aucs = np.array([a for a in null_aucs if not np.isnan(a)])
    elapsed = time.time() - t0

    p_value = (1 + np.sum(null_aucs >= observed_auc)) / (len(null_aucs) + 1)
    print(f"[{task}] observed_auc={observed_auc:.4f}  null: mean={null_aucs.mean():.4f} "
          f"std={null_aucs.std():.4f} min={null_aucs.min():.4f} max={null_aucs.max():.4f}  "
          f"p={p_value:.4f}  ({elapsed:.1f}s, n_perm={len(null_aucs)})")

    return {
        "task": task, "n_perm": len(null_aucs),
        "observed_auc": observed_auc,
        "null_auc_mean": null_aucs.mean(), "null_auc_std": null_aucs.std(),
        "null_auc_min": null_aucs.min(), "null_auc_max": null_aucs.max(),
        "z_score": (observed_auc - null_aucs.mean()) / (null_aucs.std() + 1e-12),
        "p_value": p_value,
        "elapsed_s": elapsed,
    }


if __name__ == "__main__":
    t0 = time.time()
    rows = [run_task(task) for task in TASK_ORDER]
    results = pd.DataFrame(rows)
    results.to_csv(TABLES_DIR / "track_a_significance_test.csv", index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. "
          f"Saved to {TABLES_DIR / 'track_a_significance_test.csv'}")
    print("\nSummary (alpha=0.05):")
    for _, r in results.iterrows():
        verdict = "SIGNIFICANT (but check effect size!)" if r.p_value < 0.05 else "not significant"
        print(f"  {r.task:<10} observed={r.observed_auc:.3f} null_mean={r.null_auc_mean:.3f} "
              f"z={r.z_score:+.2f} p={r.p_value:.4f} -> {verdict}")
