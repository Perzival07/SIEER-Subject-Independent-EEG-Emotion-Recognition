"""
Orchestrator: train DE+XGBoost on DEAP restricted to the DEAP<->target common
channel subset, evaluate zero-shot on the target corpus, and (if the target
corpus has enough subjects) also run a same-corpus LOSO baseline for the
"optimism gap" comparison table.

This is the piece that answers the reviewer-trust question directly:
    within-DEAP GroupKFold accuracy/AUC  vs.  DEAP->target zero-shot accuracy/AUC
      vs.  target-only LOSO accuracy/AUC

Run (after downloading SEED-IV from Kaggle and filling in the path):
    python scripts/cross_corpus/run_cross_corpus.py --seed_root "D:/EEG/seed-iv-dataset"
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from base import (DEAP_CHANNELS, build_deap_common_channel_dataset, evaluate_zero_shot,
                   extract_common_channel_de, intersect_channels)
from seed_adapter import SeedIVAdapter

RESULTS_DIR = Path(r"D:/EEG/results/tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_classifier():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                               tree_method="hist", eval_metric="logloss", random_state=42)),
    ])


def collect_target_dataset(adapter, common_channels, task="valence"):
    X_list, y_list, subj_list = [], [], []
    for trial in adapter.iter_trials():
        label = adapter.binarize_label(trial, task)
        if label is None:
            continue
        de = extract_common_channel_de(trial, common_channels)
        X_list.append(de)
        y_list.append(np.full(de.shape[0], label, dtype=np.int64))
        subj_list.append(np.full(de.shape[0], hash(trial.subject_id) % (10 ** 8), dtype=np.int64))
    return np.concatenate(X_list), np.concatenate(y_list), np.concatenate(subj_list)


def loso_baseline(X, y, subjects):
    rows = []
    for held_out in np.unique(subjects):
        train_mask = subjects != held_out
        test_mask = ~train_mask
        if len(np.unique(y[test_mask])) < 2:
            continue
        clf = get_classifier()
        clf.fit(X[train_mask], y[train_mask])
        y_pred = clf.predict(X[test_mask])
        y_prob = clf.predict_proba(X[test_mask])[:, 1]
        rows.append({
            "held_out_subject": held_out,
            "accuracy": accuracy_score(y[test_mask], y_pred),
            "f1_macro": f1_score(y[test_mask], y_pred, average="macro"),
            "roc_auc": roc_auc_score(y[test_mask], y_prob),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_root", required=True, help="Path to extracted SEED-IV dataset root")
    ap.add_argument("--task", default="valence", choices=["valence"],
                     help="SEED-IV only maps cleanly onto valence (see seed_adapter.binarize_label)")
    ap.add_argument("--deap_subjects", type=int, default=32)
    args = ap.parse_args()

    t0 = time.time()
    print("Loading SEED-IV adapter (will fail loudly if paths are wrong)...")
    adapter = SeedIVAdapter(args.seed_root)

    common_channels = intersect_channels(DEAP_CHANNELS, adapter.channel_names)
    print(f"Common DEAP<->SEED-IV channels ({len(common_channels)}): {common_channels}")

    print("Collecting SEED-IV features (this re-extracts DE per trial, may take a while)...")
    X_seed, y_seed, subj_seed = collect_target_dataset(adapter, common_channels, args.task)
    print(f"SEED-IV: X={X_seed.shape}, subjects={len(np.unique(subj_seed))}, "
          f"class balance={np.bincount(y_seed)}")

    print(f"Rebuilding DEAP DE features restricted to the {len(common_channels)} common channels...")
    X_deap, y_deap, groups_deap = build_deap_common_channel_dataset(
        common_channels, subject_ids=list(range(1, args.deap_subjects + 1)), task=args.task)
    print(f"DEAP (common-channel): X={X_deap.shape}")

    print("\n--- (1) Within-DEAP subject-independent GroupKFold baseline ---")
    gkf = GroupKFold(n_splits=5)
    within_rows = []
    for train_idx, test_idx in gkf.split(X_deap, y_deap, groups_deap):
        clf = get_classifier()
        clf.fit(X_deap[train_idx], y_deap[train_idx])
        y_pred = clf.predict(X_deap[test_idx])
        y_prob = clf.predict_proba(X_deap[test_idx])[:, 1]
        within_rows.append({
            "accuracy": accuracy_score(y_deap[test_idx], y_pred),
            "f1_macro": f1_score(y_deap[test_idx], y_pred, average="macro"),
            "roc_auc": roc_auc_score(y_deap[test_idx], y_prob),
        })
    within_df = pd.DataFrame(within_rows)
    print(within_df.mean())

    print("\n--- (2) DEAP -> SEED-IV zero-shot transfer (train on ALL of DEAP, never see SEED-IV) ---")
    final_clf = get_classifier()
    final_clf.fit(X_deap, y_deap)
    zero_shot_overall, zero_shot_per_subject = evaluate_zero_shot(final_clf, X_seed, y_seed, subj_seed)
    print(zero_shot_overall)

    print("\n--- (3) SEED-IV-only LOSO baseline (upper bound: same-corpus, subject-independent) ---")
    seed_loso_df = loso_baseline(X_seed, y_seed, subj_seed)
    print(seed_loso_df.mean(numeric_only=True))

    summary = pd.DataFrame([
        {"setting": "within_DEAP_groupkfold", **within_df.mean().to_dict()},
        {"setting": "DEAP_to_SEEDIV_zero_shot", "accuracy": zero_shot_overall["accuracy"],
         "f1_macro": zero_shot_overall["f1_macro"], "roc_auc": zero_shot_overall["roc_auc"]},
        {"setting": "SEEDIV_only_LOSO", **seed_loso_df.mean(numeric_only=True).to_dict()},
    ])
    summary.to_csv(RESULTS_DIR / "cross_corpus_deap_to_seediv_summary.csv", index=False)
    zero_shot_per_subject.to_csv(RESULTS_DIR / "cross_corpus_deap_to_seediv_per_subject.csv", index=False)
    seed_loso_df.to_csv(RESULTS_DIR / "cross_corpus_seediv_loso.csv", index=False)
    print(f"\nSaved summary to {RESULTS_DIR / 'cross_corpus_deap_to_seediv_summary.csv'}")
    print(f"Done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
