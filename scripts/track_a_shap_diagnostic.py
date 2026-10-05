"""
SHAP interpretability diagnostic for Track A -- logged in
D:/EEG/papers_EEG/literature_log.txt (sensors-25-01827) as a cheap, never-run
diagnostic: does the near-chance classifier actually lean on
neurophysiologically plausible features (frontal alpha asymmetry, occipital
gamma power, etc.) or does its (weak) decision boundary look more like noise
spread evenly across all 295 features?

This is a DIAGNOSTIC, not an intervention -- it produces no AUC delta and
answers a different question: given that AUC is near chance (sections 9-16),
is whatever weak signal these models find at least plausible, or is the
near-chance performance also accompanied by an implausible/noisy attribution
pattern (which would suggest the small amount of "signal" some folds show is
itself an artifact rather than anything neurophysiological)?

Method: for each task, fit XGBoost (the model track_a_ablation_results.csv
shows as the strongest or tied-strongest Track A performer) on the Full
feature set (DE+Hjorth+FAA+Connectivity, same 295-dim set as
track_a_ablation.py) using ONE representative GroupKFold(5) train/test split
(the first fold; not all 5 -- SHAP over all folds/models is unnecessary cost
for a diagnostic, not a headline number), compute SHAP values on the held-out
fold with shap.TreeExplainer, and aggregate mean |SHAP| both per individual
feature (top 15) and per feature family (DE per band, Hjorth, FAA,
Connectivity) so the family-level pattern is visible even though no single
feature is expected to dominate.

Run: python scripts/track_a_shap_diagnostic.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
TASK_ORDER = ["valence", "arousal", "dominance", "liking"]

CACHE_DIR = Path(r"D:/EEG/processed_cache")
RESULTS_DIR = Path(r"D:/EEG/results")
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ext = np.load(CACHE_DIR / "deap_features_ext_v1.npz")
X_full = np.concatenate([ext["X_de"], ext["X_hjorth"], ext["X_faa"], ext["X_conn"]], axis=1)
groups_all = ext["groups"]
y_by_task = {t: ext[f"y_{t}"] for t in TASK_ORDER}

feature_names = np.concatenate([
    ext["de_feature_names"], ext["hjorth_feature_names"],
    ext["faa_feature_names"], ext["conn_feature_names"],
])
feature_family = np.concatenate([
    np.full(ext["de_feature_names"].shape[0], "DE"),
    np.full(ext["hjorth_feature_names"].shape[0], "Hjorth"),
    np.full(ext["faa_feature_names"].shape[0], "FAA"),
    np.full(ext["conn_feature_names"].shape[0], "Connectivity"),
])
assert len(feature_names) == X_full.shape[1] == len(feature_family)


def run_task(task):
    y = y_by_task[task]
    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(gkf.split(X_full, y, groups_all))

    scaler = StandardScaler().fit(X_full[train_idx])
    X_train = scaler.transform(X_full[train_idx])
    X_test = scaler.transform(X_full[test_idx])

    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                           tree_method="hist", eval_metric="logloss", random_state=SEED)
    model.fit(X_train, y[train_idx])

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test)  # (n_test, n_features), binary XGBoost -> single array
    mean_abs_shap = np.abs(sv).mean(axis=0)

    fam_df = pd.DataFrame({"feature": feature_names, "family": feature_family, "mean_abs_shap": mean_abs_shap})
    fam_summary = fam_df.groupby("family")["mean_abs_shap"].sum().sort_values(ascending=False)
    fam_summary_norm = (fam_summary / fam_summary.sum()).round(4)

    top15 = fam_df.sort_values("mean_abs_shap", ascending=False).head(15).reset_index(drop=True)

    print(f"\n[{task}] feature-family share of total |SHAP|:")
    for fam, share in fam_summary_norm.items():
        print(f"    {fam:<14} {share:.1%}")
    print(f"  top feature: {top15.iloc[0]['feature']} (mean|SHAP|={top15.iloc[0]['mean_abs_shap']:.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(fam_summary.index[::-1], fam_summary.values[::-1], color="C0")
    axes[0].set_xlabel("Sum of mean |SHAP|"); axes[0].set_title(f"{task}: feature-family importance")

    axes[1].barh(top15["feature"][::-1], top15["mean_abs_shap"][::-1], color="C1")
    axes[1].set_xlabel("mean |SHAP|"); axes[1].set_title(f"{task}: top 15 individual features")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"track_a_shap_summary_{task}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for _, r in top15.iterrows():
        rows.append({"task": task, "rank": len(rows) + 1, "feature": r["feature"],
                     "family": r["family"], "mean_abs_shap": r["mean_abs_shap"]})
    for fam, share in fam_summary_norm.items():
        rows.append({"task": task, "rank": None, "feature": f"__family_share__{fam}",
                      "family": fam, "mean_abs_shap": share})
    return rows


if __name__ == "__main__":
    all_rows = []
    for task in TASK_ORDER:
        all_rows.extend(run_task(task))
    out = pd.DataFrame(all_rows)
    out.to_csv(TABLES_DIR / "track_a_shap_diagnostic.csv", index=False)
    print(f"\nSaved to {TABLES_DIR / 'track_a_shap_diagnostic.csv'}")
    print(f"Figures saved to {FIGURES_DIR}/track_a_shap_summary_<task>.png")
