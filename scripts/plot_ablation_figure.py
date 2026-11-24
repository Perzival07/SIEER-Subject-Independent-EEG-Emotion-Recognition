"""Publication-quality redraw of the feature-set ablation figure (Figure 3).

Reads results/tables/classical_ml_feature_set_ablation_results.csv and writes
results/figures/classical_ml_feature_set_ablation_roc_auc.{png,pdf}.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"

FEATURE_SETS = {
    "DE_only": "DE",
    "DE+Hjorth": "DE\n+Hjorth",
    "DE+FAA": "DE\n+FAA",
    "DE+Connectivity": "DE\n+Conn",
    "DE+Hjorth+FAA": "DE+Hjorth\n+FAA",
    "Full (DE+Hjorth+FAA+Conn)": "Full",
}
MODELS = {
    "LogisticRegression": ("Logistic Regression", "#2a78d6"),
    "RandomForest": ("Random Forest", "#eb6834"),
    "XGBoost": ("XGBoost", "#1baf7a"),
}
TASKS = ["valence", "arousal", "dominance", "liking"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def main():
    df = pd.read_csv(TABLES_DIR / "classical_ml_feature_set_ablation_results.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharey=True)
    x = np.arange(len(FEATURE_SETS))
    width = 0.26

    for ax, task in zip(axes.flat, TASKS):
        sub = df[df["task"] == task]
        for i, (model, (label, color)) in enumerate(MODELS.items()):
            rows = sub[sub["model"] == model].set_index("feature_set").loc[list(FEATURE_SETS)]
            ax.bar(x + (i - 1) * width, rows["roc_auc_mean"], width * 0.92,
                   yerr=rows["roc_auc_std"], color=color, label=label,
                   error_kw=dict(elinewidth=0.8, capsize=2, ecolor="#444444"),
                   zorder=2)
        ax.axhline(0.5, color="#222222", linestyle="--", linewidth=1, zorder=3,
                   label="Chance (AUC = 0.5)")
        ax.set_title(task.capitalize(), fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(FEATURE_SETS.values())
        ax.set_ylim(0.35, 0.65)
        ax.yaxis.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

    for ax in axes[:, 0]:
        ax.set_ylabel("ROC-AUC (mean ± SD, 5 folds)")
    for ax in axes[1, :]:
        ax.set_xlabel("Feature set")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "classical_ml_feature_set_ablation_roc_auc.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "classical_ml_feature_set_ablation_roc_auc.pdf", bbox_inches="tight")
    print("Saved", FIGURES_DIR / "classical_ml_feature_set_ablation_roc_auc.png")


if __name__ == "__main__":
    main()
