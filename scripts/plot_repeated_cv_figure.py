"""Publication-quality redraw of the repeated GroupKFold CI figure (Figure 2).

Reads results/tables/track_a_repeated_groupkfold.csv and writes
results/figures/track_a_repeated_cv_auc_ci.{png,pdf}.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"

MODELS = {
    "LogisticRegression": "Logistic Regression",
    "kNN": "k-NN",
    "SVM-RBF": "SVM (RBF)",
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
    "MLP": "MLP",
}
TASKS = ["valence", "arousal", "dominance", "liking"]
COLOR = "#2a78d6"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def main():
    df = pd.read_csv(TABLES_DIR / "track_a_repeated_groupkfold.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True, sharey=True)
    y = np.arange(len(MODELS))[::-1]

    for ax, task in zip(axes.flat, TASKS):
        sub = df[df["task"] == task].set_index("model").loc[list(MODELS)]
        mean = sub["roc_auc_mean"].to_numpy()
        lo, hi = sub["roc_auc_ci95_lo"].to_numpy(), sub["roc_auc_ci95_hi"].to_numpy()
        excludes = ~sub["chance_in_auc_ci"].to_numpy()

        ax.axvline(0.5, color="#222222", linestyle="--", linewidth=1, zorder=1)
        ax.hlines(y, lo, hi, color=COLOR, linewidth=2, zorder=2)
        for yy, l, h in zip(y, lo, hi):
            ax.vlines([l, h], yy - 0.15, yy + 0.15, color=COLOR, linewidth=1.5, zorder=2)
        ax.scatter(mean[~excludes], y[~excludes], s=48, facecolor="white",
                   edgecolor=COLOR, linewidth=1.8, zorder=3)
        ax.scatter(mean[excludes], y[excludes], s=58, color=COLOR,
                   edgecolor=COLOR, linewidth=1.8, zorder=3)
        for yy, m, h in zip(y, mean, hi):
            ax.text(h + 0.004, yy, f"{m:.3f}", va="center", fontsize=8.5, color="#444444",
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.5), zorder=4)

        ax.set_title(task.capitalize(), fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(MODELS.values())
        ax.set_xlim(0.42, 0.63)
        ax.set_xticks(np.arange(0.45, 0.63, 0.05))
        ax.set_ylim(-0.6, len(MODELS) - 0.4)
        ax.xaxis.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

    for ax in axes[1, :]:
        ax.set_xlabel("ROC-AUC (mean and 95% CI, 5 × 5 = 25 folds)")

    handles = [
        Line2D([], [], marker="o", color=COLOR, markerfacecolor=COLOR, markersize=7,
               linewidth=2, label="95% CI excludes chance"),
        Line2D([], [], marker="o", color=COLOR, markerfacecolor="white", markersize=7,
               markeredgewidth=1.8, linewidth=2, label="95% CI includes chance"),
        Line2D([], [], color="#222222", linestyle="--", linewidth=1, label="Chance (AUC = 0.5)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "track_a_repeated_cv_auc_ci.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "track_a_repeated_cv_auc_ci.pdf", bbox_inches="tight")
    print("Saved", FIGURES_DIR / "track_a_repeated_cv_auc_ci.png")


if __name__ == "__main__":
    main()
