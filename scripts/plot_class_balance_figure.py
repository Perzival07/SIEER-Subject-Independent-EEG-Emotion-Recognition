"""Publication-quality redraw of the class-balance figure (Figure 1).

Reads labels from processed_cache/deap_windows_4s_32subj_v2.npz and writes
results/figures/class_balance.{png,pdf}.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "processed_cache" / "deap_windows_4s_32subj_v2.npz"
FIGURES_DIR = ROOT / "results" / "figures"

TASKS = ["valence", "arousal", "dominance", "liking"]
COLORS = {"Low": "#2a78d6", "High": "#eb6834"}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def main():
    npz = np.load(CACHE_PATH, mmap_mode="r")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), sharey=True)

    for ax, task in zip(axes.flat, TASKS):
        y = np.asarray(npz[f"y_{task}"])
        counts = {"Low": int((y == 0).sum()), "High": int((y == 1).sum())}
        frac_high = counts["High"] / len(y)

        bars = ax.bar(list(counts), list(counts.values()), width=0.6,
                      color=list(COLORS.values()), zorder=2)
        for b, n in zip(bars, counts.values()):
            ax.text(b.get_x() + b.get_width() / 2, n + 250, f"{n:,}\n({n / len(y):.1%})",
                    ha="center", va="bottom", fontsize=12.5)

        ax.set_title(f"{task.capitalize()} ({frac_high:.1%} high)", fontweight="bold")
        ax.set_ylim(0, 19000)
        ax.yaxis.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    for ax in axes[:, 0]:
        ax.set_ylabel("Number of 4-s windows")
    for ax in axes[1, :]:
        ax.set_xlabel("Class")

    fig.tight_layout(h_pad=2.0)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "class_balance.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "class_balance.pdf", bbox_inches="tight")
    print("Saved", FIGURES_DIR / "class_balance.png")


if __name__ == "__main__":
    main()
