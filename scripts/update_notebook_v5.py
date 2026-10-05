# -*- coding: utf-8 -*-
"""
Appends section 18 (does CORAL stack on top of Euclidean Alignment?) to
DEAP_Comparative_Analysis.ipynb, same insert-before-trailing-empty-cell
pattern as update_notebook_v3.py / update_notebook_v4.py.
"""
import json
import uuid
from pathlib import Path

NB_PATH = Path(r"D:/EEG/DEAP_Comparative_Analysis.ipynb")


def cell_id():
    return uuid.uuid4().hex[:8]


def md(text):
    return {"cell_type": "markdown", "id": cell_id(), "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "id": cell_id(), "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


new_cells = []

# ---------------------------------------------------------------------------
new_cells.append(md("""## 18. Does CORAL stack on top of Euclidean Alignment?

17.1 found Euclidean Alignment (EA, raw-signal per-subject covariance
whitening) to be the strongest single alignment result in the project
(+0.044 mean AUC delta). 16.2 found CORAL (per-fold feature-level alignment)
gives a much smaller, inconsistent-sign effect (+0.0059). Both scripts'
docstrings flagged combining the two as a natural next step -- they touch
different parts of the pipeline (EA: upstream, raw-channel, per-subject;
CORAL: downstream, per-fold, feature-level) and are not obviously redundant.
`scripts/track_a_ea_coral_stacked.py` tests four conditions on the same
DE-only feature set as 17.1 (GroupKFold(5), 3 models, 4 tasks): `none`,
`coral` alone, `ea` alone, and `ea_coral` (EA first, then CORAL on the
resulting per-fold split)."""))

new_cells.append(code("""stack = pd.read_csv(TABLES_DIR / "track_a_ea_coral_stacked.csv")
piv = stack.pivot_table(index=["task", "model"], columns="condition", values="roc_auc_mean")
piv = piv[["none", "coral", "ea", "ea_coral"]]
piv["stack_vs_ea"] = piv["ea_coral"] - piv["ea"]
display(piv.style.format("{:.4f}"))

fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=(5 * len(TASK_ORDER), 4), sharey=True)
for ax, task in zip(axes, TASK_ORDER):
    sub = stack[stack.task == task]
    sns.barplot(data=sub, x="model", y="roc_auc_mean", hue="condition", ax=ax,
                hue_order=["none", "coral", "ea", "ea_coral"])
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_title(task); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=20)
    ax.set_ylabel("ROC AUC" if task == TASK_ORDER[0] else "")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "track_a_ea_coral_stacked_auc.png", dpi=150, bbox_inches="tight")
plt.show()

print("Mean AUC by condition (averaged across all 12 task/model combos):")
print(stack.groupby("condition")["roc_auc_mean"].mean().round(4).sort_values(ascending=False))
print(f"\\nea_coral beats ea alone in {(piv['stack_vs_ea'] > 0).sum()}/12 combos "
      f"(mean delta {piv['stack_vs_ea'].mean():+.4f})")
"""))

new_cells.append(md("""**Actual result: it does NOT stack -- CORAL makes EA modestly WORSE.**
Mean AUC by condition: EA alone 0.5232, EA+CORAL 0.5138, no-alignment baseline
0.4789, CORAL alone 0.4731 (CORAL alone is actually the *worst* of the four on
this DE-only feature set, consistent with 16.2's finding that CORAL's effect
is small and can go either direction). Adding CORAL on top of EA reduces AUC
in **10 of 12** task/model combinations (mean delta -0.0093) -- the two
liking/RandomForest and liking/XGBoost combos that were closest to flat in
17.1 are the only ones where the direction even flips sign, and even there the
change is small.

This is a clean, useful negative result rather than a wasted experiment: EA's
per-subject raw-signal whitening already removes most of the between-subject
covariance structure CORAL is also designed to correct for, so CORAL's
additional per-fold feature-level re-alignment is acting on residual structure
that is now mostly noise -- it has nothing systematic left to correct, and its
own alignment error (small as it is) becomes a net cost rather than a net
benefit. **Practical takeaway: use EA alone, not EA+CORAL** -- 17.1's numbers
stand as the best alignment result in this project, and this experiment
answers the "combine them" question definitively rather than leaving it as
untested future work."""))

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
insert_at = len(nb["cells"]) - 1
assert nb["cells"][insert_at]["cell_type"] == "markdown" and not "".join(nb["cells"][insert_at]["source"]).strip()
nb["cells"][insert_at:insert_at] = new_cells
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Inserted {len(new_cells)} cells at index {insert_at}. New total cell count: {len(nb['cells'])}")
