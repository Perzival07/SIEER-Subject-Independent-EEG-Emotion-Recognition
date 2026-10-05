# -*- coding: utf-8 -*-
"""
Appends two more sections (14: label-cleanup ablation, 15: per-subject
calibration/personalization) to DEAP_Comparative_Analysis.ipynb, inserted
before the trailing empty markdown cell, same pattern as update_notebook.py.
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
new_cells.append(md("""## 14. Can better labels close the gap? A label-cleanup ablation

Sections 9-13 held the label definition fixed (`rating > 5.0` on the raw 1-9 SAM
scale) and varied features/models/CV instead. This section asks the opposite
question: given the **same** Full feature set and the same three representative
models, does a different way of turning continuous ratings into binary labels
change anything?

- **`raw_threshold`** (baseline, same as everywhere else in this notebook): a
  single fixed cutoff at 5.0 for every subject.
- **`per_subject_z`**: z-score each subject's own 40 ratings before
  thresholding at 0 — corrects for individual differences in how people use a
  1-9 self-report scale (anchoring bias is a well-documented issue with SAM-style
  ratings). Note this also happens to rebalance the classes substantially
  (e.g. valence goes from ~21% "high" under the fixed threshold to ~47% under
  per-subject z-scoring) — worth remembering when comparing accuracy numbers,
  since a more balanced label set removes the "just predict the majority class"
  shortcut that was likely inflating raw accuracy in earlier sections.
- **`drop_middle`**: keep only trials rated <=3 or >=6, discarding the
  ambiguous middle — trades sample size (roughly 60-75% of windows survive,
  depending on task) for potentially less label noise.

Run via `scripts/track_a_label_cleanup.py` (GroupKFold(5), 3 variants x 3
models x 4 tasks) -- feature/rating row alignment with the section-9 cache is
verified before use (`scripts/build_raw_ratings.py`)."""))

new_cells.append(code("""label_cleanup = pd.read_csv(TABLES_DIR / "track_a_label_cleanup_results.csv")
display(label_cleanup.sort_values(["task", "roc_auc_mean"], ascending=[True, False])
        .style.format({"class_balance_high_frac": "{:.2f}", "accuracy_mean": "{:.3f}",
                        "f1_mean": "{:.3f}", "roc_auc_mean": "{:.3f}"}))

fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=(5.5 * len(TASK_ORDER), 4), sharey=True)
for ax, task in zip(axes, TASK_ORDER):
    sub = label_cleanup[label_cleanup.task == task]
    sns.barplot(data=sub, x="variant", y="roc_auc_mean", hue="model", ax=ax)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_title(task); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=20)
    ax.set_ylabel("ROC AUC" if task == TASK_ORDER[0] else "")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "track_a_label_cleanup_auc.png", dpi=150, bbox_inches="tight")
plt.show()

best_per_task = label_cleanup.loc[label_cleanup.groupby("task").roc_auc_mean.idxmax()]
baseline_per_task = label_cleanup[label_cleanup.variant == "raw_threshold"].groupby("task").roc_auc_mean.max()
print("Best AUC anywhere in the label-cleanup sweep, per task, vs. raw_threshold's best AUC:")
for task in TASK_ORDER:
    b = best_per_task[best_per_task.task == task].iloc[0]
    print(f"  {task:<10} best={b.variant}/{b.model} auc={b.roc_auc_mean:.3f} n={b.n_windows} "
          f"vs raw_threshold best={baseline_per_task[task]:.3f} "
          f"(delta={b.roc_auc_mean - baseline_per_task[task]:+.3f})")
"""))

new_cells.append(md("""**Read this table's accuracy column with the class-balance column right next to
it.** `per_subject_z`'s accuracy will very likely look *lower* than
`raw_threshold`'s for valence/arousal — that is the imbalance shortcut going
away, not the model getting worse. **ROC AUC is the fair comparison here**,
since it does not reward majority-class guessing. If `per_subject_z` and/or
`drop_middle` do not reliably beat `raw_threshold` on AUC either, that rules
out "the fixed-threshold labels were the bottleneck" the same way sections 9-10
ruled out features and hyperparameters -- another specific, testable hypothesis
for the weak subject-independent signal, checked rather than assumed."""))

# ---------------------------------------------------------------------------
new_cells.append(md("""## 15. Per-subject calibration (personalization) — does a little of the target subject's own data help?

Every result so far, including sections 9-14, evaluates strict subject
independence: zero information about the test subject is available at train
time. That is the hardest and most publishable claim ("generalizes to a brand
new person"), but it is also a very different question from "how well could
this work as a personalized system where a new user does a short calibration
session first" — which is closer to how most real BCI/affective-computing
products actually operate, and where between-subject EEG variability (a
well-documented major confound) gets partially factored out.

**Design** (`scripts/track_a_calibration_loso.py`, true Leave-One-Subject-Out —
32 folds, not 5 — since XGBoost is cheap enough to make this tractable, ~13s/subject):
for each held-out subject, a base XGBoost model is trained on the other 31
subjects exactly as before. The held-out subject's own 40 trials are then split
by **trial** (not window, so a trial's 15 windows never cross the split) into an
80/20 eval/calibration set. Two scores are computed on the **identical** eval
set: (a) the base model's zero-shot prediction, and (b) a **calibrated** model
that continues boosting ~40 extra trees warm-started from the base model, fit
only on that subject's own 8-trial calibration set. Subjects whose 8-trial
calibration set happens to be single-class (given the label imbalance already
documented) are skipped for the calibrated arm and reported as such, not
silently dropped."""))

new_cells.append(code("""calib = pd.read_csv(TABLES_DIR / "track_a_calibration_loso.csv")

summary_rows = []
for task in TASK_ORDER:
    sub = calib[calib.task == task]
    valid = sub.dropna(subset=["calibrated_auc"])
    from scipy import stats
    if len(valid) > 1:
        t_stat, p_val = stats.ttest_rel(valid.calibrated_auc, valid.zeroshot_auc)
    else:
        t_stat, p_val = np.nan, np.nan
    summary_rows.append({
        "task": task, "n_subjects": len(sub), "n_calibrated": len(valid),
        "n_skipped_single_class": int(sub.calibration_skipped_single_class.sum()),
        "zeroshot_auc_mean": sub.zeroshot_auc.mean(), "zeroshot_auc_std": sub.zeroshot_auc.std(),
        "calibrated_auc_mean": valid.calibrated_auc.mean(), "calibrated_auc_std": valid.calibrated_auc.std(),
        "delta": valid.calibrated_auc.mean() - sub.loc[valid.index, "zeroshot_auc"].mean(),
        "paired_t": t_stat, "paired_p": p_val,
    })
calib_summary = pd.DataFrame(summary_rows)
display(calib_summary.style.format({c: "{:.3f}" for c in calib_summary.columns
                                     if c not in ("task", "n_subjects", "n_calibrated", "n_skipped_single_class")}))

fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=(4.5 * len(TASK_ORDER), 4.5), sharey=True)
for ax, task in zip(axes, TASK_ORDER):
    sub = calib[calib.task == task].dropna(subset=["calibrated_auc"])
    for _, r in sub.iterrows():
        ax.plot([0, 1], [r.zeroshot_auc, r.calibrated_auc], color="gray", alpha=0.4, linewidth=1)
    ax.scatter([0] * len(sub), sub.zeroshot_auc, color="C0", zorder=3, label="zero-shot")
    ax.scatter([1] * len(sub), sub.calibrated_auc, color="C1", zorder=3, label="calibrated")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["zero-shot", "calibrated"])
    ax.set_title(task); ax.set_ylabel("ROC AUC (per subject)" if task == TASK_ORDER[0] else "")
    if task == TASK_ORDER[0]:
        ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "track_a_calibration_per_subject.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

new_cells.append(md("""**Interpretation**: each gray line is one held-out subject; if calibration
genuinely helps, most lines should slope upward (zero-shot -> calibrated) and
the paired t-test's p-value should be small. If the lines are a roughly even
mix of up/down with a non-significant p-value, an 8-trial calibration set
(with only ~40 extra boosted trees) is not enough signal to meaningfully
personalize this particular feature/model pipeline -- which would itself be
informative: it would suggest the ceiling found in sections 9-14 is not purely
a between-subject-variability problem solvable with a *little* calibration
data, and that either a larger calibration set, a different personalization
mechanism (e.g. per-subject feature normalization, fine-tuning more/earlier
layers in the deep models rather than just adding trees), or the DEAP labels'
inherent noise remains the binding constraint."""))

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
insert_at = len(nb["cells"]) - 1
assert nb["cells"][insert_at]["cell_type"] == "markdown" and not "".join(nb["cells"][insert_at]["source"]).strip()
nb["cells"][insert_at:insert_at] = new_cells
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Inserted {len(new_cells)} cells at index {insert_at}. New total cell count: {len(nb['cells'])}")
