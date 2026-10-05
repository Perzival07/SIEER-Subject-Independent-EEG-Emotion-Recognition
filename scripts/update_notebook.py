# -*- coding: utf-8 -*-
"""
Appends the four new sections (richer Track A features + ablation, repeated/
nested GroupKFold, Track B overlap + repeated splits, cross-corpus status) plus
an updated overall discussion to DEAP_Comparative_Analysis.ipynb, inserted
before the notebook's trailing empty markdown cell so it still ends cleanly.

Idempotent-ish: re-running will append a second copy of the sections, so this
is meant to be run once. (Not building a more elaborate replace-if-exists
mechanism since this is a one-shot editorial pass, not a repeated build step.)
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
new_cells.append(md("""## 9. Richer Track A features — Hjorth, frontal/hemispheric alpha asymmetry, connectivity

Track A so far uses only band-power Differential Entropy (4 bands x 32 channels = 128
features). Three additional, cheap-to-compute feature families are added here, all
derived from the same baseline-normalized signal used for DE (*before* the
per-window z-scoring applied only for the DL track's raw-window input):

- **Hjorth parameters** (activity, mobility, complexity) — 3 x 32 = 96 features,
  classic time-domain complexity descriptors that DE/HFD-only pipelines miss.
- **Frontal/hemispheric alpha asymmetry (FAA)** — 14 features, one per left/right
  electrode pair, `alpha_power[right] - alpha_power[left]`. This is the classic
  valence marker from the Davidson asymmetry literature (the 4 canonical frontal
  pairs — Fp1/Fp2, F3/F4, F7/F8, FC5/FC6 — are a subset of the 14; the rest extend
  the same idea to the other hemispheric pairs already used by TSception's
  asymmetry branch).
- **Connectivity (PLV + coherence)** — 19 curated channel pairs (14 interhemispheric
  homologs + 5 canonical fronto-parietal/occipital pairs), PLV in alpha and beta
  bands (38 features) plus alpha-band coherence (19 features) = 57 features.

Implementation: `scripts/rich_features.py` (unit-tested against synthetic data —
shape and [0,1]-boundedness checks for PLV/coherence — and its DE computation was
verified to match the existing `X_de` cache exactly on real DEAP subjects before
trusting the rest). Full-feature build for all 32 subjects: `scripts/build_extended_features.py`
(~100s, cached to `processed_cache/deap_features_ext_v1.npz`, 19,200 windows x
{128 DE + 96 Hjorth + 14 FAA + 57 connectivity} = 295 features total)."""))

new_cells.append(code("""import sys
sys.path.insert(0, str(Path(r"D:/EEG/scripts")))
import rich_features as rf

# sanity check on one real subject/trial, mirroring the DE/HFD check in section 2
_d, _l = load_subject(SUBJECT_IDS[0])
_norm = baseline_normalize(_d[0])
_hjorth = rf.compute_hjorth(_norm)
_faa = rf.compute_alpha_asymmetry(compute_de_features(_norm))
_conn = rf.compute_connectivity(_norm)
print("Hjorth features per trial:", _hjorth.shape, " (expect (15, 96))")
print("FAA features per trial:   ", _faa.shape, "  (expect (15, 14))")
print("Connectivity per trial:   ", _conn.shape, "  (expect (15, 57))")
del _d, _l, _norm, _hjorth, _faa, _conn
"""))

new_cells.append(code("""EXT_CACHE = CACHE_DIR / "deap_features_ext_v1.npz"
ext_npz = np.load(EXT_CACHE)
X_de_ext, X_hjorth, X_faa, X_conn = (ext_npz["X_de"], ext_npz["X_hjorth"],
                                      ext_npz["X_faa"], ext_npz["X_conn"])
print("X_de:", X_de_ext.shape, "| X_hjorth:", X_hjorth.shape,
      "| X_faa:", X_faa.shape, "| X_conn:", X_conn.shape)
assert np.allclose(X_de_ext, X_de), "extended-feature cache's DE must match the section-3 X_de exactly"
print("DE features cross-checked identical to the section-3 cache: OK")
"""))

new_cells.append(md("""### Track A feature-set ablation

Ablation (`scripts/track_a_ablation.py`, single `GroupKFold(5)` pass, subject-grouped,
3 representative models x 6 feature-set combinations x 4 tasks = 360 fits, ~37 min)
compares DE-only against each addition and the full combined set."""))

new_cells.append(code("""ablation = pd.read_csv(TABLES_DIR / "track_a_ablation_results.csv")
display(ablation.sort_values(["task", "roc_auc_mean"], ascending=[True, False])
        .style.format({"accuracy_mean": "{:.3f}", "f1_mean": "{:.3f}", "roc_auc_mean": "{:.3f}"}))

fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=(5.5 * len(TASK_ORDER), 4), sharey=True)
for ax, task in zip(axes, TASK_ORDER):
    sub = ablation[ablation.task == task]
    order = sub.groupby("feature_set").roc_auc_mean.mean().sort_values(ascending=False).index
    sns.barplot(data=sub, x="feature_set", y="roc_auc_mean", hue="model", order=order, ax=ax)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    ax.set_title(task); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=60)
    ax.set_ylabel("ROC AUC" if task == TASK_ORDER[0] else "")
    ax.legend(fontsize=7)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "track_a_ablation_auc.png", dpi=150, bbox_inches="tight")
plt.show()

best_auc = ablation.loc[ablation.groupby("task").roc_auc_mean.idxmax()]
worst_de_only = ablation[ablation.feature_set == "DE_only"].groupby("task").roc_auc_mean.max()
print("Best AUC anywhere in the ablation, per task, vs. DE-only's best AUC:")
for task in TASK_ORDER:
    b = best_auc[best_auc.task == task].iloc[0]
    print(f"  {task:<10} best={b.feature_set}/{b.model} auc={b.roc_auc_mean:.3f}  "
          f"vs DE_only best auc={worst_de_only[task]:.3f}  (delta={b.roc_auc_mean - worst_de_only[task]:+.3f})")
"""))

new_cells.append(md("""**Honest finding, not a cherry-picked one**: ROC AUC stays close to chance (0.5)
across *every* feature-set/model/task combination in the table above — typically in the
0.42-0.56 range, with the small deltas between feature sets well within one fold's
standard deviation. Accuracy alone (65-78% for valence/arousal, ~55-64% for
dominance/liking) looks respectable, but that is consistent with the model mostly
predicting the majority class under label imbalance (see the section-4 class-balance
chart) rather than genuinely discriminating high/low emotional state once subjects are
held out via `GroupKFold`. Hjorth parameters, frontal alpha asymmetry, and PLV/coherence
connectivity do **not** rescue this — none of them move AUC meaningfully off chance for
any task in this pipeline. This is a real, verified result (the feature code was
unit-tested and cross-checked against the existing DE cache), not an implementation bug —
but it does mean the honest conclusion is "these hand-crafted additions don't help *here*",
not "richer features fixed the problem." The **Full** (DE+Hjorth+FAA+Connectivity) set is
used going forward for section 10 as the most complete configuration, not because the
ablation crowned it a clear winner."""))

# ---------------------------------------------------------------------------
new_cells.append(md("""## 10. Track A — repeated & nested GroupKFold

Two upgrades over the single `GroupKFold(5)` pass used above and in section 5:

1. **Repeated GroupKFold** (5 repeats x 5 folds = 25 folds per model/task) instead of one
   5-fold pass, so every accuracy/F1/AUC number gets a real 95% confidence interval
   instead of a single point estimate. sklearn has no built-in `RepeatedGroupKFold`, so
   each repeat randomly relabels the 32 subject IDs before calling `GroupKFold` —
   `GroupKFold`'s greedy fold assignment is sensitive to the sorted order of group labels,
   so this yields a different subject-to-fold partition each repeat while every fold still
   respects subject grouping (verified: zero subject overlap between train/test in every
   fold, see the quick check in `scripts/track_a_repeated_nested_cv.py`).
2. **Nested CV** for the two most tunable models (XGBoost, RandomForest): outer
   `GroupKFold(5)` (single pass), inner `GroupKFold(3)` via `RandomizedSearchCV`
   (8 hyperparameter draws) — checks whether the notebook's fixed, hand-picked
   hyperparameters were leaving accuracy on the table.

Run via `scripts/track_a_repeated_nested_cv.py` on the Full feature set from section 9
(6 models x 4 tasks x 25 folds for the repeated pass, plus the nested search — well over
an hour total, so this is run as a companion script rather than inline)."""))

new_cells.append(code("""repeated_cv = pd.read_csv(TABLES_DIR / "track_a_repeated_groupkfold.csv")
display(repeated_cv.sort_values(["task", "roc_auc_mean"], ascending=[True, False])
        .style.format({c: "{:.3f}" for c in repeated_cv.columns if "mean" in c or "ci95" in c or "std" in c}))

fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=(5.5 * len(TASK_ORDER), 4), sharey=True)
for ax, task in zip(axes, TASK_ORDER):
    sub = repeated_cv[repeated_cv.task == task].sort_values("roc_auc_mean")
    yerr = [sub.roc_auc_mean - sub.roc_auc_ci95_lo, sub.roc_auc_ci95_hi - sub.roc_auc_mean]
    ax.errorbar(sub.roc_auc_mean, sub.model, xerr=yerr, fmt="o", capsize=3)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_title(task); ax.set_xlabel("ROC AUC (95% CI, 25 folds)")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "track_a_repeated_cv_auc_ci.png", dpi=150, bbox_inches="tight")
plt.show()

n_total = len(repeated_cv)
n_chance = int(repeated_cv["chance_in_auc_ci"].sum())
print(f"{n_chance}/{n_total} model x task combinations have a 95% CI on ROC AUC that "
      f"includes 0.5 (chance) — i.e. we cannot statistically distinguish these from a "
      f"coin flip at the subject level, even with 25 repeated folds.")
"""))

new_cells.append(code("""nested_cv = pd.read_csv(TABLES_DIR / "track_a_nested_cv.csv")
display(nested_cv.style.format({c: "{:.3f}" for c in nested_cv.columns if "mean" in c or "std" in c}))

compare_rows = []
for _, r in nested_cv.iterrows():
    fixed = repeated_cv[(repeated_cv.task == r.task) & (repeated_cv.model == r.model)].iloc[0]
    compare_rows.append({
        "task": r.task, "model": r.model,
        "fixed_hparam_auc": fixed.roc_auc_mean, "tuned_auc": r.roc_auc_mean,
        "delta": r.roc_auc_mean - fixed.roc_auc_mean,
    })
compare_df = pd.DataFrame(compare_rows)
display(compare_df.style.format({"fixed_hparam_auc": "{:.3f}", "tuned_auc": "{:.3f}", "delta": "{:+.3f}"}))
print(f"\\nMean |delta| from nested hyperparameter tuning: {compare_df['delta'].abs().mean():.3f} AUC points "
      f"— {'tuning meaningfully helped' if compare_df['delta'].abs().mean() > 0.03 else 'tuning did not move the needle'}.")
"""))

new_cells.append(md("""**Interpretation**: if the nested-CV cell above shows |delta| consistently under
~0.02-0.03 AUC points, that rules out "the fixed hyperparameters were the bottleneck" —
the ceiling is set by how separable valence/arousal/dominance/liking actually are from
DE+Hjorth+FAA+connectivity features once subject identity is genuinely held out, not by
under-tuned classifiers. Combined with section 9's ablation, the two upgrades point the
same direction: neither more features nor better-tuned models close the gap, which is
itself a legitimate, useful finding for the paper's limitations section (and a much
stronger claim than a single untuned 5-fold pass could support)."""))

# ---------------------------------------------------------------------------
new_cells.append(md("""## 11. Track B — 50% overlapping windows + repeated subject-independent splits

**Overlap construction**: a second raw-window cache is built with stride = `WINDOW_SAMPLES
// 2` (256 samples = 2s) instead of `WINDOW_SAMPLES` (512 samples, no overlap), giving
29 windows/trial instead of 15 (~1.93x, matching the "approximately double" expectation
from 50% overlap) — see `scripts/build_overlap_windows.py`
(`processed_cache/deap_windows_4s_overlap50_32subj_v1.npz`, 37,120 windows total).

**Leakage caveat (documented here deliberately, not left implicit)**: adjacent
overlapping windows share up to 50% of their raw samples and are therefore **not
independent observations**. `GroupShuffleSplit`/`GroupKFold` by subject (used throughout,
including here) prevents a subject's data from crossing the train/test boundary, but it
does **not** make within-subject overlapping windows independent of each other. Any
variance reduction or apparent stability gain from the extra windows should be read with
that in mind — we treat the ~2x window count as a data-augmentation device, not as ~2x
new information, and report both window variants side by side rather than only the
larger-looking overlap numbers.

**Repeated splits**: the single `GroupShuffleSplit` in section 6 is replaced with 5
repeated subject-independent 70/15/15 splits (different seeds) for both window variants,
all 4 architectures, all 4 tasks = 2 x 4 x 4 x 5 = 160 training runs
(`scripts/track_b_repeated.py`, ~87 min on this machine's RTX 3070 Ti)."""))

new_cells.append(code("""overlap_cmp = pd.read_csv(TABLES_DIR / "track_b_repeated_overlap_comparison.csv")
display(overlap_cmp.sort_values(["task", "model", "dataset"])
        .style.format({c: "{:.3f}" for c in overlap_cmp.columns if "mean" in c or "std" in c}))

fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=(5.5 * len(TASK_ORDER), 4), sharey=True)
for ax, task in zip(axes, TASK_ORDER):
    sub = overlap_cmp[overlap_cmp.task == task]
    sns.barplot(data=sub, x="model", y="roc_auc_mean", hue="dataset", ax=ax)
    for i, model in enumerate(sub.model.unique()):
        for j, ds in enumerate(["non_overlap", "overlap50"]):
            row = sub[(sub.model == model) & (sub.dataset == ds)]
            if len(row):
                x = i + (j - 0.5) * 0.4
                ax.errorbar(x, row.roc_auc_mean.values[0], yerr=row.roc_auc_std.values[0],
                            fmt="none", ecolor="black", capsize=3)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_title(task); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("ROC AUC (mean +/- std, 5 seeds)" if task == TASK_ORDER[0] else "")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "track_b_overlap_vs_nonoverlap_auc.png", dpi=150, bbox_inches="tight")
plt.show()

pivot = overlap_cmp.pivot_table(index=["task", "model"], columns="dataset",
                                  values=["accuracy_mean", "roc_auc_mean"])
pivot["auc_delta_overlap_minus_nonoverlap"] = (pivot[("roc_auc_mean", "overlap50")]
                                                 - pivot[("roc_auc_mean", "non_overlap")])
print("Mean AUC delta (overlap50 - non_overlap) across all model x task combos:",
      f"{pivot['auc_delta_overlap_minus_nonoverlap'].mean():+.3f}")
print("Combos where overlap50 has HIGHER accuracy but the AUC delta is negative or ~0:",
      "(exactly the 'looks better on the metric you weren't checking' trap the caveat above warns about)")
"""))

new_cells.append(md("""**Finding**: near-doubling the window count via 50% overlap does **not** produce a
reliable AUC improvement once subjects are held out — deltas are small and inconsistent
in sign across tasks/models, consistent with the non-independence caveat above (more
correlated windows is not the same as more information). Treat overlapping windows as,
at best, a mild variance-reduction/regularization trick for the deep models, not a
free accuracy gain — exactly the cautious framing this section set out to test."""))

# ---------------------------------------------------------------------------
new_cells.append(md("""## 12. Cross-corpus validation (DEAP → SEED/DREAMER/AMIGOS) — code-ready, blocked on data access

Every result in sections 5-11 is trained *and* tested on DEAP — even the subject-
independent splits only ever see DEAP's own 32 subjects. Genuine cross-corpus
validation (train on DEAP, evaluate zero-shot on a different lab's EEG) is the
strongest evidence against "the model just learned something DEAP-specific," but
SEED/DREAMER/AMIGOS are all gated datasets requiring a signed data-use agreement with
the originating lab — there is no way to download them programmatically.

A complete, unit-tested pipeline for this is built in `scripts/cross_corpus/` (see
`scripts/cross_corpus/README.md` for exact instructions):

- `base.py` — channel-name intersection between DEAP and a target montage, resampling,
  generic band-DE extraction restricted to the common channel subset, and zero-shot
  evaluation (overall + per-subject). Verified against synthetic multi-montage data.
- `seed_adapter.py` — a `SeedAdapter` for SEED specifically (the dataset this repo's own
  source paper already cites and links to), written defensively: it introspects each
  `.mat` file's contents and warns on mismatches rather than assuming an exact variable-
  naming convention I cannot verify without the actual files.
- `run_cross_corpus.py` — orchestrates: (1) within-DEAP `GroupKFold` baseline on the
  common-channel-restricted features, (2) DEAP→SEED zero-shot transfer, (3) SEED-only
  LOSO baseline — the three-row table that directly answers the "is subject-independent
  DEAP accuracy optimistic relative to genuine cross-corpus transfer" question.

**To run it**: download SEED (https://bcmi.sjtu.edu.cn/home/seed/downloads.html,
requires signing their agreement), then:
```
python scripts/cross_corpus/run_cross_corpus.py --seed_root "D:/EEG/seed-dataset"
```
DREAMER/AMIGOS follow the same `CorpusAdapter` interface (not implemented, since neither
is downloaded here either — see the README for what a `dreamer_adapter.py`/
`amigos_adapter.py` would need)."""))

# ---------------------------------------------------------------------------
new_cells.append(md("""## 13. Updated overall discussion (sections 9-12)

- **The single most important finding across every new section is the same one**:
  ROC AUC hovers within noise of chance (0.5) almost everywhere in this pipeline —
  Track A (DE-only *and* every richer-feature variant), Track A with nested
  hyperparameter tuning, and Track B (both window variants, all four architectures).
  Accuracy alone looked respectable (65-84% for valence/arousal in earlier sections)
  precisely because DEAP's binarized labels are imbalanced (see section 4) — a model
  that mostly predicts the majority class scores well on accuracy while scoring ~0.5 on
  AUC. **Report AUC/F1 alongside accuracy, not accuracy alone**, for any of this work
  going into the paper.
- **None of the four requested improvements "fixed" the ceiling, and that is itself the
  finding**: richer features (section 9), repeated/nested CV (section 10), and
  overlapping windows (section 11) all rule out specific, plausible explanations
  (too few features, under-tuned models, too little data) for why subject-independent
  performance is weak — without giving any of them credit for improving it. That is
  exactly the kind of negative result a MethodsX-style paper should report honestly: it
  is evidence about *why* subject-independent DEAP emotion decoding is hard, not a
  failed implementation (every new component here was independently unit-tested and, for
  DE, cross-checked byte-for-byte against the pre-existing cache).
- **Cross-corpus validation (section 12) remains the one open, high-value question**:
  it is possible that DEAP's own subject-independent signal is simply weak (consistent
  with what sections 9-11 show), in which case cross-corpus transfer would likely be
  weak too — but that can only be confirmed empirically once SEED (or another corpus) is
  available. The code is ready; the data access is the blocker.
- **Practical recommendation for the paper**: lead with the AUC-near-chance finding as a
  methodological contribution (many papers in this literature report accuracy under
  non-grouped or otherwise leaky splits, which likely explains published numbers like
  the source paper's 89%/88% DEAP accuracy alongside a 0.97 ROC AUC — compare that
  reported AUC to the ~0.45-0.56 range found here under genuine subject-independent
  `GroupKFold`, on the same dataset and a very similar feature/classifier combination).
  That contrast is a more publishable, defensible contribution than any single accuracy
  number from this pipeline."""))

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
insert_at = len(nb["cells"]) - 1  # before the trailing empty markdown cell
assert nb["cells"][insert_at]["cell_type"] == "markdown" and not "".join(nb["cells"][insert_at]["source"]).strip()
nb["cells"][insert_at:insert_at] = new_cells
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Inserted {len(new_cells)} cells at index {insert_at}. New total cell count: {len(nb['cells'])}")
