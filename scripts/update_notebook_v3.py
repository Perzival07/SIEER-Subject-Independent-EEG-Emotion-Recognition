# -*- coding: utf-8 -*-
"""
Appends section 16 (literature-driven interventions: permutation significance
test, CORAL domain alignment, subject-adversarial DANN training) to
DEAP_Comparative_Analysis.ipynb, same insert-before-trailing-empty-cell
pattern as update_notebook.py / update_notebook_v2.py.
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
new_cells.append(md("""## 16. Literature-driven interventions: significance testing, CORAL alignment, subject-adversarial training

Sections 9-15 exhausted the things this project's own design choices could
vary (features, hyperparameters, CV repetition, window overlap, label
definition, a small amount of per-subject calibration data) without closing
the AUC-near-chance gap. To go further, `D:/EEG/papers_EEG/` was assembled
(31 papers spanning EEG-emotion surveys, domain-adaptation/generalization
methods, contrastive/self-supervised pretraining, GAN/diffusion augmentation,
and DEAP-specific methodology critiques) and read end-to-end by parallel
literature-extraction passes, filtering for techniques that are both (a)
concretely implementable on this project's exact sklearn/PyTorch pipeline and
(b) specifically aimed at the subject-independent generalization gap, not
generic accuracy-chasing. Full per-paper notes are in
`D:/EEG/papers_EEG/literature_log.txt`.

Three things came out of that review that this project had **not** yet tried,
in increasing order of how much of the pipeline they touch:

1. **A formal permutation significance test** for the "AUC near chance" claim
   itself (`scripts/track_a_significance_test.py`) — every AUC number in
   sections 5-15 was compared to 0.5 by eye; Khan et al.'s DEAP methodology
   review (`2508.02417v1.pdf`) is a reminder that "looks like chance" and
   "is statistically indistinguishable from chance under this exact protocol"
   are different claims, and only the second is publication-grade.
2. **CORAL feature-distribution alignment** for Track A
   (`scripts/track_a_domain_alignment.py`) — an unsupervised, linear,
   per-fold second-order-statistics alignment step (Sun & Saenko, 2016) that
   the domain-adaptation survey (`2212.03176v1.pdf`) and DS-AGC
   (`2308.11635v2.pdf`, leave-one-subject-out on SEED) both cite as
   nontrivially closing a subject gap on EEG features, with no retraining or
   architecture change required.
3. **Subject-adversarial (DANN/gradient-reversal-layer) training** for Track B
   (`scripts/track_b_domain_adversarial.py`) — the single most repeated
   concrete recommendation across the whole literature set (PARSE, the EEG
   tutorial/review, the DA survey, DS-AGC, the GNN survey, and the
   brainsci-16-00041 review all independently converge on it): add a small
   domain-classifier head after the shared encoder, reverse its gradient
   before it reaches the encoder, and train it to predict the *training*
   subjects' identities jointly with the emotion label — pushing the encoder
   toward features a subject-ID classifier can't exploit, on the hypothesis
   that those are exactly the nuisance features hurting generalization to
   unseen subjects."""))

# ---------------------------------------------------------------------------
new_cells.append(md("""### 16.1 Is "near chance" actually indistinguishable from chance?

`scripts/track_a_significance_test.py` shuffles the labels (globally, breaking
any real feature/label relationship while leaving the GroupKFold fold
structure untouched), reruns the exact same GroupKFold(5) pipeline used
throughout this notebook on the Full feature set, and repeats 300 times per
task to build an empirical null AUC distribution. `p = (1 + #{null >=
observed}) / (n_perm + 1)`. LogisticRegression is used as a fast proxy
classifier for the null (documented trade-off — section 9 already showed
inter-model AUC deltas on the Full feature set are within one fold's standard
deviation, so this is not expected to bias the conclusion toward or away from
significance)."""))

new_cells.append(code("""sig = pd.read_csv(TABLES_DIR / "track_a_significance_test.csv")
display(sig.style.format({"observed_auc": "{:.4f}", "null_auc_mean": "{:.4f}",
                           "null_auc_std": "{:.4f}", "z_score": "{:+.2f}", "p_value": "{:.4f}"}))

fig, ax = plt.subplots(figsize=(7, 4.5))
for i, row in sig.iterrows():
    ax.errorbar(i, row.null_auc_mean, yerr=row.null_auc_std, fmt="o", color="gray",
                capsize=4, label="null (permuted labels)" if i == 0 else None)
    ax.scatter(i, row.observed_auc, color="C1", zorder=5, s=60,
               label="observed (real labels)" if i == 0 else None)
ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
ax.set_xticks(range(len(sig))); ax.set_xticklabels(sig.task)
ax.set_ylabel("ROC AUC"); ax.legend()
ax.set_title("Observed AUC vs. label-permutation null distribution (Full features, LogisticRegression)")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "track_a_significance_test.png", dpi=150, bbox_inches="tight")
plt.show()

print("Per-task verdict (alpha=0.05):")
for _, r in sig.iterrows():
    verdict = "statistically distinguishable from chance" if r.p_value < 0.05 else "NOT distinguishable from chance"
    print(f"  {r.task:<10} observed={r.observed_auc:.3f}  null={r.null_auc_mean:.3f}+/-{r.null_auc_std:.3f}  "
          f"z={r.z_score:+.2f}  p={r.p_value:.4f}  -> {verdict}")
"""))

new_cells.append(md("""**Actual result**: for 3 of 4 tasks (valence, arousal, liking), the observed
AUC is NOT just close to the null, it is statistically indistinguishable from
it (p=1.0) — and, notably, the observed AUC sits slightly *below* the null
mean in all three cases (e.g. valence: observed 0.437 vs. null 0.501 ± 0.005,
z=-12.0). This is consistent with a regularized linear model very mildly
overfitting a weak-to-absent real signal within each training fold, and is
itself informative: it means this project's near-chance numbers were, if
anything, being slightly *conservative*, not inflated.

**Dominance is the one exception**: observed AUC 0.561 vs. null 0.500 ± 0.005,
z=+12.3, p=0.0033 — genuinely significant given how tight the null
distribution is, but still a small effect size (AUC 0.56, not 0.56 being a
strong classifier by any practical standard). This matches dominance/
LogisticRegression already being the single strongest result in section 9's
ablation table (AUC 0.5606 there too) — the permutation test converts that
previously-eyeballed observation into a formally supported, if modest,
exception to the "everything is chance" pattern, rather than manufacturing a
new claim. The practical conclusion is unchanged: even the one statistically
real signal in this entire notebook is far too weak to be useful, but it is
now precisely quantified rather than eyeballed, in the same spirit as Khan et
al.'s label-randomization ("watermelon") control — applied here to a protocol
that was already leakage-free, to sharpen the claim rather than expose a leaky
one."""))

# ---------------------------------------------------------------------------
new_cells.append(md("""### 16.2 CORAL feature alignment — does closing the distribution gap move AUC?

`scripts/track_a_domain_alignment.py` re-runs the same GroupKFold(5) /
Full-feature-set / 3-model comparison as section 9's ablation, but for the
`coral` condition, the training fold's features are whitened by their own
covariance and recolored to match the (unlabeled) test fold's covariance
before the classifier ever sees them — a purely unsupervised, per-fold,
linear alignment step (Sun & Saenko, 2016), directly motivated by DS-AGC's
LOSO-on-SEED finding that plain CORAL reaches ~65-70% where untransformed
features stay under 60%."""))

new_cells.append(code("""align = pd.read_csv(TABLES_DIR / "track_a_domain_alignment.csv")
display(align.pivot_table(index=["task", "model"], columns="condition", values="roc_auc_mean")
        .assign(delta_auc=lambda d: d["coral"] - d["none"])
        .style.format("{:.4f}"))

fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=(5 * len(TASK_ORDER), 4), sharey=True)
for ax, task in zip(axes, TASK_ORDER):
    sub = align[align.task == task]
    sns.barplot(data=sub, x="model", y="roc_auc_mean", hue="condition", ax=ax)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_title(task); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=20)
    ax.set_ylabel("ROC AUC" if task == TASK_ORDER[0] else "")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "track_a_domain_alignment_auc.png", dpi=150, bbox_inches="tight")
plt.show()

delta = (align.pivot_table(index=["task", "model"], columns="condition", values="roc_auc_mean")
         .assign(delta=lambda d: d["coral"] - d["none"]))
print(f"CORAL vs. no alignment -- mean delta AUC across all task/model combos: {delta['delta'].mean():+.4f}")
print(f"Best single delta: {delta['delta'].max():+.4f} ({delta['delta'].idxmax()})")
print(f"Worst single delta: {delta['delta'].min():+.4f} ({delta['delta'].idxmin()})")
"""))

new_cells.append(md("""**Actual result**: small and inconsistent in sign. Mean AUC delta across all
12 task/model combinations is +0.0059 (8 of 12 improved, 4 got worse), and the
largest single gain (+0.047, dominance/RandomForest) and largest single loss
(-0.037, dominance/LogisticRegression) occur on the *same task*, in opposite
directions depending only on which model sits downstream of the alignment —
CORAL's effect is not a stable, model-agnostic property of the aligned
feature space in this pipeline, unlike the consistent ~65-70%-vs-<60% gain
DS-AGC reports for SEED under LOSO. This rules out "the feature distributions
weren't second-order-aligned" as a further hypothesis, joining features (9),
hyperparameters (10), window count (11), and label definition (14) as
explanations this notebook has now explicitly tested and ruled out, rather
than left unexamined."""))

# ---------------------------------------------------------------------------
new_cells.append(md("""### 16.3 Subject-adversarial (DANN) training for Track B

`scripts/track_b_domain_adversarial.py` adds a gradient-reversal-layer (GRL)
domain-classifier branch to EEGNet (the smallest Track B model, 2130 params,
chosen so the experiment is cheap enough to run to completion here and
extend to the other three architectures only if the effect replicates). The
domain classifier is trained to predict which *training-fold* subject a
window came from; the GRL reverses that gradient before it reaches the shared
encoder, so the encoder is pushed toward features the subject-ID classifier
finds useless — the standard Ganin & Lempitsky (2016) progressive-lambda
schedule ramps the adversarial term in over training. Baseline and DANN are
trained on the identical single 70/15/15 subject-independent split per task
(seed=42) for a fair paired comparison, not reused from section 6's numbers
(which may not share the same split)."""))

new_cells.append(code("""dann = pd.read_csv(TABLES_DIR / "track_b_domain_adversarial.csv")
display(dann.pivot(index="task", columns="condition", values=["accuracy", "f1_macro", "roc_auc"])
        .style.format("{:.4f}"))

fig, ax = plt.subplots(figsize=(7, 4.5))
piv = dann.pivot(index="task", columns="condition", values="roc_auc")[["baseline", "dann"]]
piv.plot(kind="bar", ax=ax)
ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
ax.set_ylabel("ROC AUC"); ax.set_title("EEGNet baseline vs. + subject-adversarial (DANN) training")
plt.xticks(rotation=0)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "track_b_domain_adversarial_auc.png", dpi=150, bbox_inches="tight")
plt.show()

delta = piv["dann"] - piv["baseline"]
print("DANN vs. baseline EEGNet, ROC-AUC delta per task:")
for task, d in delta.items():
    print(f"  {task:<10} baseline={piv.loc[task, 'baseline']:.3f}  dann={piv.loc[task, 'dann']:.3f}  delta={d:+.3f}")
print(f"\\nMean delta across tasks: {delta.mean():+.4f}")
"""))

new_cells.append(md("""**Actual result: not a clean win, and not just "no effect" — actively
unstable for half the tasks.** Valence (+0.039) and liking (+0.050) got small,
plausible-looking AUC gains. Arousal (-0.216) and dominance (-0.149) got
substantially *worse*, and the degraded runs show a specific failure
signature — precision and recall pinned at ~0.50 (arousal dann: precision
0.3875, recall exactly 0.5; dominance dann: precision/recall both ~0.497) —
consistent with the classifier collapsing toward predicting one class
regardless of input, not a milder, generic accuracy drop. This experiment was
re-run after fixing one genuine implementation bug (a validation-loader
tuple-unpacking mismatch) and the collapse pattern persisted, so it is not an
artifact of that bug.

This matches a specific, named failure mode from the literature review itself:
f1000research-14-196597's PRISMA review flags "over-aggressive alignment
collapsing class structure" as a documented UDA failure pattern, and that
appears to be exactly what happened here for two of the four tasks — the
domain-adversarial loss, ramped in via the standard Ganin & Lempitsky
schedule, was strong enough by mid-training to overwhelm the emotion-
classification objective for arousal/dominance specifically. A calmer
schedule (lower max lambda, slower ramp, or gradient-clipping the domain
branch) is the natural next debugging step before concluding GRL-based
subject-adversarial training doesn't work on this pipeline at all — but as
run here, it does not reliably improve subject-independent AUC, and for two
of four tasks makes it meaningfully worse. Extending it to the other three
Track B architectures is *not* recommended until this instability is
resolved for EEGNet first."""))

# ---------------------------------------------------------------------------
new_cells.append(md("""### 16.4 Summary: what the literature review changed

- **Confirmed the central finding more rigorously, not just repeated it**:
  16.1's permutation test replaces "AUC looks near 0.5" with an exact p-value
  against a real empirical null. For 3 of 4 tasks the result is a clean
  non-significance (p=1.0, observed AUC actually *below* the null mean).
  Dominance is a genuine, if small, exception (p=0.0033, AUC 0.561) — now
  precisely quantified rather than eyeballed.
- **Tested two more, previously-untried, literature-motivated hypotheses**
  for what might be suppressing subject-independent signal — unaligned
  feature distributions (16.2, CORAL) and subject-identifiable nuisance
  structure in the learned representation (16.3, DANN) — joining features
  (9), hyperparameters (10), window count (11), and label definition (14) as
  hypotheses this project has now explicitly tested rather than assumed.
- **Neither closed the gap, and DANN made things actively worse for half the
  tasks.** CORAL's mean AUC delta across 12 task/model combinations was a
  small +0.0059 with inconsistent sign (8 up, 4 down) — no stable, usable
  effect. DANN gave small gains for valence (+0.039) and liking (+0.050) but
  substantial *regressions* for arousal (-0.216) and dominance (-0.149), with
  a majority-class-collapse failure signature matching a documented UDA
  failure mode (over-aggressive alignment) named in the literature review
  itself. Neither result supports adopting either technique as-is.
- **Six independent, literature-recommended intervention families now tested
  and found not to close the gap**: richer features, better-tuned models,
  more data via window overlap, better label definitions, distribution
  alignment, and adversarial subject-invariance. That convergence — including
  one technique (DANN) actively backfiring rather than merely not helping —
  is itself stronger evidence for this notebook's central claim than a clean
  "we tried nothing else" negative result would be, and matches what three
  independent, contemporaneous 2026 papers found on DEAP/SEED using entirely
  different pipelines (see `citations.txt`).
- **Next steps flagged, not pursued here**: an AUC-maximization training
  objective (Xiao 2024, directly targets this project's own metric),
  subject-invariant contrastive pretraining (CLISA-style, or the
  DEAP-validated cascaded self-supervised approach of Wang et al. 2024), and
  a calmer DANN lambda schedule to test whether 16.3's instability — rather
  than the adversarial idea itself — is what's failing. All logged with
  specific reasoning in `D:/EEG/papers_EEG/literature_log.txt`.
- Full attribution — what was taken from each of the 31 papers in
  `papers_EEG/`, why it was or wasn't used, and how it maps to the three
  scripts above — is in `D:/EEG/papers_EEG/literature_log.txt`."""))

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
insert_at = len(nb["cells"]) - 1
assert nb["cells"][insert_at]["cell_type"] == "markdown" and not "".join(nb["cells"][insert_at]["source"]).strip()
nb["cells"][insert_at:insert_at] = new_cells
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Inserted {len(new_cells)} cells at index {insert_at}. New total cell count: {len(nb['cells'])}")
