# -*- coding: utf-8 -*-
"""
Appends section 17 (five literature-backed techniques deferred in section 16's
review, now actually implemented and tested: Euclidean Alignment, self-supervised
TFR-style pretraining, AUC-margin loss, AdaBN test-time adaptation, and a SHAP
interpretability diagnostic) to DEAP_Comparative_Analysis.ipynb, same
insert-before-trailing-empty-cell pattern as update_notebook.py /
update_notebook_v2.py / update_notebook_v3.py.
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
new_cells.append(md("""## 17. Five previously-deferred literature techniques, implemented and tested

Section 16 reviewed 31 papers and implemented three interventions (permutation
significance test, CORAL, DANN), explicitly deferring several more ideas as
"future work" rather than testing them. This section closes that gap: the five
most promising deferred ideas that don't require building a new model family
from scratch (GNN and custom-Transformer architectures remain out of scope,
same reasoning as section 16) are implemented here and their viability is
checked directly against this project's own subject-independent DEAP pipeline
-- not just cited. Full attribution is in
`D:/EEG/papers_EEG/literature_log.txt` ("BATCH 6").

1. **17.1 Euclidean Alignment** (`scripts/track_a_riemannian_alignment.py`) --
   CORAL's Riemannian/covariance-recentering sibling (2212.03176, 2308.11635),
   applied per-subject at the raw-signal level.
2. **17.2 Self-supervised (TFR-style) pretraining**
   (`scripts/track_b_ssl_pretrain.py`) -- the single highest-priority deferred
   idea from section 16 (2403.04041, the only paper in the whole review
   evaluated LOSO on DEAP itself while beating DA baselines).
3. **17.3 AUC-margin training loss** (`scripts/track_b_auc_maximization.py`) --
   the most directly on-metric deferred idea (2408.08979): this project's own
   headline metric is AUC.
4. **17.4 AdaBN test-time adaptation**
   (`scripts/track_b_test_time_adaptation.py`) -- the target-aware, source-free
   follow-up to DANN (2504.03707).
5. **17.5 SHAP interpretability diagnostic**
   (`scripts/track_a_shap_diagnostic.py`) -- not an intervention, a diagnostic
   (sensors-25-01827): do near-chance models lean on plausible EEG features or
   noise?

Every script's docstring states explicitly which parts are a faithful
implementation of the source paper's mechanism vs. a deliberately simplified,
self-contained version (e.g. a mini-batch pairwise AUC surrogate instead of a
full saddle-point solver; AdaBN instead of the source paper's full
pseudo-labeling system) -- same honesty convention as section 16."""))

# ---------------------------------------------------------------------------
new_cells.append(md("""### 17.1 Euclidean Alignment -- CORAL's alternative, tested

Unlike CORAL (which aligns already-extracted *feature* distributions per
GroupKFold fold), Euclidean Alignment (He & Wu 2019) operates upstream at the
raw-channel level, per SUBJECT: each subject's own raw windows are whitened by
that subject's own mean spatial covariance (label-free, using every window
that subject has), and Differential-Entropy band-power features are then
recomputed from the whitened signal. Compared against unaligned DE features,
same GroupKFold(5) / 3-model setup as section 9/16.2."""))

new_cells.append(code("""ea = pd.read_csv(TABLES_DIR / "track_a_riemannian_alignment.csv")
display(ea.pivot_table(index=["task", "model"], columns="condition", values="roc_auc_mean")
        .assign(delta_auc=lambda d: d["ea"] - d["none"])
        .style.format("{:.4f}"))

fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=(5 * len(TASK_ORDER), 4), sharey=True)
for ax, task in zip(axes, TASK_ORDER):
    sub = ea[ea.task == task]
    sns.barplot(data=sub, x="model", y="roc_auc_mean", hue="condition", ax=ax)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_title(task); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=20)
    ax.set_ylabel("ROC AUC" if task == TASK_ORDER[0] else "")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "track_a_riemannian_alignment_auc.png", dpi=150, bbox_inches="tight")
plt.show()

delta = (ea.pivot_table(index=["task", "model"], columns="condition", values="roc_auc_mean")
         .assign(delta=lambda d: d["ea"] - d["none"]))
print(f"EA vs. no alignment -- mean delta AUC across all 12 task/model combos: {delta['delta'].mean():+.4f}")
print(f"Positive deltas: {(delta['delta'] > 0).sum()}/12")
print(f"Best single delta: {delta['delta'].max():+.4f} ({delta['delta'].idxmax()})")
print(f"Worst single delta: {delta['delta'].min():+.4f} ({delta['delta'].idxmin()})")
"""))

new_cells.append(md("""**Actual result: the strongest, most consistent effect of any alignment
technique tried in this project.** Mean AUC delta across all 12 task/model
combinations is **+0.044** (vs. CORAL's +0.0059 in 16.2) -- **10 of 12**
combinations improved (vs. CORAL's 8/12), and unlike CORAL the sign is
consistent *within* three of the four tasks: valence (+0.087, +0.027, +0.069
across LR/RF/XGBoost), arousal (+0.078, +0.027, +0.065), and dominance
(+0.074, +0.038, +0.073) all improved for every model. Only liking is mixed
(LogisticRegression +0.045, but RandomForest -0.038 and XGBoost -0.014).

This is the best single AUC-delta result in the entire notebook so far --
but it is still a modest absolute improvement (best post-alignment AUCs
land around 0.53-0.58, not a qualitative jump to a strong classifier), and it
operates on the raw-signal covariance structure rather than the
already-extracted feature space CORAL touches, which is a plausible reason
the two techniques behave so differently despite both being "alignment."
It is worth testing whether combining EA (raw-signal-level) with CORAL
(feature-level) compounds further -- flagged as a natural next step, not run
here."""))

# ---------------------------------------------------------------------------
new_cells.append(md("""### 17.2 Self-supervised (TFR-style) pretraining

`scripts/track_b_ssl_pretrain.py` pretrains EEGNet's conv trunk with a
label-free pretext task (predict each window's own 32-channel x 4-band
log-band-power from the raw time-domain signal, MSE loss, training-fold
subjects only), then fine-tunes the whole network end-to-end on the labeled
task, compared against an identically-trained "scratch" (randomly
initialized) EEGNet. Motivated by 2403.04041, the only paper in the whole
literature review evaluated LOSO on DEAP itself while beating DA
baselines."""))

new_cells.append(code("""ssl = pd.read_csv(TABLES_DIR / "track_b_ssl_pretrain.csv")
display(ssl.pivot(index="task", columns="condition", values=["accuracy", "f1_macro", "roc_auc"])
        .style.format("{:.4f}"))

fig, ax = plt.subplots(figsize=(7, 4.5))
piv = ssl.pivot(index="task", columns="condition", values="roc_auc")[["scratch", "ssl_pretrained"]]
piv.plot(kind="bar", ax=ax)
ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
ax.set_ylabel("ROC AUC"); ax.set_title("EEGNet: scratch vs. SSL-pretrained trunk")
plt.xticks(rotation=0)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "track_b_ssl_pretrain_auc.png", dpi=150, bbox_inches="tight")
plt.show()

delta = piv["ssl_pretrained"] - piv["scratch"]
print("SSL-pretrained vs. scratch, ROC-AUC delta per task:")
for task, d in delta.items():
    print(f"  {task:<10} scratch={piv.loc[task, 'scratch']:.3f}  ssl={piv.loc[task, 'ssl_pretrained']:.3f}  delta={d:+.3f}")
print(f"\\nMean delta across tasks: {delta.mean():+.4f}")
"""))

new_cells.append(md("""**Actual result: a bimodal, task-split pattern, not a clean win.** Valence
(+0.074) and liking (+0.100) improved meaningfully; arousal (-0.082) and
dominance (-0.053) got worse. Mean delta across tasks is a small +0.010 --
masking the fact that no task landed close to zero. This is the *same*
two-tasks-up/two-tasks-down split seen in section 16.3's DANN result, and it
recurs again in 17.3 and 17.4 below -- see 17.6 for the cross-cutting pattern
this suggests. The pretext task itself trained fine (MSE dropped from 0.64 to
0.27 over 15 epochs on standardized band-power targets), so this is not a
pretraining-failed-to-converge artifact -- the pretrained trunk successfully
learned *something* about spectral structure, it just was not uniformly
useful as a fine-tuning initialization across all four label definitions."""))

# ---------------------------------------------------------------------------
new_cells.append(md("""### 17.3 AUC-margin training loss

`scripts/track_b_auc_maximization.py` replaces `nn.CrossEntropyLoss` with a
mini-batch pairwise squared-hinge AUC surrogate
(`mean_{i in pos, j in neg} max(0, margin - (s_i - s_j))^2`, a simplified,
self-contained version of the Ying et al.-style AUC-margin idea 2408.08979
builds its full saddle-point solver around), everything else (model, split,
seed) held identical to the cross-entropy baseline."""))

new_cells.append(code("""aucmax = pd.read_csv(TABLES_DIR / "track_b_auc_maximization.csv")
display(aucmax.pivot(index="task", columns="condition", values=["accuracy", "f1_macro", "roc_auc"])
        .style.format("{:.4f}"))

fig, ax = plt.subplots(figsize=(7, 4.5))
piv = aucmax.pivot(index="task", columns="condition", values="roc_auc")[["ce_baseline", "auc_max"]]
piv.plot(kind="bar", ax=ax)
ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
ax.set_ylabel("ROC AUC"); ax.set_title("EEGNet: cross-entropy vs. AUC-margin loss")
plt.xticks(rotation=0)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "track_b_auc_maximization_auc.png", dpi=150, bbox_inches="tight")
plt.show()

delta = piv["auc_max"] - piv["ce_baseline"]
print("AUC-margin vs. cross-entropy, ROC-AUC delta per task:")
for task, d in delta.items():
    print(f"  {task:<10} ce={piv.loc[task, 'ce_baseline']:.3f}  auc_max={piv.loc[task, 'auc_max']:.3f}  delta={d:+.3f}")
print(f"\\nMean delta across tasks: {delta.mean():+.4f}")
"""))

new_cells.append(md("""**Actual result: a real, mechanistically-explainable gain for valence, small
mixed effects elsewhere.** Valence improved the most of any single result in
17.2-17.4 (+0.083, AUC 0.474 -> 0.557) -- and the *mechanism* is visible in the
raw numbers: the cross-entropy baseline for valence had accuracy 0.835 with
AUC only 0.474, the classic signature of a classifier collapsing toward the
majority class under imbalance (high accuracy, chance-or-below AUC); the
AUC-margin loss produced a far more balanced classifier (accuracy 0.507, AUC
0.557) precisely because it is insensitive to that kind of imbalance by
construction. Dominance improved slightly (+0.008); arousal (-0.010) and
liking (-0.018) were both small, likely-noise-level regressions. Mean delta
+0.016. This is the one intervention in section 17 with a clear, understood
reason *why* it helped where it helped, rather than an unexplained task
split."""))

# ---------------------------------------------------------------------------
new_cells.append(md("""### 17.4 AdaBN test-time adaptation

`scripts/track_b_test_time_adaptation.py` recomputes every BatchNorm layer's
running statistics using ONLY a held-out test subject's own unlabeled
windows (no gradient updates, no labels), per subject, then evaluates that
subject with the adapted statistics -- the cheapest member of the source-free
UDA family 2504.03707 surveys, compared against the same trained baseline
evaluated with its original (training-population) BatchNorm statistics."""))

new_cells.append(code("""adabn = pd.read_csv(TABLES_DIR / "track_b_test_time_adaptation.csv")
display(adabn.pivot(index="task", columns="condition", values=["accuracy", "f1_macro", "roc_auc"])
        .style.format("{:.4f}"))

fig, ax = plt.subplots(figsize=(7, 4.5))
piv = adabn.pivot(index="task", columns="condition", values="roc_auc")[["no_adaptation", "adabn"]]
piv.plot(kind="bar", ax=ax)
ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
ax.set_ylabel("ROC AUC"); ax.set_title("EEGNet: no adaptation vs. AdaBN test-time adaptation")
plt.xticks(rotation=0)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "track_b_test_time_adaptation_auc.png", dpi=150, bbox_inches="tight")
plt.show()

delta = piv["adabn"] - piv["no_adaptation"]
print("AdaBN vs. no adaptation, ROC-AUC delta per task:")
for task, d in delta.items():
    print(f"  {task:<10} no_adapt={piv.loc[task, 'no_adaptation']:.3f}  adabn={piv.loc[task, 'adabn']:.3f}  delta={d:+.3f}")
print(f"\\nMean delta across tasks: {delta.mean():+.4f}")
"""))

new_cells.append(md("""**Actual result: net negative, and the clearest instance of the
arousal/dominance-vs-valence/liking split in this section.** Arousal (-0.101)
and dominance (-0.063) got substantially worse; liking improved (+0.082) and
valence was flat (+0.002). Mean delta -0.020 -- the only clearly net-negative
intervention in section 17. Recomputing BatchNorm statistics per-subject
evidently discards some of the training-population structure that arousal's
and dominance's baselines (the two strongest baseline AUCs across all of
section 17's Track B experiments, ~0.55-0.62) were relying on, without
replacing it with anything better."""))

# ---------------------------------------------------------------------------
new_cells.append(md("""### 17.5 SHAP diagnostic: plausible signal or noise?

`scripts/track_a_shap_diagnostic.py` fits XGBoost on the Full feature set (one
representative GroupKFold fold per task) and computes SHAP values on the held-out
fold, aggregated by feature family (DE, Hjorth, FAA, Connectivity) and reported
per individual feature. This is a diagnostic, not an intervention -- it asks
whether the (weak) signal these near-chance models find looks
neurophysiologically plausible or noise-like, motivated by
sensors-25-01827."""))

new_cells.append(code("""shap_df = pd.read_csv(TABLES_DIR / "track_a_shap_diagnostic.csv")
family_share = shap_df[shap_df.feature.str.startswith("__family_share__")].copy()
family_share["family"] = family_share.feature.str.replace("__family_share__", "", regex=False)
display(family_share.pivot(index="task", columns="family", values="mean_abs_shap").style.format("{:.1%}"))

top_features = shap_df[~shap_df.feature.str.startswith("__family_share__")]
for task in TASK_ORDER:
    print(f"\\n[{task}] top 5 features by mean |SHAP|:")
    print(top_features[top_features.task == task].head(5)[["feature", "family", "mean_abs_shap"]]
          .to_string(index=False))
"""))

new_cells.append(md("""**Actual result: importance is spread across families roughly in
proportion to how many features each family contributes, with two mild but
consistent departures.** Of the 295 Full-set features, DE contributes 43.4% (128),
Hjorth 32.5% (96), Connectivity 19.3% (57), FAA 4.7% (14). Across all four
tasks, DE's actual SHAP share (47-52%) and Connectivity's (22-24%) both sit
somewhat *above* their feature-count share, while Hjorth's (21-26%) sits
*below* -- a mild, consistent preference for spectral-power and
cross-channel-connectivity features over single-channel signal-complexity
features, not a random/noise-like pattern.

Notably, **frontal/hemispheric alpha asymmetry (FAA) -- the one feature family
built specifically around a named, hypothesis-driven neurophysiological
mechanism (Davidson-style approach/withdrawal asymmetry) -- carries the
*smallest* share of importance in every task (3.4-6.1%)**, roughly at or
below its 4.7% feature-count baseline. Top individual features skew toward DE
gamma/beta power at specific channels (`de_gamma_FC2` is the single top
feature for both arousal and liking) and one connectivity feature
(`plv_beta_F3-O1` for dominance), not FAA. This means the near-chance
classifiers are not obviously leaning on the literature's classic asymmetry
hypothesis, nor are they picking up nothing at all (importance is not spread
uniformly/randomly) -- the honest reading is that whatever weak, mostly
unhelpful structure these models exploit is spread across generic spectral
power and connectivity, in roughly the proportions those families are
represented in the feature set, rather than concentrated on anything
specifically meaningful."""))

# ---------------------------------------------------------------------------
new_cells.append(md("""### 17.6 Summary: what section 17 added

- **Euclidean Alignment (17.1) is the strongest single result in this
  notebook**: +0.044 mean AUC delta, 10/12 task/model combinations improved,
  and unlike every other alignment/adaptation technique tried (CORAL in 16.2,
  DANN in 16.3, AdaBN in 17.4), the improvement is *consistent in sign* for
  three of the four tasks (valence, arousal, dominance all positive for every
  one of the 3 models tested). Still modest in absolute terms (~0.53-0.58 AUC,
  not a qualitative leap), but the most reproducible positive signal found
  across sections 16-17.
- **A recurring task split, not four independent coin flips**: DANN (16.3),
  SSL pretraining (17.2), and AdaBN (17.4) all show the *same* pattern --
  valence and liking improve, arousal and dominance get *worse* -- across
  three unrelated mechanisms (adversarial training, self-supervised
  pretraining, test-time BatchNorm recalibration). Arousal and dominance are
  also the two tasks with the strongest baseline AUCs in these same Track B
  experiments (arousal ~0.55-0.62, dominance ~0.55). The natural reading: these
  three interventions all perturb the *learned representation* in some way,
  and whatever fold-specific structure the untouched baseline had already
  found for arousal/dominance is more fragile to that perturbation than
  valence/liking's much weaker baseline representations are. AUC-margin loss
  (17.3) is the exception with a clear, understood mechanism (fixing
  class-imbalance collapse for valence specifically) rather than fitting this
  representation-perturbation pattern, and it is also the only Track B
  intervention here that is a training-*loss* change rather than a
  representation/architecture/statistics change -- consistent with the
  above explanation.
- **SHAP (17.5) adds an interpretability finding, not an AUC result**: feature
  importance is spread across families roughly proportional to feature count,
  with DE and Connectivity mildly over-represented and Hjorth
  under-represented -- and, notably, the one family built around a specific
  named neurophysiological hypothesis (frontal/hemispheric alpha asymmetry)
  carries the *least* importance of all four families in every task. Neither
  a plausible concentrated signal nor pure noise -- a diffuse, generic
  spectral/connectivity pattern.
- **Practical bottom line for the paper**: of five previously-deferred,
  literature-backed techniques, one (Euclidean Alignment) gives a real,
  reproducible, if modest, AUC improvement; one (AUC-margin loss) gives a
  smaller, mechanistically-understood improvement specific to a
  class-imbalance failure mode; two (SSL pretraining, AdaBN) are net-neutral-
  to-negative on average but reveal the same task-dependent instability
  pattern DANN showed; and the SHAP diagnostic shows the underlying signal is
  diffuse rather than concentrated on any specific neurophysiological marker.
  This does not overturn sections 1-16's central finding that subject-
  independent AUC stays far below what would be practically useful -- but
  Euclidean Alignment in particular is now a concrete, evidence-backed
  candidate worth combining with the existing Full feature set and/or with
  CORAL in future work, rather than another ruled-out hypothesis.
- Full attribution -- what was taken from each of the five source papers, and
  exactly how each maps to the scripts above -- is in
  `D:/EEG/papers_EEG/literature_log.txt` ("BATCH 6")."""))

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
insert_at = len(nb["cells"]) - 1
assert nb["cells"][insert_at]["cell_type"] == "markdown" and not "".join(nb["cells"][insert_at]["source"]).strip()
nb["cells"][insert_at:insert_at] = new_cells
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Inserted {len(new_cells)} cells at index {insert_at}. New total cell count: {len(nb['cells'])}")
