# DEAP EEG Emotion Recognition -- Comparative Analysis

Subject-independent emotion classification on the DEAP dataset: a classical-ML
pipeline on Differential Entropy features (Track A) vs. a deep-learning
pipeline on raw EEG windows (Track B), plus an eighteen-section investigation
into why subject-independent performance is weak, and a cross-corpus
validation against SEED-IV. The report also includes a dedicated
Model-by-Model Analysis chapter that explains, model by model (all 6
classical + 4 deep architectures, plus every feature/alignment/loss variant
built on top of them), *why* each one scored what it scored. See
`Project_Report.pdf` / `Project_Report.tex` for the full write-up.

## Contents

- [The problem](#the-problem)
- [Key findings](#key-findings)
- [Datasets](#datasets-not-included-in-this-repo)
- [Repository layout](#repository-layout)
- [Method](#method)
- [Notebook sections](#notebook-sections)
- [Scripts](#scripts)
- [Figures](#figures-resultsfigures)
- [Tables](#tables-resultstables)
- [Reproducing the results](#reproducing-the-results)

## The problem

DEAP records 32-channel EEG from 32 participants while each watches 40
one-minute music videos, then rates each video on four 1-9 Self-Assessment
Manikin (SAM) scales: **valence**, **arousal**, **dominance** and **liking**.
Each scale is turned into a separate binary task (rating `> 5` = high, else
low).

The project asks one question throughout: **how well does a model trained on
some people recognise emotion in a person it has never seen?** Every
evaluation is therefore *subject-independent* -- all windows from a given
subject are kept on the same side of every train/test split. This is much
harder (and more honest) than the random window-level splits common in the
DEAP literature, where windows from the same trial leak into both train and
test sets.

## Key findings

1. **Accuracy is misleading on DEAP; ROC-AUC is the primary metric.** The
   binarised labels are heavily imbalanced (valence 79% low, arousal 77% low,
   dominance 61% high, liking 67% high). Several models reach 77-83%
   accuracy simply by predicting the majority class every time -- the
   confusion matrices show the best valence and arousal models never predict
   "high" at all.
2. **Subject-independent ROC-AUC sits close to chance (about 0.45-0.56)** for
   all 10 models, all 6 feature sets, and all four tasks. Repeated
   GroupKFold (25 folds) puts 15 of 24 model/task AUC confidence intervals
   across 0.5.
3. **A formal permutation test confirms it.** Against a label-shuffled null
   distribution (300 permutations), only dominance is above chance
   (p = 0.020); valence, arousal and liking are not.
4. **Nothing tried closes the gap reliably.** Richer features, nested
   hyperparameter tuning, overlapping windows, label re-binarisation, CORAL,
   DANN, self-supervised pretraining, AUC-margin loss and AdaBN each move AUC
   by a few hundredths at most, often in opposite directions on different
   tasks. The strongest single result is per-subject **Euclidean Alignment**
   (mean AUC 0.479 -> 0.523), and stacking CORAL on top of it does not add
   anything (0.514).
5. **A little of the target subject's own data helps more than any of the
   above.** Calibrating XGBoost on 8 of a new subject's 40 trials raises mean
   AUC from about 0.50 to 0.52-0.58, depending on the task.
6. **The difficulty is not DEAP-specific.** A DEAP-trained model transferred
   zero-shot to SEED-IV scores AUC 0.517, whereas a model trained and
   evaluated subject-independently on SEED-IV alone reaches 0.673.

## Datasets (not included in this repo)

The raw EEG datasets are large (multi-GB) and gated/licensed, so they are
**not committed to Git** (`deap-dataset/`, `seed-dataset/`, and the derived
`processed_cache/` are gitignored). Download them yourself and place them at
the paths below.

| Dataset | Link | Place at |
|---|---|---|
| DEAP (preprocessed Python `.dat` files, Kaggle mirror -- what this project actually used) | <https://www.kaggle.com/datasets/manh123df/deap-dataset> | `deap-dataset/data_preprocessed_python/` |
| DEAP official source -- not used here (registration/EULA required; currently returning HTTP 503, verify before relying on it) | <https://www.eecs.qmul.ac.uk/mmv/datasets/deap/> | -- |
| SEED-IV (Kaggle mirror, no agreement required) | <https://www.kaggle.com/datasets/phhasian0710/seed-iv> | `seed-dataset/` |
| SEED (original, 3-class) -- not used here | requires a signed BCMI data-use agreement | -- |
| DREAMER -- not used here | <https://zenodo.org/records/546113> (access-request form on the record page) | -- |
| AMIGOS -- not used here | <https://www.eecs.qmul.ac.uk/mmv/datasets/amigos/> (currently returning HTTP 503, verify before relying on it) | -- |

`processed_cache/` (cached `.npz` feature/window builds) is regenerated
automatically the first time the relevant notebook cell or script runs against
the raw data -- expect several minutes on a cache miss.

## Repository layout

```
Project_Report.tex / .pdf       -- the full report (methodology, results, figures/tables; kept local, gitignored)
DEAP_Comparative_Analysis.ipynb -- main notebook (Track A/B, Sections 1-18)
scripts/                        -- standalone companion experiments (Track A/B, cross-corpus)
  cross_corpus/                 -- DEAP -> SEED-IV transfer pipeline (has its own README)
results/
  tables/                       -- CSV outputs from the notebook and scripts (21 files)
  figures/                      -- PNG figures from the notebook and scripts (24 files)
papers_EEG/                     -- literature-review corpus (32 PDFs; gitignored)
deap-dataset/, seed-dataset/    -- raw data (gitignored; download separately, see above)
processed_cache/                -- cached feature builds (gitignored; regenerated automatically)
```

Result files are named after what they contain. Names starting with
`classical_ml_` belong to Track A, `deep_learning_` to Track B, and
`cross_corpus_` to the SEED-IV experiments. Figure names end in `_roc_auc`
when ROC-AUC is the plotted metric; table names usually end in `_results`.

## Method

### Preprocessing (shared by both tracks)

- All 32 DEAP subjects, 40 trials each, 32 EEG channels at 128 Hz.
- Each trial's 3-second pre-stimulus baseline is used to normalise the
  60-second stimulus segment.
- The 60 seconds are cut into **4-second windows (512 samples), 15 per
  trial**, giving **19,200 windows** in total. Every window inherits its
  trial's four binary labels.

### Track A -- Differential Entropy features + classical ML

- **Features:** Differential Entropy (DE) per channel in four frequency bands
  (theta 4-8 Hz, alpha 8-13 Hz, beta 13-30 Hz, gamma 30-45 Hz) =
  32 x 4 = **128 features** per window.
- Optional richer features (Section 9): Hjorth activity/mobility/complexity
  (+96), frontal/hemispheric alpha asymmetry "FAA" (+14), and PLV/coherence
  connectivity (+57). All of them together is the 295-feature "Full" set.
- **Models (6):** Logistic Regression, k-NN, SVM (RBF kernel), Random Forest,
  XGBoost, MLP.
- **Evaluation:** `GroupKFold(5)` grouped by subject, with a StandardScaler
  fitted on the training folds only. Results are reported as mean +/- std
  over folds.

### Track B -- raw EEG windows + deep learning

- **Input:** the baseline-normalised raw window (32 channels x 512 samples),
  z-scored per window and channel.
- **Models (4):** ShallowConvNet (54,562 parameters), EEGNet (2,130),
  CNN-LSTM (82,466) and TSception (8,765).
- **Evaluation:** one subject-independent 70/15/15 train/validation/test
  split (`GroupShuffleSplit`, seed 42) with early stopping on validation
  loss. Because it is a single split, the notebook's Track B standard
  deviations are 0. Section 11 repeats it over 5 seeds to get real spread.

### Metrics

Every table reports accuracy, macro-F1 and ROC-AUC. **ROC-AUC is the headline
metric**, because with this level of class imbalance accuracy mostly reflects
the majority-class rate. Chance is AUC = 0.5 on every task.

## Notebook sections

`DEAP_Comparative_Analysis.ipynb`, in order:

| # | Section | Outputs |
|---|---|---|
| 1-3 | Configuration, loading and preprocessing, windowed dataset (incl. 2.1 raw-EEG QC overview) | `baseline_normalized_eeg_trial1_all_32_subjects.png` |
| 4 | Class balance | `class_distribution_low_vs_high_per_emotion_dimension.png` |
| 5 | Track A: DE features -> classical ML | `classical_ml_de_features_results.csv`, `classical_ml_de_features_accuracy_per_model.png` |
| 6 | Track B: raw windows -> deep learning (architectures, training loop, results) | `deep_learning_raw_eeg_results.csv`, `deep_learning_raw_eeg_training_curves.png` |
| 7 | Combined comparison of all 10 models; confusion matrices | `all_models_results_classical_ml_and_deep_learning.csv`, `accuracy_comparison_classical_ml_vs_deep_learning.png`, `confusion_matrices_best_model_per_emotion_dimension.png` |
| 8 | Discussion | -- |
| 9 | Richer Track A features and the feature-set ablation | `classical_ml_feature_set_ablation_roc_auc.png` |
| 10 | Track A repeated and nested GroupKFold | `classical_ml_repeated_groupkfold_roc_auc_95ci.png` |
| 11 | Track B with 50%-overlapping windows and repeated splits | `deep_learning_overlap_vs_nonoverlap_windows_roc_auc.png` |
| 12 | Cross-corpus validation (DEAP -> SEED-IV) | `cross_corpus_deap_to_seediv_accuracy_and_roc_auc.png` |
| 13 | Updated discussion (Sections 9-12) | -- |
| 14 | Label-cleanup ablation | `classical_ml_label_binarization_strategies_roc_auc.png` |
| 15 | Per-subject calibration (personalisation) | `classical_ml_zero_shot_vs_calibrated_by_subject_roc_auc.png` |
| 16 | Literature-driven interventions: permutation test (16.1), CORAL (16.2), DANN (16.3), summary (16.4) | `classical_ml_permutation_test_vs_null_roc_auc.png`, `classical_ml_coral_domain_alignment_roc_auc.png`, `deep_learning_domain_adversarial_dann_roc_auc.png` |
| 17 | Five deferred literature techniques: Euclidean Alignment (17.1), SSL pretraining (17.2), AUC-margin loss (17.3), AdaBN (17.4), SHAP (17.5), summary (17.6) | `classical_ml_euclidean_alignment_roc_auc.png`, `deep_learning_self_supervised_pretraining_roc_auc.png`, `deep_learning_auc_maximization_loss_roc_auc.png`, `deep_learning_test_time_adaptation_adabn_roc_auc.png` |
| 18 | Does CORAL stack on top of Euclidean Alignment? | `classical_ml_euclidean_alignment_plus_coral_roc_auc.png` |

The heavy computations behind Sections 9-18 are in `scripts/`. The notebook
cells load those scripts' CSVs and plot them.

## Scripts

Every script runs from the repository root (`python scripts/<name>.py`) and
writes to `results/tables/` and/or `results/figures/`.

**Data and feature builders**

| Script | What it does |
|---|---|
| `build_extended_features.py` | Builds the Track A cache with DE + Hjorth + FAA + connectivity for all 32 subjects, using exactly the notebook's preprocessing. |
| `build_overlap_windows.py` | Builds the 50%-overlap (2-second stride) version of the Track B raw-window dataset. |
| `build_raw_ratings.py` | Stores the continuous 1-9 SAM ratings row-aligned with the feature cache, for the label-cleanup experiment. |
| `rich_features.py` | Hjorth, alpha-asymmetry and PLV/coherence feature functions (imported by the builders). |
| `dl_models.py` | Exact port of the notebook's EEGNet, ShallowConvNet, CNN-LSTM and TSception, so scripts can import them. |

**Track A experiments (classical ML)**

| Script | Question it answers | Output table |
|---|---|---|
| `track_a_ablation.py` | Which feature families help? 6 feature sets x 3 models x 4 tasks | `classical_ml_feature_set_ablation_results.csv` |
| `track_a_repeated_nested_cv.py` | How stable are the scores (5 x 5 repeated GroupKFold with 95% CIs)? Does hyperparameter tuning (nested CV) help? | `classical_ml_repeated_groupkfold_results_with_95ci.csv`, `classical_ml_nested_cv_tuned_results.csv` |
| `track_a_label_cleanup.py` | Does the way ratings are binarised matter? | `classical_ml_label_binarization_strategies_results.csv` |
| `track_a_calibration_loso.py` | Does a little labelled data from the new subject help? (leave-one-subject-out) | `classical_ml_zero_shot_vs_calibrated_loso_per_subject.csv` |
| `track_a_significance_test.py` | Is the observed AUC distinguishable from a permutation null? | `classical_ml_label_permutation_test_results.csv` |
| `track_a_domain_alignment.py` | Does CORAL feature alignment close the subject gap? | `classical_ml_coral_domain_alignment_results.csv` |
| `track_a_riemannian_alignment.py` | Does per-subject Euclidean Alignment of raw EEG help? | `classical_ml_euclidean_alignment_results.csv` |
| `track_a_ea_coral_stacked.py` | Do EA and CORAL add up when combined? | `classical_ml_euclidean_alignment_plus_coral_results.csv` |
| `track_a_shap_diagnostic.py` | Which features does the model actually rely on? | `classical_ml_shap_top_features_per_emotion_dimension.csv`, 4 SHAP figures |

**Track B experiments (deep learning)**

| Script | Question it answers | Output table |
|---|---|---|
| `track_b_repeated.py` | Do 50%-overlapping windows help, measured over 5 seeds instead of one split? | `deep_learning_overlap_vs_nonoverlap_windows_results.csv` |
| `track_b_domain_adversarial.py` | Does subject-adversarial (DANN, gradient-reversal) training help EEGNet? | `deep_learning_domain_adversarial_dann_results.csv` |
| `track_b_ssl_pretrain.py` | Does self-supervised pretraining of EEGNet's encoder help? | `deep_learning_self_supervised_pretraining_results.csv` |
| `track_b_auc_maximization.py` | Does training directly on a pairwise AUC-margin loss help? | `deep_learning_auc_maximization_loss_results.csv` |
| `track_b_test_time_adaptation.py` | Does re-estimating BatchNorm statistics on the test subject (AdaBN) help? | `deep_learning_test_time_adaptation_adabn_results.csv` |

**Cross-corpus, plotting and notebook maintenance**

| Script | What it does |
|---|---|
| `cross_corpus/run_cross_corpus.py` (+ `base.py`, `seed_adapter.py`) | DEAP -> SEED-IV transfer; writes the three `cross_corpus_*.csv` tables. See `scripts/cross_corpus/README.md`. |
| `plot_class_balance_figure.py`, `plot_ablation_figure.py`, `plot_repeated_cv_figure.py` | Publication-quality (300 dpi PNG + vector PDF) redraws of the class-balance, ablation and repeated-CV figures used in the report. |
| `update_notebook.py`, `update_notebook_v2.py` ... `_v5.py` | One-off scripts that appended Sections 9-18 to the notebook. Kept for provenance; re-running them appends duplicate sections. |

## Figures (`results/figures/`)

### Data overview

**`baseline_normalized_eeg_trial1_all_32_subjects.png`**
A quality-control panel: one subplot per subject (S01-S32), each showing all
32 baseline-normalised EEG channels stacked for trial 1 over its 60 seconds.
It makes the very large between-subject differences in amplitude, artefacts
and noise visible at a glance (compare S01's flat traces with S11 or S22),
which is the core reason subject-independent learning is hard.

**`class_distribution_low_vs_high_per_emotion_dimension.png`**
Four bar charts, one per emotion dimension, counting low vs. high 4-second
windows after the `> 5` threshold. Valence (79.0% low / 21.0% high) and arousal
(76.7% / 23.3%) are skewed towards low; dominance (39.1% / 60.9%) and liking
(33.5% / 66.5%) towards high. This imbalance is why accuracy is not used as
the headline metric.

### Baseline model comparison (Sections 5-7)

**`classical_ml_de_features_accuracy_per_model.png`**
Track A test accuracy (mean +/- std over 5 subject-grouped folds) for all 6
classical models on the 128 DE features, one panel per emotion dimension,
with a 0.5 reference line. Random Forest and XGBoost lead (about 0.75-0.78 on
valence/arousal), but those numbers sit at or below the majority-class rate.

**`deep_learning_raw_eeg_training_curves.png`**
A 4 x 4 grid (rows: valence, arousal, dominance, liking; columns:
ShallowConvNet, EEGNet, CNN-LSTM, TSception) of training loss (solid) and
validation loss (dashed) per epoch. Training loss falls steadily while
validation loss flattens or rises within a few epochs in almost every cell,
so early stopping triggers early: the networks fit the training subjects
without generalising to unseen ones.

**`accuracy_comparison_classical_ml_vs_deep_learning.png`**
All 10 models side by side, one panel per emotion dimension: the 6 classical
models (left of the dotted divider) vs. the 4 deep models (right). Deep models
reach about 0.83 accuracy on valence and 0.77 on arousal, almost exactly the
majority-class rates, and fall below 0.5 on dominance.

**`confusion_matrices_best_model_per_emotion_dimension.png`**
Row-normalised confusion matrices (true vs. predicted, low/high) for the
highest-accuracy model on each task: EEGNet (valence), CNN-LSTM (arousal),
TSception (dominance) and Random Forest (liking). The valence and arousal
"winners" predict "low" for 100% of windows, and the dominance and liking
winners predict "high" for most windows. The high accuracies are therefore
majority-class guessing.

### Robustness checks on Track A (Sections 9-10)

**`classical_ml_feature_set_ablation_roc_auc.png`**
ROC-AUC (mean +/- SD over 5 folds) for Logistic Regression, Random Forest and
XGBoost across six feature sets -- DE, DE+Hjorth, DE+FAA, DE+Connectivity,
DE+Hjorth+FAA and Full -- in a 2 x 2 grid of emotion dimensions. Every bar
hovers around the 0.5 chance line (about 0.42-0.56), so adding feature
families does not reliably help.

**`classical_ml_repeated_groupkfold_roc_auc_95ci.png`**
Forest plot of ROC-AUC with 95% confidence intervals from 5 x 5 = 25 repeated
GroupKFold folds, for all 6 classical models x 4 tasks. Filled markers are
intervals that exclude chance and hollow markers are intervals that include it.
Most valence/arousal intervals straddle 0.5. Dominance is the only task with
several intervals entirely above chance, peaking at 0.571 (Logistic
Regression).

### Robustness checks on Track B (Section 11)

**`deep_learning_overlap_vs_nonoverlap_windows_roc_auc.png`**
ROC-AUC (mean +/- std over 5 seeds) for each deep model trained on
non-overlapping 4-second windows (blue) vs. 50%-overlapping windows (orange),
one panel per task. Overlap doubles the training windows but moves mean AUC
only from 0.483 to 0.498, with error bars that overlap throughout.

### Cross-corpus validation (Section 12)

**`cross_corpus_deap_to_seediv_accuracy_and_roc_auc.png`**
Left: accuracy and ROC-AUC for three settings -- within-DEAP GroupKFold,
DEAP -> SEED-IV zero-shot transfer, and SEED-IV-only leave-one-subject-out
(LOSO). Right: per-subject ROC-AUC for the two SEED-IV settings. Zero-shot
transfer stays close to chance (AUC 0.517), while same-corpus SEED-IV LOSO
reaches 0.673, with individual subjects up to about 0.88.

### Label and personalisation experiments (Sections 14-15)

**`classical_ml_label_binarization_strategies_roc_auc.png`**
ROC-AUC for three ways of binarising SAM ratings -- `raw_threshold` (rating
> 5), `per_subject_z` (threshold each subject's z-scored ratings at 0) and
`drop_middle` (keep only ratings <= 3 or >= 6) -- for 3 models x 4 tasks.
Per-subject z-scoring gives the most consistent small gain (mean AUC 0.483 ->
0.514), but nothing escapes the near-chance band.

**`classical_ml_zero_shot_vs_calibrated_by_subject_roc_auc.png`**
Slope plot, one line per held-out subject: ROC-AUC of an XGBoost model trained
on the other 31 subjects (zero-shot, blue) vs. the same model after
warm-start boosting on 8 of the held-out subject's own 40 trials (calibrated,
orange), both scored on the remaining 32 trials. Most lines go up, especially
for arousal (mean 0.506 -> 0.577). Subjects with a single-class calibration
set are left out of the calibrated arm.

### Literature-driven interventions (Sections 16-18)

**`classical_ml_permutation_test_vs_null_roc_auc.png`**
For each task, the observed ROC-AUC with the real labels (orange) against the
mean +/- SD of a null distribution built from 300 within-subject trial-level
label shuffles (grey), using Logistic Regression on the Full feature set.
Only dominance sits above its null (p = 0.020). Valence and arousal score
*below* their nulls.

**`classical_ml_coral_domain_alignment_roc_auc.png`**
ROC-AUC with no alignment vs. CORAL (re-colouring the training-fold feature
covariance to match the test fold's) for 3 models x 4 tasks on the Full
feature set. The change is negligible (mean 0.483 -> 0.489) and the direction
differs between models and tasks.

**`classical_ml_euclidean_alignment_roc_auc.png`**
ROC-AUC with no alignment vs. per-subject Euclidean Alignment (EA: whitening
each subject's raw EEG by that subject's own mean spatial covariance before
DE extraction) for 3 models x 4 tasks on DE features. EA is the strongest
single intervention in the project (mean 0.479 -> 0.523).

**`classical_ml_euclidean_alignment_plus_coral_roc_auc.png`**
Four conditions side by side -- none, CORAL, EA, and EA followed by CORAL --
for 3 models x 4 tasks on DE features. EA alone (0.523) is best. Adding CORAL
on top (0.514) gives nothing extra, and CORAL alone (0.473) slightly hurts.

**`deep_learning_domain_adversarial_dann_roc_auc.png`**
EEGNet ROC-AUC per task, baseline vs. subject-adversarial training (DANN: a
gradient-reversal subject discriminator). DANN helps valence and liking
slightly (to about 0.51-0.52) but collapses arousal and dominance (0.616 ->
0.400, 0.552 -> 0.403).

**`deep_learning_self_supervised_pretraining_roc_auc.png`**
EEGNet ROC-AUC per task, trained from scratch vs. with its encoder first
pretrained (without labels) to predict each window's band-power vector. The
pretrained model gains on valence and liking and loses on arousal and
dominance.

**`deep_learning_auc_maximization_loss_roc_auc.png`**
EEGNet ROC-AUC per task with standard cross-entropy vs. a pairwise
squared-hinge AUC-margin loss. The AUC loss helps valence (0.474 -> 0.557)
and is roughly neutral or slightly worse on the other three tasks.

**`deep_learning_test_time_adaptation_adabn_roc_auc.png`**
EEGNet ROC-AUC per task without adaptation vs. with AdaBN, which re-estimates
BatchNorm statistics on each test subject's unlabelled windows. AdaBN helps
liking (0.467 -> 0.549), is neutral on valence and hurts arousal and
dominance.

**`classical_ml_shap_feature_importance_{valence,arousal,dominance,liking}.png`**
One figure per emotion dimension for XGBoost on the 295-feature Full set.
Left: total mean |SHAP| per feature family (DE, Connectivity, Hjorth, FAA).
Right: the 15 individual features with the highest mean |SHAP|. DE features
carry about half the total attribution on every task (47-52%), Hjorth and
connectivity about a quarter each, and FAA only 3-6%. The top individual
features are mostly gamma- and beta-band DE and beta-band PLV, spread across
the scalp, and they differ from task to task. The model leans on spectral power rather
than the frontal-asymmetry markers the emotion literature emphasises.

## Tables (`results/tables/`)

All tables are comma-separated with a header row. Unless stated otherwise,
`task` is one of `valence`/`arousal`/`dominance`/`liking`, metrics are on a
0-1 scale, and `*_mean`/`*_std` are taken across CV folds. Columns
`roc_auc`, `f1`/`f1_macro` and `accuracy` are ROC-AUC, macro-F1 and accuracy.
`train_time_s` is wall-clock training time in seconds.

### Baseline results

| File | Rows | Contents and columns |
|---|---|---|
| `classical_ml_de_features_results.csv` | 24 (6 models x 4 tasks) | Track A baseline on the 128 DE features, GroupKFold(5). `task, model, accuracy_mean/std, f1_mean/std, roc_auc_mean/std, train_time_s`. |
| `deep_learning_raw_eeg_results.csv` | 16 (4 models x 4 tasks) | Track B baseline, single 70/15/15 subject-independent split (so `*_std` = 0). Same metric columns plus `n_params` (trainable parameters). |
| `all_models_results_classical_ml_and_deep_learning.csv` | 40 (10 models x 4 tasks) | The two tables above stacked, sorted by task and accuracy. `n_params` is blank for classical models. |

### Track A robustness and diagnostics

| File | Rows | Contents and columns |
|---|---|---|
| `classical_ml_feature_set_ablation_results.csv` | 72 (6 feature sets x 3 models x 4 tasks) | Feature-set ablation. `feature_set` (`DE_only`, `DE+Hjorth`, `DE+FAA`, `DE+Connectivity`, `DE+Hjorth+FAA`, `Full (DE+Hjorth+FAA+Conn)`), `n_features` (128-295), then the standard metric columns. |
| `classical_ml_repeated_groupkfold_results_with_95ci.csv` | 24 (6 models x 4 tasks) | 5 repeats x GroupKFold(5) = 25 folds (`n_folds_total`) on the Full set. For each metric: `_mean`, `_std`, `_ci95_lo`, `_ci95_hi`. `chance_in_auc_ci` is True when the AUC 95% CI contains 0.5 (true for 15 of 24 rows). `elapsed_s` is runtime. |
| `classical_ml_nested_cv_tuned_results.csv` | 8 (XGBoost, Random Forest x 4 tasks) | Nested CV: inner RandomizedSearchCV with GroupKFold(3) for tuning, outer GroupKFold(5) for scoring. `n_folds`, `elapsed_s`, `accuracy/f1_macro/roc_auc _mean/_std`. Tuning does not lift AUC above about 0.52. |
| `classical_ml_label_permutation_test_results.csv` | 4 (one per task) | Permutation test, Logistic Regression on the Full set. `n_perm` (300), `observed_auc`, the null distribution's `null_auc_mean/std/min/max`, `z_score`, one-sided `p_value`, `elapsed_s`. |
| `classical_ml_shap_top_features_per_emotion_dimension.csv` | 76 (19 per task) | SHAP attributions for XGBoost on the Full set. Ranked rows (`rank` 1-15) give `feature` (e.g. `de_beta_P4`, `plv_beta_CP5-CP6`), its `family` (DE/Hjorth/FAA/Connectivity) and `mean_abs_shap`. The 4 unranked rows per task (`feature` = `__family_share__<family>`) give each family's share of total attribution (summing to 1). |

### Track A label and personalisation experiments

| File | Rows | Contents and columns |
|---|---|---|
| `classical_ml_label_binarization_strategies_results.csv` | 36 (3 variants x 3 models x 4 tasks) | `variant` (`raw_threshold`, `per_subject_z`, `drop_middle`), `n_windows` (fewer for `drop_middle`), `n_subjects`, `class_balance_high_frac` (fraction labelled high), `n_folds_valid` (folds with both classes present), then the standard metric columns. |
| `classical_ml_zero_shot_vs_calibrated_loso_per_subject.csv` | 124 (up to 32 subjects x 4 tasks) | Leave-one-subject-out XGBoost, one row per held-out `subject`. `n_calib_windows` (120 = 8 trials) and `n_eval_windows` (480 = 32 trials); `zeroshot_accuracy/f1/auc` vs. `calibrated_accuracy/f1/auc` on the same evaluation windows. Calibrated columns are empty when `calibration_skipped_single_class` is True (the calibration trials contained only one class). |

### Track A domain alignment

| File | Rows | Contents and columns |
|---|---|---|
| `classical_ml_coral_domain_alignment_results.csv` | 24 (2 conditions x 3 models x 4 tasks) | `condition` = `none` or `coral`, Full feature set, GroupKFold(5), standard metric columns. |
| `classical_ml_euclidean_alignment_results.csv` | 24 (2 conditions x 3 models x 4 tasks) | `condition` = `none` or `ea` (per-subject Euclidean Alignment), DE features. |
| `classical_ml_euclidean_alignment_plus_coral_results.csv` | 48 (4 conditions x 3 models x 4 tasks) | `condition` = `none`, `coral`, `ea` or `ea_coral` (EA then CORAL), DE features. |

### Track B experiments

All four intervention tables below use EEGNet on one subject-independent
70/15/15 split (seed 42), with one row per `task` x `condition` (8 rows), and
columns `train_time_s, accuracy, f1_macro, precision, recall, roc_auc`.

| File | Rows | Contents and columns |
|---|---|---|
| `deep_learning_overlap_vs_nonoverlap_windows_results.csv` | 32 (2 datasets x 4 models x 4 tasks) | All four deep models over 5 seeds (`n_seeds`). `dataset` = `non_overlap` (4-second stride) or `overlap50` (2-second stride); `*_mean/_std` across seeds; `train_time_s_mean`. |
| `deep_learning_domain_adversarial_dann_results.csv` | 8 | `condition` = `baseline` or `dann`. |
| `deep_learning_self_supervised_pretraining_results.csv` | 8 | `condition` = `scratch` or `ssl_pretrained`. |
| `deep_learning_auc_maximization_loss_results.csv` | 8 | `condition` = `ce_baseline` (cross-entropy) or `auc_max` (AUC-margin loss). |
| `deep_learning_test_time_adaptation_adabn_results.csv` | 8 | `condition` = `no_adaptation` or `adabn`. |

### Cross-corpus (DEAP -> SEED-IV)

These use DE + XGBoost on the channels DEAP and SEED-IV share, and the valence
task only. SEED-IV's four classes are mapped to happy = 1 vs. sad/fear = 0,
with neutral trials dropped.

| File | Rows | Contents and columns |
|---|---|---|
| `cross_corpus_deap_to_seediv_summary_three_settings.csv` | 3 | `setting` = `within_DEAP_groupkfold`, `DEAP_to_SEEDIV_zero_shot` or `SEEDIV_only_LOSO`, with `accuracy, f1_macro, roc_auc`. `held_out_subject` is only filled on the LOSO row, where it is an artefact of averaging subject IDs and should be ignored. |
| `cross_corpus_deap_to_seediv_zero_shot_per_subject.csv` | 15 (SEED-IV subjects) | Zero-shot DEAP-trained model scored on each SEED-IV `subject`: `accuracy, f1_macro, roc_auc, n_windows`. |
| `cross_corpus_seediv_only_loso_per_subject.csv` | 15 | SEED-IV-only LOSO, one row per `held_out_subject`: `accuracy, f1_macro, roc_auc`. |

## Reproducing the results

1. Download DEAP (and, for Section 12, SEED-IV) to the paths in
   [Datasets](#datasets-not-included-in-this-repo).
2. Install the Python dependencies used by the notebook and scripts: `numpy`,
   `scipy`, `pandas`, `scikit-learn`, `xgboost`, `torch`, `matplotlib`,
   `seaborn`, `tqdm`, and `shap` (SHAP diagnostic only). A CUDA GPU is
   recommended for Track B. The project was run on an RTX 3070 Ti.
3. Run `DEAP_Comparative_Analysis.ipynb` top to bottom for Sections 1-8. This
   also builds `processed_cache/`.
4. For Sections 9-18, build the extra caches (`build_extended_features.py`,
   `build_overlap_windows.py`, `build_raw_ratings.py`), run the relevant
   `scripts/track_*.py` and `scripts/cross_corpus/run_cross_corpus.py`, then
   re-run the notebook cells that read their CSVs. Some scripts are slow:
   nested CV takes about 4 hours, `track_b_repeated.py` about 90 minutes, and
   the permutation test about 45 minutes.
