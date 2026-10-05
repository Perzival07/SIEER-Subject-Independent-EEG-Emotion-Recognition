# Cross-corpus validation (DEAP -> SEED-IV/DREAMER/AMIGOS)

## Why this exists
Every result in this repo so far (Track A, Track B, the ablation, the repeated/
nested CV) is trained AND tested on DEAP -- even the "cross-subject" LOSO-style
splits only ever see DEAP's own 32 subjects. That is a genuinely harder setting
than the published MethodsX paper's ungrouped split, but it still cannot rule
out that the model is exploiting something specific to DEAP's stimuli, hardware,
or population. The only way to address that is to test on EEG collected by a
different lab, with different subjects, different electrodes, and (usually) a
different emotion-elicitation paradigm. This module does that.

## Why SEED-IV, not original SEED
Original SEED (3-class positive/neutral/negative) requires signing BCMI's
data-usage agreement directly with the lab and waiting for access -- a hard
blocker with no way to automate. **SEED-IV** (Zheng, Liu, Cheng & Lu, 2018,
"EmotionMeter") is a related, later dataset from the same lab, same 15
subjects and 62-channel montage, but with 4 emotion classes (neutral/sad/fear/
happy) recorded across 3 sessions per subject -- and it is mirrored on Kaggle
with no signed agreement required:
https://www.kaggle.com/datasets/phhasian0710/seed-iv

DREAMER and AMIGOS remain gated (see "Extending" below); SEED-IV is the one
target corpus in this pipeline that is actually downloadable without a data
request.

The code here is written and unit-tested against synthetic data (see
`base.py`'s channel-intersection / DE-extraction logic, which is the same code
already validated against the real DEAP cache in `scripts/rich_features.py`),
and is ready to run the moment you supply the actual files.

## What you need to do
1. **Download SEED-IV**: https://www.kaggle.com/datasets/phhasian0710/seed-iv
   Extract it so you have:
   - `<seed_root>/eeg_raw_data/1/*.mat`, `/2/*.mat`, `/3/*.mat` (one session
     subfolder per recording day, 15 subject files each, 24 trials/file)
   - `<seed_root>/channel-order.xlsx` (or wherever SEED-IV ships its 62-channel
     order -- check your actual download, the exact path has varied across
     distribution versions)
   - Optionally `session1_label.mat` / `session2_label.mat` / `session3_label.mat`
     if your mirror includes them (each session has its OWN trial-label order,
     unlike original SEED's single shared `label.mat`). If these aren't present,
     `SeedIVAdapter` falls back to the label sequence published in the SEED-IV
     paper and prints a loud warning -- read `ReadMe.txt` in your download and
     cross-check before trusting results.
2. Run:
   ```
   cd scripts/cross_corpus
   python run_cross_corpus.py --seed_root "D:/EEG/seed-iv-dataset"
   ```
3. **Read the diagnostic prints before trusting the output.** `SeedIVAdapter`
   prints the parsed channel count and per-session trial-label arrays on load,
   and warns if a session file's trial count doesn't match its label count, or
   if it had to fall back to the hardcoded label sequence. SEED-IV's exact
   `.mat` variable-naming convention (subject-initials-prefixed, e.g.
   `djc_eeg1..djc_eeg24`) is documented by BCMI but I cannot verify it against
   your specific download, so the adapter introspects file contents (sorts by
   trailing trial-index digits) rather than hardcoding a naming pattern that
   might not match your files -- if a warning fires, open the `.mat` file
   yourself (`scipy.io.loadmat(path).keys()`) and check.

## What it produces
`results/tables/cross_corpus_deap_to_seediv_summary.csv` with three rows:
| setting | accuracy | f1_macro | roc_auc |
|---|---|---|---|
| within_DEAP_groupkfold | DEAP-only subject-independent baseline |
| DEAP_to_SEEDIV_zero_shot | train on all of DEAP, never see SEED-IV, test on SEED-IV |
| SEEDIV_only_LOSO | upper bound: same-corpus, subject-independent |

Plus a per-subject breakdown (`cross_corpus_deap_to_seediv_per_subject.csv`) so
you can see whether zero-shot failure is uniform or concentrated in a few
SEED-IV subjects (useful for the discussion section either way).

Note: SEED-IV's 4-class scheme (neutral/sad/fear/happy) is binarized onto
DEAP's "valence" task only, as happy=1 (positive) vs. sad/fear=0 (negative),
dropping neutral trials -- see `SeedIVAdapter.binarize_label` in
`seed_adapter.py`. Arousal/dominance/liking have no clean SEED-IV counterpart
and are not attempted.

## Extending to DREAMER / AMIGOS
Both follow the same `CorpusAdapter` interface (see `base.py`). Neither adapter
is implemented here because:
- DREAMER (request form: https://ieee-dataport.org/open-access/dreamer or via
  the original authors) ships EEG as MATLAB structs per-subject with 18
  film-clip trials and continuous 1-5 valence/arousal/dominance self-reports --
  channel count is 14 (Emotiv EPOC), so the DEAP<->DREAMER common-channel
  intersection will be much smaller (Emotiv's 14 channels: AF3,F7,F3,FC5,T7,
  P7,O1,O2,P8,T8,FC6,F4,F8,AF4 -- verify against your actual download before
  relying on this list, Emotiv channel naming has had minor variants across
  firmware versions).
- AMIGOS (request form: http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/) also
  uses a 14-channel Emotiv montage and continuous self-reports, structurally
  similar to DREAMER.

To add either: write `dreamer_adapter.py` / `amigos_adapter.py` implementing
`iter_trials()` (yield one `Trial` per clip/subject with real channel names and
fs) and `binarize_label()` (threshold their 1-5 or 1-9 scale at the scale's own
midpoint, analogous to DEAP's >5 threshold on a 1-9 scale) -- then swap the
import in `run_cross_corpus.py`. The channel-intersection / DE-extraction /
zero-shot-evaluation machinery in `base.py` is already dataset-agnostic and
does not need to change.
