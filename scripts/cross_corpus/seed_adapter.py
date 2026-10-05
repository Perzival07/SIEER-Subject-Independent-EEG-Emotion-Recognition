"""
SEED-IV adapter (Zheng, Liu, Cheng & Lu, 2018 -- "EmotionMeter") -- the corpus
used for cross-corpus validation in this repo. Unlike the original 3-class SEED
dataset, SEED-IV is freely mirrored on Kaggle with no signed data-use agreement
required (https://www.kaggle.com/datasets/phhasian0710/seed-iv), which is why
it -- not original SEED -- is the target this pipeline is built against.

Expected layout after you download and extract SEED-IV, pointed to by `seed_root`:
    seed_root/
      eeg_raw_data/
        1/                     # session 1 (of 3, recorded on different days)
          1_20160518.mat       # <subject_id>_<session_date>.mat, 15 subjects
          2_20150915.mat
          ...
        2/                     # session 2 -- SAME 15 subjects, DIFFERENT trial order
        3/                     # session 3 -- again a different trial order

Each subject .mat file holds 24 trial arrays (one per film clip), each shaped
(62, n_samples) at 200 Hz -- 24 trials per session, not SEED's 15. The
per-trial emotion label is one of 4 classes: 0=neutral, 1=sad, 2=fear, 3=happy,
and -- critically -- **the trial order (and therefore the label sequence) is
different in each of the 3 sessions**, unlike original SEED's single shared
label.mat.

This adapter first tries to load session-specific label files shipped with the
download (several candidate names are tried, since Kaggle mirrors and the
official BCMI release have not always used identical filenames/paths). If none
are found, it falls back to the label sequences published in the SEED-IV paper
(Zheng et al. 2018, Table II) and reproduced consistently across public SEED-IV
loaders -- but this fallback is used LOUDLY (a warning is printed), not
silently, and you should cross-check it against your own download's
documentation (`ReadMe.txt`) before trusting downstream results, in the same
spirit as this project's other "verify, don't assume" adapters.
"""
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import scipy.io as sio

from base import CorpusAdapter, Trial

SEED_IV_FS = 200  # SEED-IV's eeg_raw_data is documented at 200 Hz
N_SESSIONS = 3
N_TRIALS_PER_SESSION = 24

# Published per-session label sequences (Zheng et al. 2018, Table II), 0=neutral,
# 1=sad, 2=fear, 3=happy. Used ONLY as a fallback when no label file can be found
# under `seed_root` -- see module docstring. Verify against your own download.
FALLBACK_SESSION_LABELS = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}

_LABEL_FILE_CANDIDATES = {
    1: ["session1_label.mat", "1_label.mat"],
    2: ["session2_label.mat", "2_label.mat"],
    3: ["session3_label.mat", "3_label.mat"],
}

# Distribution vintage varies (BCMI's own release vs. Kaggle mirrors) -- the Kaggle
# mirror of SEED-IV in particular ships "Channel Order.xlsx" (space, capitalized),
# not the hyphenated lowercase name used elsewhere in this repo's docs.
_CHANNEL_ORDER_CANDIDATES = [
    "channel-order.xlsx", "Channel Order.xlsx", "channel_order.xlsx",
    "channel-order.txt", "Channel Order.txt",
]


def _resolve_channel_order_path(seed_root: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit
    for name in _CHANNEL_ORDER_CANDIDATES:
        candidate = seed_root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No channel-order file found under {seed_root} (tried {_CHANNEL_ORDER_CANDIDATES}). "
        f"Pass `channel_order_path` explicitly if your download uses a different name."
    )


def _load_channel_order(channel_order_path: Path) -> List[str]:
    """SEED-IV shares SEED's 62-channel montage; the order list ships as xlsx
    or txt depending on distribution vintage. Fail loudly rather than silently
    mis-order channels (which would silently corrupt every downstream feature)."""
    if channel_order_path.suffix.lower() in (".xlsx", ".xls"):
        import pandas as pd
        df = pd.read_excel(channel_order_path, header=None)
        names = [str(x).strip() for x in df.iloc[:, 0].tolist() if str(x).strip().lower() != "nan"]
    else:
        names = [l.strip() for l in channel_order_path.read_text().splitlines() if l.strip()]
    if len(names) < 32:
        raise ValueError(
            f"Parsed only {len(names)} channel names from {channel_order_path}, expected ~62. "
            f"Open the file and check its format; update `_load_channel_order` to match."
        )
    return names


def _trial_key_sort_index(key: str) -> int:
    m = re.search(r"(\d+)$", key)
    if not m:
        raise ValueError(f"Could not parse a trailing trial index from SEED-IV .mat key '{key}'")
    return int(m.group(1))


def _load_session_labels(seed_root: Path, session: int) -> np.ndarray:
    for name in _LABEL_FILE_CANDIDATES[session]:
        for candidate in (seed_root / name, seed_root / "eeg_raw_data" / name):
            if candidate.exists():
                mat = sio.loadmat(candidate)
                keys = [k for k in mat.keys() if not k.startswith("__")]
                if not keys:
                    continue
                labels = mat[keys[0]].flatten().astype(int)
                print(f"[SeedIVAdapter] Loaded session {session} labels from {candidate} "
                      f"({len(labels)} trials).")
                return labels
    labels = np.array(FALLBACK_SESSION_LABELS[session], dtype=int)
    print(f"[SeedIVAdapter][WARN] No label file found for session {session} under {seed_root} "
          f"(tried {_LABEL_FILE_CANDIDATES[session]}). Falling back to the label sequence "
          f"published in Zheng et al. 2018, Table II -- VERIFY this against your own "
          f"download's ReadMe.txt before trusting results.")
    return labels


class SeedIVAdapter(CorpusAdapter):
    name = "SEED-IV"

    def __init__(self, seed_root: str, raw_data_dir: Optional[str] = None,
                 channel_order_path: Optional[str] = None):
        self.seed_root = Path(seed_root)
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else self.seed_root / "eeg_raw_data"
        self.channel_order_path = _resolve_channel_order_path(
            self.seed_root, Path(channel_order_path) if channel_order_path else None)

        if not self.raw_data_dir.exists():
            raise FileNotFoundError(
                f"SEED-IV eeg_raw_data dir not found at {self.raw_data_dir}. "
                f"Download SEED-IV from "
                f"https://www.kaggle.com/datasets/phhasian0710/seed-iv "
                f"(no signed agreement required) and point `seed_root` at the extracted folder."
            )
        self.channel_names = _load_channel_order(self.channel_order_path)
        self.session_labels = {s: _load_session_labels(self.seed_root, s) for s in range(1, N_SESSIONS + 1)}
        for s, labels in self.session_labels.items():
            print(f"[SeedIVAdapter] session {s}: {len(self.channel_names)} channels, "
                  f"{len(labels)} trial labels: {labels}")

    def iter_trials(self):
        found_any = False
        for session in range(1, N_SESSIONS + 1):
            session_dir = self.raw_data_dir / str(session)
            if not session_dir.exists():
                print(f"[SeedIVAdapter][WARN] session dir {session_dir} not found, skipping.")
                continue
            labels = self.session_labels[session]
            mat_files = sorted(p for p in session_dir.glob("*.mat"))
            if not mat_files:
                raise FileNotFoundError(f"No subject .mat files found in {session_dir}")
            for fpath in mat_files:
                found_any = True
                subject_id = fpath.stem.split("_")[0]
                mat = sio.loadmat(fpath)
                trial_keys = sorted(
                    (k for k in mat.keys() if not k.startswith("__")),
                    key=_trial_key_sort_index,
                )
                if len(trial_keys) != len(labels):
                    print(f"[SeedIVAdapter][WARN] {fpath.name}: found {len(trial_keys)} trial arrays "
                          f"but {len(labels)} labels for session {session} -- check this file "
                          f"manually before trusting results.")
                for key, label_val in zip(trial_keys, labels):
                    signal = mat[key]  # documented as (62, n_samples); verify against your actual shape
                    if signal.shape[0] != len(self.channel_names):
                        raise ValueError(
                            f"{fpath.name}/{key}: signal has {signal.shape[0]} rows but channel-order "
                            f"file lists {len(self.channel_names)} channels -- these must match 1:1."
                        )
                    yield Trial(
                        subject_id=subject_id,  # same person across all 3 sessions -> genuine LOSO grouping
                        channel_names=self.channel_names,
                        fs=SEED_IV_FS,
                        signal=signal.astype(np.float64),
                        label={"emotion": int(label_val)},
                    )
        if not found_any:
            raise FileNotFoundError(f"No session subfolders with .mat files found under {self.raw_data_dir}")

    def binarize_label(self, trial: Trial, task: str):
        if task not in ("valence",):
            return None  # SEED-IV's 4-class scheme maps onto valence only, not arousal/dominance/liking
        val = trial.label["emotion"]
        if val == 0:
            return None  # drop neutral, same convention as the original SEED adapter
        return 1 if val == 3 else 0  # happy=1 (positive); sad/fear=0 (negative)
