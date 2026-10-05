"""
Track B, upgraded: repeated subject-independent train/val/test splits (multiple
seeds) instead of the notebook's single GroupShuffleSplit, run on BOTH the
non-overlapping (WINDOW=4s, stride=4s) and 50%-overlap (stride=2s) raw-window
datasets, so the two can be compared side by side.

Scope/cost trade-off (documented rather than hidden): full Leave-One-Subject-Out
(32 folds) x 4 architectures x 4 tasks x 2 window variants would be extremely
expensive for deep nets. We use N_SEEDS repeated 70/15/15 subject-independent
splits instead -- enough to put a confidence interval around the single-split
estimate the notebook currently reports, at a small fraction of LOSO's cost.

Run: python scripts/track_b_repeated.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader

from dl_models import DL_MODEL_BUILDERS, WINDOW_SAMPLES, N_EEG_CHANNELS

SEED_BASE = 42
N_SEEDS = 5
N_EPOCHS = 40
BATCH_SIZE = 128
PATIENCE = 8
TASK_ORDER = ["valence", "arousal", "dominance", "liking"]

CACHE_DIR = Path(r"D:/EEG/processed_cache")
RESULTS_DIR = Path(r"D:/EEG/results")
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASETS = {
    "non_overlap": CACHE_DIR / "deap_windows_4s_32subj_v2.npz",
    "overlap50": CACHE_DIR / "deap_windows_4s_overlap50_32subj_v1.npz",
}


class EEGWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float().unsqueeze(1)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
    }


def subject_independent_split(groups, test_size=0.15, val_size=0.15, seed=42):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(np.zeros(len(groups)), groups=groups))
    trainval_groups = groups[trainval_idx]
    relative_val_size = val_size / (1 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed)
    train_rel, val_rel = next(gss2.split(np.zeros(len(trainval_idx)), groups=trainval_groups))
    train_idx, val_idx = trainval_idx[train_rel], trainval_idx[val_rel]
    return train_idx, val_idx, test_idx


def train_torch_model(model_builder, train_ds, val_ds, test_ds, seed, epochs=N_EPOCHS,
                       batch_size=BATCH_SIZE, lr=1e-3, weight_decay=1e-4, patience=PATIENCE):
    torch.manual_seed(seed)
    model = model_builder().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=(DEVICE.type == "cuda"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, pin_memory=True)

    best_val_loss, best_state, no_improve = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                out = model(xb)
                loss = criterion(out, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                    out = model(xb)
                    loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds, probs, trues = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(DEVICE))
            probs.append(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
            preds.append(out.argmax(1).cpu().numpy())
            trues.append(yb.numpy())
    y_true = np.concatenate(trues)
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)
    return compute_metrics(y_true, y_pred, y_prob)


def run_dataset(variant_name, npz_path):
    npz = np.load(npz_path)
    X_raw = npz["X_raw"]
    groups = npz["groups"]
    y_by_task = {t: npz[f"y_{t}"] for t in TASK_ORDER}
    print(f"\n=== dataset={variant_name}  X_raw={X_raw.shape}  windows/subj~{len(groups)//32} ===")

    rows = []
    for task in TASK_ORDER:
        y_all = y_by_task[task]
        for model_name, builder in DL_MODEL_BUILDERS.items():
            seed_metrics = []
            seed_times = []
            for s in range(N_SEEDS):
                seed = SEED_BASE + s
                train_idx, val_idx, test_idx = subject_independent_split(groups, seed=seed)
                train_ds = EEGWindowDataset(X_raw[train_idx], y_all[train_idx])
                val_ds = EEGWindowDataset(X_raw[val_idx], y_all[val_idx])
                test_ds = EEGWindowDataset(X_raw[test_idx], y_all[test_idx])

                t0 = time.time()
                metrics = train_torch_model(builder, train_ds, val_ds, test_ds, seed=seed)
                seed_times.append(time.time() - t0)
                seed_metrics.append(metrics)
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            dfm = pd.DataFrame(seed_metrics)
            rows.append({
                "dataset": variant_name, "task": task, "model": model_name, "n_seeds": N_SEEDS,
                "accuracy_mean": dfm.accuracy.mean(), "accuracy_std": dfm.accuracy.std(),
                "f1_mean": dfm.f1_macro.mean(), "f1_std": dfm.f1_macro.std(),
                "roc_auc_mean": dfm.roc_auc.mean(), "roc_auc_std": dfm.roc_auc.std(),
                "train_time_s_mean": np.mean(seed_times),
            })
            print(f"[{variant_name}] {task:<10} {model_name:<15} "
                  f"acc={dfm.accuracy.mean():.3f}+/-{dfm.accuracy.std():.3f} "
                  f"f1={dfm.f1_macro.mean():.3f}+/-{dfm.f1_macro.std():.3f} "
                  f"auc={dfm.roc_auc.mean():.3f}+/-{dfm.roc_auc.std():.3f} "
                  f"({np.mean(seed_times):.1f}s/seed x {N_SEEDS})")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
    t0 = time.time()
    all_rows = []
    for variant_name, path in DATASETS.items():
        all_rows.append(run_dataset(variant_name, path))
    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(TABLES_DIR / "track_b_repeated_overlap_comparison.csv", index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. "
          f"Saved to {TABLES_DIR / 'track_b_repeated_overlap_comparison.csv'}")
