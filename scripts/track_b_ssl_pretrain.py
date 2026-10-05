"""
Self-supervised (TFR-style) pretraining for Track B -- the single
highest-priority deferred idea in D:/EEG/papers_EEG/literature_log.txt.
Source: 2403.04041 ("Cascaded Self-supervised Learning for
Subject-independent EEG-based Emotion Recognition", Wang/Chen/Song, Fudan
2024), corroborated by the SSL-for-EEG survey (2401.05446). Uniquely among the
contrastive/self-supervised papers reviewed, 2403.04041 is evaluated LOSO on
DEAP itself and reports beating supervised domain-adaptation baselines with NO
test-time fine-tuning -- the strongest outside evidence in the whole review
that a pretraining stage (rather than a training-time loss/architecture
change) could move this project's subject-independent AUC.

Simplified pretext task (stated explicitly, not a full reproduction of the
paper's dual-stream time/frequency contrastive system): reuse EEGNet's own
conv trunk (firstconv/depthwiseConv/separableConv, same reuse pattern as
EEGNetDANN in track_b_domain_adversarial.py) as the encoder, add a small
linear regression head, and pretrain with MSE to predict each window's own
32-channel x 4-band log-band-power vector (the same Differential-Entropy
target used throughout Track A, computed directly from the raw window -- a
standard, fully label-free "time -> frequency-summary" pretext task in the
same spirit as TFR: the network must learn an internal representation from
which spectral content is linearly recoverable, without ever seeing an
emotion label). Pretraining uses ONLY the training-fold subjects' windows.

After pretraining, the trunk's weights are copied into a fresh EEGNet (with a
newly-initialized, never-pretrained classifier head) and the whole network is
fine-tuned end-to-end on the labeled training-fold data -- compared against an
identically-configured "scratch" EEGNet trained the same way but from random
initialization. Both arms use the exact same subject-independent 70/15/15
split/seed/training loop as track_b_domain_adversarial.py's baseline arm.

Compute note: subject_independent_split depends only on `groups` and `seed`,
not on the per-task labels, so the train/val/test subject partition is
IDENTICAL across all four tasks. The pretraining stage therefore runs only
ONCE (using the shared training-fold subjects' windows, no labels involved),
and its resulting trunk weights are reused as the fine-tuning initialization
for all four per-task runs -- this is a compute optimization, not a
methodological shortcut: each task still gets its own independent,
task-specific fine-tuning run and evaluation.

Run: python scripts/track_b_ssl_pretrain.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from dl_models import EEGNet, WINDOW_SAMPLES, N_EEG_CHANNELS

SEED = 42
N_EPOCHS = 40
N_EPOCHS_PRETRAIN = 15
BATCH_SIZE = 128
PATIENCE = 8
TASK_ORDER = ["valence", "arousal", "dominance", "liking"]

CACHE_DIR = Path(r"D:/EEG/processed_cache")
RESULTS_DIR = Path(r"D:/EEG/results")
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOWS_PATH = CACHE_DIR / "deap_windows_4s_32subj_v2.npz"

FS = 128
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
BAND_ORDER = ["theta", "alpha", "beta", "gamma"]
_SOS_FILTERS = {name: butter(4, band, btype="bandpass", fs=FS, output="sos") for name, band in BANDS.items()}


def compute_de_targets_batch(X):
    """X: (N, 32, T) raw windows -> (N, 32*4) log-band-power targets, same
    formula/order as build_extended_features.py::compute_de_features."""
    de_per_band = []
    for band_name in BAND_ORDER:
        filtered = sosfiltfilt(_SOS_FILTERS[band_name], X, axis=-1)
        var = filtered.var(axis=-1) + 1e-10
        de_per_band.append(0.5 * np.log(2 * np.pi * np.e * var))
    de_stack = np.stack(de_per_band, axis=-1)  # (N, 32, 4)
    return de_stack.reshape(X.shape[0], -1).astype(np.float32)


class EEGNetSSL(nn.Module):
    """Shares EEGNet's conv trunk (instantiated once via a base EEGNet to get
    correctly-shaped submodules) with a regression head for the pretext task."""

    def __init__(self, n_channels, n_samples, n_targets):
        super().__init__()
        base = EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=2)
        self.firstconv = base.firstconv
        self.depthwiseConv = base.depthwiseConv
        self.separableConv = base.separableConv
        flat_dim = base.classifier.in_features
        self.pretrain_head = nn.Linear(flat_dim, n_targets)

    def features(self, x):
        return self.separableConv(self.depthwiseConv(self.firstconv(x))).flatten(1)

    def forward(self, x):
        return self.pretrain_head(self.features(x))


class EEGWindowDataset(Dataset):
    def __init__(self, X, targets):
        self.X = torch.from_numpy(X).float().unsqueeze(1)
        self.targets = torch.from_numpy(targets).float() if targets.dtype != np.int64 \
            else torch.from_numpy(targets).long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.X[idx], self.targets[idx]


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


def pretrain_trunk(X_train_raw, seed):
    targets = compute_de_targets_batch(X_train_raw)
    target_scaler = StandardScaler().fit(targets)
    targets = target_scaler.transform(targets).astype(np.float32)

    torch.manual_seed(seed)
    model = EEGNetSSL(n_channels=N_EEG_CHANNELS, n_samples=WINDOW_SAMPLES, n_targets=targets.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    ds = EEGWindowDataset(X_train_raw, targets)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    model.train()
    for epoch in range(N_EPOCHS_PRETRAIN):
        epoch_loss, n = 0.0, 0
        for xb, tb in loader:
            xb, tb = xb.to(DEVICE), tb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, tb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        if epoch == 0 or epoch == N_EPOCHS_PRETRAIN - 1:
            print(f"  [pretrain] epoch {epoch+1}/{N_EPOCHS_PRETRAIN}  MSE={epoch_loss/n:.4f}")

    trunk_state = {
        "firstconv": model.firstconv.state_dict(),
        "depthwiseConv": model.depthwiseConv.state_dict(),
        "separableConv": model.separableConv.state_dict(),
    }
    return trunk_state


def evaluate(model, loader):
    model.eval()
    preds, probs, trues = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            out = model(xb)
            probs.append(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
            preds.append(out.argmax(1).cpu().numpy())
            trues.append(yb.numpy())
    y_true = np.concatenate(trues)
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)
    return compute_metrics(y_true, y_pred, y_prob)


def train_classifier(train_ds, val_ds, test_ds, seed, trunk_state=None):
    torch.manual_seed(seed)
    model = EEGNet(n_channels=N_EEG_CHANNELS, n_samples=WINDOW_SAMPLES, n_classes=2).to(DEVICE)
    if trunk_state is not None:
        model.firstconv.load_state_dict(trunk_state["firstconv"])
        model.depthwiseConv.load_state_dict(trunk_state["depthwiseConv"])
        model.separableConv.load_state_dict(trunk_state["separableConv"])

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True)

    best_val_loss, best_state, no_improve = float("inf"), None, 0
    for epoch in range(N_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss, best_state, no_improve = val_loss, \
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return evaluate(model, test_loader)


def run_task(task, X_raw, groups, y_all, trunk_state):
    train_idx, val_idx, test_idx = subject_independent_split(groups, seed=SEED)
    y_train_ds = EEGWindowDataset(X_raw[train_idx], y_all[train_idx])
    y_val_ds = EEGWindowDataset(X_raw[val_idx], y_all[val_idx])
    y_test_ds = EEGWindowDataset(X_raw[test_idx], y_all[test_idx])

    t0 = time.time()
    scratch_metrics = train_classifier(y_train_ds, y_val_ds, y_test_ds, seed=SEED, trunk_state=None)
    scratch_time = time.time() - t0
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    t0 = time.time()
    ssl_metrics = train_classifier(y_train_ds, y_val_ds, y_test_ds, seed=SEED, trunk_state=trunk_state)
    ssl_time = time.time() - t0
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"[{task}] scratch: acc={scratch_metrics['accuracy']:.3f} auc={scratch_metrics['roc_auc']:.3f} "
          f"({scratch_time:.1f}s)  |  ssl_pretrained: acc={ssl_metrics['accuracy']:.3f} "
          f"auc={ssl_metrics['roc_auc']:.3f} ({ssl_time:.1f}s)")

    return [
        {"task": task, "condition": "scratch", "train_time_s": scratch_time, **scratch_metrics},
        {"task": task, "condition": "ssl_pretrained", "train_time_s": ssl_time, **ssl_metrics},
    ]


if __name__ == "__main__":
    print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
    npz = np.load(WINDOWS_PATH)
    X_raw = npz["X_raw"]
    groups = npz["groups"]
    y_by_task = {t: npz[f"y_{t}"] for t in TASK_ORDER}

    # split depends only on groups/seed -> identical across tasks, computed once
    train_idx, val_idx, test_idx = subject_independent_split(groups, seed=SEED)

    print(f"Pretraining trunk on {len(train_idx)} training-fold windows "
          f"({len(np.unique(groups[train_idx]))} subjects), labels not used...")
    t0 = time.time()
    trunk_state = pretrain_trunk(X_raw[train_idx], seed=SEED)
    print(f"Pretraining done in {(time.time()-t0)/60:.1f} min")

    t0 = time.time()
    rows = []
    for task in TASK_ORDER:
        rows.extend(run_task(task, X_raw, groups, y_by_task[task], trunk_state))
    results = pd.DataFrame(rows)
    results.to_csv(TABLES_DIR / "track_b_ssl_pretrain.csv", index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. "
          f"Saved to {TABLES_DIR / 'track_b_ssl_pretrain.csv'}")

    piv = results.pivot(index="task", columns="condition", values="roc_auc")
    piv["delta_auc"] = piv["ssl_pretrained"] - piv["scratch"]
    print("\nSSL-pretrained vs. scratch EEGNet, ROC-AUC:")
    print(piv.round(4))
