"""
AUC-margin training loss for Track B -- logged in
D:/EEG/papers_EEG/literature_log.txt (2408.08979, Xiao 2024/2025) as the most
directly on-metric deferred idea in the whole literature review: this
project's own primary reported metric IS ROC-AUC, so a loss function that
directly targets AUC rather than per-sample cross-entropy is a natural,
cheap-to-test intervention (a loss swap, not a new architecture or training
stage).

This implements a simplified, self-contained version of the Ying et al.
AUC-margin idea that the source paper builds its saddle-point optimizer
around: a mini-batch pairwise squared-hinge surrogate,

    L_auc = mean over (i in positive, j in negative) of
            max(0, margin - (s_i - s_j))^2

where s = the positive-class logit. This is the same quantity the
AUC-maximization literature works with (a margin-based, differentiable proxy
for the probability that a random positive scores above a random negative,
i.e. AUC itself) computed directly and averaged within each mini-batch, rather
than via the paper's full Alt-GDA/ExtraGradient min-max solver -- stated
explicitly here as a scoped-down version of that paper's idea, same honesty
convention as the rest of this repo, not a full reproduction of their
convex-concave optimization machinery.

Everything else (model, subject-independent 70/15/15 split, seed, optimizer,
scheduler, early stopping) is identical to the baseline arm of
scripts/track_b_domain_adversarial.py, so the only thing that differs between
the two conditions here is the loss function used during training (evaluation
is always plain accuracy/F1/AUC on class probabilities, for both conditions).

Run: python scripts/track_b_auc_maximization.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader

from dl_models import EEGNet, WINDOW_SAMPLES, N_EEG_CHANNELS

SEED = 42
N_EPOCHS = 40
BATCH_SIZE = 128
PATIENCE = 8
AUC_MARGIN = 1.0
TASK_ORDER = ["valence", "arousal", "dominance", "liking"]

CACHE_DIR = Path(r"D:/EEG/processed_cache")
RESULTS_DIR = Path(r"D:/EEG/results")
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOWS_PATH = CACHE_DIR / "deap_windows_4s_32subj_v2.npz"


class EEGWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float().unsqueeze(1)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def auc_margin_loss(logits, y, margin=AUC_MARGIN):
    """Mini-batch pairwise squared-hinge AUC surrogate. logits: (B, 2) raw scores;
    uses the positive-class logit as the score s. Falls back to 0 (no gradient
    contribution) if the batch has only one class present."""
    s = logits[:, 1]
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    diff = margin - (pos.unsqueeze(1) - neg.unsqueeze(0))  # (n_pos, n_neg)
    return F.relu(diff).pow(2).mean()


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


def train_model(train_ds, val_ds, test_ds, seed, use_auc_loss):
    torch.manual_seed(seed)
    model = EEGNet(n_channels=N_EEG_CHANNELS, n_samples=WINDOW_SAMPLES, n_classes=2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    ce_criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True)

    best_val_loss, best_state, no_improve = float("inf"), None, 0
    for epoch in range(N_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = auc_margin_loss(logits, yb) if use_auc_loss else ce_criterion(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                vloss = auc_margin_loss(logits, yb) if use_auc_loss else ce_criterion(logits, yb)
                val_loss += vloss.item() * xb.size(0)
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


def run_task(task, X_raw, groups, y_all):
    train_idx, val_idx, test_idx = subject_independent_split(groups, seed=SEED)
    train_ds = EEGWindowDataset(X_raw[train_idx], y_all[train_idx])
    val_ds = EEGWindowDataset(X_raw[val_idx], y_all[val_idx])
    test_ds = EEGWindowDataset(X_raw[test_idx], y_all[test_idx])

    t0 = time.time()
    ce_metrics = train_model(train_ds, val_ds, test_ds, seed=SEED, use_auc_loss=False)
    ce_time = time.time() - t0
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    t0 = time.time()
    auc_metrics = train_model(train_ds, val_ds, test_ds, seed=SEED, use_auc_loss=True)
    auc_time = time.time() - t0
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"[{task}] ce_baseline: acc={ce_metrics['accuracy']:.3f} auc={ce_metrics['roc_auc']:.3f} "
          f"({ce_time:.1f}s)  |  auc_max: acc={auc_metrics['accuracy']:.3f} auc={auc_metrics['roc_auc']:.3f} "
          f"({auc_time:.1f}s)")

    return [
        {"task": task, "condition": "ce_baseline", "train_time_s": ce_time, **ce_metrics},
        {"task": task, "condition": "auc_max", "train_time_s": auc_time, **auc_metrics},
    ]


if __name__ == "__main__":
    print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
    npz = np.load(WINDOWS_PATH)
    X_raw = npz["X_raw"]
    groups = npz["groups"]
    y_by_task = {t: npz[f"y_{t}"] for t in TASK_ORDER}

    t0 = time.time()
    rows = []
    for task in TASK_ORDER:
        rows.extend(run_task(task, X_raw, groups, y_by_task[task]))
    results = pd.DataFrame(rows)
    results.to_csv(TABLES_DIR / "track_b_auc_maximization.csv", index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. "
          f"Saved to {TABLES_DIR / 'track_b_auc_maximization.csv'}")

    piv = results.pivot(index="task", columns="condition", values="roc_auc")
    piv["delta_auc"] = piv["auc_max"] - piv["ce_baseline"]
    print("\nAUC-margin loss vs. cross-entropy, ROC-AUC:")
    print(piv.round(4))
