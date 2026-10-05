"""
AdaBN (Adaptive Batch Normalization, Li et al. 2018) test-time adaptation for
Track B -- logged in D:/EEG/papers_EEG/literature_log.txt as the natural,
cheap, target-aware follow-up to DANN (track_b_domain_adversarial.py),
motivated by 2504.03707's source-free unsupervised-domain-adaptation family
(Dual-Loss Adaptive Regularization + Localized Consistency Learning).

This implements the single cheapest, most established member of that family
-- AdaBN -- rather than 2504.03707's full pseudo-labeling + consistency
system, stated explicitly as a scoped-down single-technique test, same
honesty convention as the rest of this repo: recompute every BatchNorm
layer's running mean/variance using ONLY a held-out test subject's own
unlabeled windows (no gradient updates, no labels, no other subjects' data),
then evaluate that subject's predictions using the adapted statistics. The
hypothesis (from the source-free UDA literature): a chunk of the
subject-independent gap comes from each subject's own EEG having a different
marginal feature distribution than the pooled training population, and
BatchNorm statistics fit to the training population are a nuisance mismatch
that can be corrected per-subject at test time without ever needing that
subject's labels.

Procedure per task:
  1. Train one baseline EEGNet exactly as in track_b_domain_adversarial.py
     (same subject-independent 70/15/15 split, seed=42, CE loss, early
     stopping) -- this produces the "no_adaptation" condition directly.
  2. For EACH held-out test subject independently (a 70/15/15 split leaves
     ~4-5 test subjects): take a fresh copy of the trained baseline, call
     `reset_running_stats()` on every BatchNorm2d layer and set its momentum
     to None (so forward passes in train() mode compute a true cumulative
     average rather than an exponential moving average), run 2 adaptation
     epochs over that subject's own windows in train() mode under
     `torch.no_grad()` (updates only the BN running-stat buffers, no weight
     gradients), then switch to eval() mode and generate that subject's
     predictions using the now target-adapted statistics.
  3. Concatenate predictions across all test subjects for both conditions and
     compute the same metrics used throughout this repo.

Run: python scripts/track_b_test_time_adaptation.py
"""
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader

from dl_models import EEGNet, WINDOW_SAMPLES, N_EEG_CHANNELS

SEED = 42
N_EPOCHS = 40
BATCH_SIZE = 128
PATIENCE = 8
ADAPT_EPOCHS = 2
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


def train_baseline(train_ds, val_ds, seed):
    torch.manual_seed(seed)
    model = EEGNet(n_channels=N_EEG_CHANNELS, n_samples=WINDOW_SAMPLES, n_classes=2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True)

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
    return model


def predict(model, loader):
    model.eval()
    preds, probs, trues = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            out = model(xb)
            probs.append(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
            preds.append(out.argmax(1).cpu().numpy())
            trues.append(yb.numpy())
    return np.concatenate(trues), np.concatenate(preds), np.concatenate(probs)


def reset_bn_for_adaptation(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average instead of exponential


def adabn_predict_one_subject(base_model, X_subj, y_subj):
    model = copy.deepcopy(base_model)
    reset_bn_for_adaptation(model)
    ds = EEGWindowDataset(X_subj, y_subj)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    model.train()
    with torch.no_grad():
        for _ in range(ADAPT_EPOCHS):
            for xb, _ in loader:
                model(xb.to(DEVICE))

    eval_loader = DataLoader(ds, batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True)
    return predict(model, eval_loader)


def run_task(task, X_raw, groups, y_all):
    train_idx, val_idx, test_idx = subject_independent_split(groups, seed=SEED)
    train_ds = EEGWindowDataset(X_raw[train_idx], y_all[train_idx])
    val_ds = EEGWindowDataset(X_raw[val_idx], y_all[val_idx])

    t0 = time.time()
    base_model = train_baseline(train_ds, val_ds, seed=SEED)
    train_time = time.time() - t0

    test_subjects = np.unique(groups[test_idx])

    no_adapt_true, no_adapt_pred, no_adapt_prob = [], [], []
    adabn_true, adabn_pred, adabn_prob = [], [], []

    t0 = time.time()
    for sid in test_subjects:
        mask = (groups[test_idx] == sid)
        subj_local_idx = test_idx[mask]
        X_subj, y_subj = X_raw[subj_local_idx], y_all[subj_local_idx]

        subj_ds = EEGWindowDataset(X_subj, y_subj)
        subj_loader = DataLoader(subj_ds, batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True)
        yt, yp, ypr = predict(base_model, subj_loader)
        no_adapt_true.append(yt); no_adapt_pred.append(yp); no_adapt_prob.append(ypr)

        yt2, yp2, ypr2 = adabn_predict_one_subject(base_model, X_subj, y_subj)
        adabn_true.append(yt2); adabn_pred.append(yp2); adabn_prob.append(ypr2)
    adapt_time = time.time() - t0

    no_adapt_metrics = compute_metrics(np.concatenate(no_adapt_true), np.concatenate(no_adapt_pred),
                                        np.concatenate(no_adapt_prob))
    adabn_metrics = compute_metrics(np.concatenate(adabn_true), np.concatenate(adabn_pred),
                                     np.concatenate(adabn_prob))

    print(f"[{task}] n_test_subj={len(test_subjects)}  "
          f"no_adaptation: acc={no_adapt_metrics['accuracy']:.3f} auc={no_adapt_metrics['roc_auc']:.3f}  |  "
          f"adabn: acc={adabn_metrics['accuracy']:.3f} auc={adabn_metrics['roc_auc']:.3f} "
          f"(train {train_time:.1f}s, adapt+eval {adapt_time:.1f}s)")

    return [
        {"task": task, "condition": "no_adaptation", "train_time_s": train_time, **no_adapt_metrics},
        {"task": task, "condition": "adabn", "train_time_s": adapt_time, **adabn_metrics},
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
    results.to_csv(TABLES_DIR / "track_b_test_time_adaptation.csv", index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. "
          f"Saved to {TABLES_DIR / 'track_b_test_time_adaptation.csv'}")

    piv = results.pivot(index="task", columns="condition", values="roc_auc")
    piv["delta_auc"] = piv["adabn"] - piv["no_adaptation"]
    print("\nAdaBN vs. no adaptation, ROC-AUC:")
    print(piv.round(4))
