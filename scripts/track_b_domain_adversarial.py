"""
Subject-adversarial (DANN-style) training for Track B, added onto EEGNet.

This is the single most repeated concrete recommendation across the literature
review in D:/EEG/papers_EEG/ for closing a subject-independent gap: PARSE
(2202.05400), the EEG tutorial/review (2203.11279), the domain-adaptation
survey (2212.03176), DS-AGC (2308.11635), the GNN survey (2402.01138), and the
brainsci-16-00041 review all independently converge on a gradient-reversal-
layer (GRL) domain discriminator (Ganin & Lempitsky, 2016) trained jointly
with the emotion classifier so the shared encoder is pushed toward features
that a discriminator *cannot* use to tell subjects apart -- the theory being
that subject-identifiable structure is exactly the kind of nuisance variance
that should hurt subject-independent generalization.

Design (kept deliberately small/fast so it can run to completion on this
machine's RTX 3070 Ti in a few minutes, not the ~87 min of track_b_repeated.py):
  - EEGNet only (smallest Track B model, 2130 params) -- if the effect
    replicates here it is worth the larger investment of adding GRL to the
    other three architectures; if it does not, the smallest/cheapest model is
    the right one to have spent the time on.
  - Single subject-independent 70/15/15 GroupShuffleSplit per task (seed=42),
    not repeated -- matches the notebook's original section-6 protocol so the
    "baseline" arm here is a fair, freshly-computed control run with the exact
    same split as the "dann" arm (not reused from track_b_dl_results.csv,
    which may have used a different split/seed).
  - Domain labels = training-fold subject IDs only (unseen validation/test
    subjects are never given domain labels or gradients -- the discriminator
    only ever sees the ~22 training subjects; the goal is an encoder that
    generalizes what it learned about "features a subject-ID classifier could
    exploit" to unseen subjects too, which is exactly the DANN hypothesis
    being tested, not assumed).
  - Standard Ganin & Lempitsky progressive lambda schedule:
    lambda_p = 2/(1+exp(-10*p)) - 1, p = training progress in [0, 1].

Run: python scripts/track_b_domain_adversarial.py
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

from dl_models import EEGNet, WINDOW_SAMPLES, N_EEG_CHANNELS

SEED = 42
N_EPOCHS = 40
BATCH_SIZE = 128
PATIENCE = 8
TASK_ORDER = ["valence", "arousal", "dominance", "liking"]

CACHE_DIR = Path(r"D:/EEG/processed_cache")
RESULTS_DIR = Path(r"D:/EEG/results")
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOWS_PATH = CACHE_DIR / "deap_windows_4s_32subj_v2.npz"


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_):
    return GradReverse.apply(x, lambda_)


class EEGNetDANN(nn.Module):
    """Reuses EEGNet's own conv blocks (instantiated once to get correctly-shaped
    submodules via its existing dummy-forward flat_dim inference) and adds a
    GRL + subject-domain-classifier branch off the same flattened features that
    feed the emotion classifier."""

    def __init__(self, n_channels=N_EEG_CHANNELS, n_samples=WINDOW_SAMPLES, n_classes=2, n_domains=1):
        super().__init__()
        base = EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes)
        self.firstconv = base.firstconv
        self.depthwiseConv = base.depthwiseConv
        self.separableConv = base.separableConv
        self.classifier = base.classifier
        flat_dim = base.classifier.in_features
        self.domain_classifier = nn.Sequential(
            nn.Linear(flat_dim, 64), nn.ReLU(), nn.Linear(64, n_domains)
        )

    def features(self, x):
        return self.separableConv(self.depthwiseConv(self.firstconv(x))).flatten(1)

    def forward(self, x, lambda_=0.0):
        feat = self.features(x)
        emotion_logits = self.classifier(feat)
        domain_logits = self.domain_classifier(grad_reverse(feat, lambda_))
        return emotion_logits, domain_logits


class EEGWindowDataset(Dataset):
    def __init__(self, X, y, domain_id=None):
        self.X = torch.from_numpy(X).float().unsqueeze(1)
        self.y = torch.from_numpy(y).long()
        self.domain_id = torch.from_numpy(domain_id).long() if domain_id is not None else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.domain_id is not None:
            return self.X[idx], self.y[idx], self.domain_id[idx]
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


def evaluate(model, loader, use_dann):
    model.eval()
    preds, probs, trues = [], [], []
    with torch.no_grad():
        for batch in loader:
            xb, yb = batch[0], batch[1]
            xb = xb.to(DEVICE)
            out = model(xb, lambda_=0.0)[0] if use_dann else model(xb)
            probs.append(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
            preds.append(out.argmax(1).cpu().numpy())
            trues.append(yb.numpy())
    y_true = np.concatenate(trues)
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)
    return compute_metrics(y_true, y_pred, y_prob)


def train_baseline(train_ds, val_ds, test_ds, seed):
    torch.manual_seed(seed)
    model = EEGNet(n_channels=N_EEG_CHANNELS, n_samples=WINDOW_SAMPLES, n_classes=2).to(DEVICE)
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
    return evaluate(model, test_loader, use_dann=False)


def train_dann(train_ds, val_ds, test_ds, seed, n_domains):
    torch.manual_seed(seed)
    model = EEGNetDANN(n_channels=N_EEG_CHANNELS, n_samples=WINDOW_SAMPLES,
                        n_classes=2, n_domains=n_domains).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    emotion_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True)

    n_batches_per_epoch = max(1, len(train_loader))
    total_steps = N_EPOCHS * n_batches_per_epoch
    global_step = 0

    best_val_loss, best_state, no_improve = float("inf"), None, 0
    for epoch in range(N_EPOCHS):
        model.train()
        for xb, yb, db in train_loader:
            p = global_step / total_steps
            lambda_p = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
            xb, yb, db = xb.to(DEVICE), yb.to(DEVICE), db.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            emotion_logits, domain_logits = model(xb, lambda_=lambda_p)
            loss = emotion_criterion(emotion_logits, yb) + domain_criterion(domain_logits, db)
            loss.backward()
            opt.step()
            global_step += 1

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                emotion_logits, _ = model(xb, lambda_=0.0)
                val_loss += emotion_criterion(emotion_logits, yb).item() * xb.size(0)
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
    return evaluate(model, test_loader, use_dann=True)


def run_task(task, X_raw, groups, y_all):
    train_idx, val_idx, test_idx = subject_independent_split(groups, seed=SEED)

    train_subjects = np.unique(groups[train_idx])
    subj_to_domain = {s: i for i, s in enumerate(train_subjects)}
    domain_id_train = np.array([subj_to_domain[s] for s in groups[train_idx]], dtype=np.int64)

    train_ds_plain = EEGWindowDataset(X_raw[train_idx], y_all[train_idx])
    val_ds_plain = EEGWindowDataset(X_raw[val_idx], y_all[val_idx])
    test_ds_plain = EEGWindowDataset(X_raw[test_idx], y_all[test_idx])

    train_ds_dann = EEGWindowDataset(X_raw[train_idx], y_all[train_idx], domain_id_train)

    t0 = time.time()
    baseline_metrics = train_baseline(train_ds_plain, val_ds_plain, test_ds_plain, seed=SEED)
    baseline_time = time.time() - t0
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    t0 = time.time()
    dann_metrics = train_dann(train_ds_dann, val_ds_plain, test_ds_plain, seed=SEED,
                               n_domains=len(train_subjects))
    dann_time = time.time() - t0
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"[{task}] n_train_subj={len(train_subjects)}  "
          f"baseline: acc={baseline_metrics['accuracy']:.3f} auc={baseline_metrics['roc_auc']:.3f} "
          f"({baseline_time:.1f}s)  |  "
          f"dann: acc={dann_metrics['accuracy']:.3f} auc={dann_metrics['roc_auc']:.3f} ({dann_time:.1f}s)")

    return [
        {"task": task, "condition": "baseline", "train_time_s": baseline_time, **baseline_metrics},
        {"task": task, "condition": "dann", "train_time_s": dann_time, **dann_metrics},
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
    results.to_csv(TABLES_DIR / "track_b_domain_adversarial.csv", index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. "
          f"Saved to {TABLES_DIR / 'track_b_domain_adversarial.csv'}")

    piv = results.pivot(index="task", columns="condition", values="roc_auc")
    piv["delta_auc"] = piv["dann"] - piv["baseline"]
    print("\nDANN vs. baseline EEGNet, ROC-AUC:")
    print(piv.round(4))
