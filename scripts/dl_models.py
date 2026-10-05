"""
Exact port of the notebook's Track B model architectures (EEGNet, ShallowConvNet,
CNN-LSTM, TSception) plus the hemisphere-pair indices TSception needs, so training
scripts can import them without re-executing the whole notebook.
"""
import numpy as np
import torch
import torch.nn as nn

FS = 128
WINDOW_SEC = 4
WINDOW_SAMPLES = WINDOW_SEC * FS
N_EEG_CHANNELS = 32

DEAP_CHANNELS = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
]
LEFT_RIGHT_PAIRS = [
    ("Fp1", "Fp2"), ("AF3", "AF4"), ("F3", "F4"), ("F7", "F8"), ("FC5", "FC6"), ("FC1", "FC2"),
    ("C3", "C4"), ("T7", "T8"), ("CP5", "CP6"), ("CP1", "CP2"), ("P3", "P4"), ("P7", "P8"),
    ("PO3", "PO4"), ("O1", "O2"),
]
LEFT_IDX = [DEAP_CHANNELS.index(l) for l, r in LEFT_RIGHT_PAIRS]
RIGHT_IDX = [DEAP_CHANNELS.index(r) for l, r in LEFT_RIGHT_PAIRS]


class EEGNet(nn.Module):
    def __init__(self, n_channels=32, n_samples=WINDOW_SAMPLES, n_classes=2,
                 F1=8, D=2, F2=16, kernel_length=64, dropout=0.5):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            flat_dim = self.separableConv(self.depthwiseConv(self.firstconv(dummy))).numel()
        self.classifier = nn.Linear(flat_dim, n_classes)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        return self.classifier(x.flatten(1))


class ShallowConvNet(nn.Module):
    def __init__(self, n_channels=32, n_samples=WINDOW_SAMPLES, n_classes=2,
                 n_filters=40, dropout=0.5):
        super().__init__()
        self.temporal_conv = nn.Conv2d(1, n_filters, (1, 25))
        self.spatial_conv = nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(dropout)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            flat_dim = self._features(dummy).numel()
        self.classifier = nn.Linear(flat_dim, n_classes)

    def _features(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        return x

    def forward(self, x):
        x = self._features(x)
        x = self.dropout(x)
        return self.classifier(x.flatten(1))


class CNNLSTM(nn.Module):
    def __init__(self, n_channels=32, n_samples=WINDOW_SAMPLES, n_classes=2,
                 cnn_channels=32, lstm_hidden=64, dropout=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, cnn_channels, (n_channels, 5), padding=(0, 2)),
            nn.BatchNorm2d(cnn_channels), nn.ELU(), nn.MaxPool2d((1, 2)),
            nn.Conv2d(cnn_channels, cnn_channels * 2, (1, 5), padding=(0, 2)),
            nn.BatchNorm2d(cnn_channels * 2), nn.ELU(), nn.MaxPool2d((1, 2)),
        )
        self.lstm = nn.LSTM(input_size=cnn_channels * 2, hidden_size=lstm_hidden,
                             num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, n_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze(2).permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        feat = torch.cat([h[-2], h[-1]], dim=1)
        return self.classifier(self.dropout(feat))


class TSception(nn.Module):
    def __init__(self, n_channels=32, n_samples=WINDOW_SAMPLES, n_classes=2, fs=FS,
                 num_t_filters=9, num_s_filters=6, t_fixed_len=32, dropout=0.5,
                 left_idx=None, right_idx=None):
        super().__init__()
        self.left_idx = left_idx if left_idx is not None else LEFT_IDX
        self.right_idx = right_idx if right_idx is not None else RIGHT_IDX
        kernel_sizes = [max(2, int(f * fs)) for f in (0.5, 0.25, 0.125)]
        self.t_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, num_t_filters, (1, k), stride=(1, max(1, k // 2))),
                nn.BatchNorm2d(num_t_filters), nn.ELU(),
                nn.AdaptiveAvgPool2d((n_channels, t_fixed_len)),
            ) for k in kernel_sizes
        ])
        n_pairs = len(self.left_idx)
        self.s_global = nn.Sequential(
            nn.Conv2d(num_t_filters * 3, num_s_filters, (n_channels, 1)),
            nn.BatchNorm2d(num_s_filters), nn.ELU())
        self.s_pair = nn.Sequential(
            nn.Conv2d(num_t_filters * 3, num_s_filters, (n_pairs, 1)),
            nn.BatchNorm2d(num_s_filters), nn.ELU())
        self.fusion = nn.Sequential(
            nn.Conv2d(num_s_filters, num_s_filters, (2, 1)),
            nn.BatchNorm2d(num_s_filters), nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 8)))
        self.dropout = nn.Dropout(dropout)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            flat_dim = self._forward_features(dummy).numel()
        self.classifier = nn.Linear(flat_dim, n_classes)

    def _forward_features(self, x):
        t_outs = [b(x) for b in self.t_branches]
        t_cat = torch.cat(t_outs, dim=1)
        s_g = self.s_global(t_cat)
        left = t_cat[:, :, self.left_idx, :]
        right = t_cat[:, :, self.right_idx, :]
        s_asym = torch.abs(self.s_pair(left) - self.s_pair(right))
        s_cat = torch.cat([s_g, s_asym], dim=2)
        return self.fusion(s_cat)

    def forward(self, x):
        feat = self.dropout(self._forward_features(x))
        return self.classifier(feat.flatten(1))


DL_MODEL_BUILDERS = {
    "ShallowConvNet": ShallowConvNet,
    "EEGNet": EEGNet,
    "CNN-LSTM": CNNLSTM,
    "TSception": TSception,
}
