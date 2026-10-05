"""
Richer Track A features for the DEAP comparative-analysis notebook:
Hjorth parameters, frontal/hemispheric alpha asymmetry, and PLV/coherence
connectivity, all derived from the same baseline-normalized, pre-DL-standardization
signal used for Differential Entropy (`trial_norm`, shape (32, 7680)).

Kept as a standalone module (rather than only notebook cells) so it can be
unit-tested quickly with `python scripts/rich_features.py` before being pasted
into the notebook.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

FS = 128
WINDOW_SEC = 4
WINDOW_SAMPLES = WINDOW_SEC * FS  # 512

BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
BAND_ORDER = ["theta", "alpha", "beta", "gamma"]
_SOS_FILTERS = {name: butter(4, band, btype="bandpass", fs=FS, output="sos") for name, band in BANDS.items()}

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
ASYM_PAIR_NAMES = [f"{l}-{r}" for l, r in LEFT_RIGHT_PAIRS]
# Classic frontal asymmetry literature (Davidson-style) uses F3/F4, F7/F8, Fp1/Fp2, FC5/FC6;
# the other 10 pairs give a fuller hemispheric-asymmetry picture as a bonus/ablation arm.
FRONTAL_PAIR_NAMES = {"Fp1-Fp2", "F3-F4", "F7-F8", "FC5-FC6"}

# Curated long-range connectivity pairs: interhemispheric homologs (reuse LEFT_RIGHT_PAIRS)
# plus canonical fronto-parietal/fronto-occipital pairs implicated in valence/arousal networks.
EXTRA_CONN_PAIRS = [("F3", "P3"), ("F4", "P4"), ("Fz", "Pz"), ("F3", "O1"), ("F4", "O2")]
CONN_PAIRS = LEFT_RIGHT_PAIRS + EXTRA_CONN_PAIRS
CONN_PAIR_NAMES = [f"{a}-{b}" for a, b in CONN_PAIRS]
CONN_A_IDX = [DEAP_CHANNELS.index(a) for a, b in CONN_PAIRS]
CONN_B_IDX = [DEAP_CHANNELS.index(b) for a, b in CONN_PAIRS]


def slice_windows_raw(trial_norm, window_samples=WINDOW_SAMPLES):
    """(32, 7680) -> (n_win, 32, window_samples), NOT z-scored (unlike make_raw_windows)."""
    n_ch, n_samp = trial_norm.shape
    n_win = n_samp // window_samples
    usable = n_win * window_samples
    return trial_norm[:, :usable].reshape(n_ch, n_win, window_samples).transpose(1, 0, 2)


def compute_hjorth(trial_norm, window_samples=WINDOW_SAMPLES):
    """-> (n_win, 3*32) = [activity x32, mobility x32, complexity x32]."""
    w = slice_windows_raw(trial_norm, window_samples).transpose(1, 0, 2)  # (32, n_win, T)
    eps = 1e-10
    activity = w.var(axis=-1)                      # (32, n_win)
    dx = np.diff(w, axis=-1)
    var_dx = dx.var(axis=-1)
    mobility = np.sqrt(var_dx / (activity + eps))
    ddx = np.diff(dx, axis=-1)
    var_ddx = ddx.var(axis=-1)
    mobility_dx = np.sqrt(var_ddx / (var_dx + eps))
    complexity = mobility_dx / (mobility + eps)
    stacked = np.stack([activity, mobility, complexity], axis=0)  # (3, 32, n_win)
    return stacked.transpose(2, 0, 1).reshape(stacked.shape[-1], -1).astype(np.float32)  # (n_win, 96)


def hjorth_feature_names():
    return [f"hjorth_{p}_{ch}" for p in ("activity", "mobility", "complexity") for ch in DEAP_CHANNELS]


def compute_alpha_asymmetry(de_feats_flat, n_channels=32, band_order=BAND_ORDER):
    """de_feats_flat: (n_win, n_channels*n_bands) as produced by the notebook's
    compute_de_features (channel-major, band-minor flatten) -> (n_win, 14) hemispheric
    alpha asymmetry = alpha_power[right] - alpha_power[left]. Positive => relatively
    greater right-hemisphere alpha power => (per Davidson-style asymmetry literature)
    relatively greater LEFT cortical activation, classically linked to approach /
    positive-valence motivational tendency. Sign convention is documented, not assumed
    causal, and should be validated empirically on this dataset's own labels."""
    n_win = de_feats_flat.shape[0]
    n_bands = len(band_order)
    de_3d = de_feats_flat.reshape(n_win, n_channels, n_bands)
    alpha_idx = band_order.index("alpha")
    alpha_power = de_3d[:, :, alpha_idx]  # (n_win, 32)
    return (alpha_power[:, RIGHT_IDX] - alpha_power[:, LEFT_IDX]).astype(np.float32)  # (n_win, 14)


def _segment(x, nperseg, noverlap):
    """x: (..., T) -> (..., n_seg, nperseg) via a strided view (no copy of the base array)."""
    step = nperseg - noverlap
    T = x.shape[-1]
    n_seg = (T - nperseg) // step + 1
    idx = np.arange(nperseg)[None, :] + step * np.arange(n_seg)[:, None]  # (n_seg, nperseg)
    return x[..., idx]


def _coherence_band(x, y, band, fs=FS, nperseg=128, noverlap=64):
    """Welch-style magnitude-squared coherence, band-averaged. x, y: (..., T) same shape -> (...,)."""
    win = np.hanning(nperseg)
    xs = _segment(x, nperseg, noverlap) * win
    ys = _segment(y, nperseg, noverlap) * win
    Xf = np.fft.rfft(xs, axis=-1)
    Yf = np.fft.rfft(ys, axis=-1)
    freqs = np.fft.rfftfreq(nperseg, d=1 / fs)
    Sxx = (Xf * np.conj(Xf)).real.mean(axis=-2)
    Syy = (Yf * np.conj(Yf)).real.mean(axis=-2)
    Sxy = (Xf * np.conj(Yf)).mean(axis=-2)
    coh = (np.abs(Sxy) ** 2) / (Sxx * Syy + 1e-12)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    return coh[..., band_mask].mean(axis=-1)


def compute_connectivity(trial_norm, window_samples=WINDOW_SAMPLES, plv_bands=("alpha", "beta")):
    """PLV (per band in plv_bands) + alpha-band coherence, over CONN_PAIRS.
    -> (n_win, len(plv_bands)*len(CONN_PAIRS) + len(CONN_PAIRS))"""
    windows = slice_windows_raw(trial_norm, window_samples)  # (n_win, 32, T)
    n_win = windows.shape[0]
    feats = []

    for band in plv_bands:
        filtered = sosfiltfilt(_SOS_FILTERS[band], trial_norm, axis=-1)  # (32, T_full)
        f_windows = slice_windows_raw(filtered, window_samples).transpose(1, 0, 2)  # (32, n_win, T)
        analytic = hilbert(f_windows, axis=-1)
        phase = np.angle(analytic)  # (32, n_win, T)
        for a_idx, b_idx in zip(CONN_A_IDX, CONN_B_IDX):
            dphi = phase[a_idx] - phase[b_idx]                       # (n_win, T)
            plv = np.abs(np.mean(np.exp(1j * dphi), axis=-1))        # (n_win,)
            feats.append(plv)

    x_all = windows.transpose(1, 0, 2)  # (32, n_win, T)
    for a_idx, b_idx in zip(CONN_A_IDX, CONN_B_IDX):
        coh = _coherence_band(x_all[a_idx], x_all[b_idx], BANDS["alpha"])  # (n_win,)
        feats.append(coh)

    return np.stack(feats, axis=-1).astype(np.float32)  # (n_win, n_feats)


def connectivity_feature_names(plv_bands=("alpha", "beta")):
    names = [f"plv_{band}_{p}" for band in plv_bands for p in CONN_PAIR_NAMES]
    names += [f"coh_alpha_{p}" for p in CONN_PAIR_NAMES]
    return names


if __name__ == "__main__":
    import time
    rng = np.random.RandomState(0)
    fake_trial = rng.randn(32, 7680) * 20.0  # rough EEG-scale synthetic signal

    t0 = time.time()
    hj = compute_hjorth(fake_trial)
    print("hjorth:", hj.shape, "expect (15, 96)  time=%.3fs" % (time.time() - t0))
    assert hj.shape == (15, 96)
    assert len(hjorth_feature_names()) == 96

    t0 = time.time()
    conn = compute_connectivity(fake_trial)
    print("connectivity:", conn.shape, "expect (15, 57)  time=%.3fs" % (time.time() - t0))
    assert conn.shape == (15, 57)
    assert len(connectivity_feature_names()) == 57
    assert np.all(conn[:, :38] >= -1e-6) and np.all(conn[:, :38] <= 1 + 1e-6), "PLV must be in [0,1]"
    assert np.all(conn[:, 38:] >= -1e-6) and np.all(conn[:, 38:] <= 1 + 1e-6), "coherence must be in [0,1]"

    # fabricate a fake DE array (n_win, 128) to test asymmetry extraction / index correctness
    fake_de = rng.randn(15, 128).astype(np.float32)
    faa = compute_alpha_asymmetry(fake_de)
    print("faa:", faa.shape, "expect (15, 14)")
    assert faa.shape == (15, 14)
    # sanity: verify index math directly against the reshape used in compute_alpha_asymmetry
    de_3d = fake_de.reshape(15, 32, 4)
    alpha_idx = BAND_ORDER.index("alpha")
    expect = de_3d[:, RIGHT_IDX, alpha_idx] - de_3d[:, LEFT_IDX, alpha_idx]
    assert np.allclose(faa, expect)

    print("ALL CHECKS PASSED")
    print("frontal pair subset:", [n for n in ASYM_PAIR_NAMES if n in FRONTAL_PAIR_NAMES])
