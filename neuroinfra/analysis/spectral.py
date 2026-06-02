"""Reusable spectral-analysis helpers for uniformly sampled traces."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import pywt
except ImportError:  # pragma: no cover - optional runtime dependency
    pywt = None

from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, lfilter, spectrogram, welch


DEFAULT_HFO_BANDS = {
    "hfo_80_130": (80.0, 130.0),
    "hfo_130_180": (130.0, 180.0),
}


def trapezoid_integral(y: Any, x: Any) -> float:
    """Integrate one sampled curve across NumPy 1.x/2.x."""
    integrator = getattr(np, "trapezoid", None)
    if integrator is None:
        integrator = np.trapz
    return float(integrator(y, x))


def uniform_trace(
    t: np.ndarray | list[float],
    y: np.ndarray | list[float],
    dt_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a trace onto a uniform time grid suitable for spectral analysis."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 2:
        return t, y
    if dt_ms is None:
        dt_ms = float(np.median(np.diff(t)))
    grid = np.arange(float(t[0]), float(t[-1]) + 0.5 * dt_ms, dt_ms)
    interp = interp1d(t, y, kind="linear", bounds_error=False, fill_value="extrapolate")
    return grid, interp(grid)


def normalize_time_modulus(modulus: float | int | None) -> float | None:
    """Return a usable positive time modulus, or None when disabled."""
    if modulus is None:
        return None
    try:
        value = float(modulus)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value <= 0.0:
        return None
    return value


def fold_time_series_by_modulus(
    t: np.ndarray | list[float],
    y: np.ndarray | list[float],
    modulus: float | int | None,
    *,
    dt_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Average a time/value trace into phase bins over one modulus period."""
    modulus_value = normalize_time_modulus(modulus)
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if modulus_value is None or len(t) < 2 or len(y) != len(t):
        return t, y

    finite = np.isfinite(t) & np.isfinite(y)
    t = t[finite]
    y = y[finite]
    if len(t) < 2:
        return t, y

    if dt_ms is None:
        diffs = np.diff(np.sort(np.unique(t)))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        dt_ms = float(np.median(diffs)) if len(diffs) else modulus_value / 200.0
    bin_count = max(2, int(round(modulus_value / max(float(dt_ms), 1e-9))))
    bin_count = min(bin_count, max(2, len(t)))
    edges = np.linspace(0.0, modulus_value, bin_count + 1)
    phase = np.mod(t, modulus_value)
    sums, _ = np.histogram(phase, bins=edges, weights=y)
    counts, _ = np.histogram(phase, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    folded = np.full(bin_count, np.nan, dtype=float)
    valid = counts > 0
    folded[valid] = sums[valid] / counts[valid]
    if np.any(~valid) and np.any(valid):
        folded[~valid] = np.interp(centers[~valid], centers[valid], folded[valid])
    elif not np.any(valid):
        folded[:] = 0.0
    return centers, folded


def fold_time_matrix_by_modulus(
    times_ms: np.ndarray | list[float],
    values: np.ndarray,
    modulus: float | int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Average matrix columns into phase bins over one modulus period."""
    modulus_value = normalize_time_modulus(modulus)
    times = np.asarray(times_ms, dtype=float)
    matrix = np.asarray(values, dtype=float)
    if modulus_value is None or matrix.ndim != 2 or len(times) != matrix.shape[1] or len(times) < 2:
        return times, matrix

    finite = np.isfinite(times)
    times = times[finite]
    matrix = matrix[:, finite]
    if len(times) < 2:
        return times, matrix

    diffs = np.diff(np.sort(np.unique(times)))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    dt_ms = float(np.median(diffs)) if len(diffs) else modulus_value / min(len(times), 200)
    bin_count = max(2, int(round(modulus_value / max(dt_ms, 1e-9))))
    bin_count = min(bin_count, max(2, len(times)))
    edges = np.linspace(0.0, modulus_value, bin_count + 1)
    phase = np.mod(times, modulus_value)
    bin_index = np.clip(np.searchsorted(edges, phase, side="right") - 1, 0, bin_count - 1)
    folded = np.zeros((matrix.shape[0], bin_count), dtype=float)
    counts = np.zeros(bin_count, dtype=float)
    for column_index, target_bin in enumerate(bin_index):
        folded[:, target_bin] += matrix[:, column_index]
        counts[target_bin] += 1.0
    valid = counts > 0
    if np.any(valid):
        folded[:, valid] /= counts[valid][None, :]
    if np.any(~valid) and np.any(valid):
        centers = (edges[:-1] + edges[1:]) / 2.0
        for row_index in range(folded.shape[0]):
            folded[row_index, ~valid] = np.interp(
                centers[~valid],
                centers[valid],
                folded[row_index, valid],
            )
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, folded


def butter_bandpass_filter(
    signal: np.ndarray | list[float],
    lowcut_hz: float,
    highcut_hz: float,
    fs_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth band-pass filter, falling back to causal filtering if needed."""
    signal = np.asarray(signal, dtype=float)
    nyquist = 0.5 * fs_hz
    b, a = butter(order, [lowcut_hz / nyquist, highcut_hz / nyquist], btype="band")
    min_len = 3 * max(len(a), len(b))
    if len(signal) <= min_len:
        return lfilter(b, a, signal)
    return filtfilt(b, a, signal)


def compute_bandpassed_signal(
    signal_t: np.ndarray | list[float],
    signal_y: np.ndarray | list[float],
    *,
    dt_ms: float | None = None,
    lowcut_hz: float = 30.0,
    highcut_hz: float = 120.0,
    order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample one trace and return a Butterworth band-passed copy."""
    t, y = uniform_trace(signal_t, signal_y, dt_ms=dt_ms)
    if len(t) < 2:
        return t, np.asarray(y, dtype=float)
    fs_hz = 1000.0 / float(np.median(np.diff(t)))
    return t, butter_bandpass_filter(y, lowcut_hz, highcut_hz, fs_hz, order=order)


def compute_welch_psd(
    signal_t: np.ndarray | list[float],
    signal_y: np.ndarray | list[float],
    *,
    dt_ms: float | None = None,
    nperseg: int | None = None,
    remove_mean: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a Welch PSD on a uniformly sampled trace."""
    t, y = uniform_trace(signal_t, signal_y, dt_ms=dt_ms)
    if len(t) < 4:
        return np.array([]), np.array([])

    values = np.asarray(y, dtype=float)
    if remove_mean:
        values = values - np.mean(values)
    fs_hz = 1000.0 / float(np.median(np.diff(t)))
    if nperseg is None:
        nperseg = min(2048, len(values))
    else:
        nperseg = min(int(nperseg), len(values))
    freqs, psd = welch(values, fs=fs_hz, nperseg=nperseg)
    return freqs, psd


def compute_spectrogram(
    signal_t: np.ndarray | list[float],
    signal_y: np.ndarray | list[float],
    dt_ms: float | None = None,
    max_freq_hz: float = 250.0,
    nperseg: int = 256,
    noverlap: int = 192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a standard spectrogram on a uniform time base."""
    t, y = uniform_trace(signal_t, signal_y, dt_ms=dt_ms)
    if len(t) < 4:
        raise ValueError("Trace is too short for spectral analysis")
    fs_hz = 1000.0 / float(np.median(np.diff(t)))
    nperseg = min(nperseg, len(y))
    noverlap = min(noverlap, max(0, nperseg - 1))

    nperseg = max(8, int(nperseg))
    noverlap = max(0, min(int(noverlap), max(0, nperseg - 1)))
    if len(y) <= nperseg:
        nperseg = max(8, min(256, len(y) // 4))
        noverlap = max(0, int(0.75 * nperseg))
        nperseg = max(8, min(int(nperseg), len(y)))
        noverlap = min(noverlap, max(0, nperseg - 1))

    while len(y) > 0 and nperseg >= 16:
        if nperseg - noverlap <= 0:
            noverlap = max(0, nperseg // 2)
        n_steps = 1 + max(0, (len(y) - nperseg) // max(1, nperseg - noverlap))
        if n_steps >= 2:
            break
        nperseg = max(16, nperseg // 2)
        noverlap = min(noverlap, max(0, nperseg // 2))
    if nperseg > len(y):
        nperseg = len(y)
        noverlap = max(0, min(noverlap, nperseg - 1))

    freqs, times_s, power = spectrogram(
        y,
        fs=fs_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="psd",
    )
    mask = freqs <= max_freq_hz
    return times_s * 1000.0, freqs[mask], power[mask]


def compute_wavelet_map(
    signal_t: np.ndarray | list[float],
    signal_y: np.ndarray | list[float],
    dt_ms: float = 0.1,
    lowcut_hz: float = 30.0,
    highcut_hz: float = 120.0,
    wavelet: str = "cgau5",
    scale_low: float = 3.0,
    scale_high: float = 32.0,
    n_scales: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the legacy-style continuous wavelet map used in the notebooks."""
    if pywt is None:
        raise ModuleNotFoundError(
            "PyWavelets is required for wavelet analysis. Install the 'pywavelets' package."
        )
    t, y = uniform_trace(signal_t, signal_y, dt_ms=dt_ms)
    fs_hz = 1000.0 / dt_ms
    y_bp = butter_bandpass_filter(y, lowcut_hz, highcut_hz, fs_hz, order=4)
    scales = np.linspace(scale_low / dt_ms, scale_high / dt_ms, n_scales)
    cfs, freqs = pywt.cwt(y_bp, scales, wavelet, dt_ms / 1000.0)
    power = np.log1p(np.abs(cfs))
    return t, y_bp, freqs, power


def compute_wavelet_band_power(
    signal_t: np.ndarray | list[float],
    signal_y: np.ndarray | list[float],
    bands: dict[str, tuple[float, float]] | None = None,
    dt_ms: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Collapse wavelet power into named frequency-band time series."""
    if bands is None:
        bands = {
            "beta": (15.0, 35.0),
            "low_gamma": (35.0, 65.0),
            "high_gamma": (65.0, 100.0),
        }
    t, _bp, freqs, power = compute_wavelet_map(signal_t, signal_y, dt_ms=dt_ms)
    traces = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        if np.any(mask):
            traces[name] = power[mask].mean(axis=0)
        else:
            traces[name] = np.zeros(power.shape[1])
    return t, freqs, power, traces


def compute_band_power_summary(
    signal_t: np.ndarray | list[float],
    signal_y: np.ndarray | list[float],
    *,
    bands: dict[str, tuple[float, float]] | None = None,
    dt_ms: float | None = 0.1,
    nperseg: int | None = None,
    relative_band: tuple[float, float] | None = (30.0, 250.0),
) -> dict[str, Any]:
    """Compute integrated Welch band powers for HFO-style summaries."""
    bands = dict(bands or DEFAULT_HFO_BANDS)
    freqs, psd = compute_welch_psd(
        signal_t,
        signal_y,
        dt_ms=dt_ms,
        nperseg=nperseg,
        remove_mean=True,
    )
    if len(freqs) == 0:
        return {
            "freqs": np.array([]),
            "psd": np.array([]),
            "band_power": {name: 0.0 for name in bands},
            "relative_band_power": {name: 0.0 for name in bands},
            "relative_band": relative_band,
        }

    if relative_band is None:
        denominator = trapezoid_integral(psd, freqs)
    else:
        relative_mask = (freqs >= relative_band[0]) & (freqs <= relative_band[1])
        denominator = trapezoid_integral(psd[relative_mask], freqs[relative_mask]) if np.any(relative_mask) else 0.0

    band_power = {}
    relative_power = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= float(lo)) & (freqs <= float(hi))
        power_value = trapezoid_integral(psd[mask], freqs[mask]) if np.any(mask) else 0.0
        band_power[name] = power_value
        relative_power[name] = power_value / denominator if denominator > 0 else 0.0

    return {
        "freqs": freqs,
        "psd": psd,
        "band_power": band_power,
        "relative_band_power": relative_power,
        "relative_band": relative_band,
    }
