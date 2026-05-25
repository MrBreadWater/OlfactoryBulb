from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from website_header_concepts import gaussian_smooth


SWEEP_INFO = "/home/alek/OlfactoryBulb/results/sweeps/gaba_gmax_20260520_185350/sweep_info.json"
DEFAULT_OUTPUT_DIR = "/home/alek/OlfactoryBulb/media/website_header_animated_concepts"
BG = "#ffffff"
TEXT = "#1e2a31"
INK = "#1e2a31"
SAND = "#f3ad5a"
CORAL = "#ff8c69"
SEA = "#43b6d9"
MINT = "#47cfaf"


@dataclass
class SweepFrameSummary:
    value: float
    run_dir: Path
    time_ms: np.ndarray
    x_norm: np.ndarray
    densities: dict[str, np.ndarray]
    lfp: np.ndarray
    lfp_energy: np.ndarray
    spectrogram: np.ndarray
    type_strengths: dict[str, float]
    active_score: float


@dataclass
class MorphologyShape:
    cell_type: str
    segments: list[tuple[float, float, float, float]]
    width: float
    anchor_x: float
    anchor_y: float
    scale: float
    rotation_deg: float
    mirror: bool = False


def hex_rgb(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def lerp(a: np.ndarray | float, b: np.ndarray | float, alpha: float):
    return (1.0 - alpha) * a + alpha * b


def soft_background(
    width: int,
    height: int,
    *,
    base_hex: str,
    glows: list[tuple[float, float, float, str, float]],
) -> np.ndarray:
    base = np.zeros((height, width, 3), dtype=float)
    base[:] = np.array(hex_rgb(base_hex), dtype=float)
    x = np.linspace(0.0, 1.0, width)
    y = np.linspace(0.0, 1.0, height)
    xx, yy = np.meshgrid(x, y)
    for cx, cy, radius, color_hex, strength in glows:
        rr = np.exp(-(((xx - cx) / radius) ** 2 + ((yy - cy) / radius) ** 2))
        base += strength * rr[..., None] * np.array(hex_rgb(color_hex))[None, None, :]
    return np.clip(base, 0.0, 1.0)


def read_swc_segments(
    path: str | Path,
    *,
    max_segments: int | None = None,
) -> list[tuple[float, float, float, float]]:
    nodes: dict[int, tuple[float, float, float]] = {}
    parents: dict[int, int] = {}
    for line in Path(path).read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        node_id = int(float(parts[0]))
        x = float(parts[2])
        y = float(parts[3])
        z = float(parts[4])
        parent = int(float(parts[6]))
        nodes[node_id] = (x, y, z)
        parents[node_id] = parent

    if not nodes:
        return []

    xs = np.array([xyz[0] for xyz in nodes.values()], dtype=float)
    ys = np.array([xyz[2] for xyz in nodes.values()], dtype=float)
    mean_x = float(xs.mean())
    mean_y = float(ys.mean())
    xs = xs - mean_x
    ys = ys - mean_y
    scale = max(np.ptp(xs), np.ptp(ys), 1.0)
    remap = {}
    for node_id, xyz in nodes.items():
        remap[node_id] = ((xyz[0] - mean_x) / scale, (xyz[2] - mean_y) / scale)

    segments: list[tuple[float, float, float, float]] = []
    for node_id, parent_id in parents.items():
        if parent_id < 0 or parent_id not in remap:
            continue
        x0, y0 = remap[parent_id]
        x1, y1 = remap[node_id]
        segments.append((x0, y0, x1, y1))
    if max_segments is not None and len(segments) > max_segments:
        keep = np.linspace(0, len(segments) - 1, max_segments).astype(int)
        segments = [segments[idx] for idx in keep]
    return segments


def load_morphology_shapes() -> list[MorphologyShape]:
    repo = Path("/home/alek/OlfactoryBulb")
    return [
        MorphologyShape(
            cell_type="MC",
            segments=read_swc_segments(
                repo / "morphology-data/MCs/urban-done/BurtonUrban2014/SDB131009c1.CNG.swc",
                max_segments=500,
            ),
            width=1.2,
            anchor_x=0.28,
            anchor_y=0.56,
            scale=0.56,
            rotation_deg=-16.0,
        ),
        MorphologyShape(
            cell_type="TC",
            segments=read_swc_segments(
                repo / "morphology-data/TCs/urban/BurtonUrban2014/SDB131008c2.CNG.swc",
                max_segments=260,
            ),
            width=1.0,
            anchor_x=0.38,
            anchor_y=0.50,
            scale=0.42,
            rotation_deg=10.0,
            mirror=True,
        ),
        MorphologyShape(
            cell_type="GC",
            segments=read_swc_segments(
                repo / "morphology-data/GCs/Guthrie-granule/WT5Grid4Sec1Cell14.CNG.swc",
                max_segments=140,
            ),
            width=0.9,
            anchor_x=0.14,
            anchor_y=0.24,
            scale=0.26,
            rotation_deg=-12.0,
        ),
    ]


def rotation_matrix(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def project_shape(shape: MorphologyShape) -> list[tuple[float, float, float, float]]:
    rot = rotation_matrix(shape.rotation_deg)
    projected: list[tuple[float, float, float, float]] = []
    for x0, y0, x1, y1 in shape.segments:
        p0 = rot @ np.array([(-x0 if shape.mirror else x0), y0], dtype=float)
        p1 = rot @ np.array([(-x1 if shape.mirror else x1), y1], dtype=float)
        x0p = shape.anchor_x + (shape.scale * p0[0])
        y0p = shape.anchor_y + (shape.scale * p0[1])
        x1p = shape.anchor_x + (shape.scale * p1[0])
        y1p = shape.anchor_y + (shape.scale * p1[1])
        projected.append((x0p, y0p, x1p, y1p))
    return projected


def load_gaba_sweep_summaries(
    sweep_info_path: str | Path = SWEEP_INFO,
    *,
    density_bins: int = 420,
    lfp_points: int = 840,
    spec_time_bins: int = 96,
    spec_freq_bins: int = 28,
) -> list[SweepFrameSummary]:
    info = json.loads(Path(sweep_info_path).read_text())
    summaries: list[SweepFrameSummary] = []

    for value, run_dir in zip(info["values"], info["run_dirs"]):
        run_dir = Path(run_dir)
        with open(run_dir / "soma_vs.pkl", "rb") as handle:
            soma_vs = pickle.load(handle)
        with open(run_dir / "lfp.pkl", "rb") as handle:
            lfp_t_ms, lfp_nv = pickle.load(handle)

        t_start = float(soma_vs[0][1][0])
        t_end = float(soma_vs[0][1][-1])
        x_density = np.linspace(t_start, t_end, density_bins)
        edges = np.linspace(t_start, t_end, density_bins + 1)
        densities: dict[str, np.ndarray] = {}
        strength_sum = 0.0
        type_strengths: dict[str, float] = {}
        spikes_by_type: dict[str, list[np.ndarray]] = {"MC": [], "TC": [], "GC": []}
        cell_counts: dict[str, int] = {"MC": 0, "TC": 0, "GC": 0}

        for label, t_ms, v_mv in soma_vs:
            cell_type = infer_cell_type(str(label))
            if cell_type not in spikes_by_type:
                continue
            cell_counts[cell_type] += 1
            t_arr = np.asarray(t_ms, dtype=float)
            v_arr = np.asarray(v_mv, dtype=float)
            crossings = np.flatnonzero((v_arr[:-1] <= 0.0) & (v_arr[1:] > 0.0)) + 1
            if crossings.size:
                spikes_by_type[cell_type].append(t_arr[crossings])

        for cell_type in ("MC", "TC", "GC"):
            if cell_counts[cell_type]:
                if spikes_by_type[cell_type]:
                    hist, _ = np.histogram(np.concatenate(spikes_by_type[cell_type], dtype=float), bins=edges)
                else:
                    hist = np.zeros(density_bins, dtype=float)
                smoothed = gaussian_smooth(hist / max(1, cell_counts[cell_type]), sigma_bins=3.2)
            else:
                smoothed = np.zeros(density_bins, dtype=float)
            densities[cell_type] = np.asarray(smoothed, dtype=float)
            type_strengths[cell_type] = float(np.percentile(smoothed, 95))
            strength_sum += type_strengths[cell_type]

        lfp_t = np.asarray(lfp_t_ms, dtype=float)
        lfp = np.asarray(lfp_nv, dtype=float)
        x_lfp = np.linspace(t_start, t_end, lfp_points)
        lfp_interp = np.interp(x_lfp, lfp_t, lfp)
        lfp_centered = lfp_interp - lfp_interp.mean()
        lfp_energy = gaussian_smooth(np.abs(lfp_centered), sigma_bins=5.0)

        spectrogram = compute_spectrogram_map(
            lfp,
            dt_ms=float(np.median(np.diff(lfp_t))) if lfp_t.size > 1 else 0.1,
            time_bins=spec_time_bins,
            freq_bins=spec_freq_bins,
            freq_min_hz=20.0,
            freq_max_hz=140.0,
        )

        summaries.append(
            SweepFrameSummary(
                value=float(value),
                run_dir=run_dir,
                time_ms=x_density,
                x_norm=np.linspace(0.0, 1.0, density_bins),
                densities=densities,
                lfp=normalize_signed(lfp_centered),
                lfp_energy=normalize_positive(lfp_energy),
                spectrogram=spectrogram,
                type_strengths=type_strengths,
                active_score=strength_sum,
            )
        )

    density_scale = max(
        max(float(np.max(summary.densities[cell_type])) for summary in summaries)
        for cell_type in ("MC", "TC", "GC")
    )
    lfp_scale = max(float(np.max(np.abs(summary.lfp))) for summary in summaries)
    energy_scale = max(float(np.max(summary.lfp_energy)) for summary in summaries)
    strength_scale = max(summary.active_score for summary in summaries)
    spec_scale = max(float(np.max(summary.spectrogram)) for summary in summaries)

    for summary in summaries:
        for cell_type in ("MC", "TC", "GC"):
            summary.densities[cell_type] = summary.densities[cell_type] / max(density_scale, 1e-9)
            summary.type_strengths[cell_type] = summary.type_strengths[cell_type] / max(density_scale, 1e-9)
        summary.lfp = summary.lfp / max(lfp_scale, 1e-9)
        summary.lfp_energy = summary.lfp_energy / max(energy_scale, 1e-9)
        summary.active_score = summary.active_score / max(strength_scale, 1e-9)
        summary.spectrogram = summary.spectrogram / max(spec_scale, 1e-9)

    return summaries


def infer_cell_type(label: str) -> str:
    for prefix in ("MC", "TC", "GC"):
        if label.startswith(prefix):
            return prefix
    return "OTHER"


def compute_spectrogram_map(
    signal: np.ndarray,
    *,
    dt_ms: float,
    time_bins: int,
    freq_bins: int,
    freq_min_hz: float,
    freq_max_hz: float,
) -> np.ndarray:
    arr = np.asarray(signal, dtype=float)
    if arr.size < 1024:
        return np.zeros((freq_bins, time_bins), dtype=float)

    downsample = 4
    arr = arr[::downsample]
    dt_s = (dt_ms * downsample) / 1000.0
    sample_rate_hz = 1.0 / dt_s
    window = 256
    hop = max(8, (arr.size - window) // max(time_bins - 1, 1))
    if hop <= 0:
        hop = window // 4

    spectra = []
    start = 0
    taper = np.hanning(window)
    freqs = np.fft.rfftfreq(window, d=dt_s)
    band_mask = (freqs >= freq_min_hz) & (freqs <= freq_max_hz)
    while start + window <= arr.size and len(spectra) < time_bins:
        chunk = arr[start:start + window]
        chunk = chunk - chunk.mean()
        power = np.abs(np.fft.rfft(chunk * taper)) ** 2
        band = power[band_mask]
        spectra.append(band)
        start += hop
    if not spectra:
        return np.zeros((freq_bins, time_bins), dtype=float)
    spec = np.asarray(spectra, dtype=float).T
    spec = np.log10(spec + 1e-9)

    target_freq = np.linspace(0, spec.shape[0] - 1, freq_bins)
    resampled = np.vstack([
        np.interp(target_freq, np.arange(spec.shape[0]), spec[:, idx])
        for idx in range(spec.shape[1])
    ]).T
    target_time = np.linspace(0, resampled.shape[1] - 1, time_bins)
    final = np.vstack([
        np.interp(target_time, np.arange(resampled.shape[1]), row)
        for row in resampled
    ])
    return normalize_positive(final - np.min(final))


def normalize_positive(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    high = float(np.percentile(arr, 99))
    if high <= 1e-9:
        return np.zeros_like(arr)
    return np.clip(arr / high, 0.0, 1.2)


def normalize_signed(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    scale = float(np.percentile(np.abs(arr), 98))
    if scale <= 1e-9:
        return np.zeros_like(arr)
    return np.clip(arr / scale, -1.2, 1.2)


def periodic_sample(values: np.ndarray, shift_frac: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return arr.copy()
    xp = np.linspace(0.0, 1.0, arr.size, endpoint=False)
    fp = np.concatenate([arr, arr[:1]])
    xp_ext = np.concatenate([xp, [1.0]])
    query = (xp - shift_frac) % 1.0
    return np.interp(query, xp_ext, fp)


def cycle_phase(value: float) -> float:
    return value % 1.0


def shift_periodic_field_horiz(values: np.ndarray, phase: float) -> np.ndarray:
    frac = cycle_phase(phase)
    if values.size == 0:
        return values.copy()
    shift = frac * values.shape[1]
    if shift == 0.0:
        return values.copy()
    cols = np.arange(values.shape[1], dtype=float)
    col0 = np.floor(cols - shift).astype(int) % values.shape[1]
    col1 = (col0 + 1) % values.shape[1]
    alpha = (cols - shift) - np.floor(cols - shift)
    return ((1.0 - alpha) * values[:, col0]) + (alpha * values[:, col1])


def rhythmic_trace(
    base: np.ndarray,
    x_norm: np.ndarray,
    phase: float,
    *,
    drift: float,
    wave_cycles: float,
    wave_strength: float,
    beat_phase: float = 0.0,
    envelope: np.ndarray | None = None,
    envelope_mix: float = 0.0,
) -> np.ndarray:
    shifted = periodic_sample(base, drift * phase)
    crest = 0.5 + 0.5 * np.sin((2.0 * math.pi * wave_cycles * x_norm) - (2.0 * math.pi * phase * 0.45) + beat_phase)
    beat = 0.90 + 0.10 * np.sin((2.0 * math.pi * phase * 1.4) + beat_phase)
    trace = shifted * (0.70 + (wave_strength * crest)) * beat
    if envelope is not None and envelope_mix > 0.0:
        env = resample_curve(envelope, trace.size)
        trace = trace + (envelope_mix * periodic_sample(env, drift * phase * 0.15))
    return np.clip(gaussian_smooth(trace, sigma_bins=0.7), 0.0, 1.4)


def resample_curve(values: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == size:
        return arr.copy()
    if arr.size < 2:
        return np.full(size, float(arr[0]) if arr.size else 0.0)
    xp = np.linspace(0.0, 1.0, arr.size)
    x = np.linspace(0.0, 1.0, size)
    return np.interp(x, xp, arr)


def interpolated_summary(summaries: list[SweepFrameSummary], position: float) -> SweepFrameSummary:
    lo = int(math.floor(position))
    hi = min(len(summaries) - 1, lo + 1)
    alpha = position - lo
    a = summaries[lo]
    b = summaries[hi]
    return SweepFrameSummary(
        value=float(lerp(a.value, b.value, alpha)),
        run_dir=a.run_dir,
        time_ms=a.time_ms,
        x_norm=a.x_norm,
        densities={cell_type: lerp(a.densities[cell_type], b.densities[cell_type], alpha) for cell_type in ("MC", "TC", "GC")},
        lfp=lerp(a.lfp, b.lfp, alpha),
        lfp_energy=lerp(a.lfp_energy, b.lfp_energy, alpha),
        spectrogram=lerp(a.spectrogram, b.spectrogram, alpha),
        type_strengths={cell_type: float(lerp(a.type_strengths[cell_type], b.type_strengths[cell_type], alpha)) for cell_type in ("MC", "TC", "GC")},
        active_score=float(lerp(a.active_score, b.active_score, alpha)),
    )


def make_positions(count: int, *, frames_per_step: int = 5, pingpong: bool = True) -> list[float]:
    positions: list[float] = []
    for idx in range(count - 1):
        for step in range(frames_per_step):
            positions.append(idx + (step / frames_per_step))
    positions.append(float(count - 1))
    if pingpong and count > 1:
        positions.extend(list(reversed(positions))[1:])
    return positions


def add_density_ribbons(
    ax: plt.Axes,
    summary: SweepFrameSummary,
    phase: float,
    *,
    x_left: float,
    x_right: float,
    baselines: dict[str, float],
    amplitudes: dict[str, float],
    colors: dict[str, str],
    fill_alpha: float = 0.18,
) -> None:
    x = np.linspace(x_left, x_right, summary.x_norm.size)
    rhythm_params = {
        "MC": {"drift": 0.03, "wave_cycles": 3.8, "wave_strength": 0.34, "beat_phase": 0.2},
        "TC": {"drift": 0.03, "wave_cycles": 4.1, "wave_strength": 0.38, "beat_phase": 1.1},
        "GC": {"drift": 0.04, "wave_cycles": 4.4, "wave_strength": 0.34, "beat_phase": 2.2},
    }
    for cell_type in ("MC", "TC", "GC"):
        density = rhythmic_trace(
            summary.densities[cell_type],
            summary.x_norm,
            phase,
            envelope=summary.lfp_energy,
            envelope_mix=0.08,
            **rhythm_params[cell_type],
        )
        y = baselines[cell_type] + amplitudes[cell_type] * density
        color = colors[cell_type]
        ax.fill_between(x, baselines[cell_type], y, color=color, alpha=fill_alpha * 0.75, zorder=2)
        ax.plot(x, y, color=color, lw=8.2, alpha=0.10, zorder=3)
        ax.plot(x, y, color=color, lw=1.9, alpha=0.90, zorder=4)


def render_aurora_currents(
    summary: SweepFrameSummary,
    *,
    width: int = 1280,
    height: int = 360,
    phase: float = 0.0,
) -> np.ndarray:
    phase = cycle_phase(phase)
    fig = plt.figure(figsize=(width / 180, height / 180), dpi=180, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_axis_off()
    ax.set_frame_on(False)
    bg = soft_background(
        width,
        height,
        base_hex="#ffffff",
        glows=[
            (0.16, 0.30, 0.45, "#b3def0", 0.11),
            (0.80, 0.52, 0.32, "#c5f1ff", 0.10),
            (0.86, 0.24, 0.30, "#f4d7ab", 0.08),
        ],
    )
    ax.imshow(bg, extent=(0, 1, 0, 1), origin="lower", aspect="auto", zorder=0)
    for x in np.linspace(0.34, 0.96, 7):
        ax.plot([x, x], [0.06, 0.94], color="#d7e2ea", lw=0.7, alpha=0.28, zorder=1)
    energy_x = resample_curve(summary.lfp_energy, summary.x_norm.size)

    add_density_ribbons(
        ax,
        summary,
        phase,
        x_left=0.34,
        x_right=0.97,
        baselines={"MC": 0.69, "TC": 0.48, "GC": 0.22},
        amplitudes={"MC": 0.22, "TC": 0.26, "GC": 0.23},
        colors={"MC": SAND, "TC": SEA, "GC": MINT},
        fill_alpha=0.20,
    )

    lfp_x = np.linspace(0.34, 0.97, summary.lfp.size)
    lfp_wave = rhythmic_trace(
        np.abs(summary.lfp),
        np.linspace(0.0, 1.0, summary.lfp.size),
        phase,
        drift=0.05,
        wave_cycles=4.5,
        wave_strength=0.26,
        beat_phase=0.7,
        envelope=summary.lfp_energy,
        envelope_mix=0.05,
    )
    lfp_y = 0.07 + 0.013 * normalize_signed(summary.lfp) + 0.012 * lfp_wave
    ax.plot(lfp_x, lfp_y, color=INK, lw=1.2, alpha=0.50, zorder=5)

    highlight = 0.5 + 0.5 * np.sin((2.0 * math.pi * (np.linspace(0.0, 1.0, summary.x_norm.size) * 4.4)) - (2.0 * math.pi * phase * 2.0))
    ax.fill_between(
        np.linspace(0.34, 0.97, summary.x_norm.size),
        0.15,
        0.15 + 0.045 * highlight * energy_x,
        color=SEA,
        alpha=0.08,
        zorder=1,
    )

    flare_x = 0.36 + 0.56 * ((phase * 0.88) % 1.0)
    for radius, alpha in ((0.09, 0.03), (0.05, 0.08)):
        circle = plt.Circle((flare_x, 0.55), radius=radius, color=SEA, alpha=alpha, ec="none", zorder=2)
        ax.add_patch(circle)

    return fig_to_rgb(fig)


def render_dendritic_pulse(
    summary: SweepFrameSummary,
    *,
    width: int = 1280,
    height: int = 360,
    phase: float = 0.0,
    shapes: list[MorphologyShape] | None = None,
) -> np.ndarray:
    phase = cycle_phase(phase)
    shapes = shapes or load_morphology_shapes()
    fig = plt.figure(figsize=(width / 180, height / 180), dpi=180, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_axis_off()
    ax.set_frame_on(False)
    bg = soft_background(
        width,
        height,
        base_hex="#ffffff",
        glows=[
            (0.18, 0.52, 0.24, "#d0eff7", 0.12),
            (0.66, 0.48, 0.24, "#c6e5f0", 0.10),
            (0.82, 0.35, 0.20, "#f3debc", 0.08),
        ],
    )
    ax.imshow(bg, extent=(0, 1, 0, 1), origin="lower", aspect="auto", zorder=0)

    wave_x = np.linspace(0.48, 0.98, summary.x_norm.size)
    energy_x = resample_curve(summary.lfp_energy, summary.x_norm.size)
    combo = 0.55 * summary.densities["MC"] + 0.80 * summary.densities["TC"] + 0.62 * summary.densities["GC"]
    wave_trace = rhythmic_trace(
        combo,
        summary.x_norm,
        phase,
        drift=0.04,
        wave_cycles=4.0,
        wave_strength=0.36,
        beat_phase=0.0,
        envelope=energy_x,
        envelope_mix=0.05,
    )
    wave = 0.11 + 0.20 * wave_trace
    ax.fill_between(wave_x, 0.05, wave, color="#0e5f74", alpha=0.26, zorder=1)
    ax.plot(wave_x, wave, color=SEA, lw=2.1, alpha=0.82, zorder=2)
    tc_trace = rhythmic_trace(
        summary.densities["TC"],
        summary.x_norm,
        phase,
        drift=0.04,
        wave_cycles=4.2,
        wave_strength=0.32,
        beat_phase=0.8,
        envelope=energy_x,
        envelope_mix=0.05,
    )
    wave2 = 0.29 + 0.12 * tc_trace
    ax.fill_between(wave_x, 0.28, wave2, color="#133947", alpha=0.18, zorder=1)
    ax.plot(wave_x, wave2, color=MINT, lw=1.4, alpha=0.42, zorder=2)

    pulse_centers = np.linspace(0.50, 0.92, 5)
    for pulse_idx, center in enumerate(pulse_centers):
        pulse = 0.035 + 0.07 * summary.active_score + 0.020 * math.sin((phase * 4 * math.pi) + pulse_idx * 0.8)
        pulse_y = 0.46 + 0.05 * math.cos((phase * 4 * math.pi) + pulse_idx * 0.7)
        ax.add_patch(plt.Circle((center, pulse_y), pulse, color=SEA, alpha=0.05, ec="none", zorder=1))
        ax.add_patch(plt.Circle((center, pulse_y), pulse * 0.55, color=SAND, alpha=0.04, ec="none", zorder=1))

    for shape in shapes:
        strength = summary.type_strengths[shape.cell_type]
        color = {"MC": SAND, "TC": SEA, "GC": MINT}[shape.cell_type]
        glow_phase = {"MC": 0.0, "TC": 0.7, "GC": 1.4}[shape.cell_type]
        glow = 0.70 + 0.30 * (0.5 + 0.5 * math.sin((phase * 4 * math.pi) + glow_phase))
        projected = project_shape(shape)
        for x0, y0, x1, y1 in projected:
            x0c, y0c, x1c, y1c = np.clip([x0, y0, x1, y1], 0.0, 1.0)
            ax.plot([x0c, x1c], [y0c, y1c], color=color, lw=shape.width * (1.25 + strength * 1.1), alpha=(0.02 + strength * 0.01) * glow, zorder=3, solid_capstyle="round")
        for x0, y0, x1, y1 in projected:
            x0c, y0c, x1c, y1c = np.clip([x0, y0, x1, y1], 0.0, 1.0)
            ax.plot([x0c, x1c], [y0c, y1c], color="#f7fbfd", lw=shape.width * 0.52, alpha=(0.06 + strength * 0.08) * glow, zorder=4, solid_capstyle="round")
            ax.plot([x0c, x1c], [y0c, y1c], color=color, lw=shape.width * 0.30, alpha=(0.52 + strength * 0.22) * glow, zorder=5, solid_capstyle="round")
        for idx, (x0, y0, x1, y1) in enumerate(projected):
            if idx % 40 != 0:
                continue
            x0c, y0c = np.clip([x0, y0], 0.0, 1.0)
            ax.scatter(x0c, y0c, s=9.0, c=color, alpha=0.22 * glow, edgecolors="none", zorder=6)

    return fig_to_rgb(fig)


def render_spike_bloom(
    summary: SweepFrameSummary,
    *,
    width: int = 1280,
    height: int = 360,
    phase: float = 0.0,
) -> np.ndarray:
    phase = cycle_phase(phase)
    fig = plt.figure(figsize=(width / 180, height / 180), dpi=180, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_axis_off()
    ax.set_frame_on(False)
    bg = soft_background(
        width,
        height,
        base_hex="#ffffff",
        glows=[
            (0.72, 0.30, 0.30, "#d9effa", 0.13),
            (0.82, 0.62, 0.26, "#f4e1c5", 0.08),
            (0.22, 0.48, 0.24, "#dff3ee", 0.10),
        ],
    )
    ax.imshow(bg, extent=(0, 1, 0, 1), origin="lower", aspect="auto", zorder=0)

    bands = {"MC": 0.72, "TC": 0.48, "GC": 0.24}
    colors = {"MC": SAND, "TC": SEA, "GC": MINT}
    counts = {"MC": 280, "TC": 420, "GC": 540}
    wave_factors = {"MC": 2.8, "TC": 3.4, "GC": 3.9}
    phase_offsets = {"MC": 0.2, "TC": 1.0, "GC": 1.8}
    x_base = np.linspace(0.28, 0.97, summary.x_norm.size)
    x_unit = np.linspace(0.0, 1.0, summary.x_norm.size)
    energy_x = resample_curve(summary.lfp_energy, summary.x_norm.size)
    density_profiles = {
        cell_type: resample_curve(summary.densities[cell_type], summary.x_norm.size)
        for cell_type in ("MC", "TC", "GC")
    }

    for cell_type in ("MC", "TC", "GC"):
        base_density = density_profiles[cell_type]
        phase_offset = phase_offsets[cell_type]
        density = rhythmic_trace(
            base_density,
            summary.x_norm,
            phase,
            drift=0.024,
            wave_cycles=4.1,
            wave_strength=0.32,
            beat_phase=phase_offset,
            envelope=energy_x,
            envelope_mix=0.03,
        )
        count = counts[cell_type]
        seed = np.linspace(0.0, 1.0, count, endpoint=False)
        shifted = (seed + (0.06 * phase)) % 1.0
        micro = 0.07 * np.sin(
            2.0 * math.pi * (2.0 * phase_offset + 4.3 * shifted + (2.0 * np.sin(phase * 2.0)))
        )
        x_pts = 0.28 + 0.69 * ((shifted + micro) % 1.0)
        amp = np.interp((x_pts - 0.28) / 0.69, x_unit, density)
        energy = np.interp((x_pts - 0.28) / 0.69, x_unit, energy_x)
        band_wave = np.sin((2.0 * math.pi * (wave_factors[cell_type] * shifted + phase)) - (2.0 * math.pi * phase_offset))
        amp_rhythm = 0.5 + 0.5 * np.sin(2.0 * math.pi * (shifted * 1.6 + phase))
        y_wave = 0.018 * amp * (1.0 + 0.4 * band_wave)
        y_pts = bands[cell_type] + y_wave + 0.022 * amp_rhythm + (0.015 * energy)

        pulse = 0.35 + 0.50 * amp + (0.12 * amp_rhythm)
        size = 6.0 + 36.0 * (0.35 * amp + 0.55 * pulse)
        color = colors[cell_type]
        glow = 0.30 + 0.70 * amp
        ax.scatter(
            x_pts,
            y_pts + 0.003 * np.cos(2.0 * math.pi * (shifted * 2.1 + phase)),
            s=size,
            color=color,
            alpha=(0.18 + 0.55 * amp) * (0.55 + 0.45 * glow),
            linewidths=0,
            zorder=2,
        )
        ax.scatter(
            x_pts,
            y_pts,
            s=size * 0.35,
            color="#ffffff",
            alpha=(0.15 + 0.12 * amp),
            linewidths=0,
            zorder=1,
        )

        streak_len = 0.009 + (0.021 * amp)
        for xp, yp, sl, strength in zip(x_pts[::7], y_pts[::7], streak_len[::7], amp[::7]):
            drift = 0.0005 + 0.001 * summary.active_score
            ax.plot(
                [xp - sl * 0.55, xp + sl],
                [yp - sl * 0.75 + drift, yp + sl * 0.55 - drift],
                color=color,
                lw=0.45 + (1.35 * strength),
                alpha=0.10 + (0.50 * strength),
                zorder=3,
            )

        ridge = bands[cell_type] - 0.07 + 0.16 * density
        ax.plot(x_base, ridge, color=color, lw=1.3, alpha=0.18, zorder=1)

    return fig_to_rgb(fig)


def smooth_2d(values: np.ndarray, sigma_x: float, sigma_y: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    tmp = np.vstack([gaussian_smooth(row, sigma_x) for row in arr])
    return np.vstack([gaussian_smooth(col, sigma_y) for col in tmp.T]).T


def upsample_2d(values: np.ndarray, out_rows: int, out_cols: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    x_src = np.arange(arr.shape[1], dtype=float)
    x_tgt = np.linspace(0, arr.shape[1] - 1, out_cols)
    interp_x = np.vstack([np.interp(x_tgt, x_src, row) for row in arr])
    y_src = np.arange(arr.shape[0], dtype=float)
    y_tgt = np.linspace(0, arr.shape[0] - 1, out_rows)
    interp_y = np.vstack([np.interp(y_tgt, y_src, interp_x[:, idx]) for idx in range(interp_x.shape[1])]).T
    return interp_y


def render_spectral_veil(
    summary: SweepFrameSummary,
    *,
    width: int = 1280,
    height: int = 360,
    phase: float = 0.0,
) -> np.ndarray:
    phase = cycle_phase(phase)
    fig = plt.figure(figsize=(width / 180, height / 180), dpi=180, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_axis_off()
    ax.set_frame_on(False)
    bg = soft_background(
        width,
        height,
        base_hex="#ffffff",
        glows=[
            (0.76, 0.52, 0.33, "#bddff2", 0.12),
            (0.86, 0.23, 0.22, "#f5e2c4", 0.07),
        ],
    )
    ax.imshow(bg, extent=(0, 1, 0, 1), origin="lower", aspect="auto", zorder=0)

    spec = smooth_2d(summary.spectrogram, sigma_x=1.2, sigma_y=0.9)
    field = upsample_2d(spec, out_rows=220, out_cols=960)
    field = shift_periodic_field_horiz(field, phase * 0.16)
    row_positions = np.linspace(0.0, 1.0, field.shape[0])[:, None]
    col_positions = np.linspace(0.0, 1.0, field.shape[1])[None, :]

    low = np.clip(1.0 - np.abs(row_positions - 0.18) / 0.28, 0.0, 1.0)
    mid = np.clip(1.0 - np.abs(row_positions - 0.50) / 0.30, 0.0, 1.0)
    high = np.clip(1.0 - np.abs(row_positions - 0.82) / 0.24, 0.0, 1.0)

    low_rgb = np.array(hex_rgb(MINT))
    mid_rgb = np.array(hex_rgb(SEA))
    high_rgb = np.array(hex_rgb(SAND))

    rgb = (
        low[..., None] * low_rgb[None, None, :]
        + mid[..., None] * mid_rgb[None, None, :]
        + high[..., None] * high_rgb[None, None, :]
    )
    rgb = rgb * field[..., None]
    shimmer = 0.55 + 0.45 * (0.5 + 0.5 * np.sin((2.0 * math.pi * 4.8 * col_positions) - (2.0 * math.pi * phase * 2.4)))
    alpha = np.clip((field ** 1.18) * shimmer, 0.0, 0.96)

    x_left, x_right = 0.34, 0.98
    y_bottom, y_top = 0.08, 0.90
    ax.imshow(rgb, extent=(x_left, x_right, y_bottom, y_top), origin="lower", aspect="auto", alpha=alpha, zorder=1)

    x_small = np.linspace(x_left, x_right, spec.shape[1])
    for level_idx, spec_idx in enumerate(np.linspace(2, spec.shape[0] - 3, 8).astype(int)):
        row = periodic_sample(spec[spec_idx], 0.08 * phase)
        baseline = y_bottom + (spec_idx / max(1, spec.shape[0] - 1)) * (y_top - y_bottom)
        pulse = 0.65 + 0.35 * (0.5 + 0.5 * np.sin((2.0 * math.pi * 3.4 * np.linspace(0.0, 1.0, row.size)) - (2.0 * math.pi * phase * 2.0) + (0.35 * level_idx)))
        y = baseline + 0.038 * row * pulse
        color = [MINT, SEA, SAND][min(2, level_idx // 3)]
        ax.plot(x_small, y, color=color, lw=1.2 + 0.2 * level_idx, alpha=0.12 + 0.05 * level_idx, zorder=2)

    lfp_y = 0.05 + 0.024 * gaussian_smooth(summary.lfp, sigma_bins=2.0)
    ax.plot(np.linspace(0.34, 0.97, summary.lfp.size), lfp_y, color=INK, lw=1.1, alpha=0.44, zorder=30)

    shimmer_x = 0.38 + 0.54 * ((phase * 0.84) % 1.0)
    ax.add_patch(plt.Circle((shimmer_x, 0.77), 0.032, color=SAND, alpha=0.05, ec="none", zorder=40))
    return fig_to_rgb(fig)


CONCEPTS: dict[str, Callable[..., np.ndarray]] = {
    "aurora_currents": render_aurora_currents,
    "dendritic_pulse": render_dendritic_pulse,
    "spike_bloom": render_spike_bloom,
    "spectral_veil": render_spectral_veil,
}


def fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    plt.close(fig)
    return np.ascontiguousarray(rgba[..., :3])


def render_concept_frames(
    summaries: list[SweepFrameSummary],
    concept_name: str,
    *,
    frames_per_step: int = 5,
) -> list[np.ndarray]:
    render_fn = CONCEPTS[concept_name]
    shapes = load_morphology_shapes() if concept_name == "dendritic_pulse" else None
    positions = make_positions(len(summaries), frames_per_step=frames_per_step, pingpong=True)
    frames = []
    for frame_index, position in enumerate(positions):
        summary = interpolated_summary(summaries, position)
        phase = cycle_phase(frame_index / max(1, len(positions) - 1))
        if shapes is None:
            frame = render_fn(summary, phase=phase)
        else:
            frame = render_fn(summary, phase=phase, shapes=shapes)
        frames.append(frame)
    return frames


def save_gif(frames: list[np.ndarray], path: str | Path, *, duration_ms: int = 70) -> Path:
    images = [Image.fromarray(frame) for frame in frames]
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    return out_path


def save_poster(frame: np.ndarray, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(out_path)
    return out_path


def save_contact_sheet(posters: dict[str, Path], output_path: str | Path) -> Path:
    ordered = list(posters.items())
    fig, axes = plt.subplots(len(ordered), 1, figsize=(16, 9.5), dpi=180, facecolor=BG)
    if len(ordered) == 1:
        axes = [axes]
    for ax, (name, path) in zip(axes, ordered):
        ax.set_axis_off()
        ax.imshow(plt.imread(path))
        ax.text(0.015, 0.08, name.replace("_", " ").title(), transform=ax.transAxes, color=TEXT, fontsize=14, ha="left", va="bottom")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.035)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path


def export_all_concepts(
    sweep_info_path: str | Path = SWEEP_INFO,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    concept_names: list[str] | None = None,
    frames_per_step: int = 5,
) -> dict[str, Path]:
    summaries = load_gaba_sweep_summaries(sweep_info_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}
    selected = concept_names or list(CONCEPTS.keys())

    for concept_name in selected:
        frames = render_concept_frames(summaries, concept_name, frames_per_step=frames_per_step)
        gif_path = save_gif(frames, out_dir / f"{concept_name}.gif")
        poster_path = save_poster(frames[len(frames) // 3], out_dir / f"{concept_name}_poster.png")
        artifacts[concept_name] = gif_path

    posters = {
        concept_name: out_dir / f"{concept_name}_poster.png"
        for concept_name in CONCEPTS
        if (out_dir / f"{concept_name}_poster.png").exists()
    }
    if posters:
        contact = save_contact_sheet(posters, out_dir / "contact_sheet.png")
        artifacts["contact_sheet"] = contact
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Render animated website-header concepts from the saved gaba_gmax sweep.")
    parser.add_argument("--sweep-info", default=SWEEP_INFO)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames-per-step", type=int, default=5)
    parser.add_argument("--concept", action="append", dest="concept_names")
    args = parser.parse_args()

    artifacts = export_all_concepts(
        sweep_info_path=args.sweep_info,
        output_dir=args.output_dir,
        concept_names=args.concept_names,
        frames_per_step=args.frames_per_step,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
