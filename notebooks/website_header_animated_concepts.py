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
                max_segments=360,
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
                max_segments=210,
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
                max_segments=120,
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


def periodic_profile(values: np.ndarray, positions: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.zeros_like(np.asarray(positions, dtype=float))
    pos = np.asarray(positions, dtype=float)
    xp = np.linspace(0.0, 1.0, arr.size, endpoint=False)
    xp_ext = np.r_[xp, 1.0]
    fp_ext = np.r_[arr, arr[0]]
    return np.interp(np.mod(pos, 1.0), xp_ext, fp_ext)


def periodic_bump_banks(
    x: np.ndarray,
    phase: float,
    *,
    count: int,
    speed: float,
    width: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if count <= 0 or x.size == 0:
        return np.zeros_like(x)
    centers = (np.arange(count, dtype=float) / count + speed * cycle_phase(phase)) % 1.0
    profile = np.zeros_like(x)
    spread = max(width, 1e-6)
    for center in centers:
        delta = np.abs(((x - center + 0.5) % 1.0) - 0.5)
        profile += np.exp(-(delta ** 2) / (2.0 * spread**2))
    return profile / float(count)


def periodic_gaussian_banks(
    x: np.ndarray,
    phase: float,
    *,
    count: int,
    speed: float,
    width: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if count <= 0:
        return np.zeros_like(x)
    centers = (np.arange(count, dtype=float) / count + speed * cycle_phase(phase)) % 1.0
    field = np.zeros_like(x)
    for center in centers:
        dist = np.abs(((x - center + 0.5) % 1.0) - 0.5)
        field += np.exp(-(dist / max(width, 1e-6)) ** 2 / 2.0)
    return field / float(count)


def clamp_unit(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def enforce_periodic_loop(values: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.full(size, 0.0)
    if size == 0:
        return np.zeros(0)
    if size == arr.size:
        circular = arr.copy()
    else:
        circular = periodic_sample(arr, 0.0)
    wrapped = periodic_profile(circular, np.linspace(0.0, 1.0, size))
    return np.asarray(wrapped, dtype=float)


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
    x = np.linspace(x_left, x_right, summary.x_norm.size * 3)
    base_x = np.linspace(0.0, 1.0, x.size)
    envelope = enforce_periodic_loop(summary.lfp_energy, x.size)
    tide = 0.10 * envelope * np.sin(2.0 * math.pi * (base_x * 0.8 + phase * 0.28))
    ripple = 0.09 * np.sin(2.0 * math.pi * (2.2 * base_x + phase * 0.65))
    for cell_type in ("MC", "TC", "GC"):
        density = enforce_periodic_loop(summary.densities[cell_type], x.size)
        density = gaussian_smooth(density, sigma_bins=1.3)
        density = clamp_unit(density)
        density_curve = 0.66 * density + 0.34 * envelope
        lane_shift = 0.024 * np.sin(
            2.0
            * math.pi
            * (
                base_x * 1.9
                + phase * 0.9
                + 0.31 * int(cell_type == "TC")
                + 0.27 * int(cell_type == "GC")
            )
        )
        y = baselines[cell_type] + amplitudes[cell_type] * density_curve
        y = y + (0.030 * density_curve) + lane_shift + (0.018 * ripple) + (0.003 * tide)
        y += 0.014 * periodic_bump_banks(
            base_x,
            phase + 0.24 * int(cell_type == "MC") - 0.11 * int(cell_type == "GC"),
            count=2,
            speed=0.12 + 0.03 * int(cell_type == "GC"),
            width=0.09,
        )
        y = np.clip(y, 0.02, 0.98)
        color = colors[cell_type]
        width = 0.024 + (0.010 * (2 - summary.type_strengths[cell_type]))
        ax.plot(x, y, color="#ffffff", lw=2.7, alpha=0.16, zorder=2)
        ax.fill_between(
            x,
            np.clip(y - width, 0.0, 1.0),
            np.clip(y + width, 0.0, 1.0),
            color=color,
            alpha=fill_alpha * 0.8,
            zorder=3,
        )
        ax.plot(x, y, color=color, lw=1.6, alpha=0.95, zorder=4)


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

    x = np.linspace(0.0, 1.0, 2000)
    energy_x = enforce_periodic_loop(summary.lfp_energy, x.size)
    lfp_pos = np.linspace(0.0, 1.0, summary.lfp.size if summary.lfp.size else x.size)
    lfp_x = np.linspace(0.0, 1.0, summary.lfp.size if summary.lfp.size else x.size)
    lfp_wave = rhythmic_trace(
        np.abs(summary.lfp) if summary.lfp.size else np.zeros_like(x),
        lfp_pos,
        phase,
        drift=0.05,
        wave_cycles=4.5,
        wave_strength=0.26,
        beat_phase=0.7,
        envelope=summary.lfp_energy,
        envelope_mix=0.05,
    )
    lfp_wave = resample_curve(lfp_wave, x.size)

    for lane in range(13):
        layer = lane / 13.0
        lane_phase = phase + layer * 0.19
        base = 0.06 + (layer * 0.065)
        mode_a = 0.060 * np.sin(2.0 * math.pi * (2.0 + (lane % 5) * 0.38) * x + 2.0 * math.pi * lane_phase)
        mode_b = 0.032 * np.sin(2.0 * math.pi * (0.7 + layer * 0.14) * x - 2.0 * math.pi * lane_phase * 0.8)
        envelope = np.clip(0.25 + 0.75 * periodic_profile(energy_x, x + 0.013 * lane), 0.0, 1.0)
        flow = 0.020 * (
            0.65 * np.sin(2.0 * math.pi * (x * (1.8 + 0.12 * lane) + phase * (0.22 + 0.03 * layer)))
            + 0.35 * mode_a
            + 0.20 * periodic_bump_banks(x, lane_phase, count=3, speed=0.14, width=0.10)
        )
        lane_wave = 0.005 * (2.0 * summary.active_score - 1.0) * np.sin(2.0 * math.pi * (x * 1.4 + phase + layer))
        y = base + 0.035 * envelope + lane_wave + mode_a + mode_b + 0.018 * flow + 0.010 * (2.0 * lfp_wave - 1.0)
        y = np.clip(y, 0.02, 0.98)

        tone = [MINT, SEA, SAND][lane % 3]
        flow_level = 0.4 + 0.6 * np.interp(x, x, energy_x)
        flow_lw = float(1.4 + (0.6 * np.clip(np.mean(flow_level), 0.1, 1.0)))
        ax.plot(x, y, color=tone, lw=flow_lw, alpha=0.10, zorder=1)
        ax.plot(x, y, color="#ffffff", lw=0.2, alpha=0.12, zorder=2)
        crest = 0.010 * periodic_bump_banks(x, lane_phase * 1.2, count=5 + lane % 4, speed=0.08 + 0.01 * layer, width=0.06)
        ax.fill_between(x, np.clip(y - 0.012, 0.0, 1.0), np.clip(y + 0.012 + crest, 0.0, 1.0), color=tone, alpha=0.05 + 0.02 * layer, zorder=1)

        packet_count = 26 + (lane % 4) * 5
        emit = np.mod(np.linspace(0.0, 1.0, packet_count, endpoint=False) + 0.11 * lane + 0.08 * phase, 1.0)
        packet_x = np.sort(emit)
        packet_y = np.interp(packet_x, x, y)
        packet_amp = np.interp(packet_x, x, 0.2 + 0.8 * envelope) * (0.2 + 0.8 * packet_count / 50.0)
        packet_glow = 0.4 + 0.6 * np.cos(2.0 * math.pi * (packet_x * 1.5 + lane_phase))
        sizes = np.clip(2.0 + (10.0 * packet_amp * packet_glow), 1.0, 12.0)
        ax.scatter(
            packet_x,
            np.clip(packet_y + 0.0015 * np.sin(2.0 * math.pi * (packet_x * 3.0 + lane_phase)), 0.02, 0.98),
            s=sizes,
            c=tone,
            alpha=0.20 + 0.13 * envelope[np.searchsorted(x, packet_x, side="right") - 1],
            linewidths=0,
            zorder=3,
        )

    add_density_ribbons(
        ax,
        summary,
        phase,
        x_left=0.0,
        x_right=1.0,
        baselines={"MC": 0.70, "TC": 0.49, "GC": 0.24},
        amplitudes={"MC": 0.23, "TC": 0.26, "GC": 0.22},
        colors={"MC": SAND, "TC": SEA, "GC": MINT},
        fill_alpha=0.20,
    )

    pulse_bus = resample_curve(np.abs(summary.lfp), x.size)
    ripple = np.interp(x, x, np.sin(2.0 * math.pi * (x * 2.8 - phase * 1.4)))
    ax.plot(
        x,
        0.15 + 0.05 * (pulse_bus * np.abs(ripple)),
        color=SEA,
        lw=0.8,
        alpha=0.16,
        zorder=4,
    )
    lfp_y = 0.08 + 0.022 * np.interp(x, lfp_x, normalize_signed(summary.lfp)) + 0.022 * lfp_wave
    ax.plot(
        x,
        lfp_y,
        color=INK,
        lw=1.1,
        alpha=0.44,
        zorder=5,
    )

    glow = 0.85 + 0.15 * periodic_bump_banks(x, phase * 1.2, count=3, speed=0.03, width=0.08)
    ax.fill_between(
        x,
        0.16,
        0.16 + 0.035 * glow * energy_x,
        color=SEA,
        alpha=0.06,
        zorder=1,
    )

    flare_x = 0.05 + (0.90 * cycle_phase(phase * 0.88))
    for radius, alpha in ((0.085, 0.028), (0.048, 0.065)):
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

    x = np.linspace(0.0, 1.0, 2000)
    pulse_field = enforce_periodic_loop(summary.lfp_energy if summary.lfp_energy.size else np.array([0.0]), x.size)
    node_profiles = {
        "MC": {
            "base_y": 0.68,
            "root_y": 0.10,
            "root_x": 0.28,
            "hue": SAND,
            "glow": "#d7edf2",
            "phase": 0.0,
            "wave": 3.2,
        },
        "TC": {
            "base_y": 0.46,
            "root_y": 0.08,
            "root_x": 0.56,
            "hue": SEA,
            "glow": "#c8eaf5",
            "phase": 1.15,
            "wave": 4.0,
        },
        "GC": {
            "base_y": 0.28,
            "root_y": 0.06,
            "root_x": 0.40,
            "hue": MINT,
            "glow": "#d6f3ec",
            "phase": 2.2,
            "wave": 2.6,
        },
    }
    anchors: list[tuple[float, float]] = []
    pulse_mask = periodic_bump_banks(x, phase * 0.45, count=6, speed=0.12, width=0.14)
    root_band = 0.05 + (0.03 * pulse_mask)

    for shape_idx, shape in enumerate(shapes):
        cfg = node_profiles[shape.cell_type]
        hue = cfg["hue"]
        glow = cfg["glow"]
        projected = project_shape(shape)
        if not projected:
            continue
        shape_wave = enforce_periodic_loop(summary.lfp_energy, max(1, len(projected)))
        seg_np = np.asarray(projected, dtype=float)
        cx0 = float(np.mean(seg_np[:, [0, 2]]))
        cy0 = float(np.mean(seg_np[:, [1, 3]]))
        if len(projected) > 220:
            keep = np.linspace(0, len(projected) - 1, 220).astype(int)
            projected = [projected[idx] for idx in keep]
        anchors.append((np.clip(cx0, 0.0, 1.0), np.clip(cy0, 0.0, 1.0)))

        root_x = np.clip(cfg["root_x"] + 0.015 * np.sin(2.0 * math.pi * (phase * 0.4 + cfg["phase"])), 0.02, 0.98)
        root_y = np.clip(cfg["root_y"] + 0.006 * np.cos(2.0 * math.pi * (phase * 0.33 + cfg["phase"])), 0.02, 0.98)
        soma_x = np.clip(cx0 + 0.008 * np.sin(2.0 * math.pi * (phase * 0.3 + cfg["phase"] + shape_idx * 0.2)), 0.03, 0.97)
        soma_y = np.clip(cy0 + 0.008 * np.cos(2.0 * math.pi * (phase * 0.27 + cfg["phase"] + shape_idx * 0.2)), 0.03, 0.97)
        stem_y = np.linspace(root_y, soma_y, 26)
        stem_t = np.linspace(0.0, 1.0, 26)
        stem_x = np.linspace(root_x, soma_x, 26) + 0.004 * np.sin(2.0 * math.pi * (stem_t * 0.9 + cfg["phase"] + phase))
        stem_profile = 0.5 + 0.5 * pulse_mask[: stem_x.size]
        ax.plot(
            np.clip(stem_x, 0.0, 1.0),
            np.clip(stem_y + 0.003 * stem_profile, 0.0, 1.0),
            color=glow,
            lw=2.2,
            alpha=0.12,
            zorder=2,
        )
        ax.plot(
            np.clip(stem_x, 0.0, 1.0),
            np.clip(stem_y + 0.003 * stem_profile, 0.0, 1.0),
            color=hue,
            lw=1.0,
            alpha=0.32,
            zorder=3,
        )

        ax.scatter(
            soma_x,
            soma_y,
            s=20.0 + (8.0 * np.clip(cfg["base_y"] * 2.0, 0.0, 1.0)),
            c=hue,
            alpha=0.60,
            linewidths=0,
            zorder=4,
        )

        body_swell = 0.5 + 0.5 * np.sin(2.0 * math.pi * (phase * 0.22 + cfg["phase"]))
        ax.plot(
            [cfg["root_x"], cfg["root_x"]],
            [0.0, cfg["root_y"]],
            color=cfg["glow"],
            lw=1.8,
            alpha=0.08,
            zorder=1,
        )

        for seg_idx, (x0, y0, x1, y1) in enumerate(projected):
            seg_mix = seg_idx / max(1, len(projected))
            wave = 2.0 * math.pi * (seg_mix * cfg["wave"] + phase * 0.75 + cfg["phase"]) + 0.3 * summary.active_score
            pulse_wave = float(shape_wave[seg_idx % len(shape_wave)]) if shape_wave.size else 0.0
            density = 0.35 + 0.65 * pulse_wave
            wave_shift = 0.006 * np.sin(wave)
            kink = 0.004 * np.cos(2.0 * math.pi * (seg_mix * 2.2 + phase + cfg["phase"]))

            sx0 = np.clip(
                0.02 + 0.96 * x0 + 0.006 * np.sin(2.0 * math.pi * (phase * 0.16 + shape_idx * 0.2 + seg_mix * 1.8)) + wave_shift,
                0.0,
                1.0,
            )
            sx1 = np.clip(
                0.02 + 0.96 * x1 + 0.006 * np.sin(2.0 * math.pi * (phase * 0.16 + shape_idx * 0.2 + 1.3 + seg_mix * 1.8)) + wave_shift,
                0.0,
                1.0,
            )
            sy0 = np.clip(
                0.02 + 0.96 * y0 + 0.006 * np.cos(2.0 * math.pi * (phase * 0.14 + cfg["phase"] + seg_mix * 0.9)) + kink,
                0.0,
                1.0,
            )
            sy1 = np.clip(
                0.02 + 0.96 * y1 + 0.006 * np.cos(2.0 * math.pi * (phase * 0.14 + cfg["phase"] + seg_mix * 0.9)) + kink,
                0.0,
                1.0,
            )
            density_lift = 0.25 + 0.75 * np.interp((sx0 + sx1) * 0.5, x, pulse_field)
            branch_alpha = 0.14 + (0.30 * density_lift * density)
            ax.plot(
                [sx0, sx1],
                [sy0, sy1],
                color=glow,
                alpha=branch_alpha * 0.35,
                lw=shape.width * 0.95 * body_swell,
                zorder=2,
                solid_capstyle="round",
            )
            ax.plot(
                [sx0, sx1],
                [sy0, sy1],
                color=hue,
                alpha=0.75 + (0.15 * summary.active_score),
                lw=shape.width * 0.58,
                zorder=3,
                solid_capstyle="round",
            )

            if seg_idx % 14 == 0:
                junction_x = 0.5 * (sx0 + sx1)
                junction_y = 0.5 * (sy0 + sy1)
                ax.scatter(
                    junction_x,
                    junction_y,
                    s=0.9 + (1.8 * density),
                    c=hue,
                    alpha=0.30,
                    linewidths=0,
                    zorder=4,
                )

        for packet_idx in range(18):
            packet_phase = np.mod((packet_idx / 18.0) + (phase * 0.35) + (0.11 * shape_idx), 1.0)
            seg_cursor = int(packet_phase * max(1, len(projected) - 1))
            local_t = packet_phase * max(1, len(projected) - 1) - seg_cursor
            seg_cursor = min(seg_cursor, len(projected) - 1)
            next_seg = min(seg_cursor + 1, len(projected) - 1)
            x0, y0, x1, y1 = projected[seg_cursor]
            nx0 = np.clip(0.02 + 0.96 * ((x0 if seg_cursor < len(projected) else x1) + local_t * ((x1 if seg_cursor < len(projected) else x0) - x0)), 0.0, 1.0)
            ny0 = np.clip(0.02 + 0.96 * ((y0 if seg_cursor < len(projected) else y1) + local_t * ((y1 if seg_cursor < len(projected) else y0) - y0)), 0.0, 1.0)
            amp = 0.2 + 0.8 * np.interp(nx0, x, pulse_field)
            glow_wave = 0.5 + 0.5 * np.cos(2.0 * math.pi * (packet_phase * 3.2 + phase + cfg["phase"]))
            ax.scatter(
                nx0,
                ny0,
                s=1.8 + (8.0 * amp * glow_wave),
                c=hue,
                alpha=0.20 + (0.45 * amp),
                linewidths=0,
                zorder=5,
            )

    if len(anchors) >= 2:
        for idx in range(len(anchors) - 1):
            x0, y0 = anchors[idx]
            x1, y1 = anchors[idx + 1]
            x_curve = np.linspace(0.0, 1.0, 130)
            base_x = 0.01 + 0.98 * np.clip(np.linspace(x0, x1, 130), 0.0, 1.0)
            base_y = 0.03 + 0.94 * np.clip(np.linspace(y0, y1, 130), 0.0, 1.0)
            bridge = 0.016 * np.sin(2.0 * math.pi * (x_curve * 2.6 + phase * 1.4) + idx)
            y_curve = np.clip(base_y + bridge, 0.0, 1.0)
            alpha = 0.07 + 0.06 * (1.0 - idx / max(1, len(anchors) - 1))
            ax.plot(base_x, y_curve, color=SEA, lw=0.85, alpha=alpha, zorder=2)

    for cell_type, cfg in node_profiles.items():
        density = enforce_periodic_loop(summary.densities[cell_type], 360)
        x_axis = np.linspace(0.0, 1.0, 360)
        lane = cfg["base_y"] + (0.020 * (2.0 * density - 1.0))
        lane = np.clip(
            lane + 0.005 * np.sin(2.0 * math.pi * (x_axis * 4.2 + phase + cfg["phase"])),
            0.0,
            1.0,
        )
        lane = np.clip(cfg["base_y"] + (0.004 * (2.0 * density - 1.0)), 0.06, 0.94)
        ax.fill_between(
            x_axis,
            np.clip(lane - 0.008, 0.0, 1.0),
            np.clip(lane + 0.008, 0.0, 1.0),
            color=cfg["glow"],
            alpha=0.09,
            zorder=1,
        )
        ax.plot(x_axis, lane, color=cfg["hue"], lw=1.1, alpha=0.30, zorder=2)

    flow = 0.042 * (0.5 + 0.5 * np.sin(2.0 * math.pi * (x * (1.9 + (0.6 * summary.active_score)) - phase * 1.9)))
    flow = flow * pulse_field
    flow_y = 0.08 + 0.011 * flow + 0.008 * (2.0 * summary.active_score - 1.0)
    ax.plot(
        x,
        np.clip(flow_y, 0.01, 0.99),
        color=SEA,
        lw=0.65,
        alpha=0.34,
        zorder=2,
    )

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

    lane_cfg = {
        "MC": {"y": 0.72, "color": SAND, "count": 240, "wave": 2.2, "phase": 0.12},
        "TC": {"y": 0.46, "color": SEA, "count": 320, "wave": 2.8, "phase": 1.0},
        "GC": {"y": 0.19, "color": MINT, "count": 390, "wave": 3.5, "phase": 2.0},
    }

    density_profiles = {
        cell_type: rhythmic_trace(
            summary.densities[cell_type],
            summary.x_norm,
            phase,
            drift=0.02,
            wave_cycles=4.3,
            wave_strength=0.28,
            beat_phase=cfg["phase"] / 2.0,
            envelope=summary.lfp_energy,
            envelope_mix=0.05,
        )
        for cell_type, cfg in lane_cfg.items()
    }

    x_norm = np.linspace(0.0, 1.0, summary.x_norm.size)
    x_dense = np.linspace(0.0, 1.0, 1900)
    global_flow = enforce_periodic_loop(summary.lfp_energy, x_dense.size)
    field = enforce_periodic_loop(np.abs(summary.lfp), x_dense.size)
    drive = periodic_bump_banks(x_dense, phase * 0.4, count=4, speed=0.08, width=0.16)
    drift = 0.018 * (0.5 + 0.5 * np.sin(2.0 * math.pi * (x_dense * 1.6 + phase)))

    for lane_idx, (cell_type, cfg) in enumerate(lane_cfg.items()):
        density = np.interp(x_dense, x_norm, density_profiles[cell_type])
        lane_wave = 0.036 * np.sin(2.0 * math.pi * (cfg["wave"] * x_dense + cfg["phase"] + phase * 1.2))
        pulse_wave = 0.012 * np.sin(2.0 * math.pi * (x_dense * 3.1 + phase * 0.8 + cfg["phase"]))
        lane = cfg["y"] + (0.030 * (2.0 * density - 1.0)) + lane_wave + pulse_wave + (0.010 * global_flow)
        lane = np.clip(lane, 0.03, 0.97)
        lane = lane + 0.008 * (2.0 * density - 1.0) * (0.45 + 0.55 * density)
        lane = np.clip(lane, 0.03, 0.97)

        base_glow = float(np.mean(1.0 + 0.4 * density))
        for glow_alpha, glow_width in ((0.28, 1.2), (0.12, 2.6)):
            ax.plot(
                x_dense,
                lane,
                color=cfg["color"],
                lw=glow_width * base_glow,
                alpha=glow_alpha,
                zorder=1,
            )
        ax.plot(
            x_dense,
            lane + 0.004 * np.cos(2.0 * math.pi * (x_dense * 3.8 + phase + cfg["phase"])),
            color="#ffffff",
            lw=0.22,
            alpha=0.20,
            zorder=2,
        )

        emit = np.linspace(0.0, 1.0, cfg["count"], endpoint=False)
        head = np.mod(emit + 0.13 * (lane_idx + 1) * phase + 0.08 * phase**2 + cfg["phase"] * 0.12, 1.0)
        lane_amp = np.interp(head, x_dense, lane)
        density_amp = np.interp(head, x_dense, density)
        core_wave = np.interp(head, x_dense, global_flow)
        shimmer = 0.6 + 0.4 * np.sin(2.0 * math.pi * (head * 1.4 + phase + cfg["phase"]))
        burst_mask = periodic_gaussian_banks(head, phase + lane_idx * 0.3, count=3, speed=0.20 + 0.03 * lane_idx, width=0.38)
        core = 1.8 + 9.0 * (0.30 + 0.70 * density_amp) * shimmer * (0.35 + 0.65 * core_wave)
        ax.scatter(
            head,
            lane_amp + (0.003 * np.sin(2.0 * math.pi * (head * 2.2 + cfg["phase"] + phase))),
            s=np.clip(core * (0.35 + 0.65 * burst_mask), 0.6, 16.0),
            c=cfg["color"],
            alpha=0.18 + (0.34 * density_amp),
            linewidths=0,
            zorder=3,
        )
        ax.scatter(
            head[::2],
            lane_amp[::2] + 0.002 * np.cos(2.0 * math.pi * (head[::2] * 3.1 + phase)),
            s=np.clip(core[::2] * (0.18 + 0.20 * shimmer[::2]), 0.5, None),
            c="#ffffff",
            alpha=0.10 + (0.12 * summary.active_score),
            linewidths=0,
            zorder=4,
        )

        for idx in range(0, emit.size, 16):
            head_x = head[idx]
            head_y = np.interp(head_x, x_dense, lane)
            tail_len = 0.022 + (0.020 * density_amp[idx])
            tail_x = np.linspace(head_x - tail_len, head_x - 0.001, 8)
            tail_x = np.mod(tail_x, 1.0)
            tail_x = np.sort(tail_x)
            tail_y = head_y + 0.004 * np.sin(
                2.0
                * math.pi
                * (np.linspace(0.0, 1.0, 8) * (cfg["wave"] + 0.7) + phase + 0.1 * idx / max(1, emit.size))
            ) + (0.003 * np.interp(tail_x, x_dense, field))
            ax.plot(
                tail_x,
                np.clip(np.interp(tail_x, x_dense, lane), 0.0, 1.0) + 0.002 * shimmer[idx],
                color=cfg["color"],
                lw=0.4 + (1.0 * density_amp[idx]),
                alpha=0.16 + (0.25 * density_amp[idx]),
                zorder=2,
            )

    pulse_bus = np.linspace(0.0, 1.0, x_dense.size)
    pulse_wave = 0.20 + 0.06 * np.sin(2.0 * math.pi * (3.0 * pulse_bus + phase * 1.4))
    pulse_wave *= summary.active_score
    for burst in range(5):
        bx = np.mod(
            pulse_bus * 0.94 + 0.03 + (0.14 * np.sin(phase * 2.2 + burst * 0.8) * burst / 5.0),
            1.0,
        )
        by = 0.18 + 0.17 * (burst / 4.0)
        crest = 0.008 * np.sin(
            2.0 * math.pi * (bx * (3.6 + burst * 0.5) + phase + burst * 0.3)
        ) + (0.010 * np.interp(bx, pulse_bus, pulse_wave))
        ax.plot(
            bx,
            np.clip(by + crest, 0.04, 0.95),
            color=["#f6ad8b", "#7fc7d8", "#4cc6a2", "#63b6ef", "#55d0bb"][burst % 5],
            lw=1.1,
            alpha=0.12 + (0.06 * float(np.mean(summary.lfp_energy))),
            zorder=1,
        )
        ax.fill_between(
            bx,
            np.clip(by + crest - 0.008, 0.02, 1.0),
            np.clip(by + crest + 0.008 + (0.004 * drift), 0.02, 1.0),
            color=["#f6ad8b", "#7fc7d8", "#4cc6a2", "#63b6ef", "#55d0bb"][burst % 5],
            alpha=0.05 + 0.03 * summary.active_score,
            zorder=1,
        )

    sweep = 0.07 * np.sin(2.0 * math.pi * (x_dense * (1.6 + 0.3 * summary.active_score) + phase * 1.1))
    ax.plot(
        x_dense,
        0.02 + 0.12 * (0.5 + 0.5 * (global_flow * (0.5 + 0.5 * np.sin(2.0 * math.pi * phase))) + sweep + drive * drift),
        color=INK,
        lw=0.95,
        alpha=0.20,
        zorder=5,
    )
    for burst in (0.16, 0.48, 0.86):
        center = np.mod(phase * 0.4 + burst, 1.0)
        ax.plot(
            [center, center],
            [0.03, 0.97],
            color=SEA,
            lw=0.4,
            alpha=0.12,
            zorder=0,
        )

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

    spec = smooth_2d(summary.spectrogram, sigma_x=1.4, sigma_y=1.0)
    field = upsample_2d(spec, out_rows=236, out_cols=1120)
    field = np.clip(field, 0.0, 1.0)
    field = shift_periodic_field_horiz(field, phase * 0.30)

    row_pos = np.linspace(0.0, 1.0, field.shape[0])
    col_pos = np.linspace(0.0, 1.0, field.shape[1])
    row_grid = row_pos[:, None]
    col_grid = col_pos[None, :]

    low = np.clip(1.0 - np.abs(row_grid - 0.2) / 0.35, 0.0, 1.0)
    mid = np.clip(1.0 - np.abs(row_grid - 0.52) / 0.3, 0.0, 1.0)
    high = np.clip(1.0 - np.abs(row_grid - 0.78) / 0.28, 0.0, 1.0)

    low_rgb = np.array(hex_rgb(MINT))
    mid_rgb = np.array(hex_rgb(SEA))
    high_rgb = np.array(hex_rgb(SAND))

    rgb = (
        low[..., None] * low_rgb[None, None, :]
        + mid[..., None] * mid_rgb[None, None, :]
        + high[..., None] * high_rgb[None, None, :]
    )
    modulation = 0.35 + 0.65 * field
    rgb = rgb * modulation[..., None]
    shimmer = 0.65 + 0.25 * np.cos((2.0 * math.pi * 5.4 * col_grid) - (2.0 * math.pi * phase * 1.7))
    field_alpha = np.clip(field ** 0.85 * shimmer, 0.0, 0.84)
    rgba = np.concatenate([np.clip(rgb, 0.0, 1.0), field_alpha[..., None]], axis=2)
    ax.imshow(rgba, extent=(0.0, 1.0, 0.0, 1.0), origin="lower", aspect="auto", zorder=1)

    for row_idx in np.linspace(2, field.shape[0] - 2, 42).astype(int):
        profile = field[row_idx]
        base_y = (row_idx / max(1, field.shape[0] - 1))
        y = 0.03 + (0.94 * base_y) + 0.012 * np.sin(2.0 * math.pi * (3.2 * col_pos + phase * 1.3 + row_idx * 0.014))
        y = np.clip(y + 0.018 * (profile - 0.5), 0.0, 1.0)
        y += 0.007 * np.sin(2.0 * math.pi * (col_pos * 2.6 + phase * 0.9 + (row_idx / 236.0)))
        color = [MINT, SEA, SAND][row_idx % 3]
        ax.plot(
            col_pos,
            np.clip(y, 0.08, 0.94),
            color=color,
            lw=0.55 + 0.25 * (1.0 - np.abs(row_idx / max(1, field.shape[0] - 1) - 0.5)),
            alpha=0.08,
            zorder=2,
        )

    for band in range(7):
        row_a = int((band / 7.0) * (field.shape[0] - 1))
        row_b = int(((band + 1.8) / 7.0) * (field.shape[0] - 1))
        path = np.linspace(0.0, 1.0, field.shape[1])
        upper = row_a / max(1, field.shape[0] - 1)
        lower = row_b / max(1, field.shape[0] - 1)
        warp = 0.018 * np.sin(2.0 * math.pi * (path * (1.8 + band * 0.38) + phase * (0.8 + 0.1 * band)))
        fill_amp = 0.011 * np.interp(path, col_pos, field[min(row_a, field.shape[0] - 1)])
        ax.fill_between(
            path,
            0.02 + (0.96 * np.clip(upper + warp - fill_amp, 0.0, 1.0)),
            0.02 + (0.96 * np.clip(lower + warp + fill_amp + 0.002, 0.0, 1.0)),
            color=[MINT, SEA, SAND][band % 3],
            alpha=0.035 + 0.01 * band,
            zorder=3,
        )

    # scanning lattice strokes to create directional movement
    scan = np.linspace(0.0, 1.0, summary.lfp.size)
    scan_amp = 0.06 * normalize_positive(summary.lfp_energy)
    scan_profile = enforce_periodic_loop(scan_amp if scan_amp.size else np.array([0.0]), scan.size)
    for beam in range(8):
        offset = (phase + beam / 8.0) % 1.0
        pulse = (scan + (0.17 * np.sin(2.0 * math.pi * phase + beam * 0.4) + offset)) % 1.0
        x = np.mod(pulse, 1.0)
        y = 0.03 + 0.94 * (0.35 + 0.30 * np.sin(2.0 * math.pi * (scan * (beam + 1) * 0.6 + phase + 0.12 * beam)))
        y += 0.04 * (scan_profile * np.sin(2.0 * math.pi * (scan * 2.2 + beam * 0.5 + phase)))
        ax.plot(
            x,
            np.clip(y, 0.02, 0.98),
            color=SEA,
            alpha=0.05 + 0.04 * summary.active_score,
            lw=0.95,
            zorder=6,
        )

    lfp_line = 0.08 + 0.030 * gaussian_smooth(summary.lfp, sigma_bins=2.0)
    ax.plot(0.02 + (0.96 * np.linspace(0.0, 1.0, summary.lfp.size)), lfp_line, color=INK, lw=0.95, alpha=0.24, zorder=10)
    for pulse in range(3):
        cx = np.mod(phase * 0.28 + 0.31 * pulse + np.linspace(0.0, 1.0, summary.lfp.size), 1.0)
        ax.plot(
            cx,
            0.02 + 0.96 * (0.16 + 0.30 * np.sin(2.0 * math.pi * (cx * 1.2 + pulse * 0.4 + phase))),
            color=[MINT, SEA, SAND][pulse % 3],
            lw=0.75,
            alpha=0.10 + (0.04 * pulse),
            zorder=7,
        )

    return fig_to_rgb(fig)


def render_population_tides(
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
            (0.13, 0.28, 0.26, "#caebf4", 0.11),
            (0.82, 0.56, 0.20, "#f0ddbc", 0.08),
        ],
    )
    ax.imshow(bg, extent=(0, 1, 0, 1), origin="lower", aspect="auto", zorder=0)

    x = np.linspace(0.0, 1.0, 2400)
    energy = enforce_periodic_loop(summary.lfp_energy if summary.lfp_energy.size else np.array([0.0]), x.size)
    tide_drive = np.array(energy)
    tide_base = 0.045 * np.sin(2.0 * math.pi * (0.95 * x + phase * 0.8))
    wave = 0.028 + 0.032 * np.sin(2.0 * math.pi * (3.8 * x + phase))

    lanes = {
        "MC": {"base": 0.72, "color": SAND, "amp": 0.10, "phase": 0.1, "harmonic": 2.3},
        "TC": {"base": 0.48, "color": SEA, "amp": 0.11, "phase": 0.8, "harmonic": 2.9},
        "GC": {"base": 0.24, "color": MINT, "amp": 0.09, "phase": 1.6, "harmonic": 2.1},
    }

    for lane_idx, (cell_type, cfg) in enumerate(lanes.items()):
        density = enforce_periodic_loop(summary.densities[cell_type], x.size)
        ridge = cfg["amp"] * (2.0 * density - 1.0)
        pulse = 0.019 * np.sin(2.0 * math.pi * (x * cfg["harmonic"] + cfg["phase"] + phase))
        breath = 0.026 * np.sin(2.0 * math.pi * (x * (1.6 + lane_idx * 0.2) - phase * 0.6))
        y = cfg["base"] + ridge + tide_base + wave + pulse + breath + (0.022 * tide_drive) + (0.010 * np.cos(2.0 * math.pi * (x * 0.8 + phase + cfg["phase"])))
        y = np.clip(y, 0.03, 0.97)
        width = 0.011 + 0.003 * (np.sin(2.0 * math.pi * (x * 0.9 + cfg["phase"])) ** 2)
        shade = 0.08 + (0.33 * np.maximum(0.0, density))
        ax.fill_between(x, y - width, y + width, color=cfg["color"], alpha=0.06 + (0.05 * shade), zorder=2)
        ax.plot(x, y, color=cfg["color"], lw=1.3 + 0.2 * lane_idx, alpha=0.82, zorder=3)

        emit = np.mod(np.linspace(0.0, 1.0, 160 + (lane_idx * 35), endpoint=False) + lane_idx * 0.13 + phase * 0.22, 1.0)
        emit = np.sort(emit)
        emit_y = np.interp(emit, x, y) + 0.004 * np.sin(2.0 * math.pi * (emit * 2.8 + phase + cfg["phase"]))
        glow = 0.6 + 0.4 * np.sin(2.0 * math.pi * (emit + cfg["phase"] + phase))
        ax.scatter(
            emit,
            np.clip(emit_y, 0.03, 0.97),
            s=1.6 + (7.5 * (0.35 + 0.65 * glow)),
            c=cfg["color"],
            alpha=0.22 + (0.35 * summary.active_score),
            linewidths=0,
            zorder=4,
        )

    x_bus = np.linspace(0.0, 1.0, summary.x_norm.size)
    crest = 0.15 + 0.11 * (summary.active_score if hasattr(summary, "active_score") else 0.5)
    crest_wave = resample_curve(
        0.028 * normalize_positive(0.8 * np.asarray(summary.lfp_energy) + 0.2 * (summary.x_norm.size > 0)),
        x_bus.size,
    )
    ax.plot(
        x_bus,
        0.08 + crest_wave * crest + 0.02 * np.sin(2.0 * math.pi * (x_bus * 2.0 + phase)),
        color=INK,
        lw=1.0,
        alpha=0.34,
        zorder=6,
    )

    horizon = 0.028 + 0.024 * np.sin(2.0 * math.pi * (x + phase * 0.7))
    ax.plot(
        x,
        np.clip(0.84 + horizon * 0.07, 0.0, 0.95),
        color=MINT,
        lw=0.8,
        alpha=0.22,
        zorder=5,
    )
    for swell in range(4):
        band = np.clip(0.35 + 0.09 * np.sin(2.0 * math.pi * (x * 2.4 + phase * (0.7 + 0.13 * swell) + 0.17 * swell)), 0.0, 1.0)
        ax.fill_between(
            x,
            np.clip(0.07 + 0.10 * band, 0.0, 0.97),
            np.clip(0.10 + 0.12 * band, 0.0, 0.97),
            color=[MINT, SEA, SAND][swell % 3],
            alpha=0.018 + (0.007 * swell),
            zorder=1,
        )

    return fig_to_rgb(fig)


CONCEPTS: dict[str, Callable[..., np.ndarray]] = {
    "aurora_currents": render_aurora_currents,
    "dendritic_pulse": render_dendritic_pulse,
    "spike_bloom": render_spike_bloom,
    "spectral_veil": render_spectral_veil,
    "population_tides": render_population_tides,
}


def fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    plt.close(fig)
    rgb = np.ascontiguousarray(rgba[..., :3])
    rgb[:2, :, :] = 255
    rgb[-2:, :, :] = 255
    rgb[:, :2, :] = 255
    rgb[:, -2:, :] = 255
    return rgb


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
