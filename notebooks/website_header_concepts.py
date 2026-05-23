from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


BG = "#07141f"
PANEL = "#0d2230"
GRID = "#244352"
TEXT = "#d8e6ed"
INPUT = "#ffd166"
CELL_COLORS = {
    "MC": "#ffb347",
    "TC": "#6ad5ff",
    "GC": "#49dcb1",
    "OTHER": "#d5c7ff",
}
CELL_ORDER = ("MC", "TC", "GC", "OTHER")


@dataclass
class TraceRecord:
    label: str
    cell_type: str
    t_ms: np.ndarray
    v_mv: np.ndarray
    spike_times_ms: np.ndarray
    spike_count: int
    peak_mv: float


@dataclass
class RunBundle:
    run_dir: Path
    traces: list[TraceRecord]
    lfp_t_ms: np.ndarray
    lfp_nv: np.ndarray
    input_times: list[tuple[str, np.ndarray]]
    dt_ms: float


def infer_cell_type(label: str) -> str:
    for prefix in ("MC", "TC", "GC"):
        if label.startswith(prefix):
            return prefix
    return "OTHER"


def gaussian_smooth(values: np.ndarray, sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return values.astype(float, copy=True)
    radius = max(1, int(math.ceil(sigma_bins * 3)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x ** 2) / (2 * sigma_bins ** 2))
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def detect_spikes(
    t_ms: np.ndarray,
    v_mv: np.ndarray,
    *,
    threshold_mv: float = 0.0,
    refractory_ms: float = 2.0,
) -> np.ndarray:
    if len(t_ms) < 2:
        return np.array([], dtype=float)
    crossings = np.flatnonzero((v_mv[:-1] <= threshold_mv) & (v_mv[1:] > threshold_mv)) + 1
    if len(crossings) == 0:
        return np.array([], dtype=float)
    spike_times = [float(t_ms[crossings[0]])]
    for idx in crossings[1:]:
        t_now = float(t_ms[idx])
        if t_now - spike_times[-1] >= refractory_ms:
            spike_times.append(t_now)
    return np.asarray(spike_times, dtype=float)


def load_pickle(path: str | Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_run(run_dir: str | Path) -> RunBundle:
    run_path = Path(run_dir)
    soma_vs = load_pickle(run_path / "soma_vs.pkl")
    lfp_t_ms, lfp_nv = load_pickle(run_path / "lfp.pkl")
    input_times_raw = load_pickle(run_path / "input_times.pkl")

    traces: list[TraceRecord] = []
    for label, t_ms, v_mv in soma_vs:
        t_arr = np.asarray(t_ms, dtype=float)
        v_arr = np.asarray(v_mv, dtype=float)
        spike_times = detect_spikes(t_arr, v_arr)
        traces.append(
            TraceRecord(
                label=label,
                cell_type=infer_cell_type(label),
                t_ms=t_arr,
                v_mv=v_arr,
                spike_times_ms=spike_times,
                spike_count=int(spike_times.size),
                peak_mv=float(np.max(v_arr)),
            )
        )

    dt_ms = float(np.median(np.diff(np.asarray(soma_vs[0][1], dtype=float)))) if soma_vs else 0.1
    input_times = [
        (segment_name, np.asarray(times, dtype=float))
        for segment_name, times in input_times_raw
    ]
    return RunBundle(
        run_dir=run_path,
        traces=traces,
        lfp_t_ms=np.asarray(lfp_t_ms, dtype=float),
        lfp_nv=np.asarray(lfp_nv, dtype=float),
        input_times=input_times,
        dt_ms=dt_ms,
    )


def summarize_run(bundle: RunBundle) -> dict[str, object]:
    type_counts = {cell_type: 0 for cell_type in CELL_ORDER}
    spike_counts = {cell_type: 0 for cell_type in CELL_ORDER}
    for trace in bundle.traces:
        type_counts[trace.cell_type] += 1
        spike_counts[trace.cell_type] += trace.spike_count
    return {
        "run_dir": str(bundle.run_dir),
        "trace_count": len(bundle.traces),
        "duration_ms": float(bundle.traces[0].t_ms[-1]) if bundle.traces else 0.0,
        "dt_ms": bundle.dt_ms,
        "type_counts": type_counts,
        "spike_counts": spike_counts,
        "lfp_range_nv": float(np.ptp(bundle.lfp_nv)) if bundle.lfp_nv.size else 0.0,
        "input_entries": len(bundle.input_times),
    }


def select_activity_window(
    bundle: RunBundle,
    *,
    window_ms: float = 650.0,
    bin_ms: float = 10.0,
) -> tuple[float, float]:
    if not bundle.traces:
        return 0.0, window_ms

    run_start = float(bundle.traces[0].t_ms[0])
    run_end = float(bundle.traces[0].t_ms[-1])
    if run_end - run_start <= window_ms:
        return run_start, run_end

    edges = np.arange(run_start, run_end + bin_ms, bin_ms, dtype=float)
    score = np.zeros(len(edges) - 1, dtype=float)
    weights = {"MC": 1.35, "TC": 1.15, "GC": 0.7, "OTHER": 0.5}
    for cell_type in CELL_ORDER:
        matching = [trace.spike_times_ms for trace in bundle.traces if trace.cell_type == cell_type]
        if not matching:
            continue
        spikes = np.concatenate([sp for sp in matching if sp.size], dtype=float) if any(sp.size for sp in matching) else np.array([], dtype=float)
        hist, _ = np.histogram(spikes, bins=edges)
        normalized = hist / max(1, len(matching))
        score += weights[cell_type] * gaussian_smooth(normalized, sigma_bins=window_ms / (8 * bin_ms))

    if not np.any(score):
        center = 0.5 * (run_start + run_end)
    else:
        peak_index = int(np.argmax(score))
        center = 0.5 * (edges[peak_index] + edges[peak_index + 1])

    window_start = np.clip(center - (window_ms * 0.58), run_start, run_end - window_ms)
    return float(window_start), float(window_start + window_ms)


def in_window(times_ms: np.ndarray, start_ms: float, end_ms: float) -> np.ndarray:
    return (times_ms >= start_ms) & (times_ms <= end_ms)


def _windowed(traces: Iterable[TraceRecord], start_ms: float, end_ms: float) -> list[tuple[TraceRecord, np.ndarray, np.ndarray]]:
    clipped: list[tuple[TraceRecord, np.ndarray, np.ndarray]] = []
    for trace in traces:
        mask = in_window(trace.t_ms, start_ms, end_ms)
        if np.count_nonzero(mask) < 3:
            continue
        clipped.append((trace, trace.t_ms[mask], trace.v_mv[mask]))
    return clipped


def _rank_traces(bundle: RunBundle, start_ms: float, end_ms: float, cell_type: str) -> list[TraceRecord]:
    candidates = [trace for trace in bundle.traces if trace.cell_type == cell_type]
    scored: list[tuple[float, TraceRecord]] = []
    for trace, _, v_win in _windowed(candidates, start_ms, end_ms):
        spikes = int(np.count_nonzero((trace.spike_times_ms >= start_ms) & (trace.spike_times_ms <= end_ms)))
        dynamic_range = float(np.percentile(v_win, 98) - np.percentile(v_win, 5))
        score = (spikes * 5.0) + dynamic_range + max(0.0, trace.peak_mv)
        scored.append((score, trace))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [trace for _, trace in scored]


def featured_traces(
    bundle: RunBundle,
    start_ms: float,
    end_ms: float,
    *,
    quotas: dict[str, int],
) -> list[TraceRecord]:
    chosen: list[TraceRecord] = []
    for cell_type in CELL_ORDER:
        quota = quotas.get(cell_type, 0)
        if quota <= 0:
            continue
        chosen.extend(_rank_traces(bundle, start_ms, end_ms, cell_type)[:quota])
    return chosen


def spike_density(
    bundle: RunBundle,
    start_ms: float,
    end_ms: float,
    *,
    cell_type: str,
    bin_ms: float = 4.0,
    sigma_ms: float = 18.0,
) -> tuple[np.ndarray, np.ndarray]:
    matching = [trace for trace in bundle.traces if trace.cell_type == cell_type]
    centers = np.arange(start_ms, end_ms, bin_ms, dtype=float)
    if len(centers) == 0:
        return centers, np.zeros(0, dtype=float)
    edges = np.append(centers, end_ms)
    spikes = np.concatenate(
        [trace.spike_times_ms[(trace.spike_times_ms >= start_ms) & (trace.spike_times_ms <= end_ms)] for trace in matching if trace.spike_times_ms.size],
        dtype=float,
    ) if any(trace.spike_times_ms.size for trace in matching) else np.array([], dtype=float)
    hist, _ = np.histogram(spikes, bins=edges)
    normalized = hist / max(1, len(matching))
    smoothed = gaussian_smooth(normalized, sigma_bins=max(0.5, sigma_ms / bin_ms))
    return centers, smoothed


def downsample_lfp(bundle: RunBundle, start_ms: float, end_ms: float, points: int = 1200) -> tuple[np.ndarray, np.ndarray]:
    mask = in_window(bundle.lfp_t_ms, start_ms, end_ms)
    t = bundle.lfp_t_ms[mask]
    v = bundle.lfp_nv[mask]
    if t.size <= points:
        return t, v
    idx = np.linspace(0, t.size - 1, points).astype(int)
    return t[idx], v[idx]


def _paint_background(ax: plt.Axes) -> None:
    w, h = 1200, 400
    x = np.linspace(0.0, 1.0, w)
    y = np.linspace(0.0, 1.0, h)
    xx, yy = np.meshgrid(x, y)
    base = np.zeros((h, w, 3), dtype=float)
    base[..., 0] = 7 / 255
    base[..., 1] = 20 / 255
    base[..., 2] = 31 / 255

    glow_teal = np.exp(-(((xx - 0.72) / 0.28) ** 2 + ((yy - 0.22) / 0.45) ** 2))
    glow_amber = np.exp(-(((xx - 0.88) / 0.22) ** 2 + ((yy - 0.74) / 0.32) ** 2))
    glow_blue = np.exp(-(((xx - 0.22) / 0.45) ** 2 + ((yy - 0.50) / 0.70) ** 2))

    base[..., 0] += 0.04 * glow_teal + 0.10 * glow_amber + 0.02 * glow_blue
    base[..., 1] += 0.10 * glow_teal + 0.06 * glow_amber + 0.05 * glow_blue
    base[..., 2] += 0.10 * glow_teal + 0.02 * glow_amber + 0.14 * glow_blue

    vignette = np.clip(1.0 - 0.75 * np.sqrt((xx - 0.58) ** 2 + (yy - 0.50) ** 2), 0.55, 1.0)
    base *= vignette[..., None]
    ax.imshow(np.clip(base, 0.0, 1.0), extent=(0, 1, 0, 1), origin="lower", aspect="auto")


def _new_figure() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(16, 5), dpi=180, facecolor=BG)
    bg_ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    bg_ax.set_axis_off()
    _paint_background(bg_ax)
    return fig, bg_ax


def _data_axes(fig: plt.Figure, rect: list[float]) -> plt.Axes:
    ax = fig.add_axes(rect, facecolor="none")
    ax.set_axis_off()
    return ax


def _add_guides(ax: plt.Axes, start_ms: float, end_ms: float, *, step_ms: float = 100.0) -> None:
    for x_ms in np.arange(math.ceil(start_ms / step_ms) * step_ms, end_ms, step_ms):
        ax.axvline(x_ms, color=GRID, lw=0.6, alpha=0.3, zorder=0)


def _trace_wave(v_mv: np.ndarray) -> np.ndarray:
    baseline = np.percentile(v_mv, 20)
    centered = v_mv - baseline
    scale = max(8.0, np.percentile(np.abs(centered), 98))
    return centered / scale


def plot_voltage_silk(bundle: RunBundle, *, window_ms: float = 650.0) -> plt.Figure:
    start_ms, end_ms = select_activity_window(bundle, window_ms=window_ms)
    featured = featured_traces(
        bundle,
        start_ms,
        end_ms,
        quotas={"MC": 4, "TC": 5, "GC": 10},
    )
    background = featured_traces(
        bundle,
        start_ms,
        end_ms,
        quotas={"MC": 7, "TC": 8, "GC": 18},
    )

    fig, _ = _new_figure()
    ax = _data_axes(fig, [0.30, 0.12, 0.66, 0.78])
    _add_guides(ax, start_ms, end_ms)

    lanes = []
    order = {"MC": 0, "TC": 1, "GC": 2, "OTHER": 3}
    featured.sort(key=lambda trace: (order[trace.cell_type], -trace.spike_count, trace.label))
    offsets = np.linspace(len(featured) - 0.5, 0.5, len(featured))
    for offset, trace in zip(offsets, featured):
        lanes.append((trace, offset))

    background_ids = {id(trace) for trace in featured}
    for trace in background:
        if id(trace) in background_ids:
            continue
        mask = in_window(trace.t_ms, start_ms, end_ms)
        if np.count_nonzero(mask) < 3:
            continue
        t = trace.t_ms[mask]
        y = _trace_wave(trace.v_mv[mask])
        color = CELL_COLORS[trace.cell_type]
        ax.plot(t, 0.12 * y + np.interp(order[trace.cell_type], [0, 2], [len(featured) - 1.2, 1.2]), color=color, lw=0.6, alpha=0.10, zorder=1)

    for trace, offset in lanes:
        mask = in_window(trace.t_ms, start_ms, end_ms)
        t = trace.t_ms[mask]
        y = _trace_wave(trace.v_mv[mask]) + offset
        color = CELL_COLORS[trace.cell_type]
        ax.plot(t, y, color=color, lw=5.0, alpha=0.08, solid_capstyle="round", zorder=2)
        ax.plot(t, y, color=color, lw=1.4, alpha=0.90, solid_capstyle="round", zorder=3)

    lfp_t, lfp = downsample_lfp(bundle, start_ms, end_ms)
    if lfp.size:
        scaled = (lfp - np.mean(lfp)) / max(1e-9, np.percentile(np.abs(lfp - np.mean(lfp)), 98))
        ax.plot(lfp_t, scaled * 0.35 - 0.55, color="#f4f7fa", lw=2.0, alpha=0.42, zorder=2)

    ax.set_xlim(start_ms, end_ms)
    ax.set_ylim(-1.2, len(featured) + 0.6)
    return fig


def plot_gamma_constellation(bundle: RunBundle, *, window_ms: float = 650.0) -> plt.Figure:
    start_ms, end_ms = select_activity_window(bundle, window_ms=window_ms)
    traces = featured_traces(
        bundle,
        start_ms,
        end_ms,
        quotas={"MC": 10, "TC": 14, "GC": 32},
    )
    order = {"MC": 0, "TC": 1, "GC": 2, "OTHER": 3}
    traces.sort(key=lambda trace: (order[trace.cell_type], -trace.spike_count, trace.label))

    fig, _ = _new_figure()
    raster_ax = _data_axes(fig, [0.30, 0.24, 0.66, 0.62])
    density_ax = _data_axes(fig, [0.30, 0.08, 0.66, 0.13])
    _add_guides(raster_ax, start_ms, end_ms)
    _add_guides(density_ax, start_ms, end_ms)

    y_positions: dict[int, float] = {}
    current_y = len(traces) + 1
    separators: list[tuple[str, float]] = []
    for cell_type in CELL_ORDER:
        matching = [trace for trace in traces if trace.cell_type == cell_type]
        if not matching:
            continue
        start_y = current_y
        for trace in matching:
            current_y -= 1
            y_positions[id(trace)] = current_y
        separators.append((cell_type, 0.5 * (start_y + current_y)))
        current_y -= 1.5

    for segment_name, spike_times in bundle.input_times[:80]:
        hits = spike_times[(spike_times >= start_ms) & (spike_times <= end_ms)]
        if hits.size:
            raster_ax.vlines(hits, ymin=-3, ymax=len(traces) + 2, color=INPUT, lw=0.8, alpha=0.03, zorder=1)

    for trace in traces:
        spikes = trace.spike_times_ms[(trace.spike_times_ms >= start_ms) & (trace.spike_times_ms <= end_ms)]
        if spikes.size == 0:
            continue
        y = np.full_like(spikes, y_positions[id(trace)], dtype=float)
        color = CELL_COLORS[trace.cell_type]
        raster_ax.scatter(spikes, y, s=18, color=color, alpha=0.14, linewidths=0, zorder=2)
        raster_ax.scatter(spikes, y, s=5, color=color, alpha=0.88, linewidths=0, zorder=3)

    for cell_type, y_mid in separators:
        if cell_type == "OTHER":
            continue
        raster_ax.text(start_ms - 8, y_mid, cell_type, color=CELL_COLORS[cell_type], ha="right", va="center", fontsize=12, alpha=0.85)

    for cell_type in ("MC", "TC", "GC"):
        centers, density = spike_density(bundle, start_ms, end_ms, cell_type=cell_type)
        if density.size == 0:
            continue
        scaled = density / max(1e-9, np.percentile(density, 99))
        color = CELL_COLORS[cell_type]
        density_ax.plot(centers, scaled, color=color, lw=4.8, alpha=0.10, zorder=1)
        density_ax.plot(centers, scaled, color=color, lw=1.8, alpha=0.95, zorder=2)
        density_ax.fill_between(centers, 0, scaled, color=color, alpha=0.12, zorder=1)

    lfp_t, lfp = downsample_lfp(bundle, start_ms, end_ms)
    if lfp.size:
        centered = lfp - np.mean(lfp)
        scaled = centered / max(1e-9, np.percentile(np.abs(centered), 99))
        density_ax.plot(lfp_t, 0.52 + (0.42 * scaled), color="#f6fbff", lw=1.3, alpha=0.55, zorder=3)

    raster_ax.set_xlim(start_ms, end_ms)
    raster_ax.set_ylim(-3, len(traces) + 2)
    density_ax.set_xlim(start_ms, end_ms)
    density_ax.set_ylim(-0.05, 1.25)
    return fig


def plot_population_tides(bundle: RunBundle, *, window_ms: float = 650.0) -> plt.Figure:
    start_ms, end_ms = select_activity_window(bundle, window_ms=window_ms)
    hero_traces = featured_traces(
        bundle,
        start_ms,
        end_ms,
        quotas={"MC": 2, "TC": 2, "GC": 3},
    )

    fig, _ = _new_figure()
    wave_ax = _data_axes(fig, [0.28, 0.12, 0.67, 0.76])
    _add_guides(wave_ax, start_ms, end_ms)

    baselines = {"MC": 2.2, "TC": 1.2, "GC": 0.25}
    for cell_type in ("MC", "TC", "GC"):
        centers, density = spike_density(bundle, start_ms, end_ms, cell_type=cell_type, sigma_ms=22.0)
        if density.size == 0:
            continue
        scaled = density / max(1e-9, np.percentile(density, 99))
        baseline = baselines[cell_type]
        color = CELL_COLORS[cell_type]
        wave_ax.fill_between(centers, baseline, baseline + (0.85 * scaled), color=color, alpha=0.20, zorder=1)
        wave_ax.plot(centers, baseline + (0.85 * scaled), color=color, lw=5.0, alpha=0.12, zorder=2)
        wave_ax.plot(centers, baseline + (0.85 * scaled), color=color, lw=2.1, alpha=0.95, zorder=3)

    input_spikes = np.concatenate(
        [times[(times >= start_ms) & (times <= end_ms)] for _, times in bundle.input_times if times.size],
        dtype=float,
    ) if any(times.size for _, times in bundle.input_times) else np.array([], dtype=float)
    if input_spikes.size:
        input_spikes = input_spikes[:: max(1, int(input_spikes.size / 120))]
        wave_ax.scatter(input_spikes, np.full_like(input_spikes, 3.48), s=10, color=INPUT, alpha=0.38, linewidths=0, zorder=4)

    order = {"MC": 0, "TC": 1, "GC": 2, "OTHER": 3}
    hero_traces.sort(key=lambda trace: (order[trace.cell_type], -trace.spike_count, trace.label))
    hero_offsets = np.linspace(4.05, 5.2, len(hero_traces))
    for offset, trace in zip(hero_offsets, hero_traces):
        mask = in_window(trace.t_ms, start_ms, end_ms)
        t = trace.t_ms[mask]
        y = 0.28 * _trace_wave(trace.v_mv[mask]) + offset
        color = CELL_COLORS[trace.cell_type]
        wave_ax.plot(t, y, color=color, lw=4.0, alpha=0.08, zorder=4)
        wave_ax.plot(t, y, color=color, lw=1.3, alpha=0.92, zorder=5)

    lfp_t, lfp = downsample_lfp(bundle, start_ms, end_ms)
    if lfp.size:
        centered = lfp - np.mean(lfp)
        scaled = centered / max(1e-9, np.percentile(np.abs(centered), 99))
        wave_ax.plot(lfp_t, -0.15 + (0.18 * scaled), color="#eef7fb", lw=1.5, alpha=0.48, zorder=2)

    wave_ax.set_xlim(start_ms, end_ms)
    wave_ax.set_ylim(-0.35, 5.55)
    return fig


CONCEPTS = {
    "voltage_silk": plot_voltage_silk,
    "gamma_constellation": plot_gamma_constellation,
    "population_tides": plot_population_tides,
}


def save_concept_previews(
    bundle: RunBundle,
    output_dir: str | Path,
    *,
    window_ms: float = 650.0,
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for concept_name, plotter in CONCEPTS.items():
        fig = plotter(bundle, window_ms=window_ms)
        out_path = out_dir / f"{concept_name}.png"
        fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        saved[concept_name] = out_path
    make_contact_sheet(saved, out_dir / "contact_sheet.png")
    return saved


def make_contact_sheet(image_paths: dict[str, Path], output_path: str | Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), dpi=180, facecolor=BG)
    for ax, (name, path) in zip(axes, image_paths.items()):
        ax.set_axis_off()
        ax.imshow(plt.imread(path))
        ax.text(0.015, 0.08, name.replace("_", " ").title(), transform=ax.transAxes, color=TEXT, fontsize=14, ha="left", va="bottom")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.035)
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render wide header-style visualizations from a soma_vs run.")
    parser.add_argument(
        "--run-dir",
        default="/home/alek/OlfactoryBulb/results/Old Debug/GammaSignature",
        help="Directory containing soma_vs.pkl, lfp.pkl, and input_times.pkl",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/alek/OlfactoryBulb/media/website_header_concepts",
        help="Directory where the preview PNG files should be written",
    )
    parser.add_argument(
        "--window-ms",
        type=float,
        default=650.0,
        help="Width of the selected highlight window in milliseconds",
    )
    args = parser.parse_args()

    bundle = load_run(args.run_dir)
    summary = summarize_run(bundle)
    print("Run summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    saved = save_concept_previews(bundle, args.output_dir, window_ms=args.window_ms)
    print("Saved previews:")
    for name, path in saved.items():
        print(f"  {name}: {path}")
    print(f"  contact_sheet: {Path(args.output_dir) / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
