"""Central HFO visualization contracts and rendering helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import signal

import obgpu_experiment_helpers as hlp
import olfactorybulb.hfo_optimizer as hfo


CELL_COLORS = {
    "MC": "#2563eb",
    "TC": "#dc2626",
    "GC": "#16a34a",
    "EPLI": "#9333ea",
    "PVCRH": "#9333ea",
    "other": "#4b5563",
}

VISUAL_STYLE_VERSION = 12
PSD_PACKET_RENDER_VERSION = 1
NOTEBOOK_ANALYSIS_DT_MS = 0.1
NOTEBOOK_TIME_MODULUS_MS = 1e10
NOTEBOOK_SPECTROGRAM_VISUAL_WINDOW_MS = 1000.0
NOTEBOOK_SPECTROGRAM_TARGET_WINDOW_COUNT = 16
NOTEBOOK_SPECTROGRAM_MIN_NPERSEG = 128
NOTEBOOK_SPECTROGRAM_MAX_NPERSEG = 1024
NOTEBOOK_SPECTROGRAM_OVERLAP_RATIO = 0.9
SPECTROGRAM_GENERATOR_ID = "tools.analysis.generate_hfo_candidate_packet.generate_packet"
SPECTROGRAM_FILE_BY_CONDITION = {
    "control": "04_spectrogram_control.png",
    "ketamine": "05_spectrogram_ketamine.png",
}
SPECTROGRAM_PIPELINE = {
    "module": "tools.analysis.generate_hfo_candidate_packet",
    "function": "_save_spectrogram",
    "helper": "obgpu_experiment_helpers.plot_spectrogram",
    "source_signal": "lfp",
    "source_metric": "windowed.result['lfp']",
    "generator": SPECTROGRAM_GENERATOR_ID,
}
PRIMARY_PSD_NAME_ORDER = (
    "03_psd_overlay.png",
    "03_power_spectrum_control_vs_ketamine.png",
    "01_lfp_psd_ketamine.png",
    "01_psd_ketamine.png",
    "01_lfp_psd_control.png",
    "01_psd_control.png",
)


@dataclass(frozen=True)
class FrequencyGroupSpec:
    label: str
    cell_types: tuple[str, ...]


@dataclass(frozen=True)
class ConditionPairSpec:
    title: str
    control_file: str
    ketamine_file: str
    dom_id_suffix: str
    open_by_default: bool = False


@dataclass(frozen=True)
class DashboardTabSpec:
    key: str
    label: str
    table_heading: str
    packet_heading: str


FREQUENCY_GROUPS = (
    FrequencyGroupSpec(label="MT", cell_types=("MC", "TC")),
    FrequencyGroupSpec(label="EPLI", cell_types=("EPLI", "PVCRH")),
    FrequencyGroupSpec(label="GC", cell_types=("GC",)),
)

FIXED_CONDITION_PAIR_SPECS = (
    ConditionPairSpec(
        title="LFP spectrogram",
        control_file=SPECTROGRAM_FILE_BY_CONDITION["control"],
        ketamine_file=SPECTROGRAM_FILE_BY_CONDITION["ketamine"],
        dom_id_suffix="spectrogram",
        open_by_default=True,
    ),
    ConditionPairSpec(
        title="Soma spike raster",
        control_file="07_raster_control.png",
        ketamine_file="08_raster_ketamine.png",
        dom_id_suffix="raster",
    ),
    ConditionPairSpec(
        title="Target-HFO phase locking",
        control_file="11_phase_control.png",
        ketamine_file="12_phase_ketamine.png",
        dom_id_suffix="phase",
    ),
)

DASHBOARD_TABS = (
    DashboardTabSpec(
        key="best",
        label="Best",
        table_heading="Top Candidates",
        packet_heading="Best Visual Packets",
    ),
    DashboardTabSpec(
        key="recent",
        label="Recent",
        table_heading="Most Recent Candidates",
        packet_heading="Recent Visual Packets",
    ),
)

PACKET_BASE_FILES = (
    SPECTROGRAM_FILE_BY_CONDITION["control"],
    SPECTROGRAM_FILE_BY_CONDITION["ketamine"],
    "06_lfp_windows.png",
    "07_raster_control.png",
    "08_raster_ketamine.png",
    "10_inputs.png",
    "11_phase_control.png",
    "12_phase_ketamine.png",
)

NOTEBOOK_FREQ_CONFIG = hlp.FrequencyPlotConfig(
    modulus=NOTEBOOK_TIME_MODULUS_MS,
    max_freq_hz=hfo.DEFAULT_SCORE_BANDS["target_hfo"][1],
    kde_bw_method="scott",
    kde1d_engine="exact",
    kde_bw_x=0.125,
    kde_bw_y=0.25,
    kde2d_engine="histogram",
    kde_resolution_t=100,
    kde_resolution_f=100,
    kde_f_resolution=1600,
    num_time_bins=32,
    bin_alpha=0.5,
    kde_cmap="inferno",
    dot_size=5,
    dot_alpha=0.2,
    strip_plot=True,
    guide_line_spacing_ms=0.0,
)


def notebook_spectrogram_max_freq_hz() -> float:
    return float(hfo.DEFAULT_SCORE_BANDS["target_hfo"][1])


def dashboard_tabs() -> tuple[DashboardTabSpec, ...]:
    return DASHBOARD_TABS


def frequency_group_specs() -> tuple[FrequencyGroupSpec, ...]:
    return FREQUENCY_GROUPS


def fixed_condition_pair_specs() -> tuple[ConditionPairSpec, ...]:
    return FIXED_CONDITION_PAIR_SPECS


def packet_manifest_files() -> list[str]:
    files = ["01_psd_control.png", "01_psd_ketamine.png", "03_psd_overlay.png", *PACKET_BASE_FILES]
    for condition in ("control", "ketamine"):
        for group in FREQUENCY_GROUPS:
            files.append(kde_filename("1d", condition, group.label))
            files.append(kde_filename("2d", condition, group.label))
    files.append("contact_sheet.png")
    return files


def visual_contract_snapshot() -> dict[str, Any]:
    return {
        "style_version": VISUAL_STYLE_VERSION,
        "frequency_groups": [asdict(group) for group in FREQUENCY_GROUPS],
        "fixed_condition_pairs": [asdict(pair) for pair in FIXED_CONDITION_PAIR_SPECS],
        "dashboard_tabs": [asdict(tab) for tab in DASHBOARD_TABS],
        "primary_psd_name_order": list(PRIMARY_PSD_NAME_ORDER),
        "packet_files": packet_manifest_files(),
        "spectrogram_pipeline": dict(SPECTROGRAM_PIPELINE),
        "spectrogram_window_ms": NOTEBOOK_SPECTROGRAM_VISUAL_WINDOW_MS,
    }


def psd_overlay_contract_snapshot() -> dict[str, Any]:
    return {
        "render_version": PSD_PACKET_RENDER_VERSION,
        "target_hfo_hz": list(hfo.DEFAULT_SCORE_BANDS["target_hfo"]),
        "high_gamma_hz": list(hfo.DEFAULT_SCORE_BANDS["high_gamma"]),
    }


def kde_filename(kind: str, condition: str, group_label: str) -> str:
    return f"13_spike_frequency_kde_{kind}_{condition}_{group_label}.png"


def parse_kde_filename(name: str, *, kind: str) -> tuple[str, str] | None:
    prefix = f"13_spike_frequency_kde_{kind}_"
    suffix = ".png"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    remainder = name[len(prefix) : -len(suffix)]
    try:
        condition, group = remainder.split("_", 1)
    except ValueError:
        return None
    if condition not in {"control", "ketamine"} or not group:
        return None
    return condition, group


def spectrogram_window_geometry(windowed: dict[str, Any]) -> tuple[int, int]:
    """Choose a dynamic spectrogram geometry that preserves time bins on 1 s slices."""
    _t_ms, values = finite_lfp(windowed)
    n_samples = int(len(values))
    if n_samples <= 1:
        return NOTEBOOK_SPECTROGRAM_MIN_NPERSEG, 0

    nperseg = max(
        NOTEBOOK_SPECTROGRAM_MIN_NPERSEG,
        min(NOTEBOOK_SPECTROGRAM_MAX_NPERSEG, max(1, n_samples // NOTEBOOK_SPECTROGRAM_TARGET_WINDOW_COUNT)),
    )
    noverlap = max(0, min(int(NOTEBOOK_SPECTROGRAM_OVERLAP_RATIO * nperseg), nperseg - 1))
    return nperseg, noverlap


def condition_windows(row: dict[str, Any]) -> dict[str, tuple[float, float]]:
    control = row.get("control_metrics") or {}
    ketamine = row.get("ketamine_metrics") or {}
    return {
        "control": (
            float(control.get("window_start_ms", 0.0)),
            float(control.get("window_stop_ms", 0.0)),
        ),
        "ketamine": (
            float(ketamine.get("window_start_ms", 0.0)),
            float(ketamine.get("window_stop_ms", 0.0)),
        ),
    }


def spectrogram_switch_time(row: dict[str, Any], result: dict[str, Any]) -> float:
    summary = result.get("summary") or {}
    params = summary.get("params") or {}
    switch = params.get("ketamine_switch")
    if isinstance(switch, dict):
        try:
            switch_time = float(switch.get("time_ms"))
        except (TypeError, ValueError):
            switch_time = math.nan
        if math.isfinite(switch_time) and switch_time > 0.0:
            return switch_time

    for container in (row.get("parameters") or {}, row.get("control_metrics") or {}, row.get("ketamine_metrics") or {}):
        for key in ("ketamine_switch_time_ms", "hfo_ketamine_switch_time_ms", "switch_time_ms"):
            try:
                switch_time = float(container.get(key))
            except (AttributeError, TypeError, ValueError):
                continue
            if math.isfinite(switch_time) and switch_time > 0.0:
                return switch_time

    t_ms = np.asarray(result.get("lfp_t", []), dtype=float)
    finite_t = t_ms[np.isfinite(t_ms)]
    if finite_t.size:
        return float(np.max(finite_t)) * 0.5
    return float(NOTEBOOK_SPECTROGRAM_VISUAL_WINDOW_MS)


def spectrogram_windows(row: dict[str, Any], result: dict[str, Any]) -> dict[str, tuple[float, float]]:
    switch_time = spectrogram_switch_time(row, result)
    vis_window = float(NOTEBOOK_SPECTROGRAM_VISUAL_WINDOW_MS)
    t_ms = np.asarray(result.get("lfp_t", []), dtype=float)
    finite_t = t_ms[np.isfinite(t_ms)]
    result_start = float(np.min(finite_t)) if finite_t.size else 0.0
    result_stop = float(np.max(finite_t)) if finite_t.size else max(switch_time + vis_window, vis_window)
    control_start = max(result_start, switch_time - vis_window)
    control_stop = min(result_stop, switch_time)
    ketamine_start = max(result_start, switch_time)
    ketamine_stop = min(result_stop, switch_time + vis_window)
    return {
        "control": (float(control_start), float(control_stop)),
        "ketamine": (float(ketamine_start), float(ketamine_stop)),
    }


def window_result(result: dict[str, Any], windows: dict[str, tuple[float, float]], condition: str) -> dict[str, Any]:
    start_ms, stop_ms = windows[condition]
    return hfo.window_result_for_condition(result, start_ms=start_ms, stop_ms=stop_ms, condition=condition)


def finite_lfp(windowed: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    t_ms = np.asarray(windowed.get("lfp_t", []), dtype=float)
    values = np.asarray(windowed.get("lfp", []), dtype=float)
    mask = np.isfinite(t_ms) & np.isfinite(values)
    return t_ms[mask], values[mask]


def save_lfp_zoom(result: dict[str, Any], windows: dict[str, tuple[float, float]], out: Path) -> None:
    t_ms = np.asarray(result.get("lfp_t", []), dtype=float)
    values = np.asarray(result.get("lfp", []), dtype=float)
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    ax.plot(t_ms, values, color="#111827", lw=0.8)
    for name, color in [("control", "#2563eb"), ("ketamine", "#dc2626")]:
        start, stop = windows[name]
        ax.axvspan(start, stop, color=color, alpha=0.08, lw=0, label=f"{name} scoring window")
    ax.axvline(windows["control"][1], color="#6b7280", lw=1.0, ls=":", label="switch")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("LFP proxy")
    ax.set_title("LFP trace and scoring windows")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, alpha=0.18)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_spectrogram(windowed: dict[str, Any], condition: str, out: Path, *, nperseg: int, noverlap: int) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.0), constrained_layout=True)
    try:
        hlp.plot_spectrogram(
            windowed,
            signal="lfp",
            dt_ms=NOTEBOOK_ANALYSIS_DT_MS,
            max_freq_hz=notebook_spectrogram_max_freq_hz(),
            nperseg=nperseg,
            noverlap=noverlap,
            modulus=None,
            ax=ax,
        )
    except Exception as exc:
        ax.text(0.5, 0.5, f"Could not render spectrogram: {exc}", ha="center", va="center", transform=ax.transAxes)
    ax.axhspan(*hfo.DEFAULT_SCORE_BANDS["high_gamma"], color="#16a34a", alpha=0.09, lw=0)
    ax.axhspan(*hfo.DEFAULT_SCORE_BANDS["target_hfo"], color="#d97706", alpha=0.10, lw=0)
    ax.set_ylim(0, notebook_spectrogram_max_freq_hz())
    ax.set_title(f"{condition} LFP spectrogram")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def spike_rows(windowed: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    spikes = windowed.get("soma_spikes") or {}
    labels = list(spikes.get("labels") or [])
    times = list(spikes.get("spike_times") or [])
    rows = []
    for label, values in zip(labels, times):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        rows.append((str(label), arr))
    return rows


def save_raster(windowed: dict[str, Any], condition: str, out: Path) -> None:
    rows = spike_rows(windowed)
    fig, ax = plt.subplots(figsize=(12, 6.0), constrained_layout=True)
    for y_index, (label, times) in enumerate(rows):
        if times.size == 0:
            continue
        cell_type = hlp.cell_type_of(label)
        ax.scatter(times, np.full(times.shape, y_index), s=4, color=CELL_COLORS.get(cell_type, CELL_COLORS["other"]), alpha=0.75)
    ax.set_xlabel("Time in window (ms)")
    ax.set_ylabel("Recorded soma index")
    ax.set_title(f"{condition} soma spike raster")
    ax.grid(True, axis="x", alpha=0.15)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_spike_frequency_kde_1d(
    windowed: dict[str, Any],
    condition: str,
    label: str,
    cell_types: tuple[str, ...],
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.0), constrained_layout=True)
    hlp.plot_spike_frequency_kde_1d(
        windowed,
        cell_types=cell_types,
        config=NOTEBOOK_FREQ_CONFIG,
        ax=ax,
        title=f"{condition} soma spike frequency 1D KDE ({label})",
    )
    ax.axvspan(*hfo.DEFAULT_SCORE_BANDS["high_gamma"], color="#16a34a", alpha=0.08, lw=0)
    ax.axvspan(*hfo.DEFAULT_SCORE_BANDS["target_hfo"], color="#d97706", alpha=0.08, lw=0)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_spike_frequency_kde_2d(
    windowed: dict[str, Any],
    condition: str,
    label: str,
    cell_types: tuple[str, ...],
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    hlp.plot_spike_frequency_kde_2d(
        windowed,
        cell_types=cell_types,
        config=NOTEBOOK_FREQ_CONFIG,
        ax=ax,
        title=f"{condition} soma spike frequency KDE ({label})",
    )
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_input_overview(result: dict[str, Any], windows: dict[str, tuple[float, float]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    all_times = []
    for _label, times in result.get("input_times", []) or []:
        values = np.asarray(times, dtype=float)
        all_times.extend(values[np.isfinite(values)].tolist())
    if all_times:
        bins = np.arange(0.0, max(all_times) + 25.0, 25.0)
        ax.hist(all_times, bins=bins, color="#0f766e", alpha=0.75)
    for name, color in [("control", "#2563eb"), ("ketamine", "#dc2626")]:
        start, stop = windows[name]
        ax.axvspan(start, stop, color=color, alpha=0.08, lw=0, label=f"{name} scoring window")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Input events / 25 ms")
    ax.set_title("Afferent input event overview")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.16)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def band_phase(t_ms: np.ndarray, values: np.ndarray, band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    if len(t_ms) < 16:
        return t_ms, np.zeros_like(t_ms)
    dt_s = float(np.median(np.diff(t_ms))) / 1000.0
    fs = 1.0 / max(dt_s, 1e-9)
    high = min(float(band[1]) / (fs / 2.0), 0.99)
    low = max(float(band[0]) / (fs / 2.0), 1e-6)
    if not 0.0 < low < high:
        return t_ms, np.zeros_like(t_ms)
    sos = signal.butter(4, [low, high], btype="bandpass", output="sos")
    filtered = signal.sosfiltfilt(sos, values - np.mean(values))
    return t_ms, np.angle(signal.hilbert(filtered))


def save_phase_hist(windowed: dict[str, Any], condition: str, out: Path) -> None:
    t_ms, values = finite_lfp(windowed)
    phase_t, phase = band_phase(t_ms, values, hfo.DEFAULT_SCORE_BANDS["target_hfo"])
    spike_phases: list[float] = []
    for label, times in spike_rows(windowed):
        if hlp.cell_type_of(label) not in {"MC", "TC", "EPLI", "PVCRH"} or not len(times):
            continue
        spike_phases.extend(np.interp(times, phase_t, phase).tolist())
    fig, ax = plt.subplots(figsize=(7.2, 5.2), subplot_kw={"projection": "polar"}, constrained_layout=True)
    if spike_phases:
        bins = np.linspace(-np.pi, np.pi, 25)
        counts, edges = np.histogram(spike_phases, bins=bins)
        centers = edges[:-1] + np.diff(edges) / 2
        ax.bar(centers, counts, width=np.diff(edges), color="#a21caf", alpha=0.75)
    ax.set_title(f"{condition} M/T/EPLI phase to target-HFO LFP")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def refresh_contact_sheet(packet_dir: Path, files: list[str]) -> None:
    image_paths = [
        (name, packet_dir / name)
        for name in files
        if name != "contact_sheet.png" and (packet_dir / name).suffix.lower() == ".png"
    ]
    image_paths = [(name, path) for name, path in image_paths if path.exists()]
    if not image_paths:
        return
    thumb_w, thumb_h = 360, 230
    cols = 3
    rows = math.ceil(len(image_paths) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (name, path) in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h - 26), Image.Resampling.LANCZOS)
        x = (index % cols) * thumb_w
        y = (index // cols) * thumb_h
        sheet.paste(image, (x + (thumb_w - image.width) // 2, y + 22 + (thumb_h - 26 - image.height) // 2))
        draw.text((x + 8, y + 5), name, fill=(20, 20, 20))
    sheet.save(packet_dir / "contact_sheet.png")


__all__ = [
    "CELL_COLORS",
    "DASHBOARD_TABS",
    "FIXED_CONDITION_PAIR_SPECS",
    "FREQUENCY_GROUPS",
    "NOTEBOOK_ANALYSIS_DT_MS",
    "NOTEBOOK_FREQ_CONFIG",
    "NOTEBOOK_SPECTROGRAM_VISUAL_WINDOW_MS",
    "PRIMARY_PSD_NAME_ORDER",
    "PSD_PACKET_RENDER_VERSION",
    "PACKET_BASE_FILES",
    "SPECTROGRAM_FILE_BY_CONDITION",
    "SPECTROGRAM_PIPELINE",
    "VISUAL_STYLE_VERSION",
    "ConditionPairSpec",
    "DashboardTabSpec",
    "FrequencyGroupSpec",
    "band_phase",
    "condition_windows",
    "dashboard_tabs",
    "finite_lfp",
    "fixed_condition_pair_specs",
    "frequency_group_specs",
    "kde_filename",
    "notebook_spectrogram_max_freq_hz",
    "packet_manifest_files",
    "parse_kde_filename",
    "psd_overlay_contract_snapshot",
    "refresh_contact_sheet",
    "save_input_overview",
    "save_lfp_zoom",
    "save_phase_hist",
    "save_raster",
    "save_spike_frequency_kde_1d",
    "save_spike_frequency_kde_2d",
    "save_spectrogram",
    "spectrogram_switch_time",
    "spectrogram_window_geometry",
    "spectrogram_windows",
    "spike_rows",
    "visual_contract_snapshot",
    "window_result",
]
