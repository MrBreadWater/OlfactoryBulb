#!/usr/bin/env python3
"""Generate diagnostic figures for one HFO optimizer candidate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import signal

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import obgpu_experiment_helpers as hlp
import olfactorybulb.hfo_optimizer as hfo
from regenerate_hfo_packet_psd import regenerate_packet_psd


CELL_COLORS = {
    "MC": "#2563eb",
    "TC": "#dc2626",
    "GC": "#16a34a",
    "EPLI": "#9333ea",
    "PVCRH": "#9333ea",
    "other": "#4b5563",
}


def _load_candidate(campaign_dir: Path, candidate_id: str) -> dict[str, Any]:
    rows = hfo.load_candidate_archive_rows(campaign_dir)
    for row in rows:
        if str(row.get("candidate_id")) == str(candidate_id):
            return row
    for batch_file in sorted((campaign_dir / "batches").glob("batch_*_scored.json"), reverse=True):
        payload = json.loads(batch_file.read_text())
        for row in payload.get("candidate_rows", []):
            if str(row.get("candidate_id")) == str(candidate_id):
                return hfo.rescore_candidate_row(row)
    raise ValueError(f"Candidate {candidate_id!r} not found under {campaign_dir}")


def _condition_windows(row: dict[str, Any]) -> dict[str, tuple[float, float]]:
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


def _window_result(result: dict[str, Any], windows: dict[str, tuple[float, float]], condition: str) -> dict[str, Any]:
    start_ms, stop_ms = windows[condition]
    return hfo.window_result_for_condition(result, start_ms=start_ms, stop_ms=stop_ms, condition=condition)


def _finite_lfp(windowed: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(windowed.get("lfp_t", []), dtype=float)
    y = np.asarray(windowed.get("lfp", []), dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    return t[mask], y[mask]


def _save_lfp_zoom(result: dict[str, Any], windows: dict[str, tuple[float, float]], out: Path) -> None:
    t = np.asarray(result.get("lfp_t", []), dtype=float)
    y = np.asarray(result.get("lfp", []), dtype=float)
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    ax.plot(t, y, color="#111827", lw=0.8)
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


def _save_spectrogram(windowed: dict[str, Any], condition: str, out: Path) -> None:
    t, y = _finite_lfp(windowed)
    fig, ax = plt.subplots(figsize=(12, 5.0), constrained_layout=True)
    if len(t) > 16:
        dt_s = float(np.median(np.diff(t))) / 1000.0
        fs = 1.0 / max(dt_s, 1e-9)
        nperseg = min(1024, max(64, int(len(y) // 2)))
        noverlap = int(nperseg * 0.75)
        freqs, bins, spec = signal.spectrogram(
            y - np.mean(y),
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
            mode="psd",
        )
        mask = freqs <= 300.0
        mesh = ax.pcolormesh(
            bins * 1000.0 + float(t[0]),
            freqs[mask],
            np.log10(spec[mask] + 1e-18),
            shading="auto",
            cmap="magma",
        )
        fig.colorbar(mesh, ax=ax, label="log10 PSD")
    ax.axhspan(*hfo.DEFAULT_SCORE_BANDS["high_gamma"], color="#16a34a", alpha=0.09, lw=0)
    ax.axhspan(*hfo.DEFAULT_SCORE_BANDS["target_hfo"], color="#d97706", alpha=0.10, lw=0)
    ax.set_ylim(0, 300)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{condition} LFP spectrogram")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _spike_rows(windowed: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    spikes = windowed.get("soma_spikes") or {}
    labels = list(spikes.get("labels") or [])
    times = list(spikes.get("spike_times") or [])
    rows = []
    for label, values in zip(labels, times):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        rows.append((str(label), arr))
    return rows


def _save_raster(windowed: dict[str, Any], condition: str, out: Path) -> None:
    rows = _spike_rows(windowed)
    fig, ax = plt.subplots(figsize=(12, 6.0), constrained_layout=True)
    for y, (label, times) in enumerate(rows):
        if times.size == 0:
            continue
        cell_type = hlp.cell_type_of(label)
        ax.scatter(times, np.full(times.shape, y), s=4, color=CELL_COLORS.get(cell_type, CELL_COLORS["other"]), alpha=0.75)
    ax.set_xlabel("Time in window (ms)")
    ax.set_ylabel("Recorded soma index")
    ax.set_title(f"{condition} soma spike raster")
    ax.grid(True, axis="x", alpha=0.15)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _save_population_rates(windows_by_condition: dict[str, dict[str, Any]], out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.0), sharex=False, constrained_layout=True)
    for ax, (condition, windowed) in zip(axes, windows_by_condition.items()):
        rows = _spike_rows(windowed)
        t, _lfp = _finite_lfp(windowed)
        stop = float(np.max(t)) if len(t) else max([float(np.max(times)) for _label, times in rows if times.size] or [0.0])
        bins = np.arange(0.0, max(stop + 25.0, 50.0), 25.0)
        for cell_type in ("MC", "TC", "EPLI", "GC"):
            merged = [
                times
                for label, times in rows
                if hlp.cell_type_of(label) == cell_type and times.size
            ]
            if not merged:
                continue
            counts, edges = np.histogram(np.concatenate(merged), bins=bins)
            n_cells = max(sum(1 for label, _times in rows if hlp.cell_type_of(label) == cell_type), 1)
            rate = counts / n_cells / (np.diff(edges) / 1000.0)
            ax.plot(edges[:-1] + np.diff(edges) / 2, rate, label=cell_type, color=CELL_COLORS.get(cell_type, "#111827"))
        ax.set_title(f"{condition} population rates")
        ax.set_ylabel("Hz / cell")
        ax.grid(True, alpha=0.16)
        ax.legend(frameon=False, ncol=4)
    axes[-1].set_xlabel("Time in window (ms)")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _save_spike_frequency_kde(
    windowed: dict[str, Any],
    condition: str,
    label: str,
    cell_types: tuple[str, ...],
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    config = hlp.FrequencyPlotConfig(
        modulus=None,
        max_freq_hz=300.0,
        kde2d_engine="histogram",
        kde_resolution_t=120,
        kde_resolution_f=120,
        kde_cmap="inferno",
    )
    hlp.plot_spike_frequency_kde_2d(
        windowed,
        cell_types=cell_types,
        config=config,
        ax=ax,
        title=f"{condition} soma spike frequency KDE ({label})",
    )
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _save_input_overview(result: dict[str, Any], windows: dict[str, tuple[float, float]], out: Path) -> None:
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


def _band_phase(t_ms: np.ndarray, y: np.ndarray, band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    if len(t_ms) < 16:
        return t_ms, np.zeros_like(t_ms)
    dt_s = float(np.median(np.diff(t_ms))) / 1000.0
    fs = 1.0 / max(dt_s, 1e-9)
    high = min(float(band[1]) / (fs / 2.0), 0.99)
    low = max(float(band[0]) / (fs / 2.0), 1e-6)
    if not 0.0 < low < high:
        return t_ms, np.zeros_like(t_ms)
    sos = signal.butter(4, [low, high], btype="bandpass", output="sos")
    filtered = signal.sosfiltfilt(sos, y - np.mean(y))
    return t_ms, np.angle(signal.hilbert(filtered))


def _save_phase_hist(windowed: dict[str, Any], condition: str, out: Path) -> None:
    t, y = _finite_lfp(windowed)
    phase_t, phase = _band_phase(t, y, hfo.DEFAULT_SCORE_BANDS["target_hfo"])
    spike_phases: list[float] = []
    for label, times in _spike_rows(windowed):
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


def _refresh_contact_sheet(packet_dir: Path, files: list[str]) -> None:
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


def generate_packet(campaign_dir: Path, candidate_id: str, output_dir: Path | None = None) -> Path:
    row = _load_candidate(campaign_dir, candidate_id)
    result_dir = Path((row.get("ketamine_metrics") or {}).get("result_dir") or (row.get("control_metrics") or {})["result_dir"])
    result = hlp.load_result(result_dir, progress=False)
    windows = _condition_windows(row)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    packet_dir = output_dir or campaign_dir / "figures" / f"short_expected_{candidate_id}_{timestamp}"
    packet_dir.mkdir(parents=True, exist_ok=True)

    windowed = {condition: _window_result(result, windows, condition) for condition in ("control", "ketamine")}
    files = [
        "04_spectrogram_control.png",
        "05_spectrogram_ketamine.png",
        "06_lfp_windows.png",
        "07_raster_control.png",
        "08_raster_ketamine.png",
        "09_population_rates.png",
        "10_inputs.png",
        "11_phase_control.png",
        "12_phase_ketamine.png",
    ]
    _save_spectrogram(windowed["control"], "control", packet_dir / files[0])
    _save_spectrogram(windowed["ketamine"], "ketamine", packet_dir / files[1])
    _save_lfp_zoom(result, windows, packet_dir / files[2])
    _save_raster(windowed["control"], "control", packet_dir / files[3])
    _save_raster(windowed["ketamine"], "ketamine", packet_dir / files[4])
    _save_population_rates(windowed, packet_dir / files[5])
    _save_input_overview(result, windows, packet_dir / files[6])
    _save_phase_hist(windowed["control"], "control", packet_dir / files[7])
    _save_phase_hist(windowed["ketamine"], "ketamine", packet_dir / files[8])
    frequency_groups = [
        ("MT_EPLI", ("MC", "TC", "EPLI", "PVCRH")),
        ("GC", ("GC",)),
    ]
    for condition in ("control", "ketamine"):
        for group_label, cell_types in frequency_groups:
            name = f"13_spike_frequency_kde_2d_{condition}_{group_label}.png"
            _save_spike_frequency_kde(
                windowed[condition],
                condition,
                group_label,
                cell_types,
                packet_dir / name,
            )
            files.append(name)

    manifest = {
        "candidate_id": candidate_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "campaign_dir": str(campaign_dir),
        "result_dir": str(result_dir),
        "pair_score": row.get("pair_score"),
        "control_peak_hz": (row.get("control_metrics") or {}).get("peak_hz"),
        "ketamine_peak_hz": (row.get("ketamine_metrics") or {}).get("peak_hz"),
        "control_window_ms": list(windows["control"]),
        "ketamine_window_ms": list(windows["ketamine"]),
        "parameters": row.get("parameters"),
        "control_metrics": row.get("control_metrics"),
        "ketamine_metrics": row.get("ketamine_metrics"),
        "files": ["01_psd_control.png", "01_psd_ketamine.png", "03_psd_overlay.png", *files, "contact_sheet.png"],
    }
    (packet_dir / "manifest.json").write_text(json.dumps(hlp._json_ready(manifest), indent=2, sort_keys=True))
    regenerate_packet_psd(packet_dir)
    _refresh_contact_sheet(packet_dir, manifest["files"])
    return packet_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("candidate_id")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    print(generate_packet(args.campaign_dir, args.candidate_id, args.output_dir))


if __name__ == "__main__":
    main()
