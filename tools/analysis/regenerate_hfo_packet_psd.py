#!/usr/bin/env python3
"""Regenerate HFO packet PSD plots with objective target overlays."""

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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import obgpu_experiment_helpers as hlp
import olfactorybulb.hfo_optimizer as hfo
import olfactorybulb.hfo_visuals as hv


PSD_TARGET_VISUAL_FLOOR = 10 ** -7.5
PSD_PACKET_RENDER_VERSION = hv.PSD_PACKET_RENDER_VERSION


def _load_manifest(packet: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = packet / "manifest.json" if packet.is_dir() else packet
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing packet manifest: {manifest_path}")
    return manifest_path, json.loads(manifest_path.read_text())


def _finite_psd(summary: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.asarray(summary["freqs"], dtype=float)
    psd = np.asarray(summary["psd"], dtype=float)
    mask = np.isfinite(freqs) & np.isfinite(psd) & (freqs >= 1.0)
    return freqs[mask], psd[mask]


def _score_band_patches(ax: Any, bands: dict[str, tuple[float, float]]) -> None:
    high_gamma = bands.get("high_gamma")
    if high_gamma is not None:
        lo_hz, hi_hz = high_gamma
        ax.axvspan(lo_hz, hi_hz, color="#16a34a", alpha=0.07, lw=0, label="high gamma band")
    lo_hz, hi_hz = bands["target_hfo"]
    ax.axvspan(lo_hz, hi_hz, color="#d97706", alpha=0.10, lw=0, label="target HFO band")
    ax.axvline(195.0, color="#d97706", alpha=0.35, lw=1.25, ls=":", label="target center")


def _set_psd_axis_limits(ax: Any, arrays: list[np.ndarray]) -> None:
    positive_parts = [values[np.isfinite(values) & (values > 0.0)] for values in arrays]
    positive = np.concatenate([values for values in positive_parts if values.size])
    if not positive.size:
        return
    ax.set_ylim(max(float(np.nanmin(positive)) * 0.5, 1e-16), float(np.nanmax(positive)) * 2.0)
    ax.set_yscale("log")


def _normalized_template_for_plot(kind: str, freqs: np.ndarray) -> np.ndarray:
    _target_freqs, target = hfo.psd_template_curve(kind, freqs, floor=hfo.PSD_TEMPLATE_VISUAL_FLOOR)
    if not target.size:
        return target
    peak = float(np.nanmax(target))
    if peak <= 0.0:
        return np.zeros_like(target)
    normalized = target / peak
    normalized = np.maximum(normalized, PSD_TARGET_VISUAL_FLOOR)
    domain = (freqs >= min(hfo.PSD_TEMPLATE_FREQS_HZ)) & (freqs <= max(hfo.PSD_TEMPLATE_FREQS_HZ))
    return np.where(domain, normalized, np.nan)


def _finish_dual_axis_legend(ax: Any, target_ax: Any) -> None:
    handles, labels = ax.get_legend_handles_labels()
    target_handles, target_labels = target_ax.get_legend_handles_labels()
    ax.legend(
        handles + target_handles,
        labels + target_labels,
        loc="upper right",
        frameon=False,
        ncol=2,
    )


def _plot_single_psd(
    *,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    condition: str,
    target_kind: str,
    output_path: Path,
    color: str,
    bands: dict[str, tuple[float, float]],
) -> None:
    freqs, psd = _finite_psd(summary)
    target_shape = _normalized_template_for_plot(target_kind, freqs)

    fig, ax = plt.subplots(figsize=(12, 6.6), constrained_layout=True)
    _score_band_patches(ax, bands)
    ax.plot(freqs, psd, color=color, lw=2.0, label=f"{condition} LFP PSD")
    ax.set_xlim(0, 300)
    _set_psd_axis_limits(ax, [psd])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density")

    target_ax = ax.twinx()
    target_ax.plot(
        freqs,
        target_shape,
        color="#a21caf",
        lw=3.2,
        ls=(0, (6, 2)),
        alpha=0.95,
        label=f"scoring template ({target_kind}, 20-300 Hz)",
        zorder=6,
    )
    valid_target = np.isfinite(target_shape)
    target_ax.fill_between(
        freqs,
        0.0,
        np.where(valid_target, target_shape, 0.0),
        where=valid_target,
        color="#a21caf",
        alpha=0.08,
        lw=0,
    )
    target_ax.set_ylim(0.0, 1.05)
    target_ax.set_ylabel("Scoring template, normalized")
    target_ax.tick_params(axis="y", colors="#86198f")
    target_ax.spines["right"].set_color("#86198f")

    ax.set_title(f"{manifest.get('candidate_id', 'candidate')} {condition} PSD with scoring template")
    relative = (summary.get("relative_band_power") or {}).get("target_hfo")
    peak = manifest.get(f"{condition.lower()}_peak_hz")
    if relative is not None and peak is not None:
        ax.text(
            0.015,
            0.035,
            f"peak {float(peak):.1f} Hz | target relative power {float(relative):.4f}",
            transform=ax.transAxes,
            fontsize=10,
        )
    ax.grid(True, which="both", alpha=0.18)
    _finish_dual_axis_legend(ax, target_ax)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_overlay_psd(
    *,
    manifest: dict[str, Any],
    control_summary: dict[str, Any],
    ketamine_summary: dict[str, Any],
    output_path: Path,
    bands: dict[str, tuple[float, float]],
) -> None:
    control_freqs, control_psd = _finite_psd(control_summary)
    ketamine_freqs, ketamine_psd = _finite_psd(ketamine_summary)
    control_target_shape = _normalized_template_for_plot("control", control_freqs)
    ketamine_target_shape = _normalized_template_for_plot("ketamine", ketamine_freqs)

    fig, ax = plt.subplots(figsize=(12, 6.8), constrained_layout=True)
    _score_band_patches(ax, bands)
    ax.plot(control_freqs, control_psd, color="#2563eb", lw=1.9, label="control LFP PSD")
    ax.plot(ketamine_freqs, ketamine_psd, color="#dc2626", lw=1.9, label="ketamine LFP PSD")
    ax.set_xlim(0, 300)
    _set_psd_axis_limits(ax, [control_psd, ketamine_psd])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density")

    target_ax = ax.twinx()
    target_ax.plot(
        control_freqs,
        control_target_shape,
        color="#7c3aed",
        lw=2.8,
        ls=(0, (3, 2)),
        alpha=0.88,
        label="scoring template (control, 20-300 Hz)",
        zorder=6,
    )
    target_ax.plot(
        ketamine_freqs,
        ketamine_target_shape,
        color="#a21caf",
        lw=3.2,
        ls=(0, (7, 2)),
        alpha=0.95,
        label="scoring template (ketamine, 20-300 Hz)",
        zorder=7,
    )
    valid_ketamine_target = np.isfinite(ketamine_target_shape)
    target_ax.fill_between(
        ketamine_freqs,
        0.0,
        np.where(valid_ketamine_target, ketamine_target_shape, 0.0),
        where=valid_ketamine_target,
        color="#a21caf",
        alpha=0.06,
        lw=0,
    )
    target_ax.set_ylim(0.0, 1.05)
    target_ax.set_ylabel("Scoring template, normalized")
    target_ax.tick_params(axis="y", colors="#86198f")
    target_ax.spines["right"].set_color("#86198f")

    ax.set_title(f"{manifest.get('candidate_id', 'candidate')} PSD overlay with scoring templates")
    ax.grid(True, which="both", alpha=0.18)
    _finish_dual_axis_legend(ax, target_ax)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _refresh_contact_sheet(packet_dir: Path, manifest: dict[str, Any]) -> None:
    files = [name for name in manifest.get("files", []) if name != "contact_sheet.png"]
    image_paths = [
        (name, packet_dir / name)
        for name in files
        if (packet_dir / name).exists() and (packet_dir / name).suffix.lower() == ".png"
    ]
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


def regenerate_packet_psd(packet: Path) -> Path:
    manifest_path, manifest = _load_manifest(packet)
    packet_dir = manifest_path.parent
    result_dir = Path(manifest["result_dir"])
    result = hlp.load_result(result_dir, progress=False)

    control_window = tuple(float(value) for value in manifest.get("control_window_ms", [0.0, 4500.0]))
    ketamine_window = tuple(float(value) for value in manifest.get("ketamine_window_ms", [5000.0, 9000.0]))
    control = hfo.window_result_for_condition(
        result,
        start_ms=control_window[0],
        stop_ms=control_window[1],
        condition="control",
    )
    ketamine = hfo.window_result_for_condition(
        result,
        start_ms=ketamine_window[0],
        stop_ms=ketamine_window[1],
        condition="ketamine",
    )

    bands = dict(hfo.DEFAULT_SCORE_BANDS)
    control_summary = hlp.compute_hfo_power_summary(control, bands=bands, dt_ms=0.1, relative_band=(15.0, 250.0))
    ketamine_summary = hlp.compute_hfo_power_summary(ketamine, bands=bands, dt_ms=0.1, relative_band=(15.0, 250.0))

    _plot_single_psd(
        manifest=manifest,
        summary=control_summary,
        condition="control",
        target_kind="control",
        output_path=packet_dir / "01_psd_control.png",
        color="#2563eb",
        bands=bands,
    )
    _plot_single_psd(
        manifest=manifest,
        summary=ketamine_summary,
        condition="ketamine",
        target_kind="ketamine",
        output_path=packet_dir / "01_psd_ketamine.png",
        color="#dc2626",
        bands=bands,
    )
    _plot_overlay_psd(
        manifest=manifest,
        control_summary=control_summary,
        ketamine_summary=ketamine_summary,
        output_path=packet_dir / "03_psd_overlay.png",
        bands=bands,
    )

    manifest["psd_target_overlay"] = {
        **hv.psd_overlay_contract_snapshot(),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "templates": ["control", "ketamine"],
        "scaling": "normalized target shape on right axis",
        "source": "tools/analysis/regenerate_hfo_packet_psd.py",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    _refresh_contact_sheet(packet_dir, manifest)
    return packet_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path, help="Packet directory or manifest.json path")
    args = parser.parse_args()
    packet_dir = regenerate_packet_psd(args.packet)
    print(packet_dir)


if __name__ == "__main__":
    main()
