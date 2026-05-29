#!/usr/bin/env python3
"""Generate diagnostic figures for one HFO optimizer candidate."""

from __future__ import annotations

import argparse
import contextlib
import concurrent.futures
import json
import fcntl
import os
import shutil
import multiprocessing as mp
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import obgpu_experiment_helpers as hlp
from olfactorybulb.hfo_features import parameter_contract_snapshot
import olfactorybulb.hfo_optimizer as hfo
import olfactorybulb.hfo_visuals as hv
from tools.analysis.regenerate_hfo_packet_psd import regenerate_packet_psd


VISUAL_STYLE_VERSION = hv.VISUAL_STYLE_VERSION
SPECTROGRAM_FILE_CONTROL = hv.SPECTROGRAM_FILE_BY_CONDITION["control"]
SPECTROGRAM_FILE_KETAMINE = hv.SPECTROGRAM_FILE_BY_CONDITION["ketamine"]
SPECTROGRAM_MOD200_FILE_CONTROL = hv.SPECTROGRAM_MOD200_FILE_BY_CONDITION["control"]
SPECTROGRAM_MOD200_FILE_KETAMINE = hv.SPECTROGRAM_MOD200_FILE_BY_CONDITION["ketamine"]
SPECTROGRAM_PIPELINE = hv.SPECTROGRAM_PIPELINE
_spectrogram_window_geometry = hv.spectrogram_window_geometry
_save_spectrogram = hv.save_spectrogram
_PACKET_RENDER_CONTEXT: dict[str, Any] | None = None


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


def packet_build_lock_path(campaign_dir: Path, candidate_id: str) -> Path:
    return campaign_dir / ".runtime" / "packet-build-locks" / f"{candidate_id}.lock"


def _packet_build_lock_path(campaign_dir: Path, candidate_id: str) -> Path:
    return packet_build_lock_path(campaign_dir, candidate_id)


@contextlib.contextmanager
def _packet_generation_lock(campaign_dir: Path, candidate_id: str):
    lock_path = _packet_build_lock_path(campaign_dir, candidate_id)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _packet_render_worker_init(context: dict[str, Any]) -> None:
    global _PACKET_RENDER_CONTEXT
    _PACKET_RENDER_CONTEXT = context


def _packet_render_task(task: dict[str, Any]) -> str:
    if _PACKET_RENDER_CONTEXT is None:
        raise RuntimeError("Packet render worker context has not been initialized")

    context = _PACKET_RENDER_CONTEXT
    packet_dir = Path(context["packet_dir"])
    kind = str(task["kind"])
    out = packet_dir / str(task["out"])

    if kind == "spectrogram":
        condition = str(task["condition"])
        windowed = context["spectrogram_windowed"][condition]
        nperseg, noverlap = context["spectrogram_geometry"][condition]
        _save_spectrogram(
            windowed,
            condition,
            out,
            nperseg=int(nperseg),
            noverlap=int(noverlap),
            modulus_ms=task.get("modulus_ms"),
        )
    elif kind == "lfp_zoom":
        hv.save_lfp_zoom(
            context["result"],
            context["windows"],
            out,
            modulus_ms=task.get("modulus_ms"),
        )
    elif kind == "raster":
        condition = str(task["condition"])
        hv.save_raster(
            context["windowed"][condition],
            condition,
            out,
            modulus_ms=task.get("modulus_ms"),
        )
    elif kind == "inputs":
        hv.save_input_overview(
            context["result"],
            context["windows"],
            out,
            modulus_ms=task.get("modulus_ms"),
        )
    elif kind == "phase_hist":
        condition = str(task["condition"])
        hv.save_phase_hist(
            context["windowed"][condition],
            condition,
            out,
        )
    elif kind == "kde1d":
        condition = str(task["condition"])
        hv.save_spike_frequency_kde_1d(
            context["windowed"][condition],
            condition,
            str(task["label"]),
            tuple(task["cell_types"]),
            out,
        )
    elif kind == "kde2d":
        condition = str(task["condition"])
        hv.save_spike_frequency_kde_2d(
            context["windowed"][condition],
            condition,
            str(task["label"]),
            tuple(task["cell_types"]),
            out,
            modulus_ms=task.get("modulus_ms"),
        )
    else:
        raise ValueError(f"Unsupported packet render task kind {kind!r}")

    return str(out)


def _packet_render_worker_count(workers: int | None) -> int:
    requested = 0 if workers is None else int(workers)
    if requested <= 0:
        requested = int(os.cpu_count() or 1)
    return max(1, requested)


def _packet_manifest_is_current(packet_dir: Path, candidate_id: str) -> bool:
    manifest_path = packet_dir / "manifest.json"
    if not packet_dir.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if str(manifest.get("candidate_id") or "") != str(candidate_id):
        return False
    if int(manifest.get("visual_style_version", -1) or -1) != int(VISUAL_STYLE_VERSION):
        return False
    overlay = manifest.get("psd_target_overlay") or {}
    if not isinstance(overlay, dict):
        return False
    if int(overlay.get("render_version", -1) or -1) != int(hv.PSD_PACKET_RENDER_VERSION):
        return False
    if list(overlay.get("target_hfo_hz") or []) != list(hfo.DEFAULT_SCORE_BANDS["target_hfo"]):
        return False
    if list(overlay.get("high_gamma_hz") or []) != list(hfo.DEFAULT_SCORE_BANDS["high_gamma"]):
        return False
    files = manifest.get("files") or hv.packet_manifest_files()
    if not isinstance(files, list):
        return False
    for name in files:
        if not (packet_dir / str(name)).exists():
            return False
    return True


def _path_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return 0.0


def _find_current_packet_dir(campaign_dir: Path, candidate_id: str) -> Path | None:
    figures_dir = campaign_dir / "figures"
    if not figures_dir.exists():
        return None
    packet_dirs = sorted(
        (path for path in figures_dir.iterdir() if path.is_dir() and not path.name.startswith(".")),
        key=_path_mtime,
        reverse=True,
    )
    for packet_dir in packet_dirs:
        if _packet_manifest_is_current(packet_dir, candidate_id):
            return packet_dir
    return None


def generate_packet(
    campaign_dir: Path,
    candidate_id: str,
    output_dir: Path | None = None,
    *,
    workers: int | None = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_packet_dir = output_dir or campaign_dir / "figures" / f"short_expected_{candidate_id}_{timestamp}"
    if output_dir is not None and _packet_manifest_is_current(final_packet_dir, candidate_id):
        return final_packet_dir
    if output_dir is None:
        current_packet = _find_current_packet_dir(campaign_dir, candidate_id)
        if current_packet is not None:
            return current_packet

    with _packet_generation_lock(campaign_dir, candidate_id):
        if output_dir is not None and _packet_manifest_is_current(final_packet_dir, candidate_id):
            return final_packet_dir
        if output_dir is None:
            current_packet = _find_current_packet_dir(campaign_dir, candidate_id)
            if current_packet is not None:
                return current_packet

        row = _load_candidate(campaign_dir, candidate_id)
        result_dir = Path((row.get("ketamine_metrics") or {}).get("result_dir") or (row.get("control_metrics") or {})["result_dir"])
        result = hlp.load_result(result_dir, progress=False)
        windows = hv.condition_windows(row)
        spectrogram_windows = hv.spectrogram_windows(row, result)
        packet_dir = final_packet_dir
        packet_dir.mkdir(parents=True, exist_ok=True)

        windowed = {condition: hv.window_result(result, windows, condition) for condition in ("control", "ketamine")}
        spectrogram_windowed = {
            condition: hv.window_result(result, spectrogram_windows, condition)
            for condition in ("control", "ketamine")
        }
        spectrogram_switch_time = hv.spectrogram_switch_time(row, result)

        control_geom = hv.spectrogram_window_geometry(spectrogram_windowed["control"])
        ketamine_geom = hv.spectrogram_window_geometry(spectrogram_windowed["ketamine"])

        tmp_packet_dir = final_packet_dir.parent / f".{final_packet_dir.name}.tmp-{os.getpid()}-{time.time_ns()}"
        tmp_packet_dir.mkdir(parents=True, exist_ok=False)
        try:
            render_tasks: list[dict[str, Any]] = [
                {
                    "kind": "spectrogram",
                    "condition": "control",
                    "out": SPECTROGRAM_FILE_CONTROL,
                },
                {
                    "kind": "spectrogram",
                    "condition": "ketamine",
                    "out": SPECTROGRAM_FILE_KETAMINE,
                },
                {
                    "kind": "spectrogram",
                    "condition": "control",
                    "out": SPECTROGRAM_MOD200_FILE_CONTROL,
                    "modulus_ms": hv.NOTEBOOK_PACKET_TIME_MODULUS_MS,
                },
                {
                    "kind": "spectrogram",
                    "condition": "ketamine",
                    "out": SPECTROGRAM_MOD200_FILE_KETAMINE,
                    "modulus_ms": hv.NOTEBOOK_PACKET_TIME_MODULUS_MS,
                },
                {"kind": "lfp_zoom", "out": "06_lfp_windows.png"},
                {
                    "kind": "lfp_zoom",
                    "out": "06_lfp_windows_mod200.png",
                    "modulus_ms": hv.NOTEBOOK_PACKET_TIME_MODULUS_MS,
                },
                {"kind": "raster", "condition": "control", "out": "07_raster_control.png"},
                {"kind": "raster", "condition": "ketamine", "out": "08_raster_ketamine.png"},
                {
                    "kind": "raster",
                    "condition": "control",
                    "out": "07_raster_control_mod200.png",
                    "modulus_ms": hv.NOTEBOOK_PACKET_TIME_MODULUS_MS,
                },
                {
                    "kind": "raster",
                    "condition": "ketamine",
                    "out": "08_raster_ketamine_mod200.png",
                    "modulus_ms": hv.NOTEBOOK_PACKET_TIME_MODULUS_MS,
                },
                {"kind": "inputs", "out": "10_inputs.png"},
                {
                    "kind": "inputs",
                    "out": "10_inputs_mod200.png",
                    "modulus_ms": hv.NOTEBOOK_PACKET_TIME_MODULUS_MS,
                },
                {"kind": "phase_hist", "condition": "control", "out": "11_phase_control.png"},
                {"kind": "phase_hist", "condition": "ketamine", "out": "12_phase_ketamine.png"},
            ]
            for condition in ("control", "ketamine"):
                for group in hv.frequency_group_specs():
                    render_tasks.append(
                        {
                            "kind": "kde1d",
                            "condition": condition,
                            "label": group.label,
                            "cell_types": list(group.cell_types),
                            "out": hv.kde_filename("1d", condition, group.label),
                        }
                    )
                    render_tasks.append(
                        {
                            "kind": "kde2d",
                            "condition": condition,
                            "label": group.label,
                            "cell_types": list(group.cell_types),
                            "out": hv.kde_filename("2d", condition, group.label),
                        }
                    )
                    render_tasks.append(
                        {
                            "kind": "kde2d",
                            "condition": condition,
                            "label": group.label,
                            "cell_types": list(group.cell_types),
                            "out": hv.kde_filename("2d", condition, group.label, suffix="mod200"),
                            "modulus_ms": hv.NOTEBOOK_PACKET_TIME_MODULUS_MS,
                        }
                    )

            render_context = {
                "packet_dir": tmp_packet_dir,
                "result": result,
                "windows": windows,
                "windowed": windowed,
                "spectrogram_windowed": spectrogram_windowed,
                "spectrogram_geometry": {
                    "control": control_geom,
                    "ketamine": ketamine_geom,
                },
            }
            render_workers = _packet_render_worker_count(workers)
            if render_workers > 1 and len(render_tasks) > 1:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=min(render_workers, len(render_tasks)),
                    mp_context=mp.get_context("spawn"),
                    initializer=_packet_render_worker_init,
                    initargs=(render_context,),
                ) as pool:
                    for _ in pool.map(_packet_render_task, render_tasks):
                        pass
            else:
                global _PACKET_RENDER_CONTEXT
                previous_context = _PACKET_RENDER_CONTEXT
                _PACKET_RENDER_CONTEXT = render_context
                try:
                    for task in render_tasks:
                        _packet_render_task(task)
                finally:
                    _PACKET_RENDER_CONTEXT = previous_context

            created_at = datetime.now().isoformat(timespec="seconds")
            manifest = {
                "candidate_id": candidate_id,
                "created_at": created_at,
                "visual_style_version": VISUAL_STYLE_VERSION,
                "visual_contract": hv.visual_contract_snapshot(),
                "parameter_contract": parameter_contract_snapshot(campaign_dir=campaign_dir),
                "campaign_dir": str(campaign_dir),
                "result_dir": str(result_dir),
                "pair_score": row.get("pair_score"),
                "pair_score_version": row.get("pair_score_version"),
                "control_peak_hz": (row.get("control_metrics") or {}).get("peak_hz"),
                "ketamine_peak_hz": (row.get("ketamine_metrics") or {}).get("peak_hz"),
                "control_window_ms": list(windows["control"]),
                "ketamine_window_ms": list(windows["ketamine"]),
                "spectrogram_window_ms": hv.NOTEBOOK_SPECTROGRAM_VISUAL_WINDOW_MS,
                "spectrogram_switch_time_ms": spectrogram_switch_time,
                "spectrogram_window_ms_by_condition": {
                    "control": list(spectrogram_windows["control"]),
                    "ketamine": list(spectrogram_windows["ketamine"]),
                },
                "spectrogram_geometry": {
                    "control": {"nperseg": control_geom[0], "noverlap": control_geom[1]},
                    "ketamine": {"nperseg": ketamine_geom[0], "noverlap": ketamine_geom[1]},
                    "dt_ms": hv.NOTEBOOK_ANALYSIS_DT_MS,
                    "max_freq_hz": hv.notebook_spectrogram_max_freq_hz(),
                },
                "spectrogram_generation": {
                    "pipeline": SPECTROGRAM_PIPELINE,
                    "control_file": SPECTROGRAM_FILE_CONTROL,
                    "ketamine_file": SPECTROGRAM_FILE_KETAMINE,
                    "generated_at": created_at,
                    "note": "lfp spectrograms produced from 1000 ms visualization windows using dense overlap and the existing helper renderer",
                },
                "packet_render_workers": render_workers,
                "parameters": row.get("parameters"),
                "control_metrics": row.get("control_metrics"),
                "ketamine_metrics": row.get("ketamine_metrics"),
                "files": hv.packet_manifest_files(),
            }
            (tmp_packet_dir / "manifest.json").write_text(json.dumps(hlp._json_ready(manifest), indent=2, sort_keys=True))
            regenerate_packet_psd(tmp_packet_dir)
            hv.refresh_contact_sheet(tmp_packet_dir, manifest["files"])
            if final_packet_dir.exists():
                shutil.rmtree(final_packet_dir)
            tmp_packet_dir.rename(final_packet_dir)
            return final_packet_dir
        except Exception:
            shutil.rmtree(tmp_packet_dir, ignore_errors=True)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("candidate_id")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Render packet figures in parallel with this many workers.")
    args = parser.parse_args()
    print(generate_packet(args.campaign_dir, args.candidate_id, args.output_dir, workers=args.workers))


if __name__ == "__main__":
    main()
