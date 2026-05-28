#!/usr/bin/env python3
"""Generate diagnostic figures for one HFO optimizer candidate."""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
from regenerate_hfo_packet_psd import regenerate_packet_psd


VISUAL_STYLE_VERSION = hv.VISUAL_STYLE_VERSION
SPECTROGRAM_FILE_CONTROL = hv.SPECTROGRAM_FILE_BY_CONDITION["control"]
SPECTROGRAM_FILE_KETAMINE = hv.SPECTROGRAM_FILE_BY_CONDITION["ketamine"]
SPECTROGRAM_MOD200_FILE_CONTROL = hv.SPECTROGRAM_MOD200_FILE_BY_CONDITION["control"]
SPECTROGRAM_MOD200_FILE_KETAMINE = hv.SPECTROGRAM_MOD200_FILE_BY_CONDITION["ketamine"]
SPECTROGRAM_PIPELINE = hv.SPECTROGRAM_PIPELINE
_spectrogram_window_geometry = hv.spectrogram_window_geometry
_save_spectrogram = hv.save_spectrogram


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


def generate_packet(campaign_dir: Path, candidate_id: str, output_dir: Path | None = None) -> Path:
    row = _load_candidate(campaign_dir, candidate_id)
    result_dir = Path((row.get("ketamine_metrics") or {}).get("result_dir") or (row.get("control_metrics") or {})["result_dir"])
    result = hlp.load_result(result_dir, progress=False)
    windows = hv.condition_windows(row)
    spectrogram_windows = hv.spectrogram_windows(row, result)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    packet_dir = output_dir or campaign_dir / "figures" / f"short_expected_{candidate_id}_{timestamp}"
    packet_dir.mkdir(parents=True, exist_ok=True)

    windowed = {condition: hv.window_result(result, windows, condition) for condition in ("control", "ketamine")}
    spectrogram_windowed = {
        condition: hv.window_result(result, spectrogram_windows, condition)
        for condition in ("control", "ketamine")
    }
    spectrogram_switch_time = hv.spectrogram_switch_time(row, result)

    control_geom = hv.spectrogram_window_geometry(spectrogram_windowed["control"])
    ketamine_geom = hv.spectrogram_window_geometry(spectrogram_windowed["ketamine"])

    final_packet_dir = packet_dir
    tmp_packet_dir = final_packet_dir.parent / f".{final_packet_dir.name}.tmp-{os.getpid()}-{time.time_ns()}"
    tmp_packet_dir.mkdir(parents=True, exist_ok=False)
    try:
        hv.save_spectrogram(
            spectrogram_windowed["control"],
            condition="control",
            out=tmp_packet_dir / SPECTROGRAM_FILE_CONTROL,
            nperseg=control_geom[0],
            noverlap=control_geom[1],
        )
        hv.save_spectrogram(
            spectrogram_windowed["ketamine"],
            condition="ketamine",
            out=tmp_packet_dir / SPECTROGRAM_FILE_KETAMINE,
            nperseg=ketamine_geom[0],
            noverlap=ketamine_geom[1],
        )
        hv.save_spectrogram(
            spectrogram_windowed["control"],
            condition="control",
            out=tmp_packet_dir / SPECTROGRAM_MOD200_FILE_CONTROL,
            nperseg=control_geom[0],
            noverlap=control_geom[1],
            modulus_ms=hv.NOTEBOOK_PACKET_TIME_MODULUS_MS,
        )
        hv.save_spectrogram(
            spectrogram_windowed["ketamine"],
            condition="ketamine",
            out=tmp_packet_dir / SPECTROGRAM_MOD200_FILE_KETAMINE,
            nperseg=ketamine_geom[0],
            noverlap=ketamine_geom[1],
            modulus_ms=hv.NOTEBOOK_PACKET_TIME_MODULUS_MS,
        )
        hv.save_lfp_zoom(result, windows, tmp_packet_dir / "06_lfp_windows.png")
        hv.save_lfp_zoom(
            result,
            windows,
            tmp_packet_dir / "06_lfp_windows_mod200.png",
            modulus_ms=hv.NOTEBOOK_PACKET_TIME_MODULUS_MS,
        )
        hv.save_raster(windowed["control"], "control", tmp_packet_dir / "07_raster_control.png")
        hv.save_raster(windowed["ketamine"], "ketamine", tmp_packet_dir / "08_raster_ketamine.png")
        hv.save_raster(
            windowed["control"],
            "control",
            tmp_packet_dir / "07_raster_control_mod200.png",
            modulus_ms=hv.NOTEBOOK_PACKET_TIME_MODULUS_MS,
        )
        hv.save_raster(
            windowed["ketamine"],
            "ketamine",
            tmp_packet_dir / "08_raster_ketamine_mod200.png",
            modulus_ms=hv.NOTEBOOK_PACKET_TIME_MODULUS_MS,
        )
        hv.save_input_overview(result, windows, tmp_packet_dir / "10_inputs.png")
        hv.save_input_overview(
            result,
            windows,
            tmp_packet_dir / "10_inputs_mod200.png",
            modulus_ms=hv.NOTEBOOK_PACKET_TIME_MODULUS_MS,
        )
        hv.save_phase_hist(windowed["control"], "control", tmp_packet_dir / "11_phase_control.png")
        hv.save_phase_hist(windowed["ketamine"], "ketamine", tmp_packet_dir / "12_phase_ketamine.png")

        for condition in ("control", "ketamine"):
            for group in hv.frequency_group_specs():
                kde_1d = hv.kde_filename("1d", condition, group.label)
                hv.save_spike_frequency_kde_1d(
                    windowed[condition],
                    condition,
                    group.label,
                    group.cell_types,
                    tmp_packet_dir / kde_1d,
                )
                kde_2d = hv.kde_filename("2d", condition, group.label)
                hv.save_spike_frequency_kde_2d(
                    windowed[condition],
                    condition,
                    group.label,
                    group.cell_types,
                    tmp_packet_dir / kde_2d,
                )
                hv.save_spike_frequency_kde_2d(
                    windowed[condition],
                    condition,
                    group.label,
                    group.cell_types,
                    tmp_packet_dir / hv.kde_filename("2d", condition, group.label, suffix="mod200"),
                    modulus_ms=hv.NOTEBOOK_PACKET_TIME_MODULUS_MS,
                )

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
    args = parser.parse_args()
    print(generate_packet(args.campaign_dir, args.candidate_id, args.output_dir))


if __name__ == "__main__":
    main()
