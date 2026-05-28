#!/usr/bin/env python3
"""Export HFO optimizer campaign metrics to TensorBoard.

The optimizer itself should stay dependency-light because it runs inside the
Phoenix/NEURON workflow.  This tool is intentionally a sidecar: it reads the
campaign JSONL archives, writes TensorBoard event files, and can optionally
watch the archive while a campaign is still running.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import olfactorybulb.hfo_optimizer as hfo


DEFAULT_LOG_SUBDIR = "tensorboard/events"
DEFAULT_TOP_N = 50
DEFAULT_WATCH_INTERVAL_S = 60.0
PAIR_METRIC_KEYS = (
    "pair_score",
    "unpenalized_pair_score",
    "target_delta",
    "target_clean_delta",
    "target_contrast_log10",
    "density_contrast_log10",
    "compound_contrast_log10",
    "peak_contrast_log10",
    "psd_template_score",
    "psd_template_loss",
    "psd_contrast_template_loss",
    "control_hfo_template_similarity",
    "ketamine_hfo_template_similarity",
    "control_leak_penalty",
    "control_target_excess_penalty",
    "same_peak_penalty",
    "negative_delta_penalty",
    "ketamine_center_penalty",
    "control_center_advantage_penalty",
    "ketamine_peak_contrast_penalty",
    "control_peak_contrast_penalty",
    "ketamine_epli_silence_penalty",
    "ketamine_epli_low_support_penalty",
    "epli_dropout_penalty",
    "ketamine_wrong_band_penalty",
    "control_wrong_band_penalty",
    "parameter_plausibility_penalty",
)
CONDITION_METRIC_KEYS = (
    "condition_score",
    "peak_hz",
    "target_centroid_hz",
    "target_centroid_match",
    "target_peak_contrast",
    "target_density_ratio",
    "dominance",
    "target_clean_fraction",
    "supra_hfo_relative",
    "beta_gamma_support",
    "phase_lock",
    "input_coverage_fraction",
    "spike_support_rate_hz",
    "epli_rate_hz",
    "rate_penalty",
    "spike_support_penalty",
    "input_dropout_penalty",
)
RATE_CELL_TYPES = ("MC", "TC", "GC", "EPLI", "PVCRH")


@dataclass(frozen=True)
class ScalarRecord:
    step: int
    tag: str
    value: float


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _safe_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_./-]+", "_", str(value)).strip("_")


def _candidate_step(row: dict[str, Any], fallback_index: int) -> int:
    candidate_id = str(row.get("candidate_id") or "")
    match = re.search(r"(\d+)$", candidate_id)
    if match:
        return int(match.group(1))
    return int(fallback_index)


def _condition_metrics(row: dict[str, Any], condition: str) -> dict[str, Any]:
    payload = row.get(f"{condition}_metrics") or {}
    return payload if isinstance(payload, dict) else {}


def _relative_band(metrics: dict[str, Any], band_name: str) -> float | None:
    relative = metrics.get("relative_band_power") or {}
    if not isinstance(relative, dict):
        return None
    return _safe_float(relative.get(band_name))


def _band_power(metrics: dict[str, Any], band_name: str) -> float | None:
    band_power = metrics.get("band_power") or {}
    if not isinstance(band_power, dict):
        return None
    return _safe_float(band_power.get(band_name))


def _rate(metrics: dict[str, Any], cell_type: str) -> float | None:
    rates = metrics.get("mean_firing_rate_by_type") or {}
    if not isinstance(rates, dict):
        return None
    return _safe_float(rates.get(cell_type))


def _iter_candidate_scalars(row: dict[str, Any], *, step: int) -> Iterable[ScalarRecord]:
    for key in PAIR_METRIC_KEYS:
        value = _safe_float(row.get(key))
        if value is not None:
            yield ScalarRecord(step=step, tag=f"score/{_safe_tag(key)}", value=value)

    for condition in ("control", "ketamine"):
        metrics = _condition_metrics(row, condition)
        for key in CONDITION_METRIC_KEYS:
            value = _safe_float(metrics.get(key))
            if value is not None:
                yield ScalarRecord(step=step, tag=f"{condition}/{_safe_tag(key)}", value=value)
        for band_name in hfo.DEFAULT_SCORE_BANDS:
            relative = _relative_band(metrics, band_name)
            if relative is not None:
                yield ScalarRecord(
                    step=step,
                    tag=f"band_relative/{condition}/{_safe_tag(band_name)}",
                    value=relative,
                )
            absolute = _band_power(metrics, band_name)
            if absolute is not None:
                yield ScalarRecord(
                    step=step,
                    tag=f"band_power/{condition}/{_safe_tag(band_name)}",
                    value=absolute,
                )
        for cell_type in RATE_CELL_TYPES:
            value = _rate(metrics, cell_type)
            if value is not None:
                yield ScalarRecord(
                    step=step,
                    tag=f"rate_hz/{condition}/{cell_type}",
                    value=value,
                )

    control = _condition_metrics(row, "control")
    ketamine = _condition_metrics(row, "ketamine")
    for band_name in hfo.DEFAULT_SCORE_BANDS:
        control_relative = _relative_band(control, band_name)
        ketamine_relative = _relative_band(ketamine, band_name)
        if control_relative is not None and ketamine_relative is not None:
            yield ScalarRecord(
                step=step,
                tag=f"band_relative_delta/{_safe_tag(band_name)}",
                value=ketamine_relative - control_relative,
            )

    parameters = row.get("parameters") or {}
    if isinstance(parameters, dict):
        for key, raw_value in parameters.items():
            value = _safe_float(raw_value)
            if value is not None:
                yield ScalarRecord(step=step, tag=f"param/{_safe_tag(key)}", value=value)


def collect_scalar_records(rows: list[dict[str, Any]], *, start_index: int = 0) -> list[ScalarRecord]:
    """Return all scalar records that would be written for archive rows."""
    records: list[ScalarRecord] = []
    best_score = float("-inf")
    best_target_delta = float("-inf")
    for index, row in enumerate(rows):
        step = _candidate_step(row, index)
        score = _safe_float(row.get("pair_score"))
        if score is not None:
            best_score = max(best_score, score)
        target_delta = _safe_float(row.get("target_delta"))
        if target_delta is not None:
            best_target_delta = max(best_target_delta, target_delta)
        if index < int(start_index):
            continue
        records.extend(_iter_candidate_scalars(row, step=step))
        if math.isfinite(best_score):
            records.append(ScalarRecord(step=step, tag="best_so_far/pair_score", value=best_score))
        if math.isfinite(best_target_delta):
            records.append(ScalarRecord(step=step, tag="best_so_far/target_delta", value=best_target_delta))
    return records


def _import_tensorboard_event_tools():
    try:
        from tensorboard.compat.proto import event_pb2, summary_pb2
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorBoard is not installed in this Python environment. Install it with:\n"
            "  python -m pip install tensorboard\n"
            "The exporter only needs the standalone tensorboard package, not TensorFlow."
        ) from exc
    return EventFileWriter, event_pb2, summary_pb2


class TensorBoardEventWriter:
    def __init__(self, log_dir: Path):
        EventFileWriter, event_pb2, summary_pb2 = _import_tensorboard_event_tools()
        self._event_pb2 = event_pb2
        self._summary_pb2 = summary_pb2
        self._writer = EventFileWriter(str(log_dir))

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        event = self._event_pb2.Event(
            wall_time=time.time(),
            step=int(step),
            summary=self._summary_pb2.Summary(
                value=[
                    self._summary_pb2.Summary.Value(
                        tag=str(tag),
                        simple_value=float(value),
                    )
                ]
            ),
        )
        self._writer.add_event(event)

    def add_png(self, tag: str, image_path: Path, step: int) -> None:
        try:
            from PIL import Image
        except ModuleNotFoundError:
            return
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
        event = self._event_pb2.Event(
            wall_time=time.time(),
            step=int(step),
            summary=self._summary_pb2.Summary(
                value=[
                    self._summary_pb2.Summary.Value(
                        tag=str(tag),
                        image=self._summary_pb2.Summary.Image(
                            height=int(height),
                            width=int(width),
                            colorspace=3,
                            encoded_image_string=image_path.read_bytes(),
                        ),
                    )
                ]
            ),
        )
        self._writer.add_event(event)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


def _top_candidate_table_rows(rows: list[dict[str, Any]], *, top_n: int) -> list[dict[str, Any]]:
    ranked = [
        row
        for row in rows
        if (score := _safe_float(row.get("pair_score"))) is not None and math.isfinite(score)
    ]
    ranked.sort(key=lambda row: float(row.get("pair_score", float("-inf"))), reverse=True)
    table_rows = []
    for row in ranked[: int(top_n)]:
        control = _condition_metrics(row, "control")
        ketamine = _condition_metrics(row, "ketamine")
        params = row.get("parameters") or {}
        table_row = {
            "candidate_id": row.get("candidate_id"),
            "batch_name": row.get("batch_name"),
            "pair_score": row.get("pair_score"),
            "target_delta": row.get("target_delta"),
            "ketamine_peak_hz": ketamine.get("peak_hz"),
            "control_peak_hz": control.get("peak_hz"),
            "ketamine_target_rel": _relative_band(ketamine, "target_hfo"),
            "control_target_rel": _relative_band(control, "target_hfo"),
            "ketamine_high_gamma_rel": _relative_band(ketamine, "high_gamma"),
            "control_high_gamma_rel": _relative_band(control, "high_gamma"),
            "ketamine_epli_rate_hz": row.get("ketamine_epli_rate_hz", ketamine.get("epli_rate_hz")),
            "control_epli_rate_hz": row.get("control_epli_rate_hz", control.get("epli_rate_hz")),
        }
        if isinstance(params, dict):
            for key, value in params.items():
                if _safe_float(value) is not None:
                    table_row[f"param_{key}"] = value
        table_rows.append(table_row)
    return table_rows


def _write_top_candidate_csv(log_dir: Path, rows: list[dict[str, Any]], *, top_n: int) -> Path:
    table_rows = _top_candidate_table_rows(rows, top_n=top_n)
    path = log_dir / "top_candidates.csv"
    if not table_rows:
        path.write_text("")
        return path
    fieldnames = list(dict.fromkeys(key for row in table_rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)
    return path


def _packet_contact_sheets(campaign_dir: Path, rows: list[dict[str, Any]]) -> list[tuple[int, str, Path]]:
    candidate_steps = {
        str(row.get("candidate_id")): _candidate_step(row, index)
        for index, row in enumerate(rows)
    }
    packets = []
    figures_dir = campaign_dir / "figures"
    if not figures_dir.exists():
        return packets
    for image_path in figures_dir.glob("*/contact_sheet.png"):
        match = re.search(r"(C\d+)", image_path.parent.name)
        if not match:
            continue
        candidate_id = match.group(1)
        step = candidate_steps.get(candidate_id)
        if step is None:
            continue
        packets.append((step, candidate_id, image_path))
    return sorted(packets)


def export_campaign_to_tensorboard(
    campaign_dir: str | Path,
    *,
    log_dir: str | Path | None = None,
    reset: bool = False,
    start_index: int = 0,
    include_existing_images: bool = True,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, Any]:
    """Write TensorBoard events and summary sidecar files for one campaign."""
    campaign_path = Path(campaign_dir).expanduser().resolve()
    log_path = Path(log_dir).expanduser().resolve() if log_dir else campaign_path / DEFAULT_LOG_SUBDIR
    if reset and log_path.exists():
        shutil.rmtree(log_path)
    log_path.mkdir(parents=True, exist_ok=True)

    rows = hfo.load_candidate_archive_rows(campaign_path)
    if reset:
        start_index = 0
    start_index = min(max(int(start_index), 0), len(rows))
    records = collect_scalar_records(rows, start_index=start_index)
    writer = TensorBoardEventWriter(log_path)
    try:
        for record in records:
            writer.add_scalar(record.tag, record.value, record.step)
        image_count = 0
        if include_existing_images:
            for step, candidate_id, image_path in _packet_contact_sheets(campaign_path, rows):
                writer.add_png(f"candidate_packet/{candidate_id}", image_path, step)
                image_count += 1
        writer.flush()
    finally:
        writer.close()

    top_csv = _write_top_candidate_csv(log_path, rows, top_n=top_n)
    manifest = {
        "campaign_dir": str(campaign_path),
        "log_dir": str(log_path),
        "generated_at_unix": time.time(),
        "candidate_rows": len(rows),
        "start_index": int(start_index),
        "scalar_records": len(records),
        "packet_images": image_count if include_existing_images else 0,
        "top_candidates_csv": str(top_csv),
        "objective_filter": hfo.load_objective_filter(campaign_path),
    }
    (log_path / "export_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def watch_campaign(
    campaign_dir: str | Path,
    *,
    log_dir: str | Path | None = None,
    interval_s: float = DEFAULT_WATCH_INTERVAL_S,
    include_existing_images: bool = True,
    top_n: int = DEFAULT_TOP_N,
) -> None:
    first = True
    last_signature: tuple[int, int, int] | None = None
    last_row_count = 0
    last_filter_mtime_ns = 0
    campaign_path = Path(campaign_dir).expanduser().resolve()
    archive = campaign_path / "candidate_archive.jsonl"
    objective_filter = campaign_path / hfo.ARCHIVE_FILTER_FILENAME
    while True:
        signature = (
            int(archive.stat().st_mtime_ns if archive.exists() else 0),
            int(archive.stat().st_size if archive.exists() else 0),
            int(objective_filter.stat().st_mtime_ns if objective_filter.exists() else 0),
        )
        if first or signature != last_signature:
            rows = hfo.load_candidate_archive_rows(campaign_path)
            filter_mtime_ns = signature[2]
            reset = first or filter_mtime_ns != last_filter_mtime_ns or len(rows) < last_row_count
            start_index = 0 if reset else last_row_count
            manifest = export_campaign_to_tensorboard(
                campaign_path,
                log_dir=log_dir,
                reset=reset,
                start_index=start_index,
                include_existing_images=include_existing_images and reset,
                top_n=top_n,
            )
            print(
                "Exported {candidate_rows} candidates from row {start_index} "
                "and wrote {scalar_records} scalars to {log_dir}".format(
                    **manifest
                ),
                flush=True,
            )
            first = False
            last_signature = signature
            last_row_count = int(manifest["candidate_rows"])
            last_filter_mtime_ns = filter_mtime_ns
        time.sleep(max(float(interval_s), 1.0))


def serve_tensorboard(
    campaign_dir: str | Path,
    *,
    log_dir: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 6006,
    skip_export: bool = False,
) -> None:
    campaign_path = Path(campaign_dir).expanduser().resolve()
    log_path = Path(log_dir).expanduser().resolve() if log_dir else campaign_path / DEFAULT_LOG_SUBDIR
    if not skip_export:
        export_campaign_to_tensorboard(campaign_path, log_dir=log_path, reset=True)
    command = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(log_path),
        "--host",
        str(host),
        "--port",
        str(int(port)),
    ]
    print("Running: " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("campaign_dir", type=Path)
        subparser.add_argument("--log-dir", type=Path, default=None)
        subparser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
        subparser.add_argument(
            "--no-images",
            action="store_true",
            help="Do not add existing candidate packet contact sheets to TensorBoard.",
        )

    export_parser = subparsers.add_parser("export", help="Export the current archive once.")
    add_common(export_parser)
    export_parser.add_argument("--reset", action="store_true", help="Clear the TensorBoard log directory first.")

    watch_parser = subparsers.add_parser("watch", help="Refresh TensorBoard logs as the campaign archive grows.")
    add_common(watch_parser)
    watch_parser.add_argument("--interval-s", type=float, default=DEFAULT_WATCH_INTERVAL_S)

    serve_parser = subparsers.add_parser("serve", help="Export once, then run a TensorBoard server.")
    serve_parser.add_argument("campaign_dir", type=Path)
    serve_parser.add_argument("--log-dir", type=Path, default=None)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=6006)
    serve_parser.add_argument("--skip-export", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            manifest = export_campaign_to_tensorboard(
                args.campaign_dir,
                log_dir=args.log_dir,
                reset=args.reset,
                include_existing_images=not args.no_images,
                top_n=args.top_n,
            )
            print(json.dumps(manifest, indent=2, sort_keys=True))
        elif args.command == "watch":
            watch_campaign(
                args.campaign_dir,
                log_dir=args.log_dir,
                interval_s=args.interval_s,
                include_existing_images=not args.no_images,
                top_n=args.top_n,
            )
        elif args.command == "serve":
            serve_tensorboard(
                args.campaign_dir,
                log_dir=args.log_dir,
                host=args.host,
                port=args.port,
                skip_export=args.skip_export,
            )
        else:
            parser.error(f"Unsupported command {args.command!r}")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
