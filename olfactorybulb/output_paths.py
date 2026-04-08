"""Helpers for timestamped result labels and per-run metadata files."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any


TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
TIMESTAMP_SUFFIX_RE = re.compile(r"(^|_)\d{8}_\d{6}$")


def make_timestamp() -> str:
    """Return a wall-clock timestamp using the project-wide label format."""
    return datetime.now().strftime(TIMESTAMP_FORMAT)


def sync_timestamp(comm: Any | None = None) -> str:
    """Return a shared run timestamp, broadcasting it across MPI ranks when needed."""
    timestamp = os.environ.get("OB_RUN_TIMESTAMP")
    if comm is not None:
        if timestamp is None and comm.Get_rank() == 0:
            timestamp = make_timestamp()
        timestamp = comm.bcast(timestamp, root=0)
    elif timestamp is None:
        timestamp = make_timestamp()

    os.environ["OB_RUN_TIMESTAMP"] = timestamp
    return timestamp


def label_has_timestamp(label: str | None) -> bool:
    """Return True when a run label already ends with the standard timestamp suffix."""
    return bool(TIMESTAMP_SUFFIX_RE.search(str(label or "")))


def label_with_timestamp(label: str | None, timestamp: str | None = None) -> str:
    """Append a timestamp suffix to a label unless one is already present."""
    label = str(label or "run")
    if label_has_timestamp(label):
        return label
    timestamp = timestamp or sync_timestamp()
    return f"{label}_{timestamp}"


def configure_output_env(
    default_label: str,
    comm: Any | None = None,
    results_base: str | os.PathLike[str] | None = None,
) -> tuple[str, str]:
    """Populate output-related environment variables for the current run."""
    timestamp = sync_timestamp(comm=comm)
    requested_label = os.environ.get("OB_RESULT_LABEL", default_label)
    final_label = label_with_timestamp(requested_label, timestamp=timestamp)
    os.environ["OB_RESULT_LABEL"] = final_label
    if results_base is not None:
        os.environ.setdefault("OB_RESULTS_BASE", str(results_base))
    return final_label, timestamp


def get_results_dir(
    default_label: str,
    base_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the timestamped results directory for the current run label."""
    base_dir = Path(base_dir or os.environ.get("OB_RESULTS_BASE", "results"))
    label = os.environ.get("OB_RESULT_LABEL", default_label)
    return base_dir / label_with_timestamp(label)


def write_run_info(results_dir: str | os.PathLike[str], payload: dict[str, Any]) -> Path:
    """Write a ``run_info.json`` file into ``results_dir`` and return its path."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_info_path = results_dir / "run_info.json"
    run_info_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return run_info_path
