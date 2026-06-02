"""Generic run-directory metadata helpers for notebook workflows."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_FILENAME = "summary.json"
DEFAULT_RUN_INFO_FILENAME = "run_info.json"
DEFAULT_STDOUT_FILENAME = "stdout.txt"
DEFAULT_STDERR_FILENAME = "stderr.txt"


@dataclass(frozen=True)
class RunRecord:
    """Metadata and captured stdout/stderr for one saved run directory."""

    label: str
    timestamp: str
    result_dir: Path
    summary: dict[str, Any]
    config: dict[str, Any]
    overrides: dict[str, Any]
    command: list[str]
    stdout: str
    stderr: str


def read_json_if_present(path: str | Path) -> Any | None:
    """Return parsed JSON when a file exists and is non-empty."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    with open(path) as handle:
        return json.load(handle)


def list_run_dirs(
    prefix: str | None = None,
    *,
    results_base: str | Path,
) -> list[Path]:
    """List saved run directories, optionally filtered by name prefix."""
    results_base = Path(results_base)
    if not results_base.exists():
        return []
    runs = [path for path in results_base.iterdir() if path.is_dir()]
    if prefix:
        runs = [path for path in runs if path.name.startswith(prefix)]
    return sorted(runs)


def resolve_run_dir(
    run_or_dir: str | os.PathLike[str] | RunRecord | None = None,
    *,
    prefix: str | None = None,
    index: int = -1,
    results_base: str | Path,
) -> Path:
    """Resolve a run identifier, path, or prefix/index pair into a run directory."""
    if run_or_dir is not None:
        return Path(run_or_dir.result_dir if isinstance(run_or_dir, RunRecord) else run_or_dir)

    runs = list_run_dirs(prefix=prefix, results_base=results_base)
    if not runs:
        raise FileNotFoundError(f"No run directories found in {results_base} with prefix={prefix!r}")
    return runs[index]


def _read_text_if_present(path: Path) -> str:
    """Return file text when present, else an empty string."""
    return path.read_text() if path.exists() else ""


def _command_list(value: Any) -> list[str]:
    """Normalize one stored command payload to a string argv list."""
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _first_present(values: tuple[Any, ...]) -> Any:
    """Return the first non-empty value from one ordered value sequence."""
    for value in values:
        if value not in (None, ""):
            return value
    return None


def load_run_record(
    run_or_dir: str | os.PathLike[str] | RunRecord | None = None,
    *,
    prefix: str | None = None,
    index: int = -1,
    results_base: str | Path,
    summary_filename: str = DEFAULT_SUMMARY_FILENAME,
    run_info_filename: str = DEFAULT_RUN_INFO_FILENAME,
    stdout_filename: str = DEFAULT_STDOUT_FILENAME,
    stderr_filename: str = DEFAULT_STDERR_FILENAME,
) -> RunRecord:
    """Load one saved run record from a results directory."""
    result_dir = resolve_run_dir(
        run_or_dir=run_or_dir,
        prefix=prefix,
        index=index,
        results_base=results_base,
    )
    summary = read_json_if_present(result_dir / summary_filename) or {}
    run_info = read_json_if_present(result_dir / run_info_filename) or {}
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(run_info, dict):
        run_info = {}

    label = _first_present(
        (
            run_info.get("label"),
            summary.get("label"),
            run_info.get("requested_label"),
            summary.get("requested_label"),
        )
    )
    timestamp = _first_present((run_info.get("timestamp"), summary.get("timestamp"))) or ""

    config = run_info.get("config")
    overrides = run_info.get("overrides")

    return RunRecord(
        label=str(label or result_dir.name),
        timestamp=str(timestamp),
        result_dir=result_dir,
        summary=summary,
        config=config if isinstance(config, dict) else {},
        overrides=overrides if isinstance(overrides, dict) else {},
        command=_command_list(run_info.get("command")),
        stdout=_read_text_if_present(result_dir / stdout_filename),
        stderr=_read_text_if_present(result_dir / stderr_filename),
    )


def load_run_config(
    run_or_dir: str | os.PathLike[str] | RunRecord | None = None,
    *,
    prefix: str | None = None,
    index: int = -1,
    results_base: str | Path,
    summary_filename: str = DEFAULT_SUMMARY_FILENAME,
    run_info_filename: str = DEFAULT_RUN_INFO_FILENAME,
    stdout_filename: str = DEFAULT_STDOUT_FILENAME,
    stderr_filename: str = DEFAULT_STDERR_FILENAME,
) -> dict[str, Any]:
    """Load a deep-copied config snapshot from one saved run directory."""
    record = load_run_record(
        run_or_dir=run_or_dir,
        prefix=prefix,
        index=index,
        results_base=results_base,
        summary_filename=summary_filename,
        run_info_filename=run_info_filename,
        stdout_filename=stdout_filename,
        stderr_filename=stderr_filename,
    )
    return deepcopy(record.config)
