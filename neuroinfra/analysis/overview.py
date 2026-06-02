"""Reusable result-overview helpers for loaded analysis results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ResultOverviewContext:
    """Normalized summary/file metadata for one loaded result mapping."""

    result: dict[str, Any]
    summary: dict[str, Any]
    params: dict[str, Any]
    timings: dict[str, Any]
    files: dict[str, Any]


def build_result_overview_context(result: dict[str, Any]) -> ResultOverviewContext:
    """Extract summary, params, timing, and file metadata views from one result."""
    summary = result.get("summary") or {}
    params = summary.get("params", {})
    timings = summary.get("timing_seconds", {})
    files = summary.get("files") or {}
    return ResultOverviewContext(
        result=result,
        summary=summary if isinstance(summary, dict) else {},
        params=params if isinstance(params, dict) else {},
        timings=timings if isinstance(timings, dict) else {},
        files=files if isinstance(files, dict) else {},
    )


def result_file_metadata(
    context: ResultOverviewContext,
    filename: str,
) -> dict[str, Any]:
    """Return one file-metadata payload from summary files, or an empty dict."""
    payload = context.files.get(str(filename))
    return payload if isinstance(payload, dict) else {}


def first_result_file_metadata(
    context: ResultOverviewContext,
    filenames: Iterable[str],
) -> dict[str, Any]:
    """Return the first available file-metadata payload across candidate names."""
    for filename in filenames:
        payload = result_file_metadata(context, str(filename))
        if payload:
            return payload
    return {}


def result_value_length(
    result: dict[str, Any],
    key: str,
) -> int:
    """Return a lazy-safe length for one result field, defaulting to zero."""
    try:
        return int(len(dict.get(result, key, [])))
    except TypeError:
        return 0


def metadata_value_or_result_length(
    context: ResultOverviewContext,
    *,
    metadata: dict[str, Any],
    metadata_key: str,
    result_key: str,
) -> int:
    """Use summary file metadata when present, otherwise fall back to local result length."""
    value = metadata.get(str(metadata_key))
    if isinstance(value, int):
        return value
    return result_value_length(context.result, result_key)


def build_result_overview(
    context: ResultOverviewContext,
    *,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standard result-overview payload plus caller-supplied fields."""
    payload = {
        "result_dir": str(context.result["result_dir"]),
        "label": context.summary.get("label"),
        "paramset": context.summary.get("paramset"),
        "nranks": context.summary.get("nranks"),
        "tstop_ms": context.params.get("tstop"),
        "sim_dt_ms": context.params.get("sim_dt"),
        "actual_dt_ms": context.params.get("actual_dt"),
        "recording_period_ms": context.params.get("recording_period"),
        "run_seconds": context.timings.get("run_max_rank"),
        "total_seconds": context.timings.get("total_max_rank"),
    }
    if extra_fields:
        payload.update(extra_fields)
    return payload
