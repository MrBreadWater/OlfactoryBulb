"""Concrete olfactory-bulb notebook run-info helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from neuroinfra.notebooks.run_info import RunInfoHooks, merge_run_info_payload, persist_run_info


@dataclass(frozen=True)
class NotebookRunInfoHooks:
    """Domain hooks for olfactory-bulb notebook run-info payloads."""

    json_ready_fn: Callable[[Any], Any]
    build_param_overrides_fn: Callable[[dict[str, Any]], dict[str, Any]]
    resolve_execution_mode_fn: Callable[[dict[str, Any]], Any]
    resolve_effective_params_fn: Callable[[dict[str, Any]], Any]
    env_keys: tuple[str, ...] = (
        "OB_RUN_TIMESTAMP",
        "OB_RESULT_LABEL",
        "OB_CORENRN_CELL_PERMUTE",
        "OB_RESULTS_BASE",
    )


def write_run_info(
    hooks: NotebookRunInfoHooks,
    result_dir,
    *,
    config,
    label,
    timestamp,
    command,
    env,
    completed,
    runner: str,
    summary=None,
    extra_payload: dict[str, Any] | None = None,
):
    """Write one standard olfactory-bulb notebook run-info payload."""
    return persist_run_info(
        result_dir,
        RunInfoHooks(
            json_ready_fn=hooks.json_ready_fn,
            build_overrides_fn=hooks.build_param_overrides_fn,
            resolve_execution_mode_fn=hooks.resolve_execution_mode_fn,
            resolve_effective_params_fn=hooks.resolve_effective_params_fn,
            env_keys=hooks.env_keys,
        ),
        config=config,
        label=label,
        timestamp=timestamp,
        command=command,
        env=env,
        completed=completed,
        runner=runner,
        summary=summary,
        extra_payload=extra_payload,
    )


def merge_extra_run_info(
    hooks: NotebookRunInfoHooks,
    result_dir,
    *,
    extra_payload: dict[str, Any],
):
    """Merge one extra payload into an olfactory-bulb notebook run-info file."""
    return merge_run_info_payload(
        result_dir,
        extra_payload=extra_payload,
        json_ready_fn=hooks.json_ready_fn,
    )
