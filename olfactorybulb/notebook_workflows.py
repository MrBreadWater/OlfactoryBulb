"""Concrete olfactory-bulb notebook workflow adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from neuroinfra.notebooks.workflows import (
    LoadRunPairHooks,
    LocalSweepHooks,
    RunAndLoadHooks,
)


@dataclass(frozen=True)
class NotebookWorkflowAdapterHooks:
    """Hooks for constructing concrete olfactory-bulb notebook workflows."""

    load_run_record_fn: Callable[..., Any]
    load_result_fn: Callable[[Any], Any]
    run_simulation_fn: Callable[..., Any]
    merge_run_info_payload_fn: Callable[[str | Path, dict[str, Any]], Any]
    save_sweep_fn: Callable[..., Path]
    sweep_item_runs_dir_fn: Callable[[dict[str, Any], str], str | Path]
    sweep_dir_fn: Callable[[dict[str, Any], str], Path]


def build_result_merge_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Return the standard olfactory-bulb result metadata merged into run_info."""
    return {
        "artifact_sizes": result.get("artifact_sizes", {}),
        "load_timing_seconds": result.get("load_timing_seconds", {}),
        "load_total_seconds": result.get("load_total_seconds"),
    }


def build_load_run_pair_hooks(
    hooks: NotebookWorkflowAdapterHooks,
) -> LoadRunPairHooks:
    """Build the notebook-facing olfactory-bulb saved-run loader hooks."""
    return LoadRunPairHooks(
        load_run_record_fn=hooks.load_run_record_fn,
        load_result_fn=hooks.load_result_fn,
    )


def build_run_and_load_hooks(
    hooks: NotebookWorkflowAdapterHooks,
) -> RunAndLoadHooks:
    """Build the notebook-facing olfactory-bulb run-and-load hooks."""
    return RunAndLoadHooks(
        run_simulation_fn=hooks.run_simulation_fn,
        load_result_fn=hooks.load_result_fn,
        merge_run_info_payload_fn=hooks.merge_run_info_payload_fn,
        build_merge_payload_fn=build_result_merge_payload,
    )


def build_local_sweep_hooks(
    hooks: NotebookWorkflowAdapterHooks,
) -> LocalSweepHooks:
    """Build the notebook-facing olfactory-bulb local sweep hooks."""
    def _run_and_load(config: dict[str, Any], label: str | None):
        run = hooks.run_simulation_fn(config, label=label)
        result = hooks.load_result_fn(run)
        hooks.merge_run_info_payload_fn(run.result_dir, build_result_merge_payload(result))
        return run, result

    return LocalSweepHooks(
        run_and_load_fn=_run_and_load,
        save_sweep_fn=hooks.save_sweep_fn,
        item_runs_dir_fn=lambda plan: hooks.sweep_item_runs_dir_fn(plan["base_config"], str(plan["sweep_label"])),
        sweep_base_dir_fn=lambda plan: hooks.sweep_dir_fn(plan["base_config"], str(plan["sweep_label"])).parent,
    )
