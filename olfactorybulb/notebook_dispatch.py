"""Concrete olfactory-bulb notebook entrypoint adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from neuroinfra.notebooks.dispatch import (
    NotebookRunDispatchHooks,
    NotebookSweepDispatchHooks,
    dispatch_grid_sweep,
    dispatch_parameter_sweep,
    dispatch_run,
)


@dataclass(frozen=True)
class NotebookRunDispatchAdapterHooks:
    """Hooks for constructing the concrete olfactory-bulb run entrypoint."""

    normalize_config_fn: Callable[[dict[str, Any] | None], dict[str, Any]]
    make_timestamp_fn: Callable[[], str]
    make_label_fn: Callable[[dict[str, Any], str], str]
    build_local_run_payload_fn: Callable[..., Any]
    local_run_payload_hooks_fn: Callable[[], Any]
    build_local_run_hooks_fn: Callable[[Any], Any]
    local_run_hook_builder_hooks_fn: Callable[[], Any]
    execute_local_run_fn: Callable[..., Any]
    execute_remote_run_fn: Callable[..., Any]
    default_results_base: str | Path
    runner_name: str = "obgpu_experiment_helpers.run_simulation"


@dataclass(frozen=True)
class NotebookSweepDispatchAdapterHooks:
    """Hooks for constructing the concrete olfactory-bulb sweep entrypoints."""

    prepare_sweep_plan_fn: Callable[..., dict[str, Any]]
    uses_remote_batch_engine_fn: Callable[[dict[str, Any]], bool]
    build_local_sweep_hooks_fn: Callable[[Any], Any]
    notebook_workflow_adapter_hooks_fn: Callable[[], Any]
    execute_local_sweep_plan_fn: Callable[[Any, dict[str, Any]], dict[str, Any]]
    execute_remote_sweep_fn: Callable[[dict[str, Any]], dict[str, Any]]


def run_notebook_simulation(
    hooks: NotebookRunDispatchAdapterHooks,
    config: dict[str, Any] | None = None,
    *,
    label: str | None = None,
) -> Any:
    """Run one notebook simulation through the generic local/remote dispatch layer."""

    def _execute_local_run(
        effective_config: dict[str, Any],
        resolved_label: str,
        timestamp: str,
        local_result_dir: Path,
    ) -> Any:
        local_payload = hooks.build_local_run_payload_fn(
            hooks.local_run_payload_hooks_fn(),
            effective_config,
            label=resolved_label,
            timestamp=timestamp,
            repo_root=Path(__file__).resolve().parent.parent,
            default_results_base=hooks.default_results_base,
        )
        return hooks.execute_local_run_fn(
            config=effective_config,
            label=resolved_label,
            timestamp=timestamp,
            result_dir=local_payload.result_dir,
            env=local_payload.env,
            command=local_payload.command,
            runner_name=hooks.runner_name,
            hooks=hooks.build_local_run_hooks_fn(hooks.local_run_hook_builder_hooks_fn()),
            success_extra_payload={"remote": None},
        )

    def _execute_remote_run(
        effective_config: dict[str, Any],
        resolved_label: str,
        timestamp: str,
        local_result_dir: Path,
    ) -> Any:
        return hooks.execute_remote_run_fn(
            effective_config,
            label=resolved_label,
            timestamp=timestamp,
            local_result_dir=local_result_dir,
        )

    return dispatch_run(
        NotebookRunDispatchHooks(
            normalize_config_fn=hooks.normalize_config_fn,
            make_timestamp_fn=hooks.make_timestamp_fn,
            make_label_fn=hooks.make_label_fn,
            execute_local_run_fn=_execute_local_run,
            execute_remote_run_fn=_execute_remote_run,
            default_results_base=hooks.default_results_base,
        ),
        config,
        label=label,
    )


def run_notebook_parameter_sweep(
    hooks: NotebookSweepDispatchAdapterHooks,
    base_config: dict[str, Any],
    sweep_path: str | list[str] | dict[str, list[Any]],
    values: list[Any] | tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    """Run one notebook parameter sweep through the generic dispatch layer."""
    return dispatch_parameter_sweep(
        NotebookSweepDispatchHooks(
            prepare_sweep_plan_fn=hooks.prepare_sweep_plan_fn,
            uses_remote_batch_engine_fn=hooks.uses_remote_batch_engine_fn,
            execute_local_sweep_fn=lambda sweep_plan: hooks.execute_local_sweep_plan_fn(
                hooks.build_local_sweep_hooks_fn(hooks.notebook_workflow_adapter_hooks_fn()),
                sweep_plan,
            ),
            execute_remote_sweep_fn=hooks.execute_remote_sweep_fn,
        ),
        base_config,
        sweep_path,
        values,
    )


def run_notebook_grid_sweep(
    hooks: NotebookSweepDispatchAdapterHooks,
    base_config: dict[str, Any],
    param_grid: dict[str, list[Any]],
) -> dict[str, Any]:
    """Run one notebook grid sweep through the generic dispatch layer."""
    return dispatch_grid_sweep(
        NotebookSweepDispatchHooks(
            prepare_sweep_plan_fn=hooks.prepare_sweep_plan_fn,
            uses_remote_batch_engine_fn=hooks.uses_remote_batch_engine_fn,
            execute_local_sweep_fn=lambda sweep_plan: hooks.execute_local_sweep_plan_fn(
                hooks.build_local_sweep_hooks_fn(hooks.notebook_workflow_adapter_hooks_fn()),
                sweep_plan,
            ),
            execute_remote_sweep_fn=hooks.execute_remote_sweep_fn,
        ),
        base_config,
        param_grid,
    )
