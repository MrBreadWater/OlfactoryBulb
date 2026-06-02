"""Reusable notebook entrypoint dispatch helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class NotebookRunDispatchHooks:
    """Hook bundle for generic local-vs-remote notebook run dispatch."""

    normalize_config_fn: Callable[[dict[str, Any] | None], dict[str, Any]]
    make_timestamp_fn: Callable[[], str]
    make_label_fn: Callable[[dict[str, Any], str], str]
    execute_local_run_fn: Callable[[dict[str, Any], str, str, Path], Any]
    execute_remote_run_fn: Callable[[dict[str, Any], str, str, Path], Any]
    default_results_base: str | Path
    local_backend_name: str = "local"
    remote_backend_names: tuple[str, ...] = ("sol_slurm", "slurm_remote")


@dataclass(frozen=True)
class NotebookSweepDispatchHooks:
    """Hook bundle for generic local-vs-remote notebook sweep dispatch."""

    prepare_sweep_plan_fn: Callable[..., dict[str, Any]]
    uses_remote_batch_engine_fn: Callable[[dict[str, Any]], bool]
    execute_local_sweep_fn: Callable[[dict[str, Any]], dict[str, Any]]
    execute_remote_sweep_fn: Callable[[dict[str, Any]], dict[str, Any]]


def dispatch_run(
    hooks: NotebookRunDispatchHooks,
    config: dict[str, Any] | None = None,
    *,
    label: str | None = None,
) -> Any:
    """Dispatch one notebook run to the configured local or remote backend."""
    normalized_config = hooks.normalize_config_fn(config)
    timestamp = hooks.make_timestamp_fn()
    resolved_label = str(label or hooks.make_label_fn(normalized_config, timestamp))
    local_result_dir = Path(normalized_config.get("results_base", hooks.default_results_base)) / resolved_label
    runner_backend = str(normalized_config.get("runner_backend", hooks.local_backend_name))

    if runner_backend in hooks.remote_backend_names:
        return hooks.execute_remote_run_fn(
            normalized_config,
            resolved_label,
            timestamp,
            local_result_dir,
        )

    if runner_backend == hooks.local_backend_name:
        return hooks.execute_local_run_fn(
            normalized_config,
            resolved_label,
            timestamp,
            local_result_dir,
        )

    raise ValueError(f"Unsupported runner_backend={runner_backend!r}")


def dispatch_parameter_sweep(
    hooks: NotebookSweepDispatchHooks,
    base_config: dict[str, Any],
    sweep_path: str | list[str] | dict[str, list[Any]],
    values: list[Any] | tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    """Prepare and dispatch one single-axis or joint notebook sweep."""
    sweep_plan = hooks.prepare_sweep_plan_fn(base_config, sweep_path, values, grid=False)
    if hooks.uses_remote_batch_engine_fn(sweep_plan["base_config"]):
        return hooks.execute_remote_sweep_fn(sweep_plan)
    return hooks.execute_local_sweep_fn(sweep_plan)


def dispatch_grid_sweep(
    hooks: NotebookSweepDispatchHooks,
    base_config: dict[str, Any],
    param_grid: dict[str, list[Any]],
) -> dict[str, Any]:
    """Prepare and dispatch one grid notebook sweep."""
    sweep_plan = hooks.prepare_sweep_plan_fn(base_config, param_grid, grid=True)
    if hooks.uses_remote_batch_engine_fn(sweep_plan["base_config"]):
        return hooks.execute_remote_sweep_fn(sweep_plan)
    return hooks.execute_local_sweep_fn(sweep_plan)
