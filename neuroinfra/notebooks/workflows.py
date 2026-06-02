"""Reusable notebook workflow helpers built on run and result primitives."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class RunAndLoadHooks:
    """Hook bundle for one generic run-and-load workflow."""

    run_simulation_fn: Callable[..., Any]
    load_result_fn: Callable[[Any], Any]
    merge_run_info_payload_fn: Callable[[str | Path, dict[str, Any]], Any] | None = None
    build_merge_payload_fn: Callable[[Any], dict[str, Any]] | None = None


@dataclass(frozen=True)
class LoadRunPairHooks:
    """Hook bundle for resolving one saved run and loading its result."""

    load_run_record_fn: Callable[..., Any]
    load_result_fn: Callable[[Any], Any]


@dataclass(frozen=True)
class LocalSweepHooks:
    """Hook bundle for one local sweep execution workflow."""

    run_and_load_fn: Callable[[dict[str, Any], str | None], tuple[Any, Any]]
    save_sweep_fn: Callable[..., Path]
    item_runs_dir_fn: Callable[[dict[str, Any]], str | Path]
    sweep_base_dir_fn: Callable[[dict[str, Any]], str | Path]


def load_run_pair(
    hooks: LoadRunPairHooks,
    *,
    run_or_dir: Any = None,
    prefix: str | None = None,
    index: int = -1,
    results_base: str | Path,
) -> tuple[Any, Any]:
    """Resolve one saved run and load its result payload."""
    run = hooks.load_run_record_fn(
        run_or_dir=run_or_dir,
        prefix=prefix,
        index=index,
        results_base=results_base,
    )
    return run, hooks.load_result_fn(run)


def run_and_load(
    hooks: RunAndLoadHooks,
    config: dict[str, Any] | None = None,
    *,
    label: str | None = None,
) -> tuple[Any, Any]:
    """Run one simulation and immediately load its standard outputs."""
    run = hooks.run_simulation_fn(config, label=label)
    result = hooks.load_result_fn(run)
    if hooks.merge_run_info_payload_fn is not None and hooks.build_merge_payload_fn is not None:
        hooks.merge_run_info_payload_fn(run.result_dir, hooks.build_merge_payload_fn(result))
    return run, result


def run_local_sweep_plan(
    hooks: LocalSweepHooks,
    sweep_plan: dict[str, Any],
) -> dict[str, Any]:
    """Execute one prepared sweep plan locally and persist the saved sweep."""
    local_item_runs_dir = Path(hooks.item_runs_dir_fn(sweep_plan))
    local_item_runs_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for item in sweep_plan["items"]:
        sweep_config = deepcopy(item["config"])
        sweep_config["results_base"] = str(local_item_runs_dir)
        run, result = hooks.run_and_load_fn(sweep_config, str(item["label"]))
        items.append({"value": item["value"], "config": sweep_config, "run": run, "result": result})

    sweep = {
        "path": sweep_plan["path"],
        "values": list(sweep_plan["values"]),
        "items": items,
        "paramset": sweep_plan["paramset"],
    }
    if sweep_plan.get("grid") is not None:
        sweep["grid"] = sweep_plan["grid"]

    hooks.save_sweep_fn(
        sweep,
        name=str(sweep_plan["sweep_label"]),
        base_dir=hooks.sweep_base_dir_fn(sweep_plan),
    )
    return sweep
