"""Concrete olfactory-bulb local notebook-run adapters."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from neuroinfra.notebooks.local_runs import LocalRunHooks


@dataclass(frozen=True)
class LocalRunPayload:
    """Prepared local notebook-run payload for one simulation."""

    result_dir: Path
    env: dict[str, str]
    command: list[str]


@dataclass(frozen=True)
class LocalRunPayloadHooks:
    """Hooks for building one concrete olfactory-bulb local run payload."""

    benchmark_param_overrides_payload_fn: Callable[[dict[str, Any]], tuple[dict[str, Any], str | None]]
    write_benchmark_overrides_file_fn: Callable[[str | Path, dict[str, Any]], Any]
    build_run_command_fn: Callable[..., list[str]]


@dataclass(frozen=True)
class NotebookLocalRunHookBuilderHooks:
    """Hooks for constructing the concrete olfactory-bulb local run protocol."""

    read_summary_fn: Callable[[Path], dict[str, Any]]
    write_run_info_fn: Callable[..., Any]
    build_param_overrides_fn: Callable[[dict[str, Any]], dict[str, Any]]
    run_record_factory_fn: Callable[..., Any]


def build_local_run_payload(
    hooks: LocalRunPayloadHooks,
    config: dict[str, Any],
    *,
    label: str,
    timestamp: str,
    repo_root: str | Path,
    default_results_base: str | Path,
) -> LocalRunPayload:
    """Prepare the concrete local env, overrides file, and command for one run."""
    result_dir = Path(config.get("results_base", default_results_base)) / label
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["OB_RUN_TIMESTAMP"] = timestamp
    env["OB_RESULT_LABEL"] = label
    env["OB_RESULTS_BASE"] = str(config.get("results_base", default_results_base))
    env["OB_CORENRN_CELL_PERMUTE"] = str(int(config.get("cell_permute", 2)))

    param_overrides, input_spec_file = hooks.benchmark_param_overrides_payload_fn(config)
    overrides_path = result_dir.parent / ".obgpu-wrapper" / label / "overrides.json"
    hooks.write_benchmark_overrides_file_fn(overrides_path, param_overrides)
    command = hooks.build_run_command_fn(
        config,
        label,
        overrides_file=overrides_path,
        param_overrides=param_overrides,
        input_spec_file=input_spec_file,
    )
    return LocalRunPayload(
        result_dir=result_dir,
        env=env,
        command=command,
    )


def build_local_run_hooks(
    hooks: NotebookLocalRunHookBuilderHooks,
) -> LocalRunHooks:
    """Build the notebook-facing olfactory-bulb local run hooks."""
    return LocalRunHooks(
        read_summary_fn=hooks.read_summary_fn,
        write_run_info_fn=hooks.write_run_info_fn,
        build_return_value_fn=lambda *,
        label,
        timestamp,
        result_dir,
        summary,
        config,
        command,
        completed: hooks.run_record_factory_fn(
            label=label,
            timestamp=timestamp,
            result_dir=result_dir,
            summary=summary,
            config=config,
            overrides=hooks.build_param_overrides_fn(config),
            command=command,
            stdout=completed.stdout,
            stderr=completed.stderr,
        ),
    )
