"""Concrete olfactory-bulb notebook reporting helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable


@dataclass(frozen=True)
class NotebookReportHooks:
    """Domain hooks for notebook run-summary presentation."""

    result_overview_fn: Callable[[dict[str, Any]], dict[str, Any]]
    build_run_config_fn: Callable[..., dict[str, Any]]
    resolve_effective_params_fn: Callable[[dict[str, Any]], dict[str, Any]]
    resolve_paramset_defaults_fn: Callable[[str], dict[str, Any]]
    diff_values_fn: Callable[[Any, Any], list[dict[str, Any]]]
    extract_runtime_control_snapshot_fn: Callable[[dict[str, Any]], dict[str, Any]]
    print_diff_section_fn: Callable[[str, list[dict[str, Any]], int | None], None]
    write_fn: Callable[[str], None] = print


def print_run_summary(
    hooks: NotebookReportHooks,
    run: Any,
    result: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> None:
    """Print the standard olfactory-bulb notebook run summary."""
    info = hooks.result_overview_fn(result)
    hooks.write_fn(json.dumps(info, indent=2, sort_keys=True))
    config = config or run.config or (result.get("run_info") or {}).get("config") or {}
    remote_info = (result.get("run_info") or {}).get("remote")
    if config:
        normalized_config = hooks.build_run_config_fn(**config)
        effective = (result.get("run_info") or {}).get("effective_params") or {}
        if "full_param_snapshot" not in effective:
            effective = hooks.resolve_effective_params_fn(normalized_config)
        hooks.write_fn("\nEffective inputs:")
        hooks.write_fn(
            json.dumps(
                {
                    "input_odors_source": effective["input_odors_source"],
                    "n_odor_presentations": effective["n_odor_presentations"],
                    "odor_names": effective["odor_names"],
                    "input_odors": effective["input_odors"],
                    "max_firing_rate_hz": effective["max_firing_rate_hz"],
                    "inhale_duration_ms": effective["inhale_duration_ms"],
                    "mc_input_weight": effective["mc_input_weight"],
                    "tc_input_weight": effective["tc_input_weight"],
                },
                indent=2,
                sort_keys=True,
            )
        )

        base_snapshot = hooks.resolve_paramset_defaults_fn(normalized_config["paramset"])
        full_snapshot = effective.get("full_param_snapshot", {})
        param_changes = hooks.diff_values_fn(base_snapshot, full_snapshot)
        hooks.print_diff_section_fn(
            "Requested/effective param changes vs clean paramset",
            param_changes,
            None,
        )

        hooks.write_fn("\nRuntime and analysis controls:")
        hooks.write_fn(
            json.dumps(
                hooks.extract_runtime_control_snapshot_fn(normalized_config),
                indent=2,
                sort_keys=True,
            )
        )
        if remote_info:
            hooks.write_fn("\nRemote execution metadata:")
            hooks.write_fn(json.dumps(remote_info, indent=2, sort_keys=True))
    hooks.write_fn(f"\nResult directory: {run.result_dir}")
    hooks.write_fn(f"Command: {' '.join(run.command)}")
