"""Reusable notebook run-info payload helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from neuroinfra.artifacts.output_paths import write_run_info as write_run_info_file


@dataclass(frozen=True)
class RunInfoHooks:
    """Hook bundle for one notebook run-info protocol."""

    json_ready_fn: Callable[[Any], Any]
    build_overrides_fn: Callable[[dict[str, Any]], dict[str, Any]]
    resolve_execution_mode_fn: Callable[[dict[str, Any]], Any] | None = None
    resolve_effective_params_fn: Callable[[dict[str, Any]], Any] | None = None
    env_keys: tuple[str, ...] = ()


def load_run_info_payload(result_dir: str | Path) -> dict[str, Any]:
    """Load one existing ``run_info.json`` payload, defaulting to an empty dict."""
    result_dir = Path(result_dir)
    run_info_path = result_dir / "run_info.json"
    if not run_info_path.exists() or run_info_path.stat().st_size == 0:
        return {}
    with open(run_info_path) as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def env_subset(env: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Return one env subset for the configured keys."""
    return {str(key): env.get(key) for key in keys}


def build_run_info_payload(
    hooks: RunInfoHooks,
    *,
    config: dict[str, Any],
    label: str,
    timestamp: str,
    command: list[str],
    env: Mapping[str, Any],
    completed: Any,
    runner: str,
    summary: Any = None,
    extra_payload: dict[str, Any] | None = None,
    existing_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one normalized notebook run-info payload."""
    payload = dict(existing_payload or {})
    payload.update(
        {
            "label": label,
            "requested_label": label,
            "timestamp": timestamp,
            "runner": str(runner),
            "config": hooks.json_ready_fn(config),
            "overrides": hooks.json_ready_fn(hooks.build_overrides_fn(config)),
            "command": list(command),
            "returncode": int(completed.returncode),
            "env": hooks.json_ready_fn(env_subset(env, hooks.env_keys)),
        }
    )

    if hooks.resolve_execution_mode_fn is not None:
        payload["resolved_execution_mode"] = hooks.json_ready_fn(hooks.resolve_execution_mode_fn(config))

    if hooks.resolve_effective_params_fn is not None:
        try:
            payload["effective_params"] = hooks.json_ready_fn(hooks.resolve_effective_params_fn(config))
            payload.pop("effective_params_error", None)
        except Exception as exc:
            payload["effective_params_error"] = f"{type(exc).__name__}: {exc}"

    if summary is not None:
        payload["summary"] = hooks.json_ready_fn(summary)

    if extra_payload:
        payload.update(hooks.json_ready_fn(extra_payload))

    return payload


def persist_run_info(
    result_dir: str | Path,
    hooks: RunInfoHooks,
    *,
    config: dict[str, Any],
    label: str,
    timestamp: str,
    command: list[str],
    env: Mapping[str, Any],
    completed: Any,
    runner: str,
    summary: Any = None,
    extra_payload: dict[str, Any] | None = None,
) -> Path:
    """Merge one notebook run-info payload into ``run_info.json`` and write it."""
    payload = build_run_info_payload(
        hooks,
        config=config,
        label=label,
        timestamp=timestamp,
        command=command,
        env=env,
        completed=completed,
        runner=runner,
        summary=summary,
        extra_payload=extra_payload,
        existing_payload=load_run_info_payload(result_dir),
    )
    return write_run_info_file(result_dir, payload)


def merge_run_info_payload(
    result_dir: str | Path,
    *,
    extra_payload: dict[str, Any],
    json_ready_fn: Callable[[Any], Any],
) -> Path:
    """Merge one extra payload into an existing run-info file and rewrite it."""
    payload = load_run_info_payload(result_dir)
    payload.update(json_ready_fn(extra_payload))
    return write_run_info_file(result_dir, payload)
