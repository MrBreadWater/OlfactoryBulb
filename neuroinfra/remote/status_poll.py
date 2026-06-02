"""Reusable remote JSON status polling helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
import time
from typing import Any, Callable


@dataclass(frozen=True)
class RemoteJSONPollHooks:
    """Callbacks injected by the notebook-facing caller for JSON status polls."""

    run_command_fn: Callable[[str, float | None], subprocess.CompletedProcess[str]]
    record_timing_fn: Callable[[str, float], None]
    sleep_fn: Callable[[float], None] = time.sleep
    perf_counter_fn: Callable[[], float] = time.perf_counter


def poll_remote_json_status(
    command: str,
    *,
    poll_json_retries: int,
    error_prefix: str,
    hooks: RemoteJSONPollHooks,
    timing_key: str = "poll_s",
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Run one remote poll command and return parsed JSON with retry-on-malformed-output."""
    retries = max(int(poll_json_retries or 1), 1)
    poll_completed: subprocess.CompletedProcess[str] | None = None
    last_exc: json.JSONDecodeError | None = None

    for attempt in range(retries):
        started = hooks.perf_counter_fn()
        poll_completed = hooks.run_command_fn(command, timeout_s)
        hooks.record_timing_fn(timing_key, started)
        if poll_completed.returncode != 0:
            raise RuntimeError(
                f"{error_prefix} failed.\n"
                f"Stdout:\n{poll_completed.stdout}\n\nStderr:\n{poll_completed.stderr}"
            )
        try:
            return json.loads((poll_completed.stdout or "").strip())
        except json.JSONDecodeError as exc:
            last_exc = exc
            if attempt + 1 >= retries:
                break
            hooks.sleep_fn(min(0.5 * (attempt + 1), 2.0))

    assert poll_completed is not None
    raise RuntimeError(
        f"{error_prefix} did not return valid JSON.\n"
        f"Stdout:\n{poll_completed.stdout}\n\nStderr:\n{poll_completed.stderr}"
    ) from last_exc
