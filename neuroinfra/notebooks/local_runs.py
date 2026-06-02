"""Reusable local notebook-run execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess
from typing import Any, Callable, Mapping

from .runs import (
    DEFAULT_STDERR_FILENAME,
    DEFAULT_STDOUT_FILENAME,
    DEFAULT_SUMMARY_FILENAME,
)


DEFAULT_COMMAND_FILENAME = "command.txt"


@dataclass(frozen=True)
class LocalRunHooks:
    """Hook bundle for one local notebook subprocess run."""

    read_summary_fn: Callable[[Path], dict[str, Any]]
    write_run_info_fn: Callable[..., Any]
    build_return_value_fn: Callable[..., Any]
    run_subprocess_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run
    command_filename: str = DEFAULT_COMMAND_FILENAME
    stdout_filename: str = DEFAULT_STDOUT_FILENAME
    stderr_filename: str = DEFAULT_STDERR_FILENAME
    summary_filename: str = DEFAULT_SUMMARY_FILENAME


def _write_command_capture_files(
    result_dir: Path,
    *,
    command: list[str],
    completed: subprocess.CompletedProcess[str],
    command_filename: str,
    stdout_filename: str,
    stderr_filename: str,
) -> None:
    """Persist one executed command plus its captured stdout/stderr."""
    (result_dir / command_filename).write_text(" ".join(command) + "\n")
    (result_dir / stdout_filename).write_text(completed.stdout or "")
    (result_dir / stderr_filename).write_text(completed.stderr or "")


def _run_failure_message(
    *,
    result_dir: Path,
    command: list[str],
    completed: subprocess.CompletedProcess[str],
    stdout_tail_chars: int = 2000,
    stderr_tail_chars: int = 4000,
) -> str:
    """Render one standard local-run failure message."""
    stderr_tail = (completed.stderr or "").strip()[-stderr_tail_chars:]
    stdout_tail = (completed.stdout or "").strip()[-stdout_tail_chars:]
    return (
        "Simulation failed.\n"
        f"Result dir: {result_dir}\n"
        f"Command: {' '.join(command)}\n"
        f"Stdout tail:\n{stdout_tail}\n\n"
        f"Stderr tail:\n{stderr_tail}"
    )


def execute_local_run(
    *,
    config: dict[str, Any],
    label: str,
    timestamp: str,
    result_dir: str | Path,
    env: Mapping[str, str],
    command: list[str],
    hooks: LocalRunHooks,
    runner_name: str,
    cwd: str | Path | None = None,
    success_extra_payload: dict[str, Any] | None = None,
    failure_extra_payload: dict[str, Any] | None = None,
) -> Any:
    """Execute one local notebook subprocess run and persist its standard artifacts."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    run_cwd = Path(cwd) if cwd is not None else result_dir

    completed = hooks.run_subprocess_fn(
        command,
        cwd=run_cwd,
        env=dict(env),
        capture_output=True,
        text=True,
        check=False,
    )
    _write_command_capture_files(
        result_dir,
        command=command,
        completed=completed,
        command_filename=hooks.command_filename,
        stdout_filename=hooks.stdout_filename,
        stderr_filename=hooks.stderr_filename,
    )

    if completed.returncode != 0:
        hooks.write_run_info_fn(
            result_dir,
            config=config,
            label=label,
            timestamp=timestamp,
            command=command,
            env=dict(env),
            completed=completed,
            runner=runner_name,
            extra_payload=failure_extra_payload,
        )
        raise RuntimeError(
            _run_failure_message(
                result_dir=result_dir,
                command=command,
                completed=completed,
            )
        )

    summary_path = result_dir / hooks.summary_filename
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected benchmark summary at {summary_path}")
    summary = hooks.read_summary_fn(summary_path)

    hooks.write_run_info_fn(
        result_dir,
        config=config,
        label=label,
        timestamp=timestamp,
        command=command,
        env=dict(env),
        completed=completed,
        runner=runner_name,
        summary=summary,
        extra_payload=success_extra_payload,
    )

    return hooks.build_return_value_fn(
        label=label,
        timestamp=timestamp,
        result_dir=result_dir,
        summary=summary,
        config=config,
        command=command,
        completed=completed,
    )
