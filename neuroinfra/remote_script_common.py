"""Remote-safe common helpers for uploaded Slurm wrapper scripts.

This module intentionally depends only on the Python standard library so it can
be bundled into remote helper caches without the rest of the local runtime.
"""

from __future__ import annotations

import shlex
from typing import Any


def shell_join(parts: list[Any] | tuple[Any, ...]) -> str:
    """Portable equivalent of ``shlex.join`` for older remote Python stacks."""
    return " ".join(shlex.quote(str(part)) for part in parts)


def path_is_within(path_value: Any, root_value: Any) -> bool:
    """Return whether one path-like string is equal to or nested under another."""
    root_text = str(root_value).rstrip("/")
    path_text = str(path_value)
    if not root_text:
        return False
    return path_text == root_text or path_text.startswith(root_text + "/")


def normalize_sbatch_args(values: list[Any] | tuple[Any, ...]) -> list[str]:
    """Normalize raw sbatch args so split flag/value pairs become one directive."""
    normalized: list[str] = []
    index = 0
    text_values = [str(value) for value in values]
    while index < len(text_values):
        current = text_values[index]
        if current.startswith("-") and "=" not in current and index + 1 < len(text_values):
            next_value = text_values[index + 1]
            if not next_value.startswith("-"):
                normalized.append(f"{current} {next_value}")
                index += 2
                continue
        normalized.append(current)
        index += 1
    return normalized


def slurm_directives(args: Any, job_name: str) -> list[str]:
    """Return ``#SBATCH`` header lines for one generated Slurm script."""
    directives = [f"#SBATCH --job-name={str(job_name)[:120]}"]
    if args.partition:
        directives.append(f"#SBATCH --partition={args.partition}")
    if args.account:
        directives.append(f"#SBATCH --account={args.account}")
    if args.time:
        directives.append(f"#SBATCH --time={args.time}")
    if args.gpus is not None:
        directives.append(f"#SBATCH --gpus={args.gpus}")
    if args.cpus_per_task is not None:
        directives.append(f"#SBATCH --cpus-per-task={args.cpus_per_task}")
    if args.mem:
        directives.append(f"#SBATCH --mem={args.mem}")
    for extra in normalize_sbatch_args(getattr(args, "sbatch_arg", [])):
        directives.append(f"#SBATCH {extra}")
    return directives


def requested_mpi_rank_count(command: list[str] | tuple[str, ...]) -> int | None:
    """Return the requested MPI rank count from one command list, if present."""
    options_with_values = {"-n", "-np", "--np", "--ntasks", "--ntasks-per-job"}
    for index, part in enumerate(command):
        if part in options_with_values and index + 1 < len(command):
            try:
                return int(command[index + 1])
            except ValueError:
                continue
        for prefix in ("-n", "-np"):
            suffix = part[len(prefix) :]
            if part.startswith(prefix) and suffix:
                try:
                    return int(suffix)
                except ValueError:
                    pass
        for prefix in ("--ntasks=", "--ntasks-per-job="):
            if part.startswith(prefix):
                try:
                    return int(part.split("=", 1)[1])
                except ValueError:
                    pass
    return None
