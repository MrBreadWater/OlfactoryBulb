"""Shared helpers for remote Slurm command wrappers.

These scripts are executed directly on remote hosts, so they bootstrap imports
from the nearest root that contains a minimal ``neuroinfra`` package payload.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _install_import_root() -> None:
    """Prepend the nearest parent directory that exposes ``neuroinfra``."""
    script_path = Path(__file__).resolve()
    candidates = [script_path.parent, *script_path.parents]
    for candidate in candidates:
        if (candidate / "neuroinfra" / "__init__.py").exists():
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)
            return


_install_import_root()

from neuroinfra.remote_script_common import (  # noqa: E402
    normalize_sbatch_args,
    path_is_within,
    requested_mpi_rank_count,
    shell_join,
    slurm_directives,
)

__all__ = [
    "normalize_sbatch_args",
    "path_is_within",
    "requested_mpi_rank_count",
    "shell_join",
    "slurm_directives",
]
