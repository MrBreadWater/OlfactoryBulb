"""Submit one timestamped benchmark run through the shared neuroinfra helper."""

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

from neuroinfra.remote_script_submit import (  # noqa: E402
    decode_command,
    main,
    neuron_mpi_preflight_suffix,
    relocate_benchmark_command,
    submit_allocation_step,
    submit_batch,
    write_batch_script,
)


if __name__ == "__main__":
    main()
