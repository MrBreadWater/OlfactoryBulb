"""Run one full OBGPU parameter sweep inside a single remote Slurm job."""

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

from neuroinfra.remote_script_sweeps import (  # noqa: E402
    add_srun_parallel_step_flags,
    build_neuron_mpi_preflight,
    decode_items,
    dll_path_from_env,
    inject_neuron_dll_args,
    launch_cwd,
    load_items_json,
    main,
    normalize_items,
    progress_payload,
    relocate_repo_paths,
    resolve_completed_result_dir,
    terminate_process_tree,
    write_json,
)


if __name__ == "__main__":
    main()
