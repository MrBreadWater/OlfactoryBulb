"""
This file is used to sequentially run the network model using different sets of parameters.

The paramsets array should contain class names found in: [repo]/olfactorybulb/paramsets/*.py

Environment overrides:
    OB_MPI_RANKS          Number of MPI ranks to launch per simulation.
    OB_MPIEXEC            MPI launcher command. Defaults to "mpiexec", or
                          "srun --mpi=pmix" inside a Slurm allocation when
                          available.
    OB_RUNTIME_MODE       One of "scientific" or "exploratory".
    OB_USE_CORENRN        Enable CoreNEURON for `initslice.py`.
    OB_USE_CORENRN_GPU    Enable CoreNEURON GPU mode for `initslice.py`.
    OB_CORENRN_CELL_PERMUTE  CoreNEURON cell permutation mode.
    OB_CORENRN_WARP_BALANCE  CoreNEURON warp-balance setting.
    When launched from the OBGPU env, the default is the rank-1 GPU scientific mode.
"""

from __future__ import annotations

import multiprocessing
import os
import shlex
import shutil
import subprocess

from olfactorybulb.output_paths import label_with_timestamp, make_timestamp

paramsets = [
    "GammaSignature",
    "GammaSignature_NoInhibition",
    "GammaSignature_NoTCGJs",
    "GammaSignature_NoMCGJs",
    "GammaSignature_EqualTCMCInputs"
]

def using_modern_gpu_env() -> bool:
    """Return True when the active conda environment is the maintained OBGPU stack."""
    return os.environ.get("CONDA_DEFAULT_ENV", "") == "OBGPU"


def env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable using shell-friendly truthy/falsey strings."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def get_mpi_ranks() -> int:
    """Return the MPI rank count for batch runs, with a GPU-friendly default of one rank."""
    raw = os.environ.get("OB_MPI_RANKS")
    if raw is None:
        if env_flag("OB_USE_CORENRN_GPU", using_modern_gpu_env()):
            return 1
        return max(2, multiprocessing.cpu_count())

    return max(2, int(raw))


def default_mpi_exec() -> str:
    """Return the preferred MPI launcher for the current environment."""
    configured = os.environ.get("OB_MPIEXEC")
    if configured:
        return configured

    if os.environ.get("SLURM_JOB_ID") and shutil.which("srun"):
        slurm_mpi_type = os.environ.get("OB_SLURM_MPI_TYPE", "pmix").strip()
        if slurm_mpi_type:
            return f"srun --mpi={slurm_mpi_type}"
        return "srun"

    return "mpiexec"


def main() -> None:
    """Launch each configured paramset sequentially with the current runtime defaults."""
    mpi_ranks = get_mpi_ranks()
    mpiexec = default_mpi_exec()
    base_env = os.environ.copy()

    # Prevent threaded math libraries from oversubscribing the CPU under MPI.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        base_env.setdefault(var, "1")

    if env_flag("OB_USE_CORENRN_GPU", using_modern_gpu_env()):
        base_env.setdefault("OB_USE_CORENRN", "1")
        base_env.setdefault("OB_USE_CORENRN_GPU", "1")
        base_env.setdefault("OB_CORENRN_CELL_PERMUTE", "2")
        base_env.setdefault("OB_CORENRN_WARP_BALANCE", "128")
        base_env.setdefault("OB_RUNTIME_MODE", "scientific")

    for i, params in enumerate(paramsets):
        runtime_mode = base_env.get("OB_RUNTIME_MODE", "scientific")
        warp_balance = base_env.get("OB_CORENRN_WARP_BALANCE", "0")
        mode = "CoreNEURON GPU" if env_flag("OB_USE_CORENRN_GPU", using_modern_gpu_env()) else "NEURON"
        run_env = base_env.copy()
        run_timestamp = make_timestamp()
        run_label = label_with_timestamp(params, timestamp=run_timestamp)
        run_env["OB_RUN_TIMESTAMP"] = run_timestamp
        run_env["OB_RESULT_LABEL"] = run_label
        print('Starting paramset: %s (%s/%s) with %s MPI ranks [%s, %s, warp=%s] -> %s...' % (params, i + 1, len(paramsets), mpi_ranks, mode, runtime_mode, warp_balance, run_label))
        command = [
            *shlex.split(mpiexec),
            "-n",
            str(mpi_ranks),
            "nrniv",
            "-mpi",
            "-python",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "initslice.py"),
            "-paramset",
            params,
            "-mpi",
        ]
        subprocess.run(command, check=True, env=run_env)


if __name__ == "__main__":
    main()
