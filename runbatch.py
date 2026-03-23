"""
This file is used to sequentially run the network model using different sets of parameters.

The paramsets array should contain class names found in: [repo]/olfactorybulb/paramsets/*.py

Environment overrides:
    OB_MPI_RANKS   Number of MPI ranks to launch per simulation. Defaults to cpu_count().
    OB_MPIEXEC     MPI launcher binary. Defaults to "mpiexec".
"""

import multiprocessing
import os
import subprocess

paramsets = [
    "GammaSignature",
    "GammaSignature_NoInhibition",
    "GammaSignature_NoTCGJs",
    "GammaSignature_NoMCGJs",
    "GammaSignature_EqualTCMCInputs"
]

def get_mpi_ranks():
    raw = os.environ.get("OB_MPI_RANKS")
    if raw is None:
        return max(2, multiprocessing.cpu_count())

    return max(2, int(raw))


mpi_ranks = get_mpi_ranks()
mpiexec = os.environ.get("OB_MPIEXEC", "mpiexec")
base_env = os.environ.copy()

# Prevent threaded math libraries from oversubscribing the CPU under MPI.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    base_env.setdefault(var, "1")

for i, params in enumerate(paramsets):
    print('Starting paramset: %s (%s/%s) with %s MPI ranks...' % (params, i + 1, len(paramsets), mpi_ranks))
    command = [
        mpiexec,
        "-n",
        str(mpi_ranks),
        "nrniv",
        "-mpi",
        "-python",
        "initslice.py",
        "-paramset",
        params,
        "-mpi",
    ]
    subprocess.run(command, check=True, env=base_env)
