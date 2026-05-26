# Changelog 2026-03-19

Historical note as of 2026-05-26: this file records the early legacy `OB`
environment recovery session. It is not the current setup guide. For current
OBGPU setup and remote execution, use `INSTALL.md`,
`MODERN_NEURON_PORT_NOTES.md`, and `SOL_REMOTE_WORKFLOW.md`.

## Summary

This log records all substantive work completed on the `OB` environment and the `OlfactoryBulb` repo during the current session.

## Environment rebuild

- Removed the previously broken `OB` conda environment and rebuilt it around Python 3.8 for compatibility with the older NEURON / neuronunit stack.
- Installed an older scientific core compatible with the repo:
  - `python 3.8.19`
  - `numpy 1.18.5`
  - `pandas 1.0.3`
  - `scipy 1.4.1`
- Installed MPI tooling into `OB`:
  - `openmpi`
  - `mpi4py`
  - `ucx`

## Permanent environment activation hooks

- Added activation and deactivation hooks under:
  - `/opt/miniconda3/envs/OB/etc/conda/activate.d/ob_env.sh`
  - `/opt/miniconda3/envs/OB/etc/conda/deactivate.d/ob_env.sh`
- Hook behavior now:
  - prepends `PATH` with NEURON binaries under `$CONDA_PREFIX/aarch64/bin`
  - prepends `LD_LIBRARY_PATH` with `$CONDA_PREFIX/aarch64/lib`
  - prepends `PYTHONPATH` with `$CONDA_PREFIX/lib/python`
  - enables `OMPI_MCA_opal_cuda_support=true`
  - sets `UCX_MEMTYPE_CACHE=n`
  - sets `MPLCONFIGDIR=$CONDA_PREFIX/.config/matplotlib`
- Removed previously forced `OMPI_MCA_pml=ucx` / `OMPI_MCA_osc=ucx` from the permanent hook after verifying they broke MPI startup on this machine.

## NEURON build and installation

- Built and installed NEURON from source into `OB` using an older compatible version line.
- Worked around broken OpenMPI wrapper compiler defaults by building with:
  - `OMPI_CC=gcc`
  - `OMPI_CXX=g++`
- Verified NEURON Python import and CLI tools inside `OB`:
  - `import neuron`
  - `nrniv`
  - `nrnivmodl`

## Python dependency installation

- Installed the repo requirements stack and resolved a large number of legacy packaging issues.
- Installed or restored the following key packages:
  - `blenderneuron`
  - `jsonpickle`
  - `neo`
  - `quantities`
  - `LFPsimpy`
  - `sciunit`
  - `neuronunit`
  - `elephant`
  - `pyNeuroML`
  - `libNeuroML`
  - `PyLEMS`
  - `natsort`
  - `networkx`
  - `pytables`
  - `airspeed`
  - `pysqlite3`
  - `numba`
  - `dask`
- Installed additional support libraries via conda:
  - `lxml`
  - `gitpython`
  - `cerberus`
  - `requests`
  - `validators`
  - `execnet`
  - `lmfit`
  - `backports.tempfile`

## Intentional version relaxations

- Kept strict compatibility where it mattered most, especially around NEURON and old neuroscience packages.
- Relaxed some legacy pins where exact versions were no longer buildable or were not needed:
  - `Cython`
  - `matplotlib`
  - `mpi4py`
  - `peewee`
  - `deap`
  - `pysqlite3`
  - general Jupyter tooling

## Local compatibility shims in `OB`

- Added `/opt/miniconda3/envs/OB/lib/python3.8/site-packages/sitecustomize.py`.
- Added a shim for a local code assumption around `quantities.__module__`.
- Added a NEURON mechanism-loading fallback so the environment can locate the repo-root compiled `aarch64/.libs/libnrnmech.so` when needed.

## Mechanism compilation

- Compiled `prev_ob_models/Birgiolas2020/Mechanisms` from the repo root using:
  - `OMPI_CC=gcc OMPI_CXX=g++ nrnivmodl prev_ob_models/Birgiolas2020/Mechanisms`
- This produced:
  - `aarch64/`
  - `aarch64/.libs/libnrnmech.so`
  - `aarch64/special`

## Import and smoke-test verification

- Verified the main import chain for:
  - `neuron`
  - `mpi4py`
  - `blenderneuron`
  - `neo`
  - `quantities`
  - `sciunit`
  - `neuronunit`
  - `elephant`
  - `numba`
  - `dask`
  - `pyneuroml`
  - `neuroml`
  - `lems`
  - `airspeed`
- Verified:
  - `from neuronunit.tests.base import VmTest`
- Verified isolated cell construction for:
  - `MC1`
  - `GC1`
  - `TC1`
- Verified real cell simulation smoke tests for isolated cells.
- Verified large direct population construction and direct NEURON simulation outside the repo model wrapper.

## MPI investigation and correction

- Determined that on this machine:
  - `mpiexec -n N python ...` does not correctly attach NEURON `ParallelContext` to the MPI world.
  - `mpiexec -n N nrniv -mpi -python ...` works correctly.
- Verified working MPI / NEURON rank visibility with:
  - `mpiexec -n 2 nrniv -mpi -python ...`
- Verified repo MPI smoke test works under:
  - `mpiexec -n 2 nrniv -mpi -python testmpi.py -mpi`

## Root-cause debugging for full-model crash

- Narrowed a native crash in `OlfactoryBulb.run(...)` to the gap-junction transfer path.
- Verified the crash was not caused by:
  - LFP electrode setup
  - status reporting callback
  - odor inputs
  - synapse loading
  - soma recording vectors
  - the base cell templates themselves
  - full population size by itself
- Identified the actual issue:
  - the code created `ParallelContext` gap-junction transfer state even when `g_gap == 0`
  - the code used serial `h.run()` on single-rank runs even when gap-junction transfer variables were active

## Repo source fix

- Updated [`olfactorybulb/model.py`](/home/alek/OlfactoryBulb/olfactorybulb/model.py):
  - `add_gap_junctions(...)` now returns immediately when `g_gap <= 0`
  - `run(...)` now uses the `psolve()` path whenever gap junction transfer state exists, even on a single rank
- This fixed the previously reproducible crash.

## Post-fix verification

- Verified:
  - `OlfactoryBulb('ParameterSetBase', autorun=False); ob.run(1)`
  - `OlfactoryBulb('OneMsTest', autorun=False); ob.run(ob.params.tstop)`
  - `python initslice.py -paramset OneMsTest`
  - `mpiexec -n 2 nrniv -mpi -python testmpi.py -mpi`
- Verified that real nonzero gap junction creation still works after the fix.
- Verified that zero-gap parameter sets no longer create useless gap-junction transfer state.

## Environment spec files created

- Added a curated reusable environment spec:
  - [`environment.yml`](/home/alek/OlfactoryBulb/environments/environment.yml)
- Added an exact exported environment snapshot:
  - [`environment-lock.yml`](/home/alek/OlfactoryBulb/environments/environment-lock.yml)
- Added an explicit conda package export for this platform:
  - [`environment-linux-aarch64-explicit.txt`](/home/alek/OlfactoryBulb/environments/environment-linux-aarch64-explicit.txt)

## Permission and repo access changes

- User granted write access to the repo for the `michael` account.
- Added the repo as a safe Git directory for the current user:
  - `git config --global --add safe.directory /home/alek/OlfactoryBulb`

## Notes

- The repo contains unrelated modified and untracked files that were not created by this session. Those were not reverted.
- The current verified MPI launch pattern for this host is:

```bash
mpiexec -n 2 nrniv -mpi -python testmpi.py -mpi
```

- The current verified single-rank model smoke run is:

```bash
python initslice.py -paramset OneMsTest
```
