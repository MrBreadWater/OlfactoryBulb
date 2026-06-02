# Install

This repo's maintained install path is **not Docker**.

The supported workflow is:

1. create/update a conda environment
2. clone/reset a pinned upstream `nrn` checkout under `external/nrn-9.0.1`
3. apply the local NEURON patch stack from `third_party_patches/nrn/`
4. build/install NEURON + CoreNEURON into the conda env
5. compile the Birgiolas mechanisms from `prev_ob_models/Birgiolas2020/Mechanisms`
6. activate the env with the repo-specific runtime hooks

The one command that drives this is:

```bash
./install-obgpu.sh
```

That wrapper delegates to `tools/setup/setup_ob_modern.sh`, which is the real source of truth.

## Host Prerequisites

You need these on the host before the conda env exists:

- `bash`
- `git`
- `python3` or `python`
- `conda`/`mamba` available directly or via a module
- `gcc` and `g++`

The script installs most Python/build dependencies into conda itself from `environments/environment-modern.yml`.

## GPU Install

Use this for the maintained OBGPU path:

```bash
ENABLE_GPU=1 ENV_NAME=OBGPU ./install-obgpu.sh
```

Additional GPU prerequisites:

- NVIDIA HPC SDK compilers: `nvc`, `nvc++`
- CUDA toolkit with `nvcc`
- a visible CUDA GPU during setup, or explicit `CUDA_ARCHITECTURES` / `NVHPC_COMPUTE_CAPABILITIES`

Notes:

- the script will try to auto-load `nvhpc` and `cuda` modules if those compilers are not already on `PATH`
- the NEURON source is rebuilt from the pinned upstream ref in `third_party_patches/nrn/manifest.json`
- the patch stack is applied on every clean rebuild; `external/nrn-9.0.1` is treated as a resettable cache, not a hand-edited fork

## CPU-Only Install

If you do not have NVHPC/CUDA, use the portable CPU build:

```bash
ENV_NAME=OBGPU-portable ENABLE_GPU=0 OBGPU_CPU_TARGET=portable ./install-obgpu.sh
```

That still builds the same patched NEURON/CoreNEURON tree, but without GPU support and with a portable mechanism/build profile.

## Activation

After setup, activate the environment from the repo root with:

```bash
source tools/setup/activate_obgpu.sh OBGPU
```

For the portable CPU env:

```bash
source tools/setup/activate_obgpu.sh OBGPU-portable
```

On Sol, use:

```bash
source tools/setup/activate_sol_obgpu.sh
```

The activation helpers do more than `conda activate`:

- activate the chosen env
- export repo/mechanism paths
- set `CORENEURONLIB`
- prepend the conda and mechanism library paths to `LD_LIBRARY_PATH`
- set `OB_MPIEXEC` appropriately for local MPI or Slurm

## First Smoke Tests

After activation, the simplest maintained benchmark smoke test is:

```bash
mpiexec -n 1 nrniv -mpi -python tools/benchmarks/benchmark_ob.py --label local_smoke --paramset OneMsTest --coreneuron
```

For the GPU path:

```bash
mpiexec -n 1 nrniv -mpi -python tools/benchmarks/benchmark_ob.py --label local_gpu_smoke --paramset OneMsTest --coreneuron --coreneuron-gpu
```

If you are inside a Slurm allocation, prefer:

```bash
$OB_MPIEXEC -n 1 nrniv -mpi -python tools/benchmarks/benchmark_ob.py --label slurm_smoke --paramset OneMsTest --coreneuron
```

The setup script already runs an import verification step at the end via `tools/setup/verify_obgpu_python_imports.py`.

For a broader maintained-surface health pass after setup or after touching
infrastructure code:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/run_audit.py repo_health --profile maintained
```

## Notebook Workflow

Once the env is active:

```bash
jupyter lab
```

The actively maintained notebook is:

- `notebooks/obgpu-working-experiment.ipynb`

The older `LFP Wavelet Analysis.ipynb` notebook still exists, but the repo's
modern setup and remote workflow are built around the OBGPU notebook/helper
path.

Remote notebook runs use Paramiko over SSH. The previous OpenSSH
control-master/rsync backend has been removed from the maintained path.

## Running Simulations

Maintained benchmark smoke, CPU-oriented:

```bash
mpiexec -n 4 nrniv -mpi -python tools/benchmarks/benchmark_ob.py --label gamma_cpu --paramset GammaSignature
```

Maintained benchmark smoke, GPU/CoreNEURON-oriented:

```bash
mpiexec -n 1 nrniv -mpi -python tools/benchmarks/benchmark_ob.py --label gamma_gpu --paramset GammaSignature --coreneuron --coreneuron-gpu
```

Historical `initslice.py` / `runbatch.py` entrypoints have been removed. Use
the benchmark runner or the notebook/helper path instead.

## Important Repo-Specific Notes

- Run commands from the repo root. This repo is not packaged as a normal `pip install -e .` project.
- Do not make ad hoc edits inside `external/nrn-9.0.1`; put NEURON changes into `third_party_patches/nrn/` and update the manifest/patch stack deliberately.
- If you change `.mod` files under `prev_ob_models/Birgiolas2020/Mechanisms`, rerun the setup script or recompile mechanisms in the active OBGPU environment before running simulations.
- `environment.yml` / `environment-lock.yml` are legacy environment files. The maintained one is `environments/environment-modern.yml`.
- `activate_obgpu.sh` is the generic local helper. `activate_sol_obgpu.sh` is the Sol-specific helper.
- Tracked generated NEURON outputs under historical `prev_ob_models` trees are
  intentionally being retired; rebuild mechanisms from `.mod` source when
  needed.

## Files That Matter

- `install-obgpu.sh`
- `tools/setup/setup_ob_modern.sh`
- `tools/setup/activate_obgpu.sh`
- `tools/setup/activate_sol_obgpu.sh`
- `environments/environment-modern.yml`
- `third_party_patches/nrn/manifest.json`
- `notes/porting/MODERN_NEURON_PORT_NOTES.md`
- `notes/porting/SOL_REMOTE_WORKFLOW.md`
