# OlfactoryBulb

This repository contains the olfactory bulb network model and the current
OBGPU workflow for running it with NEURON/CoreNEURON locally or through remote
Slurm jobs.

The maintained path is no longer Docker. Use the conda-based OBGPU setup, a
patched source build of NEURON/CoreNEURON, and the notebook helper layer in
`obgpu_experiment_helpers.py`.

## Quick Start

GPU/CoreNEURON build:

```bash
ENABLE_GPU=1 ENV_NAME=OBGPU ./install-obgpu.sh
source tools/setup/activate_obgpu.sh OBGPU
jupyter lab
```

Portable CPU-only build:

```bash
ENV_NAME=OBGPU-portable ENABLE_GPU=0 OBGPU_CPU_TARGET=portable ./install-obgpu.sh
source tools/setup/activate_obgpu.sh OBGPU-portable
jupyter lab
```

The active notebook is:

- `notebooks/obgpu-working-experiment.ipynb`

See [INSTALL.md](INSTALL.md) for host prerequisites, GPU/CUDA notes, smoke
tests, and Slurm/Phoenix/Sol setup details.

For a curated maintained-surface health pass after setup or refactors, run:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/run_audit.py repo_health --profile maintained
```

## Maintained Runtime

The modern workflow is:

1. `install-obgpu.sh` delegates to `tools/setup/setup_ob_modern.sh`.
2. The setup script creates or updates the `OBGPU` conda environment from
   `environments/environment-modern.yml`.
3. It resets `external/nrn-9.0.1` to the pinned upstream ref in
   `third_party_patches/nrn/manifest.json`.
4. It applies the repo patch stack from `third_party_patches/nrn/`.
5. It builds NEURON/CoreNEURON and compiles the Birgiolas mechanisms.
6. Runs are launched by the notebook helper or by
   `tools/benchmarks/benchmark_ob.py`.

The important active files are:

- `obgpu_experiment_helpers.py`
- `tools/benchmarks/benchmark_ob.py`
- `tools/remote/`
- `olfactorybulb/model.py`
- `olfactorybulb/paramsets/`
- `prev_ob_models/Birgiolas2020/Mechanisms/`
- `olfactorybulb/result_artifacts.py`

## Remote Slurm Runs

Remote notebook runs use Paramiko over SSH and Slurm on the remote host. The
old OpenSSH multiplex/rsync path has been removed from the maintained notebook
backend because it caused repeated authentication and stale control-socket
failures during sweeps.

Use:

- `build_slurm_remote_config(...)` for generic clusters such as Phoenix
- `build_sol_remote_config(...)` for Sol-specific activation defaults

The notebook auto-publishes the current local `HEAD` commit to the remote repo
when needed, submits through Slurm, streams live progress, and syncs results
back into local `results/notebook_runs/...`.

See [notes/porting/SOL_REMOTE_WORKFLOW.md](notes/porting/SOL_REMOTE_WORKFLOW.md)
for the current remote workflow.

## Results

Modern runs write timestamped directories under `results/notebook_runs/`.
Remote sweeps use `results/sweeps/`. Large soma trace payloads may be stored in
compact NPZ form and loaded lazily by the notebook helper.

Generated results, profiles, temporary archives, and compiled mechanism outputs
should not be committed.

## Repository Layout

- `olfactorybulb/`: core model, parameter sets, database access, and result
  artifact helpers.
- `prev_ob_models/`: historical published models and the active
  Birgiolas2020 cell/mechanism source used by the network.
- `tools/setup/`: OBGPU environment, NEURON/CoreNEURON build, and activation
  helpers.
- `tools/remote/`: Slurm submit/poll/batch helpers used by the notebook.
- `tools/benchmarks/`: maintained command-line benchmark/smoke runner.
- `notes/porting/`: current porting, setup, and remote workflow notes.
- `blender-files/` and `media/`: archival construction/visualization assets;
  not required for normal notebook runs.

## Citation

If you use this model or parts of it in your project, please cite:

```bibtex
@phdthesis{birgiolas2019towards,
  title={Towards Brains in the Cloud: A Biophysically Realistic Computational Model of Olfactory Bulb},
  author={Birgiolas, Justas},
  year={2019},
  school={Arizona State University}
}
```
