# Modern NEURON / OBGPU Notes

## Supported Model

- Env name: `OBGPU`
- Python: `3.11`
- NEURON/CoreNEURON: source-built from a pinned upstream `nrn` ref plus a repo-managed patch stack
- GPU support: enabled through NVHPC + CUDA when `ENABLE_GPU=1`

The maintained source of truth is now:

- pinned upstream ref: [third_party_patches/nrn/manifest.json](/home/alek/OlfactoryBulb/third_party_patches/nrn/manifest.json)
- patch files: [third_party_patches/nrn](/home/alek/OlfactoryBulb/third_party_patches/nrn)

`external/nrn-9.0.1` is treated as a resettable checkout cache, not the long-term home of local edits.

## Build Flow

Create or refresh `OBGPU` with:

```bash
ENABLE_GPU=1 ENV_NAME=OBGPU ./tools/setup/setup_ob_modern.sh
```

What the setup script now does:

1. creates or updates the conda env from [environment-modern.yml](/home/alek/OlfactoryBulb/environments/environment-modern.yml)
2. resets the NEURON source tree to the pinned upstream ref from the manifest
3. reapplies the repo patch stack
4. builds and installs NEURON/CoreNEURON
5. compiles the Birgiolas mechanisms with `nrnivmodl -coreneuron`
6. repairs the generated `libnrnmech.so` if the NVHPC fatbin temp-object issue is present

Important portability details:

- the setup path no longer hardcodes `aarch64/libnrnmech.so`
- the setup path no longer assumes `/opt/miniconda3`
- the main build path is intended for generic Linux GPU hosts, not just the Jetson
- [setup_nvhpc_jetson.sh](/home/alek/OlfactoryBulb/tools/setup/setup_nvhpc_jetson.sh) remains a Jetson-specific helper, not the primary OBGPU build workflow

## Runtime Defaults

The current fast local default is still:

- rank `1`
- GPU on
- `cell_permute=2`
- `warp_balance=128`

The parity-oriented mode is still available by running with `2` MPI ranks.

The notebook helper surface continues to expose both modes through `build_run_config(...)`.

## Sol Workflow

The intended Sol workflow is headless:

- notebook stays local
- notebook submits jobs to Sol through Slurm
- notebook can pin each remote run to an explicit git ref
- results sync back to local `results/notebook_runs/...`
- analysis still runs locally on the synced results

See:

- [SOL_REMOTE_WORKFLOW.md](/home/alek/OlfactoryBulb/notes/porting/SOL_REMOTE_WORKFLOW.md)
- [obgpu_experiment_helpers.py](/home/alek/OlfactoryBulb/obgpu_experiment_helpers.py)
- [tools/remote](/home/alek/OlfactoryBulb/tools/remote)

## Upgrade Policy

Upstream NEURON updates are deliberate, not automatic.

Use:

```bash
python tools/setup/check_nrn_upgrade.py --candidate-ref <tag-or-commit> --skip-build
```

or, for a real gate with build + smoke checks:

```bash
python tools/setup/check_nrn_upgrade.py --candidate-ref <tag-or-commit> --enable-gpu
```

The supported upstream ref changes only after:

1. the candidate ref is checked out cleanly
2. the patch stack applies cleanly
3. OBGPU rebuilds successfully
4. the smoke/parity checks pass

See:

- [NEURON_UPGRADE_WORKFLOW.md](/home/alek/OlfactoryBulb/notes/porting/NEURON_UPGRADE_WORKFLOW.md)
- [check_nrn_upgrade.py](/home/alek/OlfactoryBulb/tools/setup/check_nrn_upgrade.py)
