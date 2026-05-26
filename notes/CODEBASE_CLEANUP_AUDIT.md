# Codebase Cleanup Audit

Date: 2026-05-26

This audit separates the actively maintained OBGPU workflow from historical
model archives, generated artifacts, and old helper paths. It is intentionally
pragmatic: keep the modern run path clean, but do not rewrite archival model
code unless it is directly blocking current work.

`modify_model.py` is explicitly out of scope for this cleanup pass.

## Active Maintained Surface

These files are on the modern setup/run/analyze path and should receive tests,
documentation, and cleanup priority.

- `install-obgpu.sh`
- `tools/setup/setup_ob_modern.sh`
- `tools/setup/activate_obgpu.sh`
- `tools/setup/activate_sol_obgpu.sh`
- `tools/setup/verify_obgpu_python_imports.py`
- `environments/environment-modern.yml`
- `third_party_patches/nrn/`
- `obgpu_experiment_helpers.py`
- `tools/benchmarks/benchmark_ob.py`
- `tools/remote/`
- `olfactorybulb/model.py`
- `olfactorybulb/output_paths.py`
- `olfactorybulb/result_artifacts.py`
- `olfactorybulb/paramsets/`
- `notebooks/obgpu-working-experiment.ipynb`
- `single_cell_utils.py`
- `fi_curve_utils.py`
- `notebooks/fi_curve_analysis.ipynb`

`initslice.py` and `runbatch.py` remain compatibility entrypoints, but new
notebook, benchmark, and Slurm work should use the helper/benchmark path.

## Remote Execution Cleanup

The maintained remote backend is now Paramiko-only.

Removed or retired concepts:

- OpenSSH control-master transport
- `ssh_multiplex`
- `ssh_control_path`
- `ssh_control_persist_s`
- `ssh_allow_interactive_auth`
- rsync result sync path
- `rsync_binary`
- `rsync_options`

Kept concepts:

- `ssh_options`, especially `["-p", "..."]` and jump-host style options
- Paramiko persistent sessions
- streamed compressed result sync
- selected-file sync for sweeps and deferred artifacts
- reusable Slurm allocations and manual `slurm_allocation_job_id`

## Historical / Reference Code

These directories are important, but most of their contents are archival rather
than active application code.

- `prev_ob_models/`
  - Active dependency: `prev_ob_models/Birgiolas2020`.
  - Historical references: Kaplan/Lansner, Li/Cleland, Short, Saghatelyan, and
    other imported model trees.
  - Generated NEURON build outputs inside these trees should not be tracked.
- `blender-files/`
  - Network construction and visualization assets.
  - Large binary assets are not required for normal notebook runs.
- `media/`
  - Figures/videos/GIFs for documentation and presentation.
  - New generated media should be kept out of git unless it is intentionally a
    curated deliverable.
- `external/`
  - Resettable upstream dependency checkout cache. Do not hand-edit as the
    source of truth.

## Generated / Local-Only Noise

These should stay ignored and out of normal commits.

- `results/notebook_runs/`
- `results/sweeps/`
- `results/benchmarks/`
- `results/comparisons/`
- `results/profiles/`
- `results/debug_*/`
- `results/tmp_*/`
- `.jupyter-ai-state/`
- `.codex/`
- `.ipynb_checkpoints/`
- `aarch64/`
- `x86_64/`
- `corenrn_data/`
- `*.zip`
- generated `*.c`, `*.o`, `*.dll`, and shared-library outputs from NEURON
  mechanism builds

## Current High-Value Cleanup Targets

1. Keep `obgpu_experiment_helpers.py` from growing new duplicate transport,
   sync, and sweep paths. Prefer deleting old branches over adding feature
   flags.
2. Keep remote config builder options minimal. If a value can be inferred from
   Slurm resources or execution mode, avoid adding a second independent knob.
3. Keep large generated outputs out of the active tree and history. Commit
   source `.mod`, Python, JSON model definitions, and small curated examples;
   rebuild generated binaries locally.
4. Keep all active docs pointed at `install-obgpu.sh`,
   `setup_ob_modern.sh`, `benchmark_ob.py`, and the OBGPU notebook.
5. Treat old notebooks and historical imported models as reference material
   unless they are proven to be on the active path.

## Test Focus

When changing notebook-facing behavior, verify these surfaces:

- single local run
- remote single run
- remote sweep batch
- live sync/final sync
- deferred soma artifact loading
- animation and plotting helpers
- reusable allocation cleanup and manual allocation reuse

The tests should assert default behavior through the public config builders, not
only through private helper implementations.
