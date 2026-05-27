# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A biophysically realistic computational model of the olfactory bulb (OB) network, built on NEURON/CoreNEURON with Python. The network models three cell types — mitral cells (MC), tufted cells (TC), and granule cells (GC) — with reciprocal dendrodendritic synapses, gap junctions, and odor-driven OSN spike inputs. Simulations run locally via MPI or remotely on Slurm/GPU clusters.

## Running Simulations

**Active conda environment**: `OBGPU`

**Maintained benchmark run (CPU)**:
```bash
mpiexec -n 4 nrniv -mpi -python tools/benchmarks/benchmark_ob.py --label gamma_cpu --paramset GammaSignature
```

**Maintained benchmark run (GPU/CoreNEURON)**:
```bash
mpiexec -n 1 nrniv -mpi -python tools/benchmarks/benchmark_ob.py --label gamma_gpu --paramset GammaSignature --coreneuron --coreneuron-gpu
```

**Legacy compatibility entrypoints**:
```bash
mpiexec -n 4 nrniv -mpi -python initslice.py -paramset GammaSignature -mpi
python runbatch.py
```

**Notebook**: Open `notebooks/obgpu-working-experiment.ipynb` — this is the primary interactive interface and the one actively maintained.

## Key Environment Variables

| Variable | Purpose |
|---|---|
| `OB_USE_CORENRN` | Enable CoreNEURON |
| `OB_USE_CORENRN_GPU` | Enable GPU mode (implies CoreNEURON) |
| `OB_RUNTIME_MODE` | `scientific` or `exploratory` |
| `OB_RESULT_LABEL` | Override result folder name |
| `OB_RESULTS_BASE` | Override base results directory |
| `OB_MPI_RANKS` | Override MPI rank count in `runbatch.py` |
| `OBGPU_STATUS_MODE` | `stdout`, `file`, or `off` |
| `OBGPU_STATUS_INTERVAL_MS` | Status reporting interval in ms |

## Architecture

### Simulation Core (`olfactorybulb/model.py`)

`OlfactoryBulb` is the main class. Construction runs the full build→simulate→save pipeline when `autorun=True` (the default). The key steps:

1. Loads cell JSON groups from the slice directory (e.g. `olfactorybulb/slices/DorsalColumnSlice/`)
2. Distributes cells across MPI ranks using a min-heap complexity balancer
3. Uses BlenderNEURON (`OBNeuronNode`) to reconstruct morphology and synaptic wiring from JSON
4. Loads GC↔MC and GC↔TC synapse sets (reciprocal AMPA/NMDA + GABA)
5. Adds tuft gap junctions between co-glomerular MCs and TCs
6. Schedules odor input: Gaussian spike trains → `Exp2Syn` on tuft segments
7. Sets up LFP electrode (LFPsimpy, or native CoreNEURON LFP for GPU runs)
8. Runs via `h.run()` (single rank) or `pc.psolve()` (multi-rank/GJ/CoreNEURON)
9. Saves `soma_vs.pkl`, `lfp.pkl`, `input_times.pkl`, `gc_output_events.pkl` to a timestamped results directory

### Parameter Sets (`olfactorybulb/paramsets/`)

- `base.py` — `SilentNetwork`/`ParameterSetBase`: all default values (silent network, odor inputs only)
- `case_studies.py` — `GammaSignature` and its variants (NoInhibition, NoTCGJs, etc.)
- `sensitivity.py` — parameter sensitivity sweeps

A paramset is a plain Python class with class-level attributes. To override at runtime, subclass or use `extra_overrides` in the notebook.

### Notebook Helpers (`obgpu_experiment_helpers.py`)

The thick convenience layer for interactive use. Key functions:

- `build_run_config(...)` — constructs the config dict for a run
- `run_simulation(config)` / `run_and_load(config)` — launches subprocess, polls, downloads results
- `run_parameter_sweep(config, path, values)` — sweeps one config key over a list of values
- `load_result(run)` — loads result artifacts, including lazy/deferred soma trace payloads when configured
- `animate_*_sweep(sweep, ...)` — matplotlib animation wrappers for sweeps; saves GIFs
- `save_animation(anim, name)` / `save_figure(name, ...)` — persist outputs next to results
- `plot_*` family — individual plot functions (voltage traces, spike rasters, LFP overview, spectrograms, wavelets, GC output, input overview)
- `build_sol_remote_config(...)` / `build_slurm_remote_config(...)` — remote Slurm backends

Remote notebook runs are Paramiko-only. Do not reintroduce the removed OpenSSH
control-master or rsync transport path; it caused repeated authentication and
stale socket failures in sweeps.

### Slice Data (`olfactorybulb/slices/DorsalColumnSlice/`)

JSON files that define the entire network topology:
- `MCs.json`, `TCs.json`, `GCs.json` — cell morphology roots with BlenderNEURON group format
- `GCs__MCs.json`, `GCs__TCs.json` — reciprocal synapse sets
- `glom_cells.json` — maps glomerulus IDs → list of cells attached to each glomerulus

### Database (`olfactorybulb/database.py`, `olfactorybulb/model-data.sqlite`)

Peewee ORM for the SQLite database containing:
- `Odor` + `OdorGlom` — odor→glomerulus intensity mappings (used by `add_inputs()`)
- `CellModel` — cell model metadata including `tufted_dend_root` section names (used to target input synapses)

### Results Layout

Every run writes to `results/notebook_runs/<label>_<YYYYMMDD_HHMMSS>/`:
- `soma_vs.pkl` — list of `(cell_label, t_list, v_list)` tuples
- `lfp.pkl` — `(t_list, lfp_list)` in nV
- `input_times.pkl` — list of `(seg_name, spike_times)` tuples
- `gc_output_events.pkl` — list of dicts with GC→MC/TC GABA event metadata + times
- `run_info.json` — simulation metadata
- `notebook_run_info.json` — notebook-level config snapshot
- `sim_progress.json` — live progress polling file

### Model Modification (`modify_model.py`)

Utilities to add/modify synaptic connections and swap cell types on a live `OlfactoryBulb` instance. Accessed from the notebook via `add_connections`, `modify_connections`, and `swap_cell_types` config keys.

## Compiling NEURON Mechanisms

If maintained Birgiolas `.mod` files change, rerun setup or recompile the
mechanisms from the repo root:
```bash
nrnivmodl -coreneuron prev_ob_models/Birgiolas2020/Mechanisms
```
This regenerates the local architecture directory (`x86_64/`, `aarch64/`, or
the configured mechanism cache). Generated `.c`, `.o`, `.dll`, `x86_64/`, and
`aarch64/` outputs should not be committed.

## Important Constraints

- **CoreNEURON requires all synaptic sources to use gid-based NetCons** — `OBNeuronNode.force_gid_synapses = True` enforces this. Do not mix direct voltage-source NetCons with gid-based ones in the same parallel run.
- **Gap junctions require `pc.psolve()` even on a single rank** — they use ParallelContext transfer variables.
- **LFP electrode creation is deferred until `run()`** to avoid implicit `h.init()` before ParallelContext transfer state is set up.
- **Input odors are resolved through the SQLite database** — available odors are `Apple`, `Coffee`, `Mint` (and others in the DB). Custom spike trains bypass the DB entirely via `input_event_strategy`.
