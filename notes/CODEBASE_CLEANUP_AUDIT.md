# Codebase Cleanup Audit

Date: 2026-04-08

This audit separates the actively maintained OBGPU workflow from legacy model
archives, vendored code, generated artifacts, and scripts that appear stale or
experimental.

## Actively Maintained Surface

These files are on the live notebook/run path and should be treated as the
primary cleanup/documentation target.

- `initslice.py`
- `runbatch.py`
- `obgpu_experiment_helpers.py`
- `mc_gc_sweep.py`
- `olfactorybulb/model.py`
- `olfactorybulb/output_paths.py`
- `olfactorybulb/paramsets/`
- `tools/benchmarks/`
- `tools/debug/`
  - Useful, but mostly ad hoc investigation scripts rather than stable APIs.
  - They should be documented as debug probes, not treated as polished library code.
- the working notebooks under `notebooks/` that drive the current OBGPU workflow

These are the files worth deeper type-hinting, docstring work, API cleanup, and
test coverage.

## Legacy / Reference Code

These directories are important reference material, but they should not be
treated as though they are part of the actively maintained application surface.

- `prev_ob_models/`
  - Historical cell-model and fitting code from many prior publications.
  - Still important for mechanisms and isolated cell templates.
  - Most Python files here are effectively archival/reference code.
- `external/`
  - Vendored / patched upstream dependencies, especially NEURON/CoreNEURON.
- `snapshots/`
  - Historical comparison worktrees and parity references.
- `docs/`, `docs-source/`
  - Documentation build outputs and sources.

## Generated / Local-Only Noise

These should stay out of normal cleanup commits and are good `.gitignore`
targets.

- `results/benchmarks/`
- `results/comparisons/`
- `results/notebook_runs/`
- `results/profiles/`
- `results/debug_*/`
- `results/tmp_*/`
- `.jupyter-ai-state/`
- `.codex/`
- `aarch64/`
- `corenrn_data/`
- `*.zip`
- support-lib folders such as `261_22158_220-pycharm-support-libs/`
- `.ipynb_checkpoints/`

## Files That Look Stale, Experimental, or Weakly Integrated

These are not necessarily safe to delete immediately, but they should be
reviewed before spending much cleanup effort on them.

### High-confidence stale / weakly integrated

- `modify_model.py`
- `tools/debug/modify_model.py`
  - Appears experimental and currently broken.
  - Imports `olfactorybulb.parse_topology`, which does not exist in the current tree.
  - Uses `re` without importing it.
  - References globals such as `test_params` that are not defined in the file.
  - Not referenced by the active notebook or batch workflow.

- `testmpi.py`
  - Useful as a small MPI/gap-junction probe, but not part of the normal
    simulation pipeline.
  - Better treated as a debug/example script than a maintained entrypoint.

- `notebooks/cell_current_responses.py`
- `notebooks/cell_gallery.py`
- `notebooks/fitting-GC.py`
- `notebooks/fitting.py`
  - Legacy script-style analysis/fitting helpers.
  - Mentioned in older docs, but not on the active OBGPU workflow path.

### Low-confidence / manual-use only

- `build-slice.py`
  - Still referenced by the slice-recreation docs.
  - Probably intentionally manual and infrequently used.
  - Should be documented, but not treated like a hot path.

- `example_starter.py`
  - A small example/teaching script.
  - Useful, but not core to the simulation pipeline.

## Recommended Next Cleanup Steps

1. Keep documenting the active maintained surface first.
2. Move or clearly label stale scripts under a dedicated `tools/legacy/` or
   `archive/` area once they are confirmed unused.
3. Avoid large “cleanup” edits inside `prev_ob_models/` unless there is a
   concrete reason; that tree is better treated as archived/reference code.
4. Add lightweight tests around:
   - notebook run metadata persistence
   - output-path naming
   - run loading / summary helpers
   - GC output event analysis helpers
5. Consider splitting `obgpu_experiment_helpers.py` into smaller modules:
   - run launching/loading
   - parameter/introspection helpers
   - signal processing
   - plotting/animation

## Cleanup Principle

The goal should be to make the maintained OBGPU workflow clean and readable
without pretending the entire historical repository is equally active.
