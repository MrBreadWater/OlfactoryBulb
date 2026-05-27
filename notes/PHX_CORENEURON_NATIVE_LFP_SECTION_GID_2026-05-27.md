CoreNEURON Phoenix failure follow-up (2026-05-27)

Problem
- Remote and local CoreNEURON runs for `GammaSignature_EPLI_Provisional_TCOnly` were still failing with:
  - `Can't associate gid ... PreSyn already associated with gid ...`
- After fixing shared-node synapse gid aliasing, the remaining failure only reproduced when the notebook overrides kept `enable_lfp=true`.

Root cause
- Native LFP report gid reuse was too coarse.
- `get_cell_report_gid(...)` reused any gid previously remembered for the same cell name.
- That is unsafe because the existing gid may belong to a different registered source on that cell than the soma section used for native LFP reporting.
- Under CoreNEURON this led to a second `ParallelContext.cell(...)` registration against a source that NEURON already owned.

Fix
- Track gids by local source section name, not only by cell name.
- Reuse an existing gid for native LFP only when the exact soma section already has a registered source gid.
- Otherwise allocate a fresh high report gid.
- Also remember local source section gids while loading chemical synapse sets, reciprocal GC KAR sets, and gap-junction sources so the soma lookup sees the already-registered section ownership.

Files changed
- `olfactorybulb/model.py`
- `test_corenrn_native_lfp_gid_reuse.py`

Verification
1. Rebuilt NEURON/CoreNEURON mechanisms locally:
   - `nrnivmodl -coreneuron prev_ob_models/Birgiolas2020/Mechanisms`
2. Ran the exact failing benchmark path locally with MPI, CoreNEURON, and notebook overrides:
   - `mpiexec --oversubscribe -n 16 nrniv -mpi -python tools/benchmarks/benchmark_ob.py ... --overrides-file results/notebook_runs/obgpu_experiment_GammaSignature_EPLI_Provisional_TCOnly_fast_20260527_031710/overrides.json --tstop-override 50.0 --coreneuron`
3. Result:
   - initialization passed
   - simulation ran
   - output files were written successfully

Observed successful run
- label: `codex_local_tc_only_probe2_20260527_033051`
- ranks: `16`
- cells:
  - `MC=10`
  - `TC=24`
  - `GC=159`
  - `EPLI=24`
- outputs included:
  - `lfp.pkl`
  - `soma_spikes.npz`
  - `soma_vs.npz`
  - `voltage_summary.npz`
  - `gc_output_events.pkl`

Conclusion
- The original CoreNEURON gid-registration crash is fixed in the actual simulation path.
- Any remaining Phoenix failure after this point should be treated as a new issue, not the old gid bug.
