# HFO Big Optimizer Run - 2026-05-27

Active run launched from Michael's authenticated Jupyter kernel:

- Kernel connection: `/home/michael/.local/share/jupyter/runtime/kernel-300768a3-e058-4c54-a4aa-8b6496fa4c37.json`
- Kernel PID: `441223`
- Live Paramiko cache key: `jmpaniag@localhost:2223`
- Manual Phoenix allocation: `14537854`
- First remote sweep step: `14537854.1301`
- Remote nodes reported: `pcc[080-082]`
- Code commit used by the remote run: `c6dbe290820cfb794f21d43b3d8dc81b18cca1e1`

Campaign:

- Campaign dir: `/home/michael/OlfactoryBulb/results/notebook_runs/optimization/hfo_epli_big_120cpu_20260527_061046`
- Runtime log: `/home/michael/OlfactoryBulb/results/notebook_runs/optimization/codex_big_hfo_logs/big_hfo_optimizer_20260527_061046.log`
- Status JSON: `/home/michael/OlfactoryBulb/results/notebook_runs/optimization/codex_big_hfo_logs/latest_big_hfo_optimizer_status.json`

Resource plan:

- `nranks = 15`
- `slurm_step_ntasks = 15`
- `sweep_parallelism = 8`
- Effective target occupancy: `15 * 8 = 120` CPU tasks
- Each batch: `16` candidates, paired control plus ketamine conditions, so `32` simulation items per batch
- Planned batches: `96`

Run intent:

- Optimize conductance and drive parameters for a clean ketamine-specific target HFO band near `180 +/- 20 Hz`.
- Score paired control and ketamine runs so target-band power should stand out under ketamine and remain weaker in control.
- Keep time constants fixed; vary max conductances and feedforward/gap coupling knobs from `olfactorybulb.hfo_optimizer.default_hfo_search_space()`.

Startup checks:

- The first attempt failed before remote submission because the joint sweep label exceeded filesystem path length.
- Fixed by hashing long sweep labels in `_safe_sweep_path_label`.
- `python test_config_helpers.py` passed after the fix.
- The restarted run submitted successfully and progressed past the prior failure point, showing remote sweep status updates and completed item counts.

Correction during monitoring:

- The initial `hfo_epli_big_120cpu_20260527_061046` campaign revealed a scoring bug: missing/failed control runs could create artificially huge ketamine-control contrast scores.
- Fixed in commit `c08ee7722d3d7cb0f1cbfe623ec54d8ed0b137a2`: incomplete candidate pairs now receive `pair_score = -inf`.
- Stopped the stale worker and canceled its active remote step `14537854.1980`.
- Active corrected campaign:
  - Campaign dir: `/home/michael/OlfactoryBulb/results/notebook_runs/optimization/hfo_epli_big_fixedscore_120cpu_20260527_073229`
  - Runtime log: `/home/michael/OlfactoryBulb/results/notebook_runs/optimization/codex_big_hfo_logs/big_hfo_optimizer_20260527_073229.log`
  - First corrected remote sweep step: `14537854.2025`
  - Commit used by the remote run: `c08ee7722d3d7cb0f1cbfe623ec54d8ed0b137a2`

Second correction during monitoring:

- The fixed-score campaign still failed all items in batch 0 with `AttributeError: module 'olfactorybulb.model' has no attribute 'GammaSignature_EPLI_Provisional_TCOnly'`.
- Direct compute-node probes through the authenticated kernel showed the paramset was present under both `python` and `nrniv -mpi -python` at commit `c08ee77`.
- The real failure was a remote sweep wrapper race: the sweep driver wrapper was launched with `15` Slurm tasks, so 15 copies of the wrapper entered bootstrap concurrently. Some copies launched item simulations while another copy was still moving the shared checkout from `5cf41c3` to `c08ee77`.
- Fixed locally by making explicit `step_ntasks` exact for reusable-allocation wrapper submission and using `step_ntasks=1` for the remote sweep driver. The driver still launches each simulation item with `nranks=15`, and the sweep keeps `sweep_parallelism=8`, so intended simulation occupancy remains `15 * 8 = 120` CPU tasks.
- Validation:
  - `source tools/setup/activate_obgpu.sh OBGPU; python -m compileall -q obgpu_experiment_helpers.py test_config_helpers.py`
  - `source tools/setup/activate_obgpu.sh OBGPU; python test_config_helpers.py`
  - `source tools/setup/activate_obgpu.sh OBGPU; python test_hfo_optimizer.py`

Next campaign launch plan:

- Campaign dir: `/home/michael/OlfactoryBulb/results/notebook_runs/optimization/hfo_epli_big_singlewrapper_120cpu_20260527_082225`
- Runtime log: `/home/michael/OlfactoryBulb/results/notebook_runs/optimization/codex_big_hfo_logs/big_hfo_optimizer_20260527_082225.log`
- Status JSON: `/home/michael/OlfactoryBulb/results/notebook_runs/optimization/codex_big_hfo_logs/latest_big_hfo_optimizer_status.json`
- Code fix commit included in launch lineage: `8c5c78c`
- Launch git ref will be resolved from local `HEAD` immediately before the campaign starts.
- The live kernel must reload `obgpu_experiment_helpers` and `olfactorybulb.hfo_optimizer` before launching because an earlier diagnostic monkey-patched `score_hfo_batch` to stop the stale worker.

Scoring correction after batch 0:

- Batch 0 completed cleanly and confirmed the remote sweep-driver fix, but the best candidates were not ketamine-specific: several had the same 160-190 Hz peak in both control and ketamine.
- Tightened `score_candidate_pair` so target-band power in control is treated as leakage, same target-band peak frequency in control/ketamine is penalized, and positive ketamine-control target-band delta is rewarded explicitly.
- Rewrote the active campaign's batch-0 `candidate_archive.jsonl` and `batch_0000_scored.json` with the tightened formula so elite refinement will not inherit stale nonspecific scores. After rescoring, the top batch-0 candidates were ketamine-shifted cases rather than same-peak control/ketamine cases.
- Validation: `source tools/setup/activate_obgpu.sh OBGPU; python test_hfo_optimizer.py`.

Scoring correction after batch 2:

- Batch 1 and batch 2 again showed that the objective could still overvalue nonspecific target-band activity, especially candidates with the same target-band peak in control and ketamine but a moderate ketamine increase.
- Updated `score_candidate_pair` to score the paired phenotype more directly:
  - reward compound ketamine/control contrast in `target_hfo relative power * target peak ratio`
  - reward positive ketamine-control target-band delta and penalize negative delta
  - penalize control target-band leakage more strongly
  - penalize same-bin target peaks in control and ketamine as a function of control target-band power
  - add an explicit ketamine peak frequency match centered at 180 Hz
- This should keep the wide-seed batches useful while preventing the elite-refine stage from exploiting a rhythm that is already present in control.
