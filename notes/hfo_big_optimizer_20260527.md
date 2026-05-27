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
- Implementation commit: `a8e70662c48a`.
- Reloaded `olfactorybulb.hfo_optimizer` in Michael's authenticated live kernel and rescored the active campaign archive before elite refinement.
- Batch 3 then found a substantially better provisional candidate, `C00053`:
  - ketamine peak: `180.664 Hz`
  - control peak: `195.312 Hz`
  - ketamine target relative power: `0.1565`
  - control target relative power: `0.0885`
  - pair score: `3.0511`
  - parameters: `kar_mt_gmax=0.0351`, `kar_gc_gmax=0.0853`, `gaba_gmax=1.544`, `ampa_nmda_gmax=117.244`, `gap_tc=11.781`, `gap_mc=31.640`, `tc_input_weight=0.427`, `mc_input_weight=0.344`
- Batch 4 launched with remote commit `a8e70662c48a`, confirming the corrected scoring code is in the remote run lineage.

Elite proposal correction after batch 5:

- Batch 6 was the first elite-refine batch. The original elite proposal used the mean/covariance of the top 25% of all candidates; with 96 completed candidates this meant 24 elite sources, which was too broad and diluted `C00053`.
- Updated `propose_elite_batch` to cap the covariance elite set at 12 and to allocate each elite-refine batch into:
  - rank-weighted local proposals around the top 4 candidates
  - a smaller covariance proposal across the capped elite set
  - a retained Latin-hypercube exploration tail
- Validation:
  - `source tools/setup/activate_obgpu.sh OBGPU; python -m compileall -q olfactorybulb/hfo_optimizer.py test_hfo_optimizer.py`
  - `source tools/setup/activate_obgpu.sh OBGPU; python test_hfo_optimizer.py`

Focused-refine monitoring:

- Implementation commit: `442901eb08fc`.
- Batch 7 was the first batch using the focused proposal mix. Its plan recorded `proposal_counts = {"local": 6, "covariance": 4, "explore": 6}`, `local_source_ids = ["C00053", "C00093", "C00103", "C00072"]`, and a capped 12-candidate elite source set.
- Batch 7 completed cleanly on Phoenix step `14537854.2789`; its best new candidate was `C00116` with score `0.9797`, so it did not displace `C00053`.
- Batch 8 launched immediately on commit `442901eb08fc` as Phoenix step `14537854.2823`; the campaign remains active in Michael's authenticated notebook kernel.
- Because the first focused-refine batch still did not improve on `C00053`, the next proposal update splits local elite proposals into tight-best samples around the current best candidate and broader weighted samples around the top 4 candidates.
- Implementation commit: `a03fc77`.
- Validation: `source tools/setup/activate_obgpu.sh OBGPU; python -m compileall -q olfactorybulb/hfo_optimizer.py test_hfo_optimizer.py && python test_hfo_optimizer.py`.
- Reloaded `olfactorybulb.hfo_optimizer` in the live authenticated notebook kernel; future locally generated batch plans should include `local_detail_counts`.
