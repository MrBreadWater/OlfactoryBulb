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

Tight-refine monitoring:

- Batch 9 was the first batch generated after the tight-best local refinement update. Its plan recorded `local_detail_counts = {"tight_best": 3, "broad_weighted": 3}`.
- Batch 9 completed cleanly on Phoenix step `14537854.2857`.
- Best new candidate: `C00144`, score `2.9806`, ketamine peak `175.781 Hz`, control peak `161.133 Hz`, ketamine target relative power `0.1799`, control target relative power `0.1328`.
- `C00144` is a strong near-miss: it increases ketamine target-band power relative to `C00053`, but the control target-band leakage remains higher than desired.
- Batch 10 launched immediately on Phoenix step `14537854.2891`; its local sources are `["C00053", "C00144", "C00152", "C00147"]`.

Elite rebalancing after batch 10:

- Batch 10 did not improve the archive, suggesting the next useful move is more local exploitation around both `C00053` and `C00144`.
- Updated `propose_elite_batch` so campaigns with at least 128 valid candidates cap the exploration tail at 25% of each batch and draw tight local samples from the top two candidates instead of only the single current best.
- Implementation commit: `7b2fddf`.
- Validation: `source tools/setup/activate_obgpu.sh OBGPU; python -m compileall -q olfactorybulb/hfo_optimizer.py test_hfo_optimizer.py && python test_hfo_optimizer.py`.
- Reloaded `olfactorybulb.hfo_optimizer` in the live authenticated notebook kernel; batch 12 and later should use `local_detail_counts = {"tight_top": ..., "broad_weighted": ...}`.

Targeted probe refinement after batch 11:

- Batch 11 completed cleanly but did not improve on `C00053`; its best new candidate was `C00184` with score `0.8822`.
- Batch 12 is the first batch generated from commit `7b2fddf`; its plan records the top-two local mix: `proposal_counts = {"local": 7, "covariance": 5, "explore": 4}` and `local_detail_counts = {"tight_top": 4, "broad_weighted": 3}`.
- Added a next-stage targeted-probe component for campaigns with at least 192 valid candidates:
  - seeded line probes between the current best and the strongest near-miss
  - small one-coordinate probes around the top two candidates
  - reduced exploration tail to keep the batch concentrated on the promising `C00053`/`C00144` neighborhood while preserving covariance proposals
- Intended batch-13 shape for 16 candidates: `targeted=4`, `local=7`, `covariance=3`, `explore=2`.
- Validation: `source tools/setup/activate_obgpu.sh OBGPU; python -m compileall -q olfactorybulb/hfo_optimizer.py test_hfo_optimizer.py && python test_hfo_optimizer.py`.
- Implementation commit: `fcdd5e4`.
- Reloaded `olfactorybulb.hfo_optimizer` in the live authenticated notebook kernel.
- Batch 12 completed cleanly but did not improve on `C00053`; its best new candidate was `C00201` with score `1.1712`.
- Batch 13 launched from commit `fcdd5e4` on Phoenix step `14537854.2993`. Its plan records `proposal_counts = {"targeted": 4, "local": 7, "covariance": 3, "explore": 2}` and `targeted_detail = {"top_pair": ["C00053", "C00144"], "line_probe_count": 3, "coordinate_probe_count": 1}`.

Coordinate-stencil refinement after batch 13:

- Batch 13 completed cleanly but did not improve on `C00053`; its best new candidate was `C00210` with score `1.2310`.
- `C00210` was essentially an interpolation between `C00053` and `C00144`: it kept ketamine target power near the best candidate but increased control target-band leakage substantially. That argues against continuing to spend many proposals on the C00053-C00144 line.
- Updated the elite proposal policy so campaigns with at least 224 valid candidates switch the targeted component from line probes to a heavier coordinate stencil around the current best candidate. For 16-candidate batches the intended mix is `targeted=8`, `local=5`, `covariance=2`, `explore=1`.
- Validation: `source tools/setup/activate_obgpu.sh OBGPU; python -m compileall -q olfactorybulb/hfo_optimizer.py test_hfo_optimizer.py && python test_hfo_optimizer.py`.
- Implementation commit: `45b33a8`.
- Reloaded `olfactorybulb.hfo_optimizer` in Michael's authenticated live notebook kernel while batch 14 was running.
- Batch 14 completed cleanly but did not improve on `C00053`; its best new candidate was `C00226` with score `1.2310`, ketamine peak `180.664 Hz`, ketamine target relative power `0.1533`, control peak `195.312 Hz`, and control target relative power `0.1455`.
- Batch 15 launched from commit `45b33a8` on Phoenix step `14537854.3061`.
- Batch 15 plan confirmed the coordinate-stencil policy: `proposal_counts = {"targeted": 8, "local": 5, "covariance": 2, "explore": 1}` and `targeted_detail.mode = "stencil"`.

Visible-path and combo-stencil refinement after batch 15:

- Fixed optimizer defaults to prefer the user-facing checkout path `~/OlfactoryBulb` when present. In Michael's live notebook kernel this resolves campaign defaults to `/home/michael/OlfactoryBulb/results/notebook_runs/optimization`, avoiding accidental `/home/alek/...` paths from the symlink target. Implementation commit: `79dd79f`.
- Batch 15 completed cleanly. Its best new candidate was `C00247`, the single-axis AMPA-down probe around `C00053`:
  - ketamine peak: `180.664 Hz`
  - control peak: `195.312 Hz`
  - ketamine target relative power: `0.1608`
  - control target relative power: `0.0835`
  - pair score: `2.9415`
  - changed parameter: `ampa_nmda_gmax=112.468` versus `117.244` in `C00053`
- `C00247` improves target-band delta and control leakage but has worse peak-ratio contrast, so it remains below `C00053`.
- Added a next-stage combo-stencil policy for campaigns with at least 256 valid candidates. It concentrates 10 of 16 proposals on two-knob combinations around the current best, especially AMPA-down plus TC-input/GABA/gap/KAR-weight moves. Implementation commit: `716167e`.
- Validation: `source tools/setup/activate_obgpu.sh OBGPU; python -m compileall -q olfactorybulb/hfo_optimizer.py test_hfo_optimizer.py && python test_hfo_optimizer.py`.
- Reloaded `olfactorybulb.hfo_optimizer` in Michael's authenticated live notebook kernel.
- Batch 16 launched from commit `716167e` on Phoenix step `14537854.3095`. Its plan records `proposal_counts = {"targeted": 10, "local": 4, "covariance": 1, "explore": 1}` and `targeted_detail.mode = "combo"`.

First improved combo candidate:

- Batch 16 completed cleanly and produced a new best candidate, `C00261`, from the GABA-up plus TC-gap-down combo:
  - ketamine peak: `180.664 Hz`
  - control peak: `195.312 Hz`
  - ketamine target relative power: `0.1318`
  - control target relative power: `0.0655`
  - pair score: `3.8415`
  - parameters changed from `C00053`: `gaba_gmax=1.6549` and `gap_tc=10.8407`
- This is the first candidate to beat `C00053` by a substantial margin. It trades away some absolute ketamine target-band power but improves ketamine/control specificity and control leakage.
- Batch 17 launched on Phoenix step `14537854.3129`; its plan uses the combo policy around `C00261` with top pair `["C00261", "C00053"]`.
- Added a micro-refinement stage for campaigns with at least 288 valid candidates. It spends 12 of 16 proposals on smaller local steps around the current best, focused on GABA, TC gap, AMPA, TC input, KAR weight, and GC A-current scale. Implementation commit: `84a8018`.
- Validation: `source tools/setup/activate_obgpu.sh OBGPU; python -m compileall -q olfactorybulb/hfo_optimizer.py test_hfo_optimizer.py && python test_hfo_optimizer.py`.
- Reloaded `olfactorybulb.hfo_optimizer` in Michael's authenticated live notebook kernel; batch 18 and later should use `targeted_detail.mode = "micro"` once the archive reaches 288 valid candidates.

Micro-refinement launch:

- Batch 17 completed cleanly but did not improve the archive. Its best new candidate was `C00275`, score `0.7385`, ketamine peak `195.312 Hz`, and control peak `161.133 Hz`.
- Batch 18 launched on Phoenix step `14537854.3163` from commit `e6b1cc4`.
- Batch 18 plan verified the micro policy:
  - `proposal_counts = {"targeted": 12, "local": 2, "covariance": 1, "explore": 1}`
  - `targeted_detail.mode = "micro"`
  - `targeted_detail.top_pair = ["C00261", "C00053"]`
- Current archive best remains `C00261`, score `3.8415`.
- Batch 18 completed cleanly but did not improve the archive. Its best new candidate was `C00294`, score `2.7377`, with ketamine peak shifted down to `166.016 Hz`, ketamine target relative power `0.1449`, control peak `195.312 Hz`, and control target relative power `0.0676`.
- Batch 19 launched on Phoenix step `14537854.3197` from commit `4180d89`. Its plan continues the micro policy around `C00261` and `C00053`.

Ridge steering for the next stage:

- The micro stage found useful near-miss structure but did not improve the archive: `C00247` retains high ketamine target power at `180.664 Hz`, while `C00294` keeps control leakage low but shifts ketamine peak down to `166.016 Hz`.
- Added a `ridge` proposal mode for campaigns with at least 320 valid candidates. It keeps the top candidate as the main anchor but deliberately proposes small moves around the stronger-power and lower-control-leak near-misses too, instead of repeatedly perturbing only `C00261`.
- For 16-candidate batches the intended post-320 mix is `targeted=11`, `local=3`, `covariance=1`, and `explore=1`.
- Validation: `source tools/setup/activate_obgpu.sh OBGPU; python -m compileall -q olfactorybulb/hfo_optimizer.py test_hfo_optimizer.py && python test_hfo_optimizer.py`.
- Reloaded `olfactorybulb.hfo_optimizer` in Michael's authenticated live notebook kernel before batch 19 finished.
- Batch 19 completed cleanly but did not improve the archive. Its best new candidate was `C00310`, score `2.7377`, matching the same score class as `C00294` with ketamine peak `166.016 Hz` and control peak `195.312 Hz`.
- Batch 20 launched on Phoenix step `14537854.3231` from commit `ccb7535`.
- Batch 20 plan verified the ridge policy:
  - `proposal_counts = {"targeted": 11, "local": 3, "covariance": 1, "explore": 1}`
  - `targeted_detail.mode = "ridge"`

First ridge improvement:

- Batch 20 completed cleanly and produced a new archive best, `C00327`:
  - ketamine peak: `180.664 Hz`
  - control peak: `195.312 Hz`
  - ketamine target relative power: `0.1547`
  - control target relative power: `0.0680`
  - pair score: `4.2544`
  - key parameters: `ampa_nmda_gmax=112.4676`, `gaba_gmax=1.7132`, `gap_tc=10.8407`, `tc_input_weight=0.4268`
- This result supports the ridge interpretation: combining the lower AMPA value from the high-power near-miss with the higher GABA / lower TC-gap setting from `C00261` improved ketamine/control separation while keeping the ketamine peak centered in the requested band.
- Batch 21 launched from commit `8669ab2`; its plan continues ridge refinement around top pair `["C00327", "C00261"]`.
- Batch 21 completed cleanly but did not beat `C00327`. Its best new candidate was `C00343`, score `3.4613`, with ketamine peak `180.664 Hz`, ketamine target relative power `0.1507`, control peak `195.312 Hz`, and control target relative power `0.0858`.

Late trust-region steering:

- Added a post-368-candidate `needle` proposal mode for very small trust-region moves around the current best ridge candidates. The intent is to keep the ketamine peak centered at `180.664 Hz` while testing smaller AMPA/GABA/TC-gap combinations than the broader ridge stage.
- For 16-candidate batches the intended post-368 mix is `targeted=12`, `local=2`, `covariance=1`, and `explore=1`.
- Validation: `source tools/setup/activate_obgpu.sh OBGPU; python -m compileall -q olfactorybulb/hfo_optimizer.py test_hfo_optimizer.py && python test_hfo_optimizer.py`.
- The next step is to reload `olfactorybulb.hfo_optimizer` in Michael's authenticated live notebook kernel before batch 22 finishes, so batch 23 and later can use `targeted_detail.mode = "needle"` once the archive reaches 368 valid candidates.
- Reloaded the live kernel and confirmed batch 23 launched from commit `8f2fcc0` with `targeted_detail.mode = "needle"`.
- Batch 22 completed cleanly but did not improve the archive. Its best new candidate was `C00359`, score `-0.8908`; the important diagnostic is that many upward-GABA / downward-TC-gap ridge probes shifted both ketamine and control to a same-peak `195.312 Hz` solution.
- Revised the `needle` proposal plan to bracket `C00327` more conservatively: test slight GABA reductions, slight AMPA reductions, slight TC-gap increases, and only smaller moves from `C00261`/`C00343` back toward the current best. Validation was repeated with the same compile/test command above.
- Batch 23 launched with the first, pre-bracket needle plan and did not improve the archive. Its best new candidate was `C00372`, score `3.4581`, preserving ketamine `180.664 Hz` but losing target relative power (`0.1404`) and raising control leakage (`0.0742`).
- Batch 24 launched with the bracketed needle plan and also did not improve the archive. Its best new candidate was `C00386`, score `3.3077`, again preserving ketamine `180.664 Hz` but weakening the ketamine/control target-power contrast.
- Added a post-416-candidate `basin` proposal mode. This uses the full candidate archive, not just the top-scoring elite rows, to pick alternative centers by ketamine target power, low control target power, and distance from `C00327`. The intent is to keep the long optimization from over-polishing a local optimum when the local neighborhood is no longer improving.

EPLI weight-scaling audit:

- Batches 32-34 showed an implementation-level diagnostic: paired runs that differed only in `epli_ampa_weight_scale` and/or `epli_gaba_weight_scale` produced byte-identical `lfp.pkl`, `soma_spikes.npz`, `soma_vs.npz`, and `gc_output_events.pkl` outputs.
- Root cause: upstream BlenderNEURON `NeuronNode.create_synapses()` mutates `self.synapse_sets[set_name]` in place and returns `None`. Our wrapper forwarded that `None`, so `load_synapse_set()` could not apply direction-specific reciprocal NetCon weight scaling after the earlier guard fix.
- Fix: `OBNeuronNode.create_synapses()` now returns the stored synapse list from `self.synapse_sets[syn_set["name"]]`. Added a smoke test that monkeypatches the parent method to the same in-place/no-return behavior and asserts that the wrapper returns the populated set, then separately verifies EPLI forward GABA and reciprocal AMPA weight scaling.
