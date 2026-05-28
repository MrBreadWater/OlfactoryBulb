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

Objective and LFP-source correction after visual audit:

- The current best plot pack for `C00327` did not satisfy the visual criterion despite ranking highest under the old objective. The failure modes were: weak/ugly HFO separation, a visible control spectrogram line just above the old HFO band, and afferent input events ending near 1.9 s while the run continued to 9 s.
- Changed the optimizer objective from a narrow peak-centered 180 +/- 20 Hz score to integrated power density across 160-230 Hz. The pair score now rewards ketamine/control total target-band contrast and clean target-band concentration, while penalizing control target power, control above-target power, ketamine above-target leakage, missing MC/TC/EPLI/PVCRH spike support, and input dropout.
- `default_campaign_run_config()` now emits a sustained 200 ms odor schedule covering the full optimization `tstop`, with `inhale_duration_ms=125`. This prevents the optimizer from selecting late LFP artifacts that persist after the input has stopped.
- Added `lfp_include_cell_types` and `lfp_exclude_cell_types` as model/runtime controls. These change LFP source registration without changing the circuit, so `lfp_exclude_cell_types=["GC"]` can test whether the apparent HFO is generated primarily by GC currents. Added `lfp_source_diagnostic_configs()` to build all-source, GC-excluded, non-GC-only, and shifted-probe run configs from the same candidate.
- Validation: `source tools/setup/activate_obgpu.sh OBGPU; python -m compileall -q olfactorybulb/hfo_optimizer.py obgpu_experiment_helpers.py olfactorybulb/model.py olfactorybulb/paramsets/base.py tools/benchmarks/benchmark_ob.py test_hfo_optimizer.py test_corenrn_native_lfp_gid_reuse.py && python test_hfo_optimizer.py && MPLCONFIGDIR=/tmp/mpl python test_corenrn_native_lfp_gid_reuse.py`.
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
- Search-space correction: restored `kar_osn_weight_scale`, `kar_gc_weight_scale`, and `gc_ka_gbar_scale` to the default optimizer dimensions alongside the new EPLI scales. The active campaign was originally seeded with those three dimensions, and its best candidate depends on them; dropping them made later batches search a reduced basin with defaults instead of the actual archive optimum.

Duplicate-proposal correction:

- The live notebook was briefly pointed at a mistyped local commit hash (`35bbcb450...` instead of the actual `35bbcb40...`), causing three quick bundle-ref failures before any simulations launched. Corrected the live kernel globals and restarted the authenticated optimizer worker from the same campaign. Batch 39 then launched and completed under `35bbcb40a65c36dc1631b5b2824895c89da33320`.
- Batch 39 verified that EPLI scaling is now functional: probes with different `epli_ampa_weight_scale` and `epli_gaba_weight_scale` no longer collapse to identical scores. Best new result from that batch was `C00626`, score `3.1880`, with ketamine peak `180.664 Hz`, control peak `195.312 Hz`, `epli_ampa_weight_scale=0.496`, and `epli_gaba_weight_scale=1.0`.
- Batches 40-41 revealed a search-efficiency issue: the late `frontier` proposer deterministically regenerated the same coordinate probes around `C00327`/`C00261` in each batch once those remained the top pair.
- Added archive-level duplicate suppression in `propose_elite_batch`. Candidate rows are compared in encoded parameter space against the scored archive; duplicates are dropped and refilled with jittered local samples around elite centers plus fallback uniform exploration. The batch plan records `targeted_detail.archive_duplicate_rows_dropped` so future audits can see when the refill path was used.
- Follow-up refinement: duplicate structured probes now first become small jittered variants of the intended probe before falling back to broader elite-centered refill. This keeps late-stage batches aligned with the frontier direction instead of replacing every exhausted EPLI stencil with unrelated local samples.
- Batch 46 surfaced a different high-power basin (`C00745`) with ketamine peak `185.547 Hz` and target relative power `0.2412`, but with excessive control target power (`0.1774`). The frontier `power` center previously excluded that row with a hard `control <= 0.16` gate. Widened the gate to `control <= 0.20` so late batches can try to reduce leakage around this stronger ketamine-power basin instead of only polishing `C00327`.
- Validation: `source tools/setup/activate_obgpu.sh OBGPU; python -m compileall -q olfactorybulb/hfo_optimizer.py test_hfo_optimizer.py && python test_hfo_optimizer.py`.

GC-excluded LFP pivot after source audit:

- After visual review, the clearest spectrogram lines appeared to persist even when MC, TC, and EPLI spike activity fell off. That makes a GC-dominated LFP source a plausible artifact path.
- Runtime LFP source filtering was already implemented in `olfactorybulb.model` and exposed through notebook/benchmark config as `lfp_include_cell_types` and `lfp_exclude_cell_types`.
- While batch 50 was already running, the authenticated live notebook kernel globals were patched for subsequent batches:
  - `remote_git_ref = 739d8dff813749df1d23659c4c99b39961e9efb9`
  - `lfp_include_cell_types = None`
  - `lfp_exclude_cell_types = ["GC"]`
- Batch 50 itself remains an all-source LFP batch because its run config had already been copied into the remote sweep. Batch 51 and later should score the same circuit with GC sections excluded from LFP registration, testing whether candidate HFOs survive without GC current dominating the LFP proxy.

Scoring/objective consistency correction:

- Inspecting the live worker source showed the notebook loop still called `score_hfo_batch(... target_hz=180.0, target_half_width_hz=20.0)`, even after the objective had moved to integrated `target_hfo = 160-230 Hz`.
- Fixed `score_condition_result()` so the named `target_hfo` band is authoritative whenever it is present in the configured score bands. This prevents stale notebook arguments from silently narrowing the target mask back to `160-200 Hz`.
- Added a campaign-local `objective_filter.json` mechanism so an objective pivot can exclude earlier archive rows from ranking/proposal while preserving the old data on disk.
- Reloaded the fixed module into Michael's authenticated live kernel while batch 52 was running and wrote:
  - `/home/michael/OlfactoryBulb/results/notebook_runs/optimization/hfo_epli_big_singlewrapper_120cpu_20260527_082225/objective_filter.json`
  - `min_batch_index = 52`
  - `lfp_exclude_cell_types = ["GC"]`
  - `target_hfo_hz = [160.0, 230.0]`
- Live-kernel smoke check confirmed that a 220 Hz synthetic signal still scores inside `target_band_hz = [160.0, 230.0]` even when stale `target_hz=180.0, target_half_width_hz=20.0` arguments are passed.
- Implementation commit: `13488a8`.

Iteration-speed audit:

- Completed GC-excluded batches sync about `98 MB` of compact artifacts for 32 paired items.
- Largest synced artifacts in batch 52:
  - `lfp.pkl`: about `52 MB`
  - `gc_output_events.pkl`: about `39 MB`
  - `soma_spikes.npz`: about `3.7 MB`
  - `input_times.pkl`: about `1.4 MB`
- HFO optimizer scoring uses LFP, soma spikes, input times, and summaries. It does not use GC output events.
- Changed `default_campaign_run_config()` for optimizer campaigns to set `record_gc_output_events=False`. Future batches launched from this config should stop generating/syncing `gc_output_events.pkl`, cutting roughly `40%` of the compact sync payload.
- Reloaded the live kernel and patched `CODEX_HFO_BIG_BASE_CONFIG["record_gc_output_events"] = False`; current batch 53 was already launched, so this should take effect from batch 54 onward.
- Implementation commit: `f35747c`.
- Item summaries show remote save time around `18.5 s` per simulation item. The remote run still wrote raw soma traces and voltage-summary arrays even though the optimizer only needs `soma_spikes.npz`.
- Added non-breaking controls:
  - `save_soma_traces`
  - `save_voltage_summary`
- Defaults remain `True` for ordinary notebook runs, but `default_campaign_run_config()` now sets both to `False` for optimizer batches. Spike detection is still saved before those optional writes are skipped.
- Implementation commit: `979f1b8`.
- LFP recording quality was left unchanged at the existing `recording_period_ms = 0.1`. A quick downsample/rescore probe showed that coarse LFP sampling can move peak-bin identity, so LFP decimation should only be considered after a dedicated A/B validation batch.
- Batch 54 was verified in the live worker stack with:
  - `record_gc_output_events = False`
  - `save_soma_traces = False`
  - `save_voltage_summary = False`
  - `recording_period_ms = 0.1`
  - `analysis_dt_ms = 0.1`
- Batch 54 completed cleanly with `32/32` successful remote items. Local compact payload dropped from about `93 MB` in batch 53 to about `58 MB` in batch 54, with `lfp.pkl` still present at the unchanged 0.1 ms sampling period.
- Batch 54 did not beat the current filtered leaders. Its best row was `C00870`, score `-2.8279`, ketamine peak `205.078 Hz`, control peak `209.961 Hz`, ketamine target relative power `0.1644`, and control target relative power `0.1584`. The high control leakage means this is not a useful improvement.
- Batch 55 launched after batch 54 with the same lean artifact settings and GC-excluded LFP. It is still using simulation commit `a4111f2`; the later `bde6212` commit only touched notes, so this does not affect simulation behavior.
- Batch 55 produced a new numerical leader, `C00884`: score `-0.0714`, ketamine peak `214.844 Hz`, control peak `219.727 Hz`, ketamine target relative power `0.1984`, and control target relative power `0.1397`. This is promising by the objective but still not clean because the control condition has visible target-band power and a nearby peak.
- Generated local diagnostic figures, including 2D spike-frequency KDEs, at `results/notebook_runs/optimization/hfo_epli_big_singlewrapper_120cpu_20260527_082225/figures/best_C00884_20260527_194143`.
- C00884 revealed a scoring bug: the same-peak penalty still used the old hard-coded `160-200 Hz` window even though the current objective target is `160-230 Hz`. Fixed pair scoring so same-peak contamination uses the recorded `target_band_hz`, added an explicit excess-control-target penalty above `0.12` relative target-band power, and bumped `PAIR_SCORE_VERSION` to `3`.
- Reloaded the fixed scorer into Michael's live authenticated kernel while batch 56 was still running, before local scoring. Future submissions were pointed at commit `ea89e28`; LFP sampling and source filtering were left unchanged.
- Batch 56 was scored with `PAIR_SCORE_VERSION = 3` and produced `C00906` as the new leader: score `0.3206`, ketamine peak `214.844 Hz`, control peak `161.133 Hz`, ketamine target relative power `0.1850`, control target relative power `0.1299`, same-peak penalty `0`, and excess-control penalty `0.348`. This is the cleanest leader so far by the corrected objective, though the control target-band power is still not negligible.
- Generated local diagnostic figures for `C00906` at `results/notebook_runs/optimization/hfo_epli_big_singlewrapper_120cpu_20260527_082225/figures/best_C00906_20260527_195525`.
- Batch 57 improved the corrected objective. New leader `C00914`: score `2.1736`, ketamine peak `219.727 Hz`, control peak `229.492 Hz`, ketamine target relative power `0.1861`, control target relative power `0.1142`, same-peak penalty `0`, and excess-control penalty `0`. This is much closer to the desired ketamine-specific HFO regime, but the control condition still has a peak at the upper edge of the target band.
- Generated local diagnostic figures for `C00914` at `results/notebook_runs/optimization/hfo_epli_big_singlewrapper_120cpu_20260527_082225/figures/best_C00914_20260527_200740`.
- Biological plausibility audit: `C00914` depends on `kar_mt_gmax = 276.07 uS` and `kar_osn_weight_scale = 5.20`, which conflicts with `notes/EFFECTIVE_CONDUCTANCE_SCALING.md` where the working MT KAR range is `0.01..0.05 uS` with `0.03 uS` as the representative value. Treating C00914 as a valid biological solution would therefore repeat the implausibly huge-excitatory-drive failure mode.
- Added a soft KAR plausibility penalty and narrowed future default KAR search bounds: `kar_mt_gmax = 0.01..0.08`, `kar_gc_gmax = 0.001..0.025`, `kar_osn_weight_scale = 0.25..2.0`, and `kar_gc_weight_scale = 0.25..4.0`. The live kernel was reloaded at commit `4767acf`; future scoring and proposals should use these constraints while leaving LFP quality unchanged.
- Batch 58 was launched before the narrowed proposal bounds took effect, so it still contained broad-KAR candidates; rescoring demoted those rows through the plausibility penalty. The strongest plausible rows after this correction are weaker than the huge-KAR solutions, which is the expected result if the previous clean bands were mostly an unphysiological drive artifact.
- Batch 59 plan inspection confirmed the live proposer is now constrained: `kar_mt_gmax = 0.0105..0.0637`, `kar_gc_gmax = 0.00126..0.0220`, `kar_osn_weight_scale = 0.25..1.99`, and `kar_gc_weight_scale = 0.255..3.18`. Raw LFP and analysis sampling remain unchanged at `0.1 ms`; any future decimation should be an offline A/B analysis copy, not a simulation-quality default.
- Batch 59 completed `32/32` remote items with no failures and did not improve the plausible leader. Best new row was `C00945`, score `-0.8993`, ketamine peak `175.781 Hz`, control peak `161.133 Hz`, ketamine target relative power `0.0835`, and control target relative power `0.0790`. The campaign leader remains `C00837` at score `-0.8685`.
- Batch 60 completed `32/32` remote items with no failures and found a new constrained/plausible numerical leader, `C00964`: score `-0.2171`, ketamine peak `161.133 Hz`, control peak `185.547 Hz`, ketamine target relative power `0.0995`, and control target relative power `0.0842`. This improves the objective but is not yet a convincing target-regime solution because the ketamine peak is pinned to the lower target-band edge while control has the more central HFO-band peak.
- Scoring correction after batch 60: the scorer already computed a peak-based frequency match but did not use it, allowing lower-edge `160 Hz` target-band power to win despite the intended HFO center being closer to `180-200 Hz`. Added a target-band centroid match so the objective still uses integrated `160-230 Hz` power but penalizes edge-dominated target-band solutions and cases where control is more centrally HFO-like than ketamine. Reloaded the live kernel at `PAIR_SCORE_VERSION = 4`; with the corrected score, `C00837` is again the plausible leader (`-0.6393`) and `C00964` drops to `-0.9363`.
- Batch 61 completed `32/32` remote items with no failures. Its best new row was `C00987`, score `-1.0301`, so it did not beat the corrected archive leader. Generated the post-batch-61 diagnostic graph set for current best `C00837` at `results/notebook_runs/optimization/hfo_epli_big_singlewrapper_120cpu_20260527_082225/figures/best_C00837_after_batch61_20260527_210934`, including PSDs, spectrograms, rasters, MT/EPLI/GC 2D spike-frequency KDEs for both conditions, input plots, a target-band phase histogram, manifest, and contact sheet.
- Visual review of the post-batch-61 graph set showed two objective failures that were not sufficiently represented in `PAIR_SCORE_VERSION = 4`: the target-band line was faint relative to broadband background, and the ketamine condition had no EPLI spikes. Added actual target-band peak contrast against target/shoulder/background floors, added an EPLI silence penalty, and penalized ketamine EPLI dropout relative to control. The scorer is now `PAIR_SCORE_VERSION = 5`; `python test_hfo_optimizer.py` covers low-contrast and silent-EPLI demotion.
- Reloaded Michael's authenticated live kernel with scorer version 5 at commit `3d3052f`. Under the corrected objective, silent-EPLI rows such as `C00837` and `C00964` are demoted. The current plausible archive leaders (`C00954`, `C01000`, `C00989`, and neighbors) have active ketamine EPLI rates of about `3-7 Hz`, but their absolute peak contrast remains weak, confirming that the previous candidate was not a convincing HFO solution.
- Batch 63 was already planned under the prior objective before the scorer reload, so it may be partially stale. At `2026-05-27 21:26 EDT`, it was still healthy on Phoenix with `12/32` paired items complete, `8` running, `12` pending, and `0` failures. Batch 64 should be checked to verify that proposal selection is using the version-5 archive leaders rather than the old silent-EPLI rows.
- Generated a version-5 visual baseline for the corrected archive leader `C00954` at `results/notebook_runs/optimization/hfo_epli_big_singlewrapper_120cpu_20260527_082225/figures/best_v5_C00954_20260527_212935`. This candidate has active ketamine EPLI (`6.21 Hz`) but weak target-line contrast (`0.238` ketamine vs `0.213` control) and only a small target relative-power lift (`0.074` vs `0.066`), so it is a better negative/weak baseline than a final solution.
- Proposer correction for post-v5 batches: late `frontier` mode now chooses an additional `contrast_support` center from archive rows that have ketamine target-band peaks, nonzero ketamine EPLI support, ketamine peak contrast at least as strong as control, and bounded control target leakage. It then adds targeted perturbations around EPLI AMPA/GABA balance, GC KAR, gap coupling, and AMPA/NMDA gain. This should bias batch 64+ toward actual clean-line/EPLI-supported candidates instead of only total target-band power.
- Follow-up: batch 64 was correctly sourced from the v5 leaders (`C00954`, `C01000`) but still used old `line` mode because the objective filter reset the usable archive count below the historical `frontier` threshold. Added an objective-filter-specific early-frontier path so campaigns with an explicit target-band pivot enter contrast-aware `frontier` mode once they have at least `192` filtered candidate pairs. Regression coverage now checks that an objective-filtered archive at this size proposes `frontier` rather than `line`.
- Batch 64 completed `32/32` remote items with no failures. It improved the v5 archive leader from `C00954` to `C01030`: pair score `-1.3956`, ketamine peak `200.195 Hz`, control peak `219.727 Hz`, ketamine peak contrast `0.377`, and ketamine EPLI rate `6.0 Hz`. This is not yet a clean solution because absolute contrast is still below the scorer's desired floor and control still has target-band structure, but it is a real improvement over the silent/low-contrast failure.
- Batch 65 launched immediately after with the early-frontier patch active: `targeted_detail.mode = frontier`, `12` targeted probes, top pair `C01030`/`C01024`, and future remote submissions pointed at `1281f21`. At launch it had `8` running, `24` pending, and `0` failed items.
- Generated diagnostic figures for the new v5 leader `C01030` at `results/notebook_runs/optimization/hfo_epli_big_singlewrapper_120cpu_20260527_082225/figures/best_v5_C01030_20260527_215420`, including summary, PSD, spectrogram, rasters, population-rate traces, manifest, and contact sheet.
- Generated the full expected diagnostic packet for `C01030` at `results/notebook_runs/optimization/hfo_epli_big_singlewrapper_120cpu_20260527_082225/figures/full_expected_C01030_20260527_220004`. This includes individual and overlay PSDs, paired spectrograms, LFP zoom, control/ketamine rasters, MT/EPLI/GC 2D spike KDEs for both conditions, input overviews, MT/EPLI HFO-phase histograms, manifest, and contact sheet.
- Process note: future long waits and remote progress monitoring should be kept in the background/nonblocking path; foreground work should be reserved for immediate artifacts, scoring/proposer fixes, and recoverable failure handling.
- Attempted a lightweight local background status logger for the active campaign at `results/notebook_runs/optimization/codex_big_hfo_logs/background_status_monitor.log`, but the local shell wrapper reaped the sidecar after one snapshot. Do not rely on that sidecar for persistence; the actual nonblocking long-running process remains Michael's authenticated notebook optimizer worker.
- Batch 65 completed `32/32` remote items with no failures and produced a better v5 leader, `C01043`: pair score `0.1531`, ketamine peak `195.312 Hz`, control peak `161.133 Hz`, ketamine peak contrast `0.4496`, control peak contrast `0.2437`, and ketamine EPLI rate `6.60 Hz`. Batch 66 was then planned in `frontier` mode from `C01043`/`C01030`.
- Objective pivot: added a machine-learning-style theoretical PSD template loss (`PAIR_SCORE_VERSION = 6`). Each newly scored condition now stores a compact normalized PSD-shape vector on a fixed 20-300 Hz grid. Pair scoring compares ketamine to a clean HFO-plus-gamma/beta template, control to a no-clean-HFO template, and the positive ketamine-control PSD difference to an HFO contrast template. Old archive rows without vectors fall back to a coarse band-power shape, so the existing archive remains rankable under v6. A local v6 rescore still ranks `C01043` first, now with `psd_template_loss = 1.425` and `psd_contrast_template_loss = 0.124`.
- Generated the full expected diagnostic packet for current v6 leader `C01043` at `results/notebook_runs/optimization/hfo_epli_big_singlewrapper_120cpu_20260527_082225/figures/full_expected_C01043_20260527_221502`. Summary: v6 pair score `2.2669`, ketamine peak `195.312 Hz`, control peak `161.133 Hz`, ketamine target relative power `0.0931`, control target relative power `0.0562`, ketamine peak contrast `0.4496`, ketamine EPLI `6.60 Hz`, `psd_template_loss = 1.425`, and `psd_contrast_template_loss = 0.124`.
- Batch 66/67 status snapshot: the v6 archive has a new numerical leader, `C01068`, with pair score `2.4159`, ketamine peak `195.312 Hz`, control peak `229.492 Hz`, ketamine target relative power `0.1073`, control target relative power `0.0701`, and active ketamine EPLI rate `7.74 Hz`. This is an improvement by objective but still needs visual review because control retains upper-target-band structure.
- Implemented an opt-in mid-run ketamine-switch mode to reduce paired-evaluation overhead in future batches. `AmpaNmdaSyn.mod` now supports `ketamine_switch_time` and `ketamine_block_after`; before the switch it uses the existing `ketamine_block`, after the switch it uses the post-switch multiplier. Notebook configs expose `ketamine_switch_time_ms` and `ketamine_block_after_switch`. `run_hfo_batch(..., condition_mode="switch")` emits one simulation per candidate, and `score_hfo_batch()` splits the saved trace into control and ketamine windows with a configurable washout. This does not alter the existing default separate-control/ketamine workflow.
- Verification for the switch implementation: `python test_hfo_optimizer.py` passed, normal `nrnivmodl` compile passed, CoreNEURON `nrnivmodl -coreneuron` compile passed, and a direct mechanism load confirmed the new `AmpaNmdaSyn` attributes are accessible (`ketamine_block`, `ketamine_switch_time`, `ketamine_block_after`). The local CUDA fatbin warning during Python startup is the same pre-existing harmless warning seen in prior tests.
- Batch 67 completed and batch 68 launched from commit `0267d5d`, so future remote submissions have the mid-run switch-capable code available. A transient Paramiko heartbeat read appeared stalled after batch 67 final sync, but the worker recovered without closing the authenticated transport and advanced to batch 68.
- Generated the full expected diagnostic packet for the current v6 leader `C01068` at `results/notebook_runs/optimization/hfo_epli_big_singlewrapper_120cpu_20260527_082225/figures/full_expected_C01068_20260527_223650`. The packet has 19 PNGs plus `manifest.json`, including paired PSDs, spectrograms, LFP zoom, spike rasters, MT/EPLI/GC 2D KDEs, input overviews, population rates, HFO phase histograms, and a contact sheet. Summary: archive pair score `2.4159`, ketamine peak `195.312 Hz`, control peak `229.492 Hz`, ketamine target relative power `0.1073`, control target relative power `0.0701`, and ketamine EPLI `7.74 Hz`.
- Batch 68 completed `32/32` remote items with no failures. Its best new row was `C01090`, score `-1.9637`, ketamine peak `205.078 Hz`, control peak `166.016 Hz`; it did not beat archive leader `C01068`. While batch 68 was running, the live notebook worker hit a recoverable Paramiko channel stall in the remote poll path. Added bounded Paramiko shell reads (`remote_ssh_command_timeout_s`, default `300 s`) and bounded status-poll reads (`remote_poll_command_timeout_s`, default `60 s`) so future stale SSH reads cannot freeze the optimizer indefinitely; regression test `python test_hfo_optimizer.py` passed.
- Batch 69 then launched from commit `8a9fb14`. The worker hit the same class of stall at Paramiko `exec_command` acknowledgement while publishing the next local commit, so added a separate `remote_ssh_exec_timeout_s` guard (default `30 s`) and reloaded Michael's authenticated kernel with commit `f871e15` for future batches. Released only the stuck channel; the authenticated transport stayed alive. Scratch-side check for batch 69 at `2026-05-27 22:51 EDT` showed the Slurm step running on `pcc082` with `6` finished, `8` running, `18` pending, and `0` failed items.
- Batch 69 completed `32/32` remote items with no failures. Its best new row was `C01104`, score `0.0355`, ketamine peak `180.664 Hz`, control peak `161.133 Hz`; it did not beat archive leader `C01068`. Added config-driven switch support to `run_hfo_batch()` so a running notebook worker can opt into `hfo_condition_mode = "switch"` through `base_config` without rewriting the worker loop; regression test `python test_hfo_optimizer.py` passed.
- Reloaded Michael's authenticated kernel at commit `2f7d0ed` and set future optimizer base configs to `hfo_condition_mode = "switch"`, `hfo_ketamine_switch_time_ms = 4500.0`, and `hfo_ketamine_switch_washout_ms = 500.0`. Batch 70 launched from commit `2f7d0ed` with `16` items instead of `32`; manifest inspection confirmed each item uses `AmpaNmdaSyn.ketamine_block = 1.0`, `ketamine_switch_time = 4500.0`, and `ketamine_block_after = 0.0`.
- Batch 70 completed under the mid-run ketamine-switch protocol. New archive leader `C01123`: pair score `5.1655`, control peak `229.492 Hz`, ketamine peak `190.430 Hz`, control target relative power `0.0746`, ketamine target relative power `0.1369`, control EPLI `7.27 Hz`, and ketamine EPLI `8.47 Hz`. Its main parameters are `kar_mt_gmax = 0.05763`, `kar_gc_gmax = 0.00147`, `kar_osn_weight_scale = 1.735`, `kar_gc_weight_scale = 0.265`, `ampa_nmda_gmax = 33.33`, `gaba_gmax = 2.48`, `epli_ampa_weight_scale = 2.58`, and `epli_gaba_weight_scale = 0.891`.
- Generated the full expected diagnostic packet for `C01123` at `results/notebook_runs/optimization/hfo_epli_big_singlewrapper_120cpu_20260527_082225/figures/full_expected_C01123_20260527_231406`. The packet includes paired PSDs, spectrograms, LFP zoom, control/ketamine spike rasters, MT/EPLI/GC 2D KDEs, input plots, population rates, HFO phase histograms, `manifest.json`, and `contact_sheet.png`.
- Batch 71 launched immediately afterward in the same switch mode with `16` candidates. At the latest live-kernel check, the authenticated notebook worker and watchdog were still alive; the worker was polling Phoenix through the existing Paramiko transport, not requesting new authentication.
- Added reusable PSD-template plotting helpers, `psd_template_curve()` and `scaled_psd_template_curve()`, so diagnostics can overlay the same theoretical PSD shapes used by the v6 objective. Also added `tools/analysis/regenerate_hfo_packet_psd.py` to regenerate those PSD panels from any packet `manifest.json`. Regenerated the C01123 packet's `01_psd_control.png`, `01_psd_ketamine.png`, `03_psd_overlay.png`, `contact_sheet.png`, and `manifest.json` with condition-specific target PSD overlays.
- Follow-up plot fix: area-scaled target overlays could be hard to see on the measured log-PSD axis. Updated `regenerate_hfo_packet_psd.py` to plot the theoretical target as an explicit normalized purple curve on a right-hand axis, then regenerated the C01123 PSD panels and contact sheet again.
- Follow-up plot semantics fix: the PSD template is only defined on the optimizer's 20-300 Hz scoring grid, so the diagnostic overlay now masks values outside that domain instead of interpolating to zero below 20 Hz. The plot label was changed from "target PSD" to "scoring template" to avoid implying that the curve is an empirical full-spectrum PSD target.
- Live-worker recovery note: after batch 73 planning, the authenticated notebook worker stalled inside Paramiko `channel.send()` while uploading a git bundle for the next remote commit. Added `remote_ssh_upload_timeout_s` and applied it directly to the shell-backed upload channel so this failure mode becomes bounded instead of freezing the optimizer thread indefinitely.
