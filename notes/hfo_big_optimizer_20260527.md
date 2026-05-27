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
