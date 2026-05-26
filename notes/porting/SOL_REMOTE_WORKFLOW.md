# Remote Slurm Workflow

## Goal

Use the local notebook as the control surface while a remote Slurm cluster runs
the simulations.

The supported model is:

- local machine runs Jupyter and the notebook kernel
- remote cluster holds a clone of this repo plus an `OBGPU` env
- notebook authenticates once through Paramiko SSH
- notebook auto-publishes the current local `HEAD` commit to the remote repo
  when the remote does not already have it
- notebook submits Slurm jobs or Slurm steps
- completed results stream back into local `results/notebook_runs/...`
- local analysis cells work on synced results unchanged

The old OpenSSH control-master and rsync transport is retired. Do not configure
`ssh_multiplex`, `ssh_control_path`, `rsync_binary`, or `rsync_options`; those
knobs are no longer part of the maintained notebook path.

## Required Remote State

The remote cluster needs:

- a clone of this repo
- a working `OBGPU` build created with `tools/setup/setup_ob_modern.sh`
- Slurm access
- a normal SSH path from the notebook host
- `sbatch`, `sacct`, `scontrol`, `srun`, `tar`, and at least one compressor
  among `zstd`, `pigz`, `gzip`, or `xz`

You do not need to run Jupyter on the cluster.

## Configuration Builders

Use:

- `build_slurm_remote_config(...)` for generic Slurm clusters such as Phoenix
- `build_sol_remote_config(...)` for Sol-specific activation defaults

Common controls:

- `runner_backend`
- `remote_host`
- `remote_repo_root`
- `remote_results_root`
- `remote_conda_activate_cmd`
- `remote_runtime_profiles`
- `remote_git_ref`
- `remote_git_fetch`
- `remote_mpi_exec`
- `remote_poll_interval_s`
- `remote_log_poll_interval_s`
- `remote_live_status`
- `remote_live_logs`
- `remote_preserve_paramiko_session`
- `slurm_partition`
- `slurm_account`
- `slurm_time`
- `slurm_gpus`
- `slurm_cpus_per_task`
- `slurm_mem`
- `slurm_extra_args`
- `slurm_allocation_job_id`
- `slurm_reuse_allocation`
- `ssh_options`

`ssh_transport` remains only as a compatibility guard. `auto` and `paramiko`
are accepted; `openssh` is intentionally rejected.

## Minimal Example

```python
SOL_REMOTE_CONFIG = build_sol_remote_config(
    remote_host="jmpaniag@localhost",
    remote_repo_root="/scratch/jmpaniag/OlfactoryBulb",
    remote_results_root="/scratch/jmpaniag/OlfactoryBulb/results/notebook_runs",
    slurm_account="grp_scrook",
    slurm_time="00:30:00",
    slurm_gpus=1,
    slurm_partition="arm",
    ssh_options=["-p", "2222"],
)

RUN_CONFIG.update(SOL_REMOTE_CONFIG)
run, result = run_and_load(RUN_CONFIG)
```

For Phoenix-style CPU runs, omit GPU-specific fields and choose a partition/QOS
with `slurm_partition` and `slurm_extra_args` only when the cluster requires
them.

## Reusable Allocations

Two reuse modes are supported:

- `slurm_allocation_job_id="<jobid>"` uses an existing allocation that you
  created manually.
- `slurm_reuse_allocation=True` lets the notebook create and cache a reusable
  allocation, then launch runs as Slurm steps inside it.

Notebook-managed reusable allocations write heartbeat files. The notebook tries
to rediscover matching live allocations after a kernel restart and cancels stale
notebook-managed allocations before creating new ones. Manual allocations are
not cancelled by the notebook.

## SSH Authentication

Paramiko keeps one authenticated SSH transport per notebook runtime and endpoint.
With `remote_preserve_paramiko_session=True`, the helper fails closed instead of
silently opening a new login prompt mid-run after a connection was already
authenticated. Restart the kernel or explicitly reconnect if the remote SSH
session dies.

Common SSH tunnel example:

```bash
ssh -N -L 2222:sol-login02:22 jmpaniag@sol.asu.edu
```

Then configure the notebook with:

```python
remote_host="jmpaniag@localhost"
ssh_options=["-p", "2222"]
```

## Interactive Sol Shells

For interactive builds or smoke tests on Sol, do not run heavy work on the
login node. Allocate a node first:

```bash
salloc -p arm -G 1 -c 8 -t 02:00:00
source tools/setup/activate_sol_obgpu.sh
```

Inside a Slurm allocation, use `$OB_MPIEXEC` instead of guessing the launcher:

```bash
$OB_MPIEXEC -n 1 nrniv -mpi -python tools/benchmarks/benchmark_ob.py --label sol_smoke --paramset OneMsTest --coreneuron --coreneuron-gpu
```

The Sol activation helper:

- prefers already loaded Sol modules
- otherwise scans `module avail` for compatible `mamba`, `nvhpc`, and `cuda`
  entries
- activates `OBGPU`
- exports `OB_MPIEXEC` to a working Slurm launcher

## Code Update Workflow

Use committed code, not dirty notebook-local state.

Recommended flow:

1. commit locally
2. leave `remote_git_ref=None` so the notebook uses current local `HEAD`, or set
   `remote_git_ref` explicitly
3. let the notebook check whether the remote repo already has that commit
4. let the notebook upload an incremental git bundle if needed
5. run the Slurm job at the resolved commit

The remote commit is recorded in `git_ref.txt`, `git_commit.txt`, and
`run_info.json`.

## Result Contract

Remote runs sync into the same local layout as local notebook runs:

- `summary.json`
- `run_info.json`
- `command.txt`
- `stdout.txt`
- `stderr.txt`
- `input_times.pkl`
- `lfp.pkl`
- soma trace/spike artifacts when requested

Remote orchestration also keeps diagnostics such as:

- `bootstrap.log`
- `submit_stdout.txt`
- `submit_stderr.txt`
- `sync_stdout.txt`
- `sync_stderr.txt`
- `sim_progress.json`

Large result transfers use streamed tar compression over the existing Paramiko
connection and extract locally while downloading.
