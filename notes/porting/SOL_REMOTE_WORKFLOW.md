# Sol Remote Workflow

## Goal

Use the local notebook as the control surface while Sol runs the actual simulation jobs through Slurm.

The supported model is:

- local machine runs Jupyter and the notebook kernel
- Sol holds a clone of this repo plus an `OBGPU` env
- notebook submits Slurm jobs to Sol over SSH
- completed results sync back into local `results/notebook_runs/...`
- local analysis cells work on those synced results unchanged

## Required Sol State

Sol needs:

- a clone of this repo
- a working `OBGPU` build created with [setup_ob_modern.sh](/home/alek/OlfactoryBulb/tools/setup/setup_ob_modern.sh)
- Slurm access
- `ssh`, `sbatch`, `sacct`, and `rsync` available in the normal user path

You do not need to run Jupyter on Sol.

## Notebook Controls

The notebook helper layer now supports these remote controls:

- `runner_backend="sol_slurm"`
- `remote_host`
- `remote_repo_root`
- `remote_results_root`
- `remote_conda_activate_cmd`
- `remote_git_ref`
- `remote_git_fetch`
- `remote_mpi_exec`
- `remote_poll_interval_s`
- `slurm_partition`
- `slurm_account`
- `slurm_time`
- `slurm_gpus`
- `slurm_cpus_per_task`
- `slurm_mem`
- `slurm_extra_args`
- `ssh_binary`
- `ssh_options`
- `rsync_binary`
- `rsync_options`

The public notebook interface is unchanged:

- `run, result = run_and_load(RUN_CONFIG)`
- `run, result = load_run_pair(...)`

## Minimal Example

```python
RUN_CONFIG = build_run_config(
    mode="fast",
    paramset="GammaSignature",
    runner_backend="sol_slurm",
    remote_host="youruser@sol.asu.edu",
    remote_repo_root="/path/on/sol/OlfactoryBulb",
    remote_results_root="/path/on/sol/OlfactoryBulb/results/notebook_runs",
    remote_conda_activate_cmd='source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate OBGPU',
    remote_git_ref="0123abcd...",  # optional; defaults to the current local HEAD commit
    slurm_partition="gpu",
    slurm_time="02:00:00",
    slurm_gpus=1,
)

run, result = run_and_load(RUN_CONFIG)
```

## Code Update Workflow

Use committed code on Sol, not notebook-local dirty state.

Recommended flow:

1. commit locally
2. push that commit to your remote
3. either:
   - leave `remote_git_ref=None` and let the notebook use the current local `HEAD` commit, or
   - set `remote_git_ref` explicitly to the commit, tag, or branch you want Sol to run

When the remote backend runs:

- it can `git fetch` on Sol first when `remote_git_fetch=True`
- it checks out the requested ref before launching the simulation
- it syncs back `git_ref.txt` and `git_commit.txt`
- the resolved remote commit is recorded in `run_info.json`

## Remote Helper Scripts

The remote backend uses:

- [submit_sol_run.py](/home/alek/OlfactoryBulb/tools/remote/submit_sol_run.py)
- [poll_sol_run.py](/home/alek/OlfactoryBulb/tools/remote/poll_sol_run.py)
- [run_obgpu_batch.sh](/home/alek/OlfactoryBulb/tools/remote/run_obgpu_batch.sh)

These scripts are committed so the local notebook does not need to embed large ad hoc shell templates.

## Result Contract

Remote runs are synced back into the same local layout as local notebook runs:

- `summary.json`
- `run_info.json`
- `command.txt`
- `stdout.txt`
- `stderr.txt`
- output pickle files

Additional orchestration artifacts are also written locally when using the remote backend:

- `submit_stdout.txt`
- `submit_stderr.txt`
- `sync_stdout.txt`
- `sync_stderr.txt`

That keeps debugging local even when the compute happened on Sol.
