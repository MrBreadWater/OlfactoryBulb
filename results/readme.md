# Results

This directory is for generated simulation outputs. Most contents under it are
local artifacts and should not be committed.

Modern notebook runs write to:

- `results/notebook_runs/<label>_<YYYYMMDD_HHMMSS>/`

Remote sweeps write to:

- `results/sweeps/<sweep_name>_<YYYYMMDD_HHMMSS>/`

Typical run artifacts include:

- `summary.json`
- `run_info.json`
- `notebook_run_info.json`
- `command.txt`
- `stdout.txt`
- `stderr.txt`
- `sim_progress.json`
- `input_times.pkl`
- `lfp.pkl`
- `gc_output_events.pkl`
- soma trace/spike artifacts, often compact NPZ files

Remote runs may also include orchestration diagnostics such as
`bootstrap.log`, `submit_stdout.txt`, `submit_stderr.txt`, `sync_stdout.txt`,
and `sync_stderr.txt`.

Older `initslice.py` and `runbatch.py` outputs may still appear here, but the
maintained notebook/benchmark path is the timestamped `notebook_runs` layout.
