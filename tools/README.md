# Tools

Support scripts created during the NEURON/CoreNEURON port and performance work are grouped here:

- `benchmarks/`
  Benchmark runners and result-comparison helpers.
- `debug/`
  One-off diagnostics used to isolate parity and performance issues.
- `remote/`
  Headless Slurm helpers used by the notebook's `sol_slurm` and
  `slurm_remote` backends for submit/poll/run orchestration.
- `setup/`
  Environment/bootstrap helpers, including the `OBGPU` NEURON/CoreNEURON setup, NVHPC `libnrnmech.so` repair, and upgrade-gate checks.

The maintained OBGPU build path is now:

1. read [third_party_patches/nrn/manifest.json](/home/alek/OlfactoryBulb/third_party_patches/nrn/manifest.json)
2. reset the cached NEURON checkout to the pinned upstream ref
3. apply the repo patch stack
4. build/install via [setup_ob_modern.sh](/home/alek/OlfactoryBulb/tools/setup/setup_ob_modern.sh)

For the supported remote workflow, see:

- [SOL_REMOTE_WORKFLOW.md](/home/alek/OlfactoryBulb/notes/porting/SOL_REMOTE_WORKFLOW.md)
- [submit_sol_run.py](/home/alek/OlfactoryBulb/tools/remote/submit_sol_run.py)
- [poll_sol_run.py](/home/alek/OlfactoryBulb/tools/remote/poll_sol_run.py)

The notebook remote path is Paramiko-only. The old OpenSSH multiplex and rsync
transport branch has been removed.

For interactive Sol shells, use:

- [activate_sol_obgpu.sh](/home/alek/OlfactoryBulb/tools/setup/activate_sol_obgpu.sh)

For one maintained-surface health command that stays inside the official audit
system, use:

- `python tools/run_audit.py repo_health --profile maintained`

For grouped developer-facing Python tests presented through the same audit
surface, use:

- `python tools/run_audit.py test_suite_status --list-suites`
- `python tools/run_audit.py test_suite_status --suite maintained_core`

For the maintained unified web shell that combines:

- audit results
- HFO optimization review
- current docs

use:

- `python -m olfactorybulb.dashboard.control_center`

That maintained command auto-detects the active optimization campaign when it
can, serves immediately at `http://127.0.0.1:6010/` by default, and renders
the default maintained audit into the page in the background. Use
`--open-browser` if you want it to launch a local browser automatically.

For a local docs portal that points at the maintained markdown docs first and
historical generated pages second, open:

- `docs/index.html`
