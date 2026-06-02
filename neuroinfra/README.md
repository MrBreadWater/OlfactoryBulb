# neuroinfra

`neuroinfra` is the internal extraction target for the reusable infrastructure
that has accumulated inside this repository.

It is **not** yet the runtime owner of the main simulation, notebook, or remote
execution paths. Right now it serves two purposes:

1. provide a stable place to define what should become reusable framework code
2. force the extraction plan to become structured, inspectable, and testable

## Current scope

The first implemented artifact is the component inventory:

- `neuroinfra.inventory`

The first live internal extraction is now also present:

- `neuroinfra.artifacts.loading`
- `neuroinfra.artifacts.result_view`
- `neuroinfra.artifacts.output_paths`
- `neuroinfra.artifacts.result_artifacts`

The next standardized seam is also in place:

- `neuroinfra.analysis.overview`
- `neuroinfra.analysis.frequency_plots`
- `neuroinfra.analysis.grouped_views`
- `neuroinfra.analysis.phase_locking`
- `neuroinfra.analysis.events`
- `neuroinfra.analysis.plotting`
- `neuroinfra.analysis.profiles`
- `neuroinfra.analysis.signal_views`
- `neuroinfra.analysis.spectral`
- `neuroinfra.analysis.sweeps`
- `neuroinfra.analysis.signals`
- `neuroinfra.notebooks.config_store`
- `neuroinfra.notebooks.runs`
- `neuroinfra.remote.config`
- `neuroinfra.remote.helper_bundle`
- `neuroinfra.remote.command_launch`
- `neuroinfra.remote.notebook_runtime`
- `neuroinfra.remote.paramiko_transport`
- `neuroinfra.remote.sftp_sync`
- `neuroinfra.remote.archive_stream`
- `neuroinfra.remote.slurm_launch`
- `neuroinfra.remote.slurm_state`
- `neuroinfra.remote_script_common`
- `neuroinfra.remote_script_submit`
- `neuroinfra.remote_script_polling`
- `neuroinfra.remote_script_allocations`
- `neuroinfra.remote_script_sweeps`
- `neuroinfra.remote.git_sync`
- `neuroinfra.remote.helper_cache`
- `neuroinfra.remote.allocation_cache`
- `neuroinfra.remote.allocation_runtime`
- `neuroinfra.remote.stream_sync`
- `neuroinfra.remote.result_sync`
- `neuroinfra.remote.deferred_artifacts`
- `neuroinfra.remote.status_poll`
- `neuroinfra.remote.run_artifacts`
- `neuroinfra.remote.run_monitor`
- `neuroinfra.remote.sweep_monitor`
- `neuroinfra.remote.sweep_artifacts`
- `neuroinfra.models.registry`
- `neuroinfra.campaigns.store`
- `neuroinfra.contracts.parameters`
- `neuroinfra.contracts.visuals`
- `neuroinfra.dashboard.packets`
- `neuroinfra.dashboard.runtime`

It captures:

- candidate reusable subsystems
- current source file locations
- generic capabilities
- repo-specific couplings
- extraction confidence
- proposed extraction phase

## Usage

Text summary:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m neuroinfra
```

JSON inventory:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m neuroinfra --json
```

## Near-term intent

The expected first-wave extractions are:

1. audit core / CLI / registry
2. shared Slurm helper layer

The result-artifact and output-path helpers have already been mirrored into
`neuroinfra.artifacts` and left behind compatibility wrappers under
`olfactorybulb.*`.

The generic lazy-result container plus the timed local artifact-loading loop
that powers notebook result ingestion now also live under
`neuroinfra.artifacts.loading`, while the concrete OBGPU artifact plan and
field mapping still remain in `obgpu_experiment_helpers.py`.

The generic result-view planner that reads summary/run-info metadata, decides
which artifacts should load eagerly, wires deferred remote artifacts, and
attaches lazy local/remote loaders now also lives under
`neuroinfra.artifacts.result_view`. That same module now also owns the generic
result-schema machinery for default fields, artifact application behavior, and
lazy-path bookkeeping, while the concrete OBGPU signal names and higher-level
presentation policy still remain in `obgpu_experiment_helpers.py`.

The generic result-catalog helpers, result-overview builders, shared plotting
primitives, named-signal trace, band-pass, PSD overview, time-frequency view
helpers, resolved-signal phase-locking summaries, frequency KDE and
time-binned sample renderers, labeled-row instantaneous-frequency sample
collectors, trace-derived instantaneous-frequency sample collectors,
spectral-analysis core, sweep plot protocol, sweep persistence and reload
helpers, streamed GIF rendering pipeline, event-rate/raster analysis helpers,
shared event-rate plotting, reusable event-rate series assembly,
prepared labeled event display rows, overview-layout derivation from prepared
rows, label-prefix row filtering, normalization-driven event-rate
computation, result-backed event-family specs for filtering, sample
collection, normalized rate computation, and raster/rate/overview composition,
result-backed frequency plot families plus family-bound plotting suites for
KDE/time-binned rendering, stable ordered-name helpers,
fair round-robin subgroup truncation, ordered group-row flattening with
per-bucket limits, grouped row-display policies, grouped stacked-trace and
event-raster suites, domain analysis profiles that aggregate concrete signal,
event, frequency, and sweep definitions, raster-plus-rate overview layout, and sweep plot
registries with named built-ins plus deprecation handling, and
the named-signal provider registry, ordered named-signal registries,
registry-backed resolved-signal view suites, result-backed event family and
event plot suites, and provider-factory layer for keyed traces, suffix
variants, pattern-matched signals, labeled traces, and aligned mean traces,
plus stacked labeled trace plotting that list,
summarize, transform, plot, animate, persist, reload, and resolve analysis
signals in a stable ordered way now also live under `neuroinfra.analysis`,
while the concrete OBGPU result semantics and profile are now assembled
explicitly in `olfactorybulb.analysis_data` and
`olfactorybulb.analysis_profile`, with grouped soma presentation policy now in
`olfactorybulb.analysis_views`, HFO/LFP overview policy now in
`olfactorybulb.analysis_hfo_views`, and standard notebook output plus sweep
animation presets now in `olfactorybulb.analysis_presentations`; the remaining
notebook-facing glue still remains in `obgpu_experiment_helpers.py`.

The generic run-directory catalog and metadata loader that list saved notebook
runs, resolve prefix/index selections, recover captured stdout/stderr, and
reload saved config snapshots now also live under `neuroinfra.notebooks.runs`,
while the concrete config normalization and simulation/result wiring remain in
`obgpu_experiment_helpers.py`.

The generic JSON-ready conversion plus notebook-config save/load/list helpers
now also live under `neuroinfra.notebooks.config_store`, while the
olfactory-bulb-specific odor normalization, built-in paramset catalog, and
effective-param diff logic now live in `olfactorybulb.notebook_configs`.

The remote Slurm layer is not extracted yet, but the helper-bundle
manifest/signature protocol that packages those scripts for remote upload, plus
the local command builders that launch uploaded or inline helpers, now have a
standardized home under `neuroinfra.remote`.

The remote endpoint parsing, timeout normalization, retry policy, and generic
Paramiko-backed Slurm config builder that the notebook layer uses now also live
under `neuroinfra.remote.config`, while the notebook-facing wrappers remain in
`obgpu_experiment_helpers.py`.

The notebook-shared remote runtime keys, Paramiko prompt-cache handling, and
fail-closed reconnect policy that sit underneath the live notebook SSH path now
also live under `neuroinfra.remote.notebook_runtime`, while the transport I/O
and prompt plumbing still remain in `obgpu_experiment_helpers.py`.

The reusable Paramiko transport/session surface now also lives under
`neuroinfra.remote.paramiko_transport`, including cached connection reuse,
interactive authentication, lazy SFTP opening, and remote shell execution,
while the notebook-facing wrappers still remain in `obgpu_experiment_helpers.py`.

The SFTP transfer planning and copy loops that power selected-file and full
result syncs now live under `neuroinfra.remote.sftp_sync`, while the notebook
progress-bar wiring still remains in `obgpu_experiment_helpers.py`.

The remote archive probe/stream command builders and local decompressor helpers
that power compressed Paramiko syncs now also live under
`neuroinfra.remote.archive_stream`, while the live transport plumbing remains
in `obgpu_experiment_helpers.py`.

The higher-level argv and helper-launch assembly for allocation submit, run
submit, stale-allocation cleanup, and polling now also live under
`neuroinfra.remote.slurm_launch`, while the repo-specific config mapping and
live orchestration still remain in `obgpu_experiment_helpers.py`.

The remote preflight command builder, one-session preflight cache policy,
remote result-directory listing command, cancel command builder, and Slurm
state-query normalization that sit underneath the live notebook run path now
also live under `neuroinfra.remote.slurm_state`.

The remote-safe common helpers shared by uploaded Slurm wrapper scripts now
live under `neuroinfra.remote_script_common`, while `tools/remote/slurm_common.py`
remains as a compatibility bootstrap so the live entrypoints keep working from
both repo and helper-cache execution roots.

The remote-safe single-run submit helpers now live under
`neuroinfra.remote_script_submit`, while `tools/remote/submit_sol_run.py`
remains as a compatibility bootstrap/CLI wrapper around that module.

The remote-safe polling helpers shared by uploaded Slurm wrapper scripts now
live under `neuroinfra.remote_script_polling`, while `tools/remote/poll_sol_run.py`
remains as a compatibility bootstrap/CLI wrapper around that module.

The remote-safe allocation lifecycle helpers shared by uploaded Slurm wrapper
scripts now live under `neuroinfra.remote_script_allocations`, while
`tools/remote/submit_slurm_allocation.py` and
`tools/remote/cleanup_stale_allocations.py` remain as compatibility
bootstrap/CLI wrappers.

The remote-safe sweep runner helpers now live under
`neuroinfra.remote_script_sweeps`, while
`tools/remote/remote_sweep_driver.py` remains as a compatibility bootstrap
wrapper around that module.

The local Git publication/base-resolution helpers that support notebook-driven
remote syncs now also live under `neuroinfra.remote.git_sync`, while the live
Paramiko upload/orchestration path still remains in
`obgpu_experiment_helpers.py`.

The helper-cache runtime key, remote cache directory layout, manifest probe
logic, and upload-plan assembly that sit between helper-bundle metadata and the
live Paramiko transport now also live under `neuroinfra.remote.helper_cache`.

The reusable-allocation cache signature, cache key, runtime-config subset, and
normalized allocation record shape that sit underneath notebook-managed
allocation reuse now also live under `neuroinfra.remote.allocation_cache`.

The notebook-managed allocation orchestration layer that decides when to
refresh heartbeats, throttle stale-allocation cleanup, rediscover reusable
allocations, submit new ones, and release them now also lives under
`neuroinfra.remote.allocation_runtime`, while the notebook-facing wrappers
remain in `obgpu_experiment_helpers.py`.

The low-level Paramiko archive-stream, direct-file stream, and selected-file
probe helpers that power notebook result sync now also live under
`neuroinfra.remote.stream_sync`.

The higher-level Paramiko result-sync retry/fallback policy that sits above
those low-level stream helpers now also lives under
`neuroinfra.remote.result_sync`, while deferred-artifact sync policy and the
larger notebook load/orchestration flow still remain in
`obgpu_experiment_helpers.py`.

The deferred remote-artifact sync layer that parses notebook `run_info`,
retrieves one named payload, retries selected-file sync, optionally falls back
to direct file streaming for preferred artifact classes, and then escalates to
full result-dir sync now also lives under `neuroinfra.remote.deferred_artifacts`,
while the surrounding result-loading policy still remains in
`obgpu_experiment_helpers.py`.

The shared JSON status-poll retry/parsing helper that both the single-run and
remote-sweep notebook paths use now also lives under
`neuroinfra.remote.status_poll`, while the higher-level run/sweep monitoring
loops still remain in `obgpu_experiment_helpers.py`.

The remote single-run final sync, retry-on-empty-diagnostics, partial-payload
warning handling, failure listing fallback, and local artifact-collection
policy that sit between status monitoring and `run_info` persistence now also
live under `neuroinfra.remote.run_artifacts`, while the surrounding live
monitoring loop and notebook-specific metadata writing still remain in
`obgpu_experiment_helpers.py`.

The remote single-run live monitoring policy that manages poll cadence,
summary-aware full-log repolls, progress-bar lifecycle, live tail emission,
missing-artifact retry handling, and interrupt/error-driven cancel-plus-partial-sync
behavior now also lives under `neuroinfra.remote.run_monitor`, while the
notebook-specific poll command construction and `run_info` persistence still
remain in `obgpu_experiment_helpers.py`.

The remote sweep live monitoring policy that manages sacct poll cadence,
UNKNOWN-state forced repolls, sweep progress status summaries, incremental
finished-item sync triggering, and interrupt-driven cancellation now also
lives under `neuroinfra.remote.sweep_monitor`.

The remote sweep compact final-sync, summary-recovery, bulk compact-item sync,
and merged item-status finalization policy now also lives under
`neuroinfra.remote.sweep_artifacts`, while the notebook-specific sweep-item
loading and final `run`/`result` object assembly still remain in
`obgpu_experiment_helpers.py`.

The generic parameter-space and contract helpers that back the HFO optimizer's
search-space registry now live under `neuroinfra.contracts.parameters`, while
the HFO-specific parameter catalog remains in `olfactorybulb.hfo_features`.

The generic visualization-contract metadata types and snapshot builder that
back the HFO packet/dashboard schema now live under
`neuroinfra.contracts.visuals`, while the concrete HFO plot families and
render helpers remain in `olfactorybulb.hfo_visuals`.

The generic manifest-backed packet discovery and stale-packet cleanup helpers
that the HFO dashboard uses now live under `neuroinfra.dashboard.packets`,
while the HFO-specific packet freshness rules and HTML rendering remain in
`tools.analysis.hfo_visual_dashboard`.

The generic sidecar/runtime process primitives that the HFO dashboard uses now
live under `neuroinfra.dashboard.runtime`, while the HFO-specific command
assembly and freshness policy still remain in
`tools.analysis.hfo_visual_dashboard`.

The main file blocking deeper extraction is still:

- `obgpu_experiment_helpers.py`

That file needs to be split by responsibility before a clean reusable package
boundary can exist.
