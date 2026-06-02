# Reusable Infrastructure Extraction Map - 2026-06-01

## Goal

Map the parts of this repository that are strong candidates for release as a
general-purpose Python module for computational neuroscience workflows, and
separate them from the parts that should remain olfactory-bulb-specific.

This document is based on the current repository state, not on an idealized
future architecture.

## Executive Summary

Yes: the repo now contains several substantial pieces of reusable
infrastructure.

The strongest extraction candidates are:

1. `audit` framework
2. result artifact / run-label / summary helpers
3. remote Slurm execution helpers
4. campaign/archive machinery for long-running batch optimization
5. metadata registries and contract-driven rendering patterns
6. named result-signal registry helpers

The main things preventing clean extraction today are:

1. `obgpu_experiment_helpers.py` is a very large mixed-responsibility notebook
   layer
2. some reusable systems still depend directly on `olfactorybulb.*` modules
3. the HFO optimizer mixes generic campaign logic with highly specific scoring
   logic
4. the dashboard and packet generation system is generic in shape but still
   hard-coded to HFO-specific metrics and file conventions

So the right direction is not "extract the whole repo." The right direction is:

- extract a **framework package**
- keep this repo as a **domain plugin / reference application**

## Current Feature Map

### 1. Audit framework

Current files:

- `olfactorybulb/audit/core.py`
- `olfactorybulb/audit/cli.py`
- `olfactorybulb/audit/registry.py`
- `olfactorybulb/audit/neuron_protocols.py`
- domain audits:
  - `env_install.py`
  - `burton_urban_fi.py`
  - `epli_correctness.py`
  - `hfo_feature_contracts.py`

What is already generic:

- `AuditItem` / `AuditReport`
- text + JSON reporting
- colored CLI formatting
- registry-based audit discovery
- `new_sweep` composition across multiple audits

What is domain-specific:

- the individual audit modules
- reference data sources
- NEURON protocols specific to this model family

Extraction confidence:

- **High**

Recommended public package target:

- `neuroinfra.audit`

### 2. Result artifact and output-path helpers

Current files:

- `olfactorybulb/result_artifacts.py`
- `olfactorybulb/output_paths.py`

What is already generic:

- timestamped run-label generation
- run-info writing
- compact NPZ/PKL artifact persistence
- saved trace discovery
- spike detection from saved voltage traces
- versioned artifact formats

What is domain-specific:

- naming like `soma_vs`, `lfp`, and some OBGPU defaults

Extraction confidence:

- **High**

Recommended public package target:

- `neuroinfra.artifacts`

Current progress:

- the artifact/output-path helpers already live under `neuroinfra.artifacts`
- the generic lazy-result container and timed local artifact-loading loop now
  live under `neuroinfra.artifacts.loading`
- the generic result-view planner that reads summary/run-info metadata,
  decides eager vs deferred artifacts, and wires lazy local/remote loaders now
  lives under `neuroinfra.artifacts.result_view`
- that same module now also owns the generic result-schema machinery for
  default fields, artifact application behavior, and lazy-path bookkeeping
- the remaining concrete OBGPU signal names and field mapping still live in
  `obgpu_experiment_helpers.py`

### 2b. Notebook run catalog and metadata

Current files:

- `neuroinfra/notebooks/runs.py`
- `obgpu_experiment_helpers.py`

What is already generic:

- saved run metadata datatypes
- run directory listing and prefix filtering
- prefix/index-based run resolution
- captured stdout/stderr recovery
- saved config snapshot reload

What is domain-specific:

- the default OBGPU results-base location
- config normalization after reload
- run summary presentation
- simulation and result-loading wiring

Extraction confidence:

- **High**

Recommended public package target:

- `neuroinfra.notebooks`

Current progress:

- the generic run-directory catalog and metadata loader now live under
  `neuroinfra.notebooks.runs`
- `obgpu_experiment_helpers.py` now delegates the generic saved-run listing,
  resolution, metadata loading, and config snapshot reload paths there

### 2c. Notebook config persistence and catalog

Current files:

- `neuroinfra/notebooks/config_store.py`
- `olfactorybulb/notebook_configs.py`
- `obgpu_experiment_helpers.py`

What is already generic:

- JSON-ready conversion for notebook config payloads
- config save and reload helpers
- saved config file discovery

What is domain-specific:

- odor-schedule normalization after reload
- the built-in paramset catalog
- effective-param diffing against paramset defaults

Extraction confidence:

- **High**

Recommended public package target:

- `neuroinfra.notebooks`

Current progress:

- the generic notebook config save/load/list helpers now live under
  `neuroinfra.notebooks.config_store`
- the olfactory-bulb-specific normalization, built-in paramset catalog, and
  config diff logic now live in `olfactorybulb/notebook_configs.py`
- `obgpu_experiment_helpers.py` now delegates that config layer instead of
  owning it inline

### 2d. Notebook reporting and figure output

Current files:

- `neuroinfra/notebooks/reporting.py`
- `olfactorybulb/notebook_reports.py`
- `obgpu_experiment_helpers.py`

What is already generic:

- nested payload flattening for diff reports
- stable nested value diff generation
- human-readable diff section rendering
- figure save helpers with run-aware and sweep-aware output directories

What is domain-specific:

- run-summary content and section ordering
- effective-param and runtime-control summaries
- default figure output roots from the notebook helper

Extraction confidence:

- **High**

Recommended public package target:

- `neuroinfra.notebooks`

Current progress:

- the generic notebook diff/report/save helpers now live under
  `neuroinfra.notebooks.reporting`
- the olfactory-bulb-specific run-summary presentation now lives in
  `olfactorybulb/notebook_reports.py`
- `obgpu_experiment_helpers.py` now delegates that reporting layer instead of
  owning it inline

### 2e. Notebook sweep planning

Current files:

- `neuroinfra/notebooks/sweeps.py`
- `obgpu_experiment_helpers.py`

What is already generic:

- nested config path splitting with indexed list support
- nested config value assignment for dict/list payloads
- single-axis sweep expansion
- joint sweep expansion
- grid sweep expansion
- hook-driven timestamp and label policy

What is domain-specific:

- concrete run-config normalization
- sweep label naming policy
- local and remote sweep execution
- sweep persistence and artifact loading

Extraction confidence:

- **High**

Recommended public package target:

- `neuroinfra.notebooks`

Current progress:

- the generic sweep-planning layer now lives under
  `neuroinfra.notebooks.sweeps`
- `obgpu_experiment_helpers.py` now delegates nested path mutation and sweep
  plan expansion there while still owning concrete execution and persistence

### 2f. Notebook local subprocess execution

Current files:

- `neuroinfra/notebooks/local_runs.py`
- `obgpu_experiment_helpers.py`

What is already generic:

- local subprocess execution with captured stdout/stderr
- command and capture artifact persistence
- required summary-file enforcement
- standard failure-message rendering
- hook-driven run metadata persistence and return-value construction

What is domain-specific:

- concrete run-config normalization
- env construction
- benchmark command construction
- run-info payload semantics
- remote dispatch

Extraction confidence:

- **High**

Recommended public package target:

- `neuroinfra.notebooks`

Current progress:

- the generic local notebook subprocess runner now lives under
  `neuroinfra.notebooks.local_runs`
- `obgpu_experiment_helpers.py` now delegates the local captured-command
  execution path there while still owning env setup, command construction,
  and remote dispatch

### 2g. Result analysis and signal registry

Current files:

- `neuroinfra/analysis/events.py`
- `neuroinfra/analysis/catalog.py`
- `neuroinfra/analysis/frequency_plots.py`
- `neuroinfra/analysis/grouped_views.py`
- `neuroinfra/analysis/overview.py`
- `neuroinfra/analysis/phase_locking.py`
- `neuroinfra/analysis/plotting.py`
- `neuroinfra/analysis/profiles.py`
- `neuroinfra/analysis/signal_views.py`
- `neuroinfra/analysis/spectral.py`
- `neuroinfra/analysis/sweeps.py`
- `neuroinfra/analysis/signals.py`
- `olfactorybulb/analysis_data.py`
- `olfactorybulb/analysis_hfo_views.py`
- `olfactorybulb/analysis_presentations.py`
- `olfactorybulb/analysis_profile.py`
- `olfactorybulb/analysis_views.py`
- `obgpu_experiment_helpers.py`

What is already generic:

- stable category and label cataloging
- stable ordered-name helpers with preferred ordering and unknown-last handling
- fair round-robin subgroup truncation for merged display buckets
- ordered group-row flattening across display buckets with per-bucket limits
- grouped row-display policies for stable bucketed ordering and per-bucket limits
- grouped stacked-trace and event-raster suites built on those row policies
- domain analysis profiles that aggregate concrete signal, event, frequency,
  and sweep suites behind one plugin-style boundary
- result-overview context and summary builders
- stacked labeled trace plotting with configurable offsets and styling
- shared plotting primitives for traces, time-frequency maps, and band-power summaries
- uniform-trace interpolation and time-modulus folding
- spectrogram, wavelet, and band-power analysis on plain time/value arrays
- named-signal trace, band-pass, PSD overview, spectrogram, wavelet, and band-power view helpers built on provider-style resolvers
- phase-locking summaries from resolved signals and labeled spike-time rows
- frequency KDE and time-binned plotting from precomputed sample arrays
- result-backed frequency plot families for 1D KDE, 2D KDE, and time-binned rendering
- family-bound result-frequency plotting suites
- instantaneous frequency sample collection from labeled event rows
- trace-derived instantaneous frequency sample collection from labeled continuous-trace rows
- row filtering by label-prefix families
- normalization-driven event-rate computation from arbitrary event rows
- result-backed event-family specs for filtering, frequency sample collection, and normalized rate computation
- result-backed event family suites with reusable `t_stop` inference hooks
- result-backed event plot suites for raster, rate, and overview composition
- reusable event-rate series assembly for named subset plots
- prepared labeled event display rows and overview-layout derivation from them
- event-frequency conversion, event-rate binning, shared rate-plot helpers, and reusable raster-plot primitives
- shared raster-plus-rate overview layout for notebook summaries
- sweep plot specification, named plot registries, placeholder rendering, per-frame rendering, and GIF assembly
- sweep metadata persistence, reload, and saved-sweep discovery
- ordered named-signal providers
- ordered named-signal registries
- registry-backed resolved-signal view suites
- dynamic signal enumeration
- provider-based signal resolution
- provider factories for keyed traces, suffix variants, pattern-matched signals, and labeled traces
- aligned mean traces from grouped time/value rows
- separation between signal registry mechanics and concrete signal families

What is domain-specific:

- the concrete result-semantics layer that understands saved soma traces,
  saved spike artifacts, cell-family aliases, and olfactory-bulb result labels
- the concrete profile assembly that binds the reusable suites into this
  repo's OBGPU-facing analysis surface
- the concrete grouped soma presentation layer that defines MT grouping,
  colors, and bucket-specific display limits for this notebook workflow
- the concrete HFO/LFP overview layer that defines PSD template overlays and
  the standard LFP/HFO summary figure policy for this notebook workflow
- the concrete notebook presentation layer that defines standard output bundles
  and sweep-animation presets for this notebook workflow
- concrete OBGPU signal families like `lfp`, `gc_output_rate`, and
  `mean_MC_voltage`
- the remaining notebook entrypoint glue that still lives in the notebook helper

Extraction confidence:

- **Medium**

Current progress:

- `neuroinfra.analysis.profiles` now provides the reusable profile boundary
- the concrete OBGPU result semantics have moved into
  `olfactorybulb/analysis_data.py`
- the concrete HFO/LFP overview policy has moved into
  `olfactorybulb/analysis_hfo_views.py`
- the concrete notebook presentation presets have moved into
  `olfactorybulb/analysis_presentations.py`
- the concrete OBGPU profile assembly has moved into
  `olfactorybulb/analysis_profile.py`
- the concrete grouped soma presentation policy has moved into
  `olfactorybulb/analysis_views.py`
- `obgpu_experiment_helpers.py` now consumes that explicit domain module instead
  of carrying the result semantics and concrete profile assembly inline

### 3. Remote Slurm execution layer

Current files:

- `tools/remote/slurm_common.py`
- `tools/remote/submit_sol_run.py`
- `tools/remote/submit_slurm_allocation.py`
- `tools/remote/remote_sweep_driver.py`
- `tools/remote/poll_sol_run.py`

What is already generic:

- sbatch directive generation
- MPI-rank parsing
- remote wrapper scripting
- batch-item fanout inside a long-lived allocation
- path relocation for per-run worktrees
- launch preflight for NEURON MPI

What is domain-specific:

- some filenames and env vars with `OBGPU_` prefixes
- benchmark command assumptions
- remote bootstrap defaults specific to this repo

Extraction confidence:

- **Medium-high**

Recommended public package target:

- `neuroinfra.remote.slurm`

Important note:

This layer is already close to reusable, but it should be extracted as a
general "remote command campaign" framework rather than as an olfactory-bulb
runner.

Current progress:

- the remote endpoint parsing, timeout normalization, retry policy, and
  generic Paramiko-backed Slurm config builder that the notebook layer uses
  now live under `neuroinfra.remote.config`
- the notebook-shared remote runtime keys, Paramiko prompt-cache handling,
  and fail-closed reconnect policy that sit underneath the live notebook SSH
  path now live under `neuroinfra.remote.notebook_runtime`
- the reusable Paramiko transport/session surface now also lives under
  `neuroinfra.remote.paramiko_transport`, including cached connection reuse,
  interactive authentication, lazy SFTP opening, and remote shell execution
- the SFTP transfer planning and copy loops that power selected-file and full
  result syncs now live under `neuroinfra.remote.sftp_sync`
- the remote archive probe/stream command builders and local decompressor
  helpers that power compressed Paramiko syncs now live under
  `neuroinfra.remote.archive_stream`
- the helper-bundle manifest/signature protocol that packages remote scripts
  for upload is now standardized under `neuroinfra.remote.helper_bundle`
- the local command builders that launch uploaded or inline helper scripts are
  now standardized under `neuroinfra.remote.command_launch`
- the higher-level argv and helper-launch assembly for allocation submit, run
  submit, stale-allocation cleanup, and polling now live under
  `neuroinfra.remote.slurm_launch`
- the remote preflight command builder, one-session preflight cache policy,
  result-directory listing command, cancel command builder, and Slurm state
  query normalization now live under `neuroinfra.remote.slurm_state`
- the remote-safe common helpers shared by uploaded Slurm wrapper scripts now
  live under `neuroinfra.remote_script_common`, while
  `tools/remote/slurm_common.py` remains as a compatibility bootstrap
- the remote-safe single-run submit helpers now live under
  `neuroinfra.remote_script_submit`, while
  `tools/remote/submit_sol_run.py` remains as a compatibility bootstrap/CLI wrapper
- the remote-safe polling helpers shared by uploaded Slurm wrapper scripts now
  live under `neuroinfra.remote_script_polling`, while
  `tools/remote/poll_sol_run.py` remains as a compatibility bootstrap/CLI wrapper
- the remote-safe allocation lifecycle helpers shared by uploaded Slurm wrapper
  scripts now live under `neuroinfra.remote_script_allocations`, while
  `tools/remote/submit_slurm_allocation.py` and
  `tools/remote/cleanup_stale_allocations.py` remain as compatibility wrappers
- the remote-safe sweep runner helpers now live under
  `neuroinfra.remote_script_sweeps`, while
  `tools/remote/remote_sweep_driver.py` remains as a compatibility wrapper
- the local Git publication/base-resolution helpers that support notebook-
  driven remote syncs now live under `neuroinfra.remote.git_sync`
- the helper-cache runtime key, cache directory layout, manifest probe logic,
  and upload-plan assembly now live under `neuroinfra.remote.helper_cache`
- the reusable-allocation cache signature, cache key, runtime-config subset,
  and normalized allocation record shape now live under
  `neuroinfra.remote.allocation_cache`
- the notebook-managed reusable-allocation orchestration layer that refreshes
  heartbeats, throttles stale-allocation cleanup, rediscovers allocations,
  submits new ones, and releases them now lives under
  `neuroinfra.remote.allocation_runtime`, while the notebook-facing wrappers
  remain in `obgpu_experiment_helpers.py`
- the low-level Paramiko archive-stream, direct-file stream, and selected-file
  probe helpers that power notebook result sync now live under
  `neuroinfra.remote.stream_sync`
- the higher-level Paramiko result-sync retry/fallback policy that sits above
  those low-level stream helpers now lives under
  `neuroinfra.remote.result_sync`, while deferred-artifact sync policy and the
  larger notebook load/orchestration flow still remain in
  `obgpu_experiment_helpers.py`
- the deferred remote-artifact sync layer that parses notebook `run_info`,
  retries selected-file sync for one payload, optionally falls back to direct
  file streaming for preferred artifact classes, and then escalates to full
  result-dir sync now lives under `neuroinfra.remote.deferred_artifacts`,
  while the surrounding result-loading and lazy-loader policy still remains in
  `obgpu_experiment_helpers.py`
- the shared JSON status-poll retry/parsing helper that both the single-run
  and remote-sweep notebook paths use now lives under
  `neuroinfra.remote.status_poll`, while the higher-level monitoring loops
  still remain in `obgpu_experiment_helpers.py`
- the remote single-run final sync, retry-on-empty-diagnostics, partial-payload
  warning handling, failure listing fallback, and local artifact-collection
  policy now live under `neuroinfra.remote.run_artifacts`, while the live
  monitoring loop and notebook-specific `run_info` persistence still remain in
  `obgpu_experiment_helpers.py`
- the remote single-run live monitoring policy that manages poll cadence,
  summary-aware full-log repolls, progress-bar lifecycle, live tail emission,
  missing-artifact retry handling, and interrupt/error-driven
  cancel-plus-partial-sync behavior now lives under
  `neuroinfra.remote.run_monitor`, while notebook-specific poll-command
  construction and `run_info` persistence still remain in
  `obgpu_experiment_helpers.py`
- the remote sweep live monitoring policy that manages sacct poll cadence,
  UNKNOWN-state forced repolls, sweep progress status summaries, incremental
  finished-item sync triggering, and interrupt-driven cancellation now lives
  under `neuroinfra.remote.sweep_monitor`
- the remote sweep compact final-sync, summary-recovery, bulk compact-item
  sync, and merged item-status finalization policy now lives under
  `neuroinfra.remote.sweep_artifacts`, while notebook-specific sweep-item
  loading and final `run`/`result` object assembly still remain in
  `obgpu_experiment_helpers.py`
- the launcher scripts themselves still live under `tools/remote/`

### 4. Campaign / optimizer archive framework

Current files:

- `olfactorybulb/hfo_optimizer.py`
- `tools/run_hfo_campaign.py`

What is already generic:

- campaign directory layout
- state files
- archive files
- batch plan / batch run / batch score lifecycle
- Latin-hypercube seeding
- elite/refinement proposal patterns
- criteria-based candidate ranking
- reusable-allocation campaign execution style

What is domain-specific:

- HFO scoring templates
- ketamine/control paired-condition semantics
- PSD-specific objective metrics
- explicit knowledge of olfactory-bulb cell types and analysis readouts

Extraction confidence:

- **Medium**

Recommended public package target:

- generic layer: `neuroinfra.campaigns`
- domain layer retained here: `olfactorybulb.hfo_optimizer`

Best split:

- generic `CampaignStore`, `BatchPlan`, `BatchResult`, proposal engine APIs
- domain-specific scorer/proposer implementations live in the plugin repo

Current progress:

- the generic campaign filesystem/state/archive layer now lives under
  `neuroinfra.campaigns.store`
- HFO-specific scoring and proposal logic still lives in
  `olfactorybulb.hfo_optimizer`

### 5. Metadata registry / contract system

Current files:

- `olfactorybulb/hfo_features.py`
- `olfactorybulb/hfo_visuals.py`
- `olfactorybulb/audit/hfo_feature_contracts.py`
- `prev_ob_models/cell_registry.py`

What is already generic:

- single-source-of-truth parameter registry
- versioned parameter contracts
- versioned visual contracts
- model registry pattern with explicit metadata
- contract audits to prevent drift

What is domain-specific:

- specific HFO parameters and plot families
- olfactory-bulb cell-model catalog

Current progress:

- the generic parameter-space and contract helpers that back the HFO optimizer
  now live under `neuroinfra.contracts.parameters`
- the generic visualization-contract metadata types and snapshot builder that
  back the HFO packet/dashboard schema now live under
  `neuroinfra.contracts.visuals`
- `olfactorybulb.hfo_features` still owns the concrete HFO parameter catalog,
  runtime defaults, and override wiring
- `olfactorybulb.hfo_visuals` still owns the concrete HFO plot families,
  filenames, and render helpers

Extraction confidence:

- **High for the pattern**
- **Medium for the concrete implementation**

Recommended public package target:

- `neuroinfra.contracts`
- `neuroinfra.registry`

This is one of the most valuable general patterns in the repo.

### 6. Dashboard / packet system

Current files:

- `neuroinfra/dashboard/packets.py`
- `neuroinfra/dashboard/runtime.py`
- `tools/analysis/hfo_visual_dashboard.py`
- `tools/analysis/generate_hfo_candidate_packet.py`
- `tools/analysis/regenerate_hfo_packet_psd.py`
- `tools/analysis/hfo_tensorboard_dashboard.py`

What is already generic:

- packet manifest directory scanning
- latest-packet selection per candidate
- stale packet cleanup
- detached sidecar process spawning
- pid metadata/status file management
- process liveness and command matching
- sidecar termination helpers
- packet manifest pattern
- background packet generation
- runtime supervision
- served static dashboard with live refresh
- recent/best candidate views
- packet stale detection via manifest versioning

What is domain-specific:

- HFO packet file names
- PSD overlays
- candidate score summaries
- fixed plot families

Extraction confidence:

- **Medium**

Recommended public package target:

- generic layer: `neuroinfra.dashboard`
- domain plugin layer: packet schema + renderers supplied by caller

Current progress:

- the generic packet manifest discovery and stale-packet cleanup helpers now
  live under `neuroinfra.dashboard.packets`
- the generic sidecar/runtime process primitives now live under
  `neuroinfra.dashboard.runtime`
- the HFO dashboard still owns the HFO-specific packet freshness rules,
  packet-generation queueing, command assembly, and HTML/server behavior

### 7. Cell-model registry

Current files:

- `prev_ob_models/cell_registry.py`
- `prev_ob_models/utils.py`

What is already generic:

- discoverable model metadata
- family / role / source / citation / target-use metadata
- canonical-key resolution
- default family-role mapping
- dynamic import and instantiation

What is domain-specific:

- actual registered olfactory-bulb models
- assumptions about family and role names

Extraction confidence:

- **Medium-high**

Recommended public package target:

- generic layer: `neuroinfra.models`
- repo-specific registry contents remain here

Current progress:

- the generic registry skeleton now lives under `neuroinfra.models.registry`
- `prev_ob_models.cell_registry` remains the first concrete provider/catalog

### 8. Slice geometry / connectivity evaluation

Current files:

- `olfactorybulb/slice_connectivity_optimizer.py`
- `tools/optimize_slice_connectivity.py`

What is already generic:

- exported-geometry loading
- section/terminal representations
- proximity-based connection evaluation
- candidate rule scoring against a reference slice

What is domain-specific:

- slice file conventions
- neuronal section-family assumptions
- olfactory-bulb group names and synapse-set conventions

Extraction confidence:

- **Medium**

Recommended public package target:

- maybe `neuroinfra.geometry` later

This should not be first-wave extraction. It has real value, but it still leans
heavily on this repo's exported slice schema.

## What Should Stay Repo-Specific

These should remain in the domain repo, not in the framework package:

- `olfactorybulb/model.py`
- `olfactorybulb/inputs.py`
- `olfactorybulb/epli.py`
- `olfactorybulb/paramsets/`
- `olfactorybulb/slicebuilder/`
- `prev_ob_models/*` model implementations
- HFO scoring semantics and biology-specific objective design

These are the science application, not the reusable infrastructure.

## Biggest Extraction Blocker: `obgpu_experiment_helpers.py`

This file is the clearest signal that the repo has outgrown a single helper
module.

Current responsibilities mixed into one file include:

- run-config schema and defaults
- local run launching
- remote SSH/Paramiko logic
- Slurm execution logic
- sweep orchestration
- result loading
- artifact loading
- plotting
- spectrogram/wavelet helpers
- notebook convenience utilities
- live dashboard support glue

This file is useful, but it is the least extractable part in its current form.

### Recommended split

Proposed decomposition:

- `neuroinfra.runconfig`
  - config defaults
  - help text
  - config serialization
- `neuroinfra.runners.local`
  - local subprocess launch
- `neuroinfra.runners.remote`
  - Paramiko/SSH session management
  - remote command execution
  - sync helpers
- `neuroinfra.sweeps`
  - sweep planning and execution wrappers
- `neuroinfra.artifacts`
  - result loading and artifact lookup
- `neuroinfra.plots`
  - generic plotting primitives
- repo adapter:
  - olfactory-bulb-specific config keys
  - olfactory-bulb-specific plotting defaults

Until that split happens, the extraction path will stay awkward.

## Proposed Public Package Layout

Recommended architecture:

```text
neuroinfra/
  audit/
    core.py
    cli.py
    registry.py
    protocols.py
  artifacts/
    output_paths.py
    traces.py
    summaries.py
  remote/
    ssh.py
    slurm.py
    sweep_driver.py
  campaigns/
    store.py
    proposals.py
    execution.py
    scoring.py
  dashboard/
    runtime.py
    packets.py
    manifest.py
    static_ui.py
  contracts/
    parameters.py
    visuals.py
    validation.py
  models/
    registry.py
    spec.py
  notebooks/
    convenience.py
```

And then this repo becomes:

```text
olfactorybulb/
  adapters/
    neuroinfra_config.py
    neuroinfra_dashboard.py
  domain/
    model.py
    inputs.py
    paramsets/
    slice_tools/
    scoring/
    cell_catalog/
```

## Standardization Rules Worth Preserving

These patterns are strong and should become part of the extracted framework's
design rules.

### 1. Contract-first feature definition

Already demonstrated by:

- `hfo_features.py`
- `hfo_visuals.py`
- `hfo_feature_contracts.py`

Rule:

- no hand-maintained duplicate parameter or visualization lists
- every public feature has one registry entry
- dashboards, packets, and audits consume the same contract

### 2. Artifact versioning

Already demonstrated by:

- manifest versions
- packet render versions
- trace artifact format versions

Rule:

- every cached/generated artifact needs an explicit version
- stale detection should be automatic

### 3. Long-running campaign state on disk

Already demonstrated by the HFO optimizer.

Rule:

- progress should live in files, not only memory
- batch plan / run / score separation is good
- resumed execution should be first-class

### 4. Fail-loud audit surfaces

Already demonstrated by the audit CLI.

Rule:

- a framework package should make drift visible
- human-facing descriptions, acceptable ranges, and provenance belong in the
  framework, not only in docs

### 5. Runtime sidecars should be supervised

Already demonstrated by the dashboard runtime work.

Rule:

- watchers, servers, and exporters need supervision and manifest-backed health
  checks

## Recommended Extraction Order

### Phase 1: easiest, highest payoff

1. extract `audit.core`, `audit.cli`, `audit.registry`
2. extract `result_artifacts.py` and `output_paths.py`
3. extract `slurm_common.py`

This gives a real reusable package quickly with low biology coupling.

### Phase 2: remote execution framework

1. extract remote Slurm wrapper generation
2. extract SSH/session utilities from `obgpu_experiment_helpers.py`
3. expose a generic remote campaign runner API

### Phase 3: contracts and registries

1. extract the parameter/visual contract pattern
2. extract the model-registry skeleton
3. make domain-specific registries plug into it

### Phase 4: campaign framework

1. split generic campaign/archive/proposal logic out of `hfo_optimizer.py`
2. keep HFO scoring and candidate interpretation inside this repo

### Phase 5: dashboard runtime

1. extract runtime supervision, packet manifests, and refresh protocol
2. keep the HFO packet schema as a domain plugin

## Concrete Immediate Refactors Inside This Repo

Before any external release, these repo-local changes would pay off:

1. split `obgpu_experiment_helpers.py`
2. remove `olfactorybulb`-specific imports from generic audit core
3. move remote execution code behind a small runner interface
4. separate generic campaign storage from HFO scoring logic
5. formalize packet schema objects instead of ad hoc dicts

## Practical Recommendation

Do **not** start by publishing the whole repo as a package.

Instead:

1. create a new internal package directory, probably `neuroinfra/`, inside this
   repo first
2. migrate the high-confidence generic layers into it while keeping this repo as
   the first consumer
3. only once that stabilizes, split it into a separate repository/module

That path keeps the real notebook workflow intact while forcing the abstractions
to prove themselves against the current application.

## Bottom Line

This repo is no longer just a model repository. It already contains the early
shape of a general computational-neuroscience workflow framework.

The strongest reusable assets are:

- audit/reporting infrastructure
- artifact/versioning infrastructure
- remote Slurm campaign machinery
- contract-driven metadata and dashboard patterns

The main discipline required now is to extract **framework from biology** rather
than extracting everything at once.
