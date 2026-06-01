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

- the helper-bundle manifest/signature protocol that packages remote scripts
  for upload is now standardized under `neuroinfra.remote.helper_bundle`
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

Extraction confidence:

- **High for the pattern**
- **Medium for the concrete implementation**

Recommended public package target:

- `neuroinfra.contracts`
- `neuroinfra.registry`

This is one of the most valuable general patterns in the repo.

### 6. Dashboard / packet system

Current files:

- `tools/analysis/hfo_visual_dashboard.py`
- `tools/analysis/generate_hfo_candidate_packet.py`
- `tools/analysis/regenerate_hfo_packet_psd.py`
- `tools/analysis/hfo_tensorboard_dashboard.py`

What is already generic:

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
