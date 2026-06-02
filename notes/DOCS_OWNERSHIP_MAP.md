# Documentation Ownership Map

This file says which document is the source of truth for each major topic in
the maintained OBGPU workflow. Use it to avoid duplicating the same guidance in
multiple places and to decide where a doc change should land first.

## Rule of thumb

- Update the narrowest source-of-truth document first.
- Let broader docs point to that document rather than restating all of it.
- If two docs disagree, fix the lower-level owner and then prune or redirect
  the higher-level one.

## Ownership table

### Top-level user entrypoints

- `readme.md`
  - Purpose: shortest maintained repo overview and quick-start path.
  - Owns:
    - what this repo is
    - the maintained install/run path at a high level
    - where the active notebook/workflow starts
  - Should not own:
    - detailed remote workflow mechanics
    - full install troubleshooting
    - reference-data or validation framework internals

- `INSTALL.md`
  - Purpose: maintained host/install/bootstrap instructions.
  - Owns:
    - prerequisites
    - install commands
    - activation
    - first smoke tests
    - install-specific cautions
  - Should not own:
    - notebook orchestration policy
    - full remote architecture rationale

### Tooling and environment ownership

- `tools/README.md`
  - Purpose: map of tool families and the important maintained entrypoints.
  - Owns:
    - what lives under `tools/`
    - which tool families are maintained
    - where to look for setup/remote helpers
    - where the canonical repo-health command lives
  - Should not own:
    - deep procedural setup steps already covered by `INSTALL.md`

- `environments/README.md`
  - Purpose: environment-spec ownership.
  - Owns:
    - which environment files are maintained vs legacy
    - which spec backs OBGPU
  - Should not own:
    - general install workflow beyond environment-file selection

### Remote and build workflow ownership

- `notes/porting/SOL_REMOTE_WORKFLOW.md`
  - Purpose: maintained remote notebook/Slurm workflow.
  - Owns:
    - supported remote architecture
    - Sol/Phoenix remote execution expectations
    - remote bootstrap / submit / poll / sync behavior
  - Should not own:
    - generic install instructions already covered elsewhere

- `notes/porting/NEURON_UPGRADE_WORKFLOW.md`
  - Purpose: upstream `nrn` bump / patch-stack workflow.
  - Owns:
    - patch-stack replay expectations
    - upgrade-gate checks
    - how to change the pinned NEURON ref safely

- `notes/porting/MODERN_NEURON_PORT_NOTES.md`
  - Purpose: historical but still relevant port/build notes for the maintained
    modern path.
  - Owns:
    - important porting caveats that do not fit cleanly in `INSTALL.md`

### Maintained vs archival boundaries

- `notes/CODEBASE_CLEANUP_AUDIT.md`
  - Purpose: active-vs-archival boundary document.
  - Owns:
    - what is currently maintained
    - what is compatibility-only
    - what is archival/reference-only
  - Should not own:
    - detailed install or remote execution procedures

- `AGENTS.md`
  - Purpose: coding-agent operating contract.
  - Owns:
    - durable workflow rules for future agent sessions
    - anti-rot expectations
    - verification standards
    - source-of-truth pointers
  - Should not own:
    - long user HOWTO content already documented elsewhere
    - dynamic inventories better discovered via CLI/config

### Reference-data and validation ownership

- `notes/REFERENCE_DATASET_HOWTO.md`
  - Purpose: how to add/maintain declarative reference datasets.
  - Owns:
    - dataset config workflow
    - downloader/extractor usage
    - manual extraction boundaries

- `notes/REFERENCE_VALIDATION_HOWTO.md`
  - Purpose: how to add/maintain declarative simulation-backed literature
    validations.
  - Owns:
    - validation config workflow
    - protocol-runner/rule extension workflow
    - human-review and band-selection expectations

- `notes/REFERENCE_VALIDATION_SYSTEM_OVERVIEW.md`
  - Purpose: short system map.
  - Owns:
    - the concise conceptual overview only

- `research_context/README.md`
  - Purpose: boundaries inside `research_context/`.
  - Owns:
    - what is raw source data
    - what is manual intake
    - what is generated canonical output
    - what should and should not be hand-edited

- `research_context/manual_reference_templates/README.md`
  - Purpose: manual intake workflow for unresolved literature data.
  - Owns:
    - raw human-curated intake expectations
    - row-shape and provenance expectations for manual capture

## Common cleanup targets

- If `readme.md`, `INSTALL.md`, and `tools/README.md` all say the same thing,
  the detailed version should usually live in `INSTALL.md` or the specialized
  tool doc, with `readme.md` shortened to a pointer.

- If `AGENTS.md` and a HOWTO both describe procedure, keep the procedure in the
  HOWTO and leave only the contract-level reminder in `AGENTS.md`.

- If a generated reference CSV appears to need manual correction, update the
  dataset config, raw source, or extraction engine first, then regenerate.
