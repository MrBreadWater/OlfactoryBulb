# Repository Agent Notes

This file is for coding agents operating inside `/home/alek/OlfactoryBulb`.
It is not user-facing product documentation. Treat it as the stable operating
contract for future sessions.

## 0. Maintenance duty and reproducibility

- Maintain `AGENTS.md` as a stable operating contract, not a session journal.
  Update it when a lesson is durable enough to guide future agents across
  tasks; do not record every UI tweak, bug fix, command run, or temporary
  workaround.

- Add or revise this file when the new information is one of:
  - a repo-wide invariant or source-of-truth boundary
  - a maintained workflow or entrypoint future agents should prefer
  - a recurring failure mode plus its correct recovery path
  - a verification standard that prevents false "done" reports
  - an anti-sprawl rule about where new logic should live
  - a reproducibility rule that must survive across chats

- Do **not** use this file for:
  - dynamic inventories that can be discovered live
  - one-off visual preferences unless they generalize into a UI contract
  - exact test logs, commit summaries, or historical play-by-play
  - current counts, current audit rosters, or current campaign names
  - implementation details already enforced by code/tests/config

- For dynamic inventory, point to live mechanisms instead of copying lists:
  - audits: `python tools/run_audit.py --list`
  - reference validations: `python tools/run_reference_validation.py --list-validations`
  - validation protocols: `python tools/run_reference_validation.py --list-protocols`
  - dataset configs: `research_context/reference_datasets/`
  - validation configs: `research_context/reference_validations/`

- If a behavior matters for reproducibility, encode it in code defaults,
  declarative config, tests, CLI discovery, or HOWTO docs. Use `AGENTS.md` to
  point at those sources of truth, not to replace them.

- Keep detailed documentation ownership in `notes/DOCS_OWNERSHIP_MAP.md`.
  Keep `AGENTS.md` focused on contract-level rules and anti-rot expectations.

## 1. Core defaults

- Always use the `OBGPU` environment for repo-local Python work.
  - Preferred forms:
    - `source tools/setup/activate_obgpu.sh OBGPU`
    - `/opt/miniconda3/envs/OBGPU/bin/python ...`
  - Do not use base/system Python for:
    - tests
    - notebook-facing scripts
    - simulation helpers
    - audit/reference-data tooling

- `/home/michael/OlfactoryBulb` is the user-facing checkout path and may be a
  symlink to `/home/alek/OlfactoryBulb`.
  - Prefer `/home/michael/OlfactoryBulb` in user-facing paths and notebook
    discussions.
  - Preserve Michael's authenticated notebook/Jupyter session when working on
    live notebook-managed workflows.

- When making user-requested repository changes, create a targeted git commit
  before the final response unless the user explicitly says not to.

- Stage only files that belong to the current task.
  - Leave unrelated dirty files and user changes unstaged.
  - This worktree is often noisy. Do not treat unrelated staged/untracked files
    as yours to clean up.

- For notebooks, avoid committing transient execution output unless the output
  is intentionally part of the deliverable.

## 1a. Active maintained surface and archival boundaries

- Prefer the actively maintained OBGPU workflow over historical entrypoints.
  - For source-of-truth boundaries, check:
    - `notes/CODEBASE_CLEANUP_AUDIT.md`
    - `notes/DOCS_OWNERSHIP_MAP.md`
  - The maintained surface is organized by category, not by every file name:
    - setup/activation: `tools/setup/`
    - benchmark/smoke entrypoints: `tools/benchmarks/`
    - notebook/local/remote orchestration facade: `obgpu_experiment_helpers.py`
    - extracted reusable infrastructure: `neuroinfra/`
    - core runtime/model/config surface: `olfactorybulb/`, `single_cell_utils.py`, `fi_curve_utils.py`
    - active notebooks: `notebooks/obgpu-working-experiment.ipynb`, `notebooks/fi_curve_analysis.ipynb`

- Treat compatibility and archival paths cautiously.
  - `prev_ob_models/` contains important references, but most trees are
    archival rather than active runtime surfaces.
  - Do not treat a historical model tree as active just because it exists.
    Use the current runtime registry/configuration path as the source of truth.

- Keep generated and local-only noise out of normal commits.
  - In particular:
    - `results/notebook_runs/`
    - `results/sweeps/`
    - `results/benchmarks/`
    - `results/comparisons/`
    - `results/profiles/`
    - `results/debug_*/`
    - `results/tmp_*/`
    - `.jupyter-ai-state/`
    - `.codex/`
    - `.ipynb_checkpoints/`
    - architecture build outputs such as `aarch64/`, `x86_64/`,
      `corenrn_data/`, generated mechanism C/object/shared-library files

- Keep `external/` treated as a resettable upstream/dependency cache, not a
  hand-edited source of truth.

- Use the central cell-model registry as the source of truth for family, role,
  aliases, and `network_ready` status.
  - Do not bypass it with ad hoc direct imports when the registry path exists.

- Current biological/runtime boundaries:
  - `Birgiolas2020` MC/TC/GC models are the maintained current-network family.
  - The canonical maintained slice path is MC/TC/GC-first.
  - The synthetic `SyntheticEPL2026.PVCRH_FSI1` external plexiform layer
    interneuron surrogate is provisional and not network-ready by default.
  - Published candidate/proxy families in the registry are useful references,
    but should not be silently treated as maintained runtime defaults.

## 1b. OBGPU build and setup path

- The maintained build/setup path is the modern OBGPU path.
  - Start with:
    - `install-obgpu.sh`
    - `tools/setup/setup_ob_modern.sh`
    - `tools/setup/activate_obgpu.sh`
    - `tools/setup/activate_sol_obgpu.sh`
    - `tools/setup/verify_obgpu_python_imports.py`
  - Prefer current setup docs under:
    - `tools/README.md`
    - `notes/porting/SOL_REMOTE_WORKFLOW.md`
    - `notes/porting/NEURON_UPGRADE_WORKFLOW.md`

- Do not resurrect older build/setup branches when the modern path is the
  maintained one.

- The generic Linux/GPU OBGPU path is the primary build path.
  - Jetson-specific helpers remain secondary and should not become the main
    source of truth for the general workflow.

- Any NEURON/CoreNEURON portability or compatibility fix must land in:
  - `third_party_patches/nrn/`
  - and the pinned patch-stack workflow
  - not as an ad hoc local edit inside `external/nrn-*`

- Upstream NEURON bumps are deliberate and gated.
  - Use:
    - `python tools/setup/check_nrn_upgrade.py --candidate-ref <ref> ...`
  - The supported upstream ref changes only after:
    - patch stack replay succeeds
    - OBGPU rebuild succeeds
    - smoke/parity checks pass
  - Do not silently retarget the pinned upstream dependency.

- If the environment shows NVHPC stale temp-object loader failures such as
  `dlopen failed ... /tmp/pgcudafat...`:
  - use `tools/setup/fix_nvhpc_libnrnmech.sh`
  - and verify with:
    - `python tools/run_audit.py env_install`
  - Do not treat that warning as harmless without checking the actual mechanism
    libraries.

- After changing environment/bootstrap/import/runtime-surface code, run the
  maintained environment audit:
  - `python tools/run_audit.py env_install`
  - Use that instead of relying on ad hoc import spot-checks alone.

## 2. Verification standard

- Do not claim a fix without checking the actual result path the user will rely
  on.
- For CLI/reporting work, run the real command.
- For dashboard/runtime work, hit the served page or live artifact, not just
  the generator.
  - For interactive dashboard changes, verify the actual control behavior
    (draft preservation, reruns, loading/progress states, and tab content) in
    the served page, not only exported HTML or static snapshots.
  - For visual/layout dashboard changes, verify at least one rendered
    post-CSS state that exercises the changed layout. DOM presence/order tests
    are not enough when the task is spacing, overflow, collapse behavior,
    z-order, or responsive fit.
- For notebook-facing defaults, verify the actual notebook/runtime paths that
  consume them.
  - If a user reports that a just-delivered change failed, reproduce that exact
    command or launch path before changing code again, then add or tighten a
    regression for the missed path when practical.

- When changing a notebook-facing default, verify the actual user paths that
  should observe it:
  - single run
  - remote run
  - sweep batch
  - live sync
  - final sync
  - load/animation helpers
  - Add or update tests for the default behavior, not only the helper
    implementation.

- Prefer proving behavior with the maintained live path when practical.
  - In this repo, helper-only validation is often insufficient.

- When changing result loading, remote sync, or notebook monitoring behavior,
  preserve the maintained performance/reliability semantics:
  - selected-file sync is an optimization, not the only success path
  - deferred artifact loading and lazy local/remote loaders are intentional
  - compact sync and partial-result recovery are intentional
  - heartbeat refresh/poll paths must stay bounded rather than inheriting
    unbounded remote waits
  - file/JSON-backed simulation progress is preferred over fragile stdout
    scraping

## 3. Live HFO optimizer and notebook-managed remote workflows

- If a live HFO workflow is blocked, recover the running campaign first and
  harden the failure path second.

- The live HFO campaign status file is:
  - `results/notebook_runs/optimization/codex_big_hfo_logs/latest_big_hfo_optimizer_status.json`

- The standard repair entrypoint for notebook-managed HFO recovery is:
  - `tools/analysis/resume_live_hfo_optimizer.py`

- For notebook-managed remote HFO reconnect issues:
  - inspect the latest campaign status first
  - prefer resuming the pending planned batch before proposing new work
  - treat fresh Paramiko handshake/banner-read failures as retryable transport
    errors, not only dead cached-session reuse

- If notebook kernel auto-discovery drifts, use the explicit connection-file
  path sooner instead of relying on ambient discovery.

## 3a. Remote execution architecture

- The supported remote-workflow model is:
  - local notebook/kernel as the control surface
  - remote cluster as the execution host
  - remote clone of this repo plus an `OBGPU` environment
  - notebook-managed submit/poll/sync lifecycle
  - local analysis on synced results

- The remote cluster should stay headless in the maintained workflow.
  - Do not redesign the supported path toward “run Jupyter on the cluster”.

- The maintained remote notebook/backend path is Paramiko-only.
  - Do not reintroduce or fork behavior around:
    - OpenSSH control-master transport
    - `ssh_multiplex`
    - `ssh_control_path`
    - `ssh_control_persist_s`
    - rsync result-sync paths
    - `rsync_binary`
    - `rsync_options`
    - `ssh_transport=\"openssh\"`

- The supported concepts are:
  - Paramiko persistent sessions
  - streamed compressed result sync
  - selected-file sync for sweeps and deferred artifacts
  - reusable Slurm allocations and manual `slurm_allocation_job_id`
  - `ssh_options` for port/jump-host behavior

- Keep remote config builders minimal.
  - If a value can be inferred from execution mode or Slurm resources, avoid
    adding a second independent knob.

- For remote execution, use committed code rather than dirty notebook-local
  state.
  - The maintained workflow expects notebook-managed git publication of the
    current local commit when needed.
  - Prefer the recorded git-ref path over ad hoc source copying.

- Preserve the remote result contract.
  - Remote runs should sync back into the same local result layout expected by
    local analysis helpers and notebooks, rather than inventing a second result
    format for remote execution.

- Selected-file sync and deferred-artifact fetches are maintained optimization
  layers.
  - Keep the fallback path to full/safer sync behavior intact.
  - Do not convert an optimization failure into silent data loss or a false
    success report.

- Preserve the distinction between local execution knobs and remote resource
  requests.
  - Local notebook runs still use settings such as `use_corenrn` and `use_gpu`.
  - Remote Slurm runs infer execution mode from the resource request unless
    explicitly overridden.
  - Do not collapse those two layers into one ambiguous configuration surface.

- On Sol, do not run heavy jobs on the login node.
  - Allocate first, then activate:
    - `salloc ...`
    - `source tools/setup/activate_sol_obgpu.sh`
  - When inside an allocation, use `$OB_MPIEXEC` instead of guessing the MPI
    launcher.

- The Sol module autoload convenience path is intentionally opt-in.
  - Preserve that behavior; do not make cluster-module side effects surprise
    generic Linux hosts on plain environment activation.

## 4. Dashboard/runtime expectations

- Maintained dashboard layers:
  - shell: `neuroinfra.dashboard.shell`
  - audit renderer: `olfactorybulb.audit.dashboard`
  - unified control center: `olfactorybulb.dashboard.control_center`
  - optimization/HFO module: `tools/analysis/hfo_visual_dashboard.py`

- Prefer the unified control center for end-user dashboard behavior:
  - `python -m olfactorybulb.dashboard.control_center`
  - The zero-argument path should serve promptly, print the reachable URL,
    leave audits idle until requested, and fall forward if port `6006` is busy.

- Verify dashboard changes against the served UI, not only generated files.
  At minimum, check the real launch path, relevant HTTP endpoints, and any
  changed interactive behavior with a DOM-capable browser test when JavaScript
  or layout state is involved.

- Dashboard UI contracts:
  - keep one shared shell rather than creating standalone dashboard stacks
  - mount new surfaces as tabs or modules in the unified shell
  - keep tab panels as full working surfaces, not card-in-card layouts
  - keep global status/progress compact and truthful
  - preserve local form drafts across background polling
  - make audit reports glanceable by default, with collapsed groups/items and
    explicit expansion controls
  - keep warnings visible, specific, and non-duplicated
  - when audit evidence contains an explicit series or curve, such as
    f-I/current-clamp arrays or `fi_curve_rows`, render the curve first and
    keep the key/value grid as fallback metadata rather than the primary
    visual
  - render explicit series/curve graphics in the persistent card body so they
    stay visible when the item is collapsed; do not hide them only inside the
    expanded detail body
  - do not force scalar numeric evidence into a chart; if the evidence is not
    explicitly series-shaped, keep it as metadata or use a dedicated interval
    visual when bounds are present
  - keep persistent audit-card body regions visually coherent in collapsed and
    expanded states; numeric summaries, notes/caveats, and warning blocks
    should share deliberate spacing/order instead of relying on each block's
    incidental first-child margins
  - when changing audit-card body layout, check representative combinations:
    no persistent body, numeric-only, notes-only, warning-only, notes plus
    warning, and numeric plus notes/warning when applicable
  - prefer responsive spacing such as `clamp()` over fixed desktop-only values

- Audit dashboard behavior that should stay stable:
  - sequential audit runs in one session remain visible instead of replacing
    earlier groups
  - `all` / new-sweep output preserves constituent audit groups
  - progress indicators reflect live audit progress, not just a spinner
  - numeric interval checks use structured visual summaries when possible

- The docs tab serves rendered maintained markdown under `docs/maintained/`.
  After changing maintained markdown or the render template, run:
  - `python tools/build_maintained_docs_portal.py`
  - `python tools/run_audit.py maintained_docs_integrity`

- Do not assume dashboard runtimes are healthy from status files alone. Verify
  the listener and rendered page. Shut down local servers you start during
  validation unless the user explicitly asks to leave them running.

- If `6006` is unexpectedly reoccupied after killing a dashboard child process,
  check for an older HFO dashboard watchdog/runtime that is respawning it and
  stop the runtime through the maintained HFO stop path.

## 5. Audit system: current architecture

- Keep the layer distinction clear:
  - `tools/run_audit.py` is the main human-facing surface for live system/model
    audits.
  - `tools/run_reference_validation.py` is the generic declarative
    literature-validation runner.
  - Low-level developer tests live under `tests/` and should usually be run as
    modules, for example `python -m tests.reference.test_reference_data_sanity`.
  - `python tools/run_audit.py test_suite_status --suite <suite_id>` is the
    maintained grouped presentation layer for those low-level tests.
  - `python tools/run_audit.py reference_dataset_status --dataset-id <id>` /
    `python tools/run_audit.py reference_dataset_contracts --dataset-id <id>`
    validate extracted literature bundles and provenance; they are not, by
    themselves, simulation-backed model-vs-literature audits.
  - Do not stop at reference-data extraction tests if the user asked for an
    actual literature-validation audit.

- Human-facing audit CLI:
  - `python tools/run_audit.py --list`
  - `python tools/run_audit.py <audit_id>`
  - stable aliases:
    - `python tools/run_audit.py default` -> maintained repo-health profile
    - `python tools/run_audit.py all` -> full registered audit sweep
  - `python tools/run_audit.py test_suite_status --list-suites`
  - For large grouped output, the default text mode may collapse fully passing
    groups.
  - Use:
    - `python tools/run_audit.py --expand`
    - `python tools/run_audit.py --failures-only`
    when you need the full or narrowed text view.

- Generic literature-validation CLI:
  - `python tools/run_reference_validation.py --list-validations`
  - `python tools/run_reference_validation.py --list-protocols`
  - `python tools/run_reference_validation.py --validation-id <id>`

- Audit web presentation:
  - `python -m olfactorybulb.audit.dashboard <audit_id> --output-dir <dir> [-- <audit args>]`
  - The audit dashboard should be driven from `AuditReport.to_dict()` rather
    than a separate ad hoc report schema.
  - Numeric interval-style validation items should render as structured
    reference-interval visuals rather than raw evidence dumps when the evidence
    includes accepted bounds plus the observed/reference mean values.
  - In the control-center shell, sequential audit runs should accumulate as
    separate audit groups for the current session.

- Do not hardcode the current audit roster here.
  - Audit IDs change over time.
  - The live source of truth is `python tools/run_audit.py --list` plus the
    audit registry in code.

- Do not add new root-level `test_*.py` files.
  - Low-level tests belong under `tests/`.
  - If a new check needs a user-facing maintained surface, prefer:
    - grouped suite audit for a family of tests
    - or a real operational audit if it is not fundamentally a developer test
  - Do not create one first-class audit per tiny helper regression.

- Keep the CLI output standards intact:
  - explicit `Description`
  - `Acceptable result`
  - `How Acceptable Result Was Determined`
  - explicit, concise `Warning` text for `WARN` items, especially when the
    warning comes from pending/provisional validation-design review state or
    an intentionally surfaced caveat; avoid repeating boilerplate phrasing like
    "why this is a warning" when a shorter warning label plus a specific
    reason is clearer
  - evidence blocks readable in plain text
  - grouped summaries should collapse cleanly in large multi-audit reports
    without hiding warning/failure detail

## 5a. Contract and registry discipline

- Keep feature/validation/dashboard surfaces mechanically linked to their
  registries or contracts.
  - Do not reintroduce hand-maintained duplicate lists when a contract or
    registry already exists.

- Keep user-facing parameter surfaces centralized.
  - When adding or changing notebook/run/network/cell parameters, prefer one
    canonical config or contract path rather than parallel helper-only knobs.
  - If human-facing parameter catalog docs are still relevant to the task, keep
    them aligned with the real runtime/config surface:
    - `build_run_config_parameters.md`
    - `notes/porting/NETWORK_AND_CELL_PARAMETER_CATALOG.md`
  - Do not let helper wrappers or notebooks become the only place a new
    user-facing knob exists.

- For HFO-facing parameter and visualization surfaces:
  - use the current contract/registry path rather than ad hoc whitelists
  - keep dashboard, packets, controls, and optimizer/search-space views derived
    from the same contract layer
  - run the contract audit after changes:
    - `python tools/run_audit.py hfo_feature_contracts`
  - if a new maintained dashboard surface is added, wire it through the shared
    shell and keep its visual style aligned with the other maintained tabs

- For literature validation:
  - dataset membership belongs in dataset configs
  - validation behavior belongs in validation configs, protocol runners, and
    rule kinds
  - human review status belongs in validation metadata
  - default behavior should preserve separation across incompatible targets
    rather than silently pooling them

## 5b. Reusable infrastructure extraction rules

- `neuroinfra/` is the internal extraction target for reusable infrastructure.
  - Use:
    - `neuroinfra/README.md`
    - `python -m neuroinfra`
  - to inspect the current extraction inventory and source-of-truth locations.

- Do not grow new mixed-responsibility monoliths when a reusable layer already
  exists under `neuroinfra`.
  - Prefer extending the extracted generic modules and keeping repo-specific
    wiring in the `olfactorybulb.*` adapters or notebook-facing glue.

- `obgpu_experiment_helpers.py` is still a major notebook-facing facade, but it
  should keep shrinking toward orchestration glue rather than regaining
  ownership of:
  - remote transport internals
  - result-artifact schemas
  - generic analysis primitives
  - dashboard runtime supervision
  - generic config/run catalog logic

- Before adding new helper logic, check whether the reusable home already
  exists in:
  - `neuroinfra.notebooks`
  - `neuroinfra.remote`
  - `neuroinfra.analysis`
  - `neuroinfra.artifacts`
  - `neuroinfra.dashboard`
  - `neuroinfra.contracts`
  - `neuroinfra.campaigns`
  - `neuroinfra.models`

- The `tools/remote/*.py` entrypoints now often act as compatibility bootstrap
  wrappers around `neuroinfra.remote_*` modules.
  - Keep user-facing CLI wrappers working, but put new generic logic in the
    extracted module rather than the wrapper script.

- `tools/debug/` scripts are one-off diagnostics, not maintained public
  workflow entrypoints.
  - Do not infer supported workflow contracts from a debug helper unless the
    same rule is anchored in the maintained setup/notebook/audit surfaces.

- The maintained import/runtime audit scope is the OBGPU workflow surface.
  - Blender-only paths and the older `neuronunit` stack are not part of the
    maintained import-verification contract for current OBGPU work.

## 6. Declarative literature-validation rules

- Use the declarative validation framework instead of bespoke one-off audit code
  when adding literature-backed validation:
  - configs live in `research_context/reference_validations/`
  - template:
    - `research_context/reference_validations/TEMPLATE.validation.toml`
  - guide:
    - `notes/REFERENCE_VALIDATION_HOWTO.md`
  - system overview:
    - `notes/REFERENCE_VALIDATION_SYSTEM_OVERVIEW.md`

- `burton_urban_fi` is one registered protocol-backed validation in this
  system, not the architecture itself.

- If a paper needs a new protocol:
  - add a registered protocol runner
  - keep custom measurements in the protocol runner output

- If a paper needs a new judgment rule:
  - add a new validation rule kind
  - keep comparison logic in the rule layer, not buried inside the protocol
    runner

- If a validation needs cheap static checks plus optional NEURON-backed checks,
  prefer the declarative path and use:
  - `skip_neuron_mode = "protocol_handles_skip"`
  - This is how `epli_correctness` now works.

## 7. Validation-design review metadata is mandatory

- Every declarative validation item should resolve to a validation-design
  review state.

- Supported review statuses:
  - `accepted`
  - `provisional`
  - `pending_review`
  - `not_applicable`

- Use `[human_review]` in validation configs.
  - The key name is historical/compatibility-focused.
  - The semantic meaning is review of the validation-design choice itself.
  - At minimum:
    - `[human_review]`
    - `default_status = "pending_review"`

- For `reference_band_rows`, per-property review state is preferred when not
  all metrics are at the same maturity:
  - `property_human_review_statuses`
  - `property_human_review_notes`
  - `property_human_review_reviewers`

- Run the coverage audit after changing validation configs:
  - `python tools/run_audit.py human_review_status`

- The `human_review_status` audit should:
  - fail on missing review status coverage
  - fail on unknown status strings
  - warn on `pending_review`
  - warn on `provisional`

- Keep the semantics clear in human-facing renderers.
  - Human-review metadata describes the review state of the underlying
    validation rule or reference-band choice, not whether one specific observed
    audit result row was manually checked after the fact.
  - Good examples:
    - whether a reference-band distribution shape is acceptable
    - whether pooling/separation choices are acceptable
    - whether a protocol-equivalence assumption is acceptable
    - whether a manual extraction or mapping choice is acceptable
  - Non-examples:
    - a human has verified the full code stack used by the audit
    - a human has manually confirmed this specific PASS/WARN/FAIL outcome
    - a human has re-reviewed every upstream source document in full for this run
  - Do not render human-review metadata inline on individual audit result items
    in the CLI or HTML dashboards unless the wording is explicitly reframed to
    avoid that confusion.

## 8. Reference-band selection rules

- `reference_band_rows` requires an explicit band mode for every property in
  `property_metric_map`.
  - No silent fallback is allowed.

- Supported band modes currently include:
  - `symmetric_sd`
  - `lognormal_sd`
  - `beta_sd`
  - `quantile_interval`
  - `binary_indicator`

- Choose the mode manually per metric. Do not assume every positive-valued
  metric should use a lognormal reconstruction.

- Keep a hard distinction between:
  - what the paper reported
  - what the audit reconstructed as an acceptance band

- If a band is an audit-side reconstruction or other temporary stopgap, mark it
  explicitly with:
  - review status `provisional`
  - visible note/caveat in the relevant notes CSV when appropriate

## 9. Reference-data pipeline: current architecture

- Use the generic declarative reference-data system for literature bundle work:
  - dataset configs:
    - `research_context/reference_datasets/`
  - template:
    - `research_context/reference_datasets/TEMPLATE.dataset.toml`
  - guide:
    - `notes/REFERENCE_DATASET_HOWTO.md`
  - manual intake templates:
    - `research_context/manual_reference_templates/`

- Main generic commands:
  - `python tools/download_reference_dataset_sources.py --dataset-id <id>`
  - `python tools/extract_reference_dataset.py --dataset-id <id>`

- Human-readable bundle summaries:
  - `python tools/run_audit.py reference_dataset_status --dataset-id <id>`
  - `python tools/run_audit.py reference_dataset_contracts --dataset-id <id>`
  - Do not assume dataset-specific helper wrappers exist.

- Treat normalized reference bundle outputs in `research_context/` as generated
  artifacts unless a file is explicitly a manual-intake template or a raw
  downloaded source file.
  - Prefer changing:
    - dataset config
    - extraction engine
    - source downloader/manifest
    - manual intake artifact
  - Then regenerate the normalized CSV/README outputs.
  - Do not hand-edit generated canonical outputs just to “fix the data” unless
    the task is explicitly about a one-off manual correction and the provenance
    implications are understood.

- Preserve stable provenance URLs.
  - When download URLs redirect through signed/object-store links, keep the
    stable publisher URL in `source_url` and treat the redirected fetch target
    as transport detail rather than canonical provenance.

## 10. Reference-data boundaries

- The pipeline directly supports:
  - downloaded supplemental tables
  - local manual tables
  - local manually digitized CSVs

- The pipeline does **not** do built-in screenshot or figure digitization.
- If actual numeric values cannot be extracted reliably:
    - do not guess
    - use `needs_manual_extraction.csv`
    - use the manual reference templates where applicable

- Manual intake templates live under:
  - `research_context/manual_reference_templates/`
  - They are raw human-curated intake artifacts, not canonical validation CSVs.

- Manual intake standards:
  - leave unknown fields blank
  - do not guess
  - keep `source_location` specific
  - preserve exact wording in `reported_text`
  - keep subtype and condition separated
  - do not pool different cell classes or protocols in one row
  - for f-I point templates, do not back-project points from summary metrics
    such as max rate or gain alone
  - keep one row per metric for summary tables, one row per current step for
    f-I tables, and one row per protocol variant for protocol templates

- Default literature-validation discipline:
  - do not silently pool different cell classes
  - do not silently pool subtype-specific and generic targets
  - do not silently pool different protocols
  - do not silently pool baseline with pharmacology/modulation conditions
  - do not silently pool across species, age, or maturity when those
    distinctions are part of the reference meaning
  - if incompatible targets appear together, keep the relevant notes/caveats
    visible in downstream outputs

- Preserve provenance in all extracted rows:
  - `source_file`
  - `source_url`
  - `source_location`
  - `reported_value_raw`

## 11. Mixed-unit table warning

- Be careful with `formatted_summary_rules`.

- If a formatted table mixes properties with different unit semantics, do **not**
  use one blanket `transform_scale` unless every mapped property truly needs the
  same scaling.

- Preferred solution:
  - `property_transform_scales = { ... }`

- The engine now rejects risky mixed-property blanket scaling by default unless
  explicitly overridden with:
  - `allow_blanket_transform_scale = true`

- This rule exists because it previously caused real GC extraction corruption:
  gain conversion was correct, but the same blanket scale also inflated
  rheobase, latency, and peak-rate rows by a factor of one thousand.

## 12. Sanity tests for extracted bundles

- Run the reference-data sanity test when changing extraction logic:
  - `python -m tests.reference.test_reference_data_sanity`

- Also rerun the relevant dataset extractor and verifier, not just unit tests.
  - Generic rebuild:
    - `python tools/extract_reference_dataset.py --dataset-id <id>`
  - Generic/source checks:
    - `python tools/download_reference_dataset_sources.py --dataset-id <id>`
  - Human-readable verification:
    - `python tools/run_audit.py test_suite_status --suite reference_bundles`
    - `python tools/run_audit.py reference_dataset_status --dataset-id <id>`
    - `python tools/run_audit.py reference_dataset_contracts --dataset-id <id>`

- That test is heuristic, not exhaustive. It is intended to catch obvious
  mistakes such as:
  - absurd magnitudes for standard units
  - negative standard deviations
  - reversed quantiles
  - impossible signs for CV, resistance, capacitance, rheobase, and similar
    metrics

- Still inspect suspicious rows against source files manually. Passing sanity
  checks does not prove semantic correctness.

## 13. Notes/caveats must travel with data

- Validation caveats belong in notes tables and must show up downstream.

- Use the notes system rather than burying important warnings inside free-text
  comments:
  - protocol differences
  - subtype separation
  - modulation-state separation
  - provisional or audit-side reconstruction caveats

- For outputs that compare incompatible protocols or populations, the notes
  section should remain visible in CLI/report/HTML outputs.

## 14. Good commands to know

- Audit/discovery:
  - `python tools/run_audit.py --list`
  - `python tools/run_audit.py`
  - `python tools/run_audit.py default`
  - `python tools/run_audit.py test_suite_status --list-suites`

- Dashboards/docs:
  - `python -m olfactorybulb.dashboard.control_center`
  - `python -m olfactorybulb.audit.dashboard new_sweep --output-dir /tmp/full_audit -- --skip-neuron`
  - `python tools/build_maintained_docs_portal.py`
  - `docs/index.html`

- Reference workflows:
  - `python tools/run_reference_validation.py --list-validations`
  - `python tools/run_reference_validation.py --list-protocols`
  - `python tools/extract_reference_dataset.py --dataset-id <id>`

## 15. Commit hygiene reminder

- Before final response:
  - review the diff
  - stage only task files
  - commit once with a targeted message
  - leave unrelated modifications alone

- If you accidentally stage unrelated files:
  - fix the index/commit scope cleanly
  - do not rewrite or discard user worktree changes
