# Repository Agent Notes

This file is for coding agents operating inside `/home/alek/OlfactoryBulb`.
It is not user-facing product documentation. Treat it as the stable operating
contract for future sessions.

## 0. Maintenance duty and reproducibility

- Keep this file continuously up to date when durable repo expectations change.
  - If a session establishes a new long-lived workflow rule, failure pattern,
    recovery path, validation requirement, or important caution that future
    agents should know, update `AGENTS.md` in the same task when practical.
  - Do not wait for undocumented tribal knowledge to accumulate elsewhere.

- Prefer stable rules over stale inventories.
  - This file should capture:
    - invariants
    - durable expectations
    - source-of-truth locations
    - verification standards
    - failure patterns and recovery rules
  - This file should **not** hardcode dynamic lists that are expected to change
    over time unless the list itself is the invariant being documented.

- For dynamic inventory, point to the live source of truth instead of copying
  the current contents into prose.
  - Examples:
    - registered audits -> `python tools/run_audit.py --list`
    - registered reference validations ->
      `python tools/run_reference_validation.py --list-validations`
    - registered validation protocols ->
      `python tools/run_reference_validation.py --list-protocols`
    - dataset configs -> `research_context/reference_datasets/`
    - validation configs -> `research_context/reference_validations/`
  - If a future agent adds a new audit/protocol/validation/dataset, update the
    registry or config and keep `AGENTS.md` pointing at that live mechanism.

- When documenting behavior here, distinguish clearly between:
  - stable contract
  - current example
  - live-discoverable state
  - temporary workaround

- If a behavior matters for reproducibility, do not leave it documented only in
  `AGENTS.md`.
  - Encode it in one or more of:
    - code defaults
    - declarative config
    - tests
    - CLI/discovery surfaces
    - HOWTO/reference docs
  - Then use `AGENTS.md` to point future agents to that source of truth.

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
  - High-priority maintained surfaces include:
    - `tools/setup/setup_ob_modern.sh`
    - `tools/setup/activate_obgpu.sh`
    - `tools/setup/activate_sol_obgpu.sh`
    - `tools/setup/verify_obgpu_python_imports.py`
    - `tools/remote/`
    - `tools/benchmarks/benchmark_ob.py`
    - `obgpu_experiment_helpers.py`
    - `olfactorybulb/model.py`
    - `olfactorybulb/paramsets/`
    - `single_cell_utils.py`
    - `fi_curve_utils.py`
    - `notebooks/obgpu-working-experiment.ipynb`
    - `notebooks/fi_curve_analysis.ipynb`
  - For the longer active-vs-archival reasoning, also check:
    - `notes/CODEBASE_CLEANUP_AUDIT.md`

- Treat compatibility and archival paths cautiously.
  - `initslice.py` and `runbatch.py` are compatibility entrypoints, not the
    preferred path for new notebook, benchmark, or Slurm work.
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

## 2. Verification standard

- Do not claim a fix without checking the actual result path the user will rely
  on.
  - For CLI/reporting work, run the real command.
  - For dashboard/runtime work, hit the served page or live artifact, not just
    the generator.
  - For notebook-facing defaults, verify the actual notebook/runtime paths that
    consume them.

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
  - Do not move the supported path toward “run Jupyter on the cluster”.

- Do not redesign the maintained remote workflow around running Jupyter on the
  cluster. That is not the supported model.

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

- If touching the HFO dashboard/runtime:
  - verify the served result, not just static HTML generation
  - if relevant, check:
    - `http://127.0.0.1:6006/`
    - `http://127.0.0.1:6006/visual_dashboard/`

- Do not assume a watcher/server is healthy because a status file says so.
  - Verify the listener and the rendered page.

- If changing packet generation or dashboard refresh behavior:
  - check both:
    - pre-render behavior
    - manual/queued packet refresh behavior

## 5. Audit system: current architecture

- Human-facing audit CLI:
  - `python tools/run_audit.py --list`
  - `python tools/run_audit.py <audit_id>`

- Generic literature-validation CLI:
  - `python tools/run_reference_validation.py --list-validations`
  - `python tools/run_reference_validation.py --list-protocols`
  - `python tools/run_reference_validation.py --validation-id <id>`

- Do not hardcode the current audit roster here.
  - Audit IDs change over time.
  - The live source of truth is `python tools/run_audit.py --list` plus the
    audit registry in code.

- Keep the CLI output standards intact:
  - explicit `Description`
  - `Acceptable result`
  - `How Acceptable Result Was Determined`
  - visible `Human Review` line when review metadata exists
  - evidence blocks readable in plain text

## 5a. Contract and registry discipline

- Keep feature/validation/dashboard surfaces mechanically linked to their
  registries or contracts.
  - Do not reintroduce hand-maintained duplicate lists when a contract or
    registry already exists.

- For HFO-facing parameter and visualization surfaces:
  - use the current contract/registry path rather than ad hoc whitelists
  - keep dashboard, packets, controls, and optimizer/search-space views derived
    from the same contract layer
  - run the contract audit after changes:
    - `python tools/run_audit.py hfo_feature_contracts`

- For literature validation:
  - dataset membership belongs in dataset configs
  - validation behavior belongs in validation configs, protocol runners, and
    rule kinds
  - human review status belongs in validation metadata

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

## 7. Human review metadata is mandatory

- Every declarative validation item should resolve to a human-review state.

- Supported review statuses:
  - `accepted`
  - `provisional`
  - `pending_review`
  - `not_applicable`

- Use `[human_review]` in validation configs.
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

- Dataset-specific wrappers may exist for convenience, but they are not the
  stable source of truth. Prefer the generic commands plus dataset IDs.

- Human-readable bundle summaries:
  - Use dataset-specific `tools/verify_*_reference_data.py` scripts when
    present.
  - Do not assume the exact wrapper set is stable over time.

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
  - `python test_reference_data_sanity.py`

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

- List audits:
  - `python tools/run_audit.py --list`

- Run all audits:
  - `python tools/run_audit.py`

- Run human review coverage audit:
  - `python tools/run_audit.py human_review_status`

- List reference validations:
  - `python tools/run_reference_validation.py --list-validations`

- List registered validation protocols:
  - `python tools/run_reference_validation.py --list-protocols`

- Rebuild any reference bundle:
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
