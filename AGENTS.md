# Repository Agent Notes

This file is for coding agents operating inside `/home/alek/OlfactoryBulb`.
It is not user-facing product documentation. Treat it as the stable operating
contract for future sessions.

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

- Registered simulation-backed or structural audits currently include:
  - `env_install`
  - `burton_urban_fi`
  - `gc_intrinsic_validation`
  - `epl_fsi_intrinsic_validation`
  - `epli_correctness`
  - `human_review_status`
  - `hfo_feature_contracts`

- Keep the CLI output standards intact:
  - explicit `Description`
  - `Acceptable result`
  - `How Acceptable Result Was Determined`
  - visible `Human Review` line when review metadata exists
  - evidence blocks readable in plain text

## 6. Declarative literature-validation rules

- Use the declarative validation framework instead of bespoke one-off audit code
  when adding literature-backed validation:
  - configs live in `research_context/reference_validations/`
  - template:
    - `research_context/reference_validations/TEMPLATE.validation.toml`
  - guide:
    - `notes/REFERENCE_VALIDATION_HOWTO.md`

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

- Main generic commands:
  - `python tools/download_reference_dataset_sources.py --dataset-id <id>`
  - `python tools/extract_reference_dataset.py --dataset-id <id>`

- Dataset-specific wrappers exist for convenience, for example:
  - `tools/download_gc_reference_sources.py`
  - `tools/extract_gc_reference_data.py`
  - `tools/download_epl_fsi_reference_sources.py`
  - `tools/extract_pv_crh_epl_fsi_reference_data.py`

- Human-readable bundle summaries:
  - `tools/verify_gc_reference_data.py`
  - `tools/verify_pv_crh_epl_fsi_reference_data.py`

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

- Rebuild GC reference bundle:
  - `python tools/extract_gc_reference_data.py`

- Rebuild EPL-FSI reference bundle:
  - `python tools/extract_pv_crh_epl_fsi_reference_data.py`

## 15. Commit hygiene reminder

- Before final response:
  - review the diff
  - stage only task files
  - commit once with a targeted message
  - leave unrelated modifications alone

- If you accidentally stage unrelated files:
  - fix the index/commit scope cleanly
  - do not rewrite or discard user worktree changes
