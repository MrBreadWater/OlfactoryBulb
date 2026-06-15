# Reference Validation Tutorial

Use this document when you want the full maintained paper-to-audit workflow in
one place.

This tutorial is intentionally more detailed and more step-by-step than the
system overview or the validation HOWTO. It is the maintained onboarding path
for:

- adding one new paper metric to an existing validation
- adding a new paper that needs normalized reference rows
- deciding whether you need dataset work, protocol work, rule work, or only a
  validation-config change
- choosing and documenting validation-design review status at every stage

If you only want the short conceptual explanation first, start with
[REFERENCE_VALIDATION_SYSTEM_OVERVIEW.md](/home/michael/OlfactoryBulb/notes/REFERENCE_VALIDATION_SYSTEM_OVERVIEW.md).

If you already know the workflow and only need the extension points, use
[REFERENCE_VALIDATION_HOWTO.md](/home/michael/OlfactoryBulb/notes/REFERENCE_VALIDATION_HOWTO.md).

## What this system does

Reference validation turns "the paper says this metric should look like X"
into a reproducible audit item.

It does that by keeping four responsibilities separate:

1. reference dataset config
   - turns papers and supplements into normalized reference rows
2. reference validation config
   - says which protocol to run and which checks to apply
3. protocol runner
   - measures model-side quantities
4. rule handler
   - turns those measurements into judgments

The core discipline is:

- protocol runners produce measurements
- rule handlers produce judgments

Do not hide judgment logic inside a protocol runner.
Do not hide measurement logic inside a validation config.

## The quickest decision tree

When you want to add a paper metric, ask these questions in order:

1. Is the paper value already present in a normalized reference dataset?
   - If no: update or add a dataset config first.
2. Does an existing protocol runner already emit the corresponding model-side
   metric?
   - If no: extend or add a protocol runner.
3. Can an existing built-in rule kind express the comparison?
   - If yes: edit only the validation config.
   - If no: add a new rule kind.
4. Has a human reviewed the validation-design choice behind the comparison?
   - If no: mark it `pending`.
   - If yes, but it is still a temporary stopgap: mark it `provisional`.
   - If yes, and the design choice is accepted: mark it `approved`.

That decision tree is the shortest reliable way to avoid writing the wrong kind
of code.

## End-to-end example: add one paper metric

This section walks through the full path as if you had a paper with one metric
you wanted to compare against one of the maintained models.

Assume the paper reports a metric called "Membrane Resting Voltage" and you
want to compare it against an existing validation.

### Step 1: write down the comparison you actually want

Before touching code or TOML, write down all four of these:

- paper/source identity
- the exact literature value you want to use
- the model-side quantity that should match it
- the comparison rule

Good example:

- paper: "Example et al. (2026)"
- literature metric: "Membrane Resting Voltage"
- model metric key: `resting_potential_mV`
- comparison style: mean-with-accepted-band

Bad example:

- "compare the model to the paper somehow"

If you cannot name the model-side metric key or the comparison style yet, you
do not know which layer needs work.

### Step 2: decide which layer is missing

Use this checklist:

- If the paper value is not yet in a normalized CSV used by validations:
  dataset work is missing.
- If the paper value exists, but the protocol result does not emit the
  model-side metric key:
  protocol work is missing.
- If both values exist, but no built-in rule kind can compare them:
  rule work is missing.
- If both values exist and a built-in rule kind fits:
  only validation-config work is missing.

### Step 3: if needed, add the paper data to a dataset

If the paper is not yet normalized, update or add a dataset config under
`research_context/reference_datasets/`.

The dataset layer owns:

- source files
- extraction mapping
- normalized outputs such as `ephys`, `fi_curve`, `protocols`, `identity`
- note rows and manual-extraction backlog

The dataset layer does not own:

- model execution
- judgment logic
- PASS/WARN/FAIL decision rules

Recommended order when adding a new paper to the dataset layer:

1. add the source files
2. add protocol rows
3. add note rows for protocol caveats
4. add summary or point extraction rules
5. add manual-extraction rows for anything unresolved
6. run the downloader and extractor
7. inspect the generated CSVs
8. run the dataset audits

Commands:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/download_reference_dataset_sources.py --dataset-id <dataset_id>
python tools/extract_reference_dataset.py --dataset-id <dataset_id>
python tools/run_audit.py reference_dataset_contracts --dataset-id <dataset_id>
python tools/run_audit.py reference_dataset_status --dataset-id <dataset_id>
python tools/run_audit.py test_suite_status --suite reference_bundles
```

Rules for this stage:

- do not guess missing values
- do not silently pool incompatible cell classes
- do not hide protocol caveats
- do not manually patch generated CSVs when the config or extractor should be
  fixed instead

### Step 4: confirm the protocol runner emits the model-side metric

Once the paper value exists in normalized form, check whether the relevant
protocol runner already emits the model-side metric you need.

Examples of emitted metrics in current validations include:

- `resting_potential_mV`
- `input_resistance_MOhm`
- `rheobase_pA`
- `fi_gain_Hz_per_50pA`
- `cv_isi`
- `FWHM_ms`
- `AP_onset_mV`

If the protocol runner already emits the metric, keep moving.

If it does not, extend the appropriate protocol runner in
`olfactorybulb/audit/reference_validation_protocols.py`.

Protocol runners should emit measurements, not pass/fail decisions.

### Step 5: choose the right rule kind

Most paper metrics map cleanly to one of the built-in rule kinds:

- exact protocol execution sanity:
  `protocol_executed`
- "every cell has a usable finite value":
  `all_finite_metric`
- "every cell equals a fixed expected value":
  `all_exact_metric`
- "group B should be larger or smaller than group A":
  `group_ordering`
- "two groups should stay close":
  `group_abs_diff_max`
- "these groups should remain positive":
  `group_positive`
- "a summary value should be above a threshold":
  `summary_metric_min`
- "a summary value should be below a threshold":
  `summary_metric_max`
- "a summary value should be inside a fixed range":
  `summary_metric_range`
- "a summary status code maps to PASS/WARN/FAIL":
  `summary_metric_status_map`
- "a normalized literature row defines an accepted band":
  `reference_band_rows`
- "a reference curve and model curve should stay near each other":
  `reference_curve_match`
- "protocol or reference caveats must remain visible":
  `note_presence`

If one of those fits, stay in config.

If none of those fits, add a new rule kind and document it in the same task.

For the built-in SciUnit-backed suite families, the maintained report now adds
one suite-overview item ahead of the detailed cases. That overview card is for
presentation only: it is marked `summary_rollup_exempt`, so the report summary
and group summary still count the detailed cases rather than double-counting
the overview.

Those overview cards and their case-matrix payloads are intentionally shared.
Future SciUnit-backed suite families should extend
`olfactorybulb.neuronunit.suite_presentation` rather than inventing another
per-family overview schema.

### Step 6: create or edit the validation config

Validation configs live under:

- `research_context/reference_validations/`

Copy the template when creating a new one:

```bash
cp \
  research_context/reference_validations/TEMPLATE.validation.toml \
  research_context/reference_validations/my_validation.validation.toml
```

At minimum, a config needs:

```toml
validation_id = "my_validation"
title = "My literature validation audit"
description = "One-line summary of what this validation compares."
protocol_runner = "replace_with_registered_protocol_id"

[[checks]]
kind = "protocol_executed"
check_id = "my_protocol_executed"
title = "My protocol executed"
criterion = "Describe the required protocol."
description = "This confirms the configured protocol actually ran."
acceptable = "At least one metric row is produced."
acceptable_basis = "This is an execution sanity check."
```

### Step 7: add the specific metric comparison

If the paper metric is best represented as a literature reference band, add it
to `reference_band_rows`.

Example:

```toml
[[checks]]
kind = "reference_band_rows"
loader = "csv:research_context/EXAMPLE_ephys.csv"
reference_source = "Example et al. (2026)"
group_field = "cell_type"
sigma_arg_name = "reference_sigma_multiplier"
property_metric_map = {
  "Membrane Resting Voltage" = "resting_potential_mV",
}
property_band_modes = {
  "Membrane Resting Voltage" = "symmetric_sd",
}
title = "Reference band placeholder"
criterion = "Unused placeholder text; this rule auto-generates one item per mapped reference row."
description = "Unused placeholder text; this rule auto-generates one item per mapped reference row."
acceptable = "Unused placeholder text; this rule auto-generates one item per mapped reference row."
acceptable_basis = "Unused placeholder text; this rule auto-generates one item per mapped reference row."
```

If the metric is not a reference-band comparison, use the rule kind that best
matches the paper's claim instead.

### Step 8: set validation-design review status

This is mandatory.

Every declarative validation item must resolve to one of these statuses:

- `approved`
- `provisional`
- `pending`
- `not_applicable`

Start new work with:

```toml
[validation_design_review]
default_status = "pending"
```

Then override it where needed.

Examples:

```toml
[validation_design_review]
default_status = "pending"
```

```toml
[[checks]]
kind = "protocol_executed"
...
validation_design_review_status = "approved"
validation_design_review_reviewer = "qualified_domain_expert"
validation_design_review_note = "Protocol equivalence manually reviewed."
```

```toml
[[checks]]
kind = "reference_band_rows"
...
property_validation_design_review_statuses = {
  "Membrane Resting Voltage" = "approved",
  "AHP Duration" = "provisional",
}
property_validation_design_review_notes = {
  "AHP Duration" = "Temporary stopgap until source-backed quantiles are available.",
}
```

### Step 9: do a cheap smoke test first

Most validations should first be run in skip mode so you can validate the
config path before paying for the full model execution.

Command:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/run_reference_validation.py --validation-id <validation_id> --skip-neuron
```

For simulation-backed validations with `skip_neuron_mode = "short_circuit"`,
this returns the configured `[skip_item]`.

For mixed validations with
`skip_neuron_mode = "protocol_handles_skip"`, the protocol runner still emits
cheap status-style checks while the expensive parts stay skipped.

### Step 10: run the full validation

Once the config path looks right, run the full validation:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/run_reference_validation.py --validation-id <validation_id>
```

If a richer audit wrapper exists, run that too:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/run_audit.py <audit_id>
```

### Step 11: run the review-status coverage audit

Always check review coverage explicitly:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/run_audit.py validation_design_review_status
```

This audit:

- fails if any item has no resolved review status
- fails if any status string is unknown
- warns on `pending`
- warns on `provisional`

### Step 12: decide what can be promoted from pending review

After the validation runs end-to-end, promote only the design choices that have
actually been reviewed.

Good pattern:

- start with `pending`
- promote stable, justified choices to `approved`
- mark temporary but acknowledged stopgaps as `provisional`
- keep control-flow-only items as `not_applicable`

That sequence makes review debt visible without blocking all work.

## Validation-design review statuses

These statuses are about the validation-design choice, not about whether a
human personally inspected one specific observed run result.

### `approved`

Definition:

- a human has reviewed the validation-design choice and considers it acceptable

Use it for:

- accepted protocol-equivalence assumptions
- accepted pooling/separation choices
- accepted reference-band modes
- accepted manual extraction or mapping choices

Example:

```toml
[[checks]]
kind = "reference_band_rows"
...
property_validation_design_review_statuses = {
  "Membrane Resting Voltage" = "approved",
}
```

### `provisional`

Definition:

- a human reviewed the design choice, but it is still a deliberate stopgap

Use it for:

- positive-only band reconstructions that are acceptable for now but not ideal
- temporary protocol mapping assumptions
- source-backed but caveated approximations

Example:

```toml
[[checks]]
kind = "reference_band_rows"
...
property_validation_design_review_statuses = {
  "AHP Duration" = "provisional",
}
property_validation_design_review_notes = {
  "AHP Duration" = "Temporary stopgap until source-backed quantiles are available.",
}
```

### `pending`

Definition:

- the design choice exists and is runnable, but an appropriately qualified
  person has not yet signed off on it

Use it for:

- new validations
- new rule instances
- new mapping choices
- newly added literature bands

Important:

- this is the normal status for LLM-authored or LLM-assisted draft work
- an LLM can help write the config, but that does not count as validation-design
  review
- `pending` is the explicit marker for "drafted and runnable, but not yet
  scientifically approved by the right kind of reviewer"

Example:

```toml
[validation_design_review]
default_status = "pending"
```

### `not_applicable`

Definition:

- the item is not a scientific validation-design choice and should not count as
  review debt

Use it for:

- skip items
- pure command/control-flow items
- structural non-literature execution scaffolding when appropriate

Example:

```toml
[skip_item]
check_id = "example_validation_skipped"
status = "WARN"
title = "Example validation skipped"
criterion = "The audit should make it explicit when the expensive protocol was not run."
description = "This keeps the CLI informative in skip mode."
acceptable = "The report clearly says no protocol-backed measurements were produced."
acceptable_basis = "This item is generated by control flow rather than by scientific data."
validation_design_review_status = "not_applicable"
```

## Accepted top-level validation TOML keys and sections

The maintained loader and validation engine recognize these top-level keys and
sections.

If you need a new top-level section, add loader support and update this
document, the template, and the HOWTO in the same task.

## What "Expert" Means In This Repo

In this repo, `expert`, `qualified_domain_expert`, and validation-design review
all exist partly because LLM agents are an intended workflow surface.

LLM agents can:

- draft validation configs
- suggest rule choices
- help map paper metrics into normalized forms
- write caveat text and documentation

LLM agents cannot be treated as scientifically reliable reviewers.

They do not count as validation-design review, even when the resulting config
looks clean and runs successfully.

For this metadata, `expert` means a human with the relevant domain background
to judge the scientific validity of the design choice being made, for example:

- whether a protocol mapping is biologically comparable
- whether pooling or subtype separation is justified
- whether a reconstructed reference band is acceptable
- whether a manual extraction/mapping decision is scientifically defensible

### `validation_id`

Definition:

- stable identifier for the validation

Example:

```toml
validation_id = "gc_intrinsic_validation"
```

### `title`

Definition:

- human-facing validation title

Example:

```toml
title = "Granule cell intrinsic validation"
```

### `description`

Definition:

- one-line description of what the validation compares

Example:

```toml
description = "Audit maintained granule-cell models against generic and subtype-specific intrinsic references."
```

### `protocol_runner`

Definition:

- registered protocol runner ID used to produce model-side metrics

Example:

```toml
protocol_runner = "gc_intrinsic_current_clamp"
```

### `metric_group_field`

Definition:

- metric field used to group measurements before group-mean and summary rules
  run

Examples:

```toml
metric_group_field = "cell_type"
```

```toml
metric_group_field = "gc_subtype"
```

### `notes_path`

Definition:

- optional path to a notes table used by `note_presence`

Example:

```toml
notes_path = "research_context/GC_validation_notes.csv"
```

### `skip_neuron_mode`

Definition:

- how `--skip-neuron` behaves for this validation

Accepted values:

- `short_circuit`
- `protocol_handles_skip`

Examples:

```toml
skip_neuron_mode = "short_circuit"
```

```toml
skip_neuron_mode = "protocol_handles_skip"
```

### `extensions`

Definition:

- optional module specs that register new protocol runners or new rule kinds

Examples:

```toml
extensions = ["my_project.validation_extensions"]
```

```toml
extensions = ["my_project.validation_extensions:register"]
```

### `[validation_design_review]`

Definition:

- default validation-design review metadata for items that do not override it

Example:

```toml
[validation_design_review]
default_status = "pending"
default_reviewer = "human"
default_note = "Shared note for items that do not override it."
```

### `[defaults]`

Definition:

- validation-level defaults copied onto parsed CLI args when those args do not
  already have values

Example:

```toml
[defaults]
cell_models = "GC1,GC2,GC3,GC4,GC5"
reference_sigma_multiplier = 2.0
```

### `[protocol]`

Definition:

- default protocol-runner parameter values passed into the registered protocol
  runner

Example:

```toml
[protocol]
target_vm_mV = -58.0
step_start_nA = 0.0
step_stop_nA = 0.30
step_increment_nA = 0.05
```

### `[skip_item]`

Definition:

- optional synthetic item returned when `--skip-neuron` short-circuits the
  validation

Example:

```toml
[skip_item]
check_id = "gc_intrinsic_validation_skipped"
status = "WARN"
title = "Granule-cell intrinsic validation was skipped"
criterion = "The audit should make it explicit when the NEURON-backed intrinsic protocol was not run."
description = "This keeps the declarative validation CLI informative even when the expensive protocol execution is skipped."
acceptable = "The report clearly says that no protocol-backed granule-cell measurements were produced."
acceptable_basis = "This item is generated by command-line control flow rather than scientific data."
evidence_arg_keys = ["jobs", "reference_sigma_multiplier", "cell_models", "reference_gc_subtypes"]
validation_design_review_status = "not_applicable"
```

### `[[checks]]`

Definition:

- one rule instance per array entry

Every check should define:

- `kind`
- `check_id`
- `title`
- `criterion`
- `description`
- `acceptable`
- `acceptable_basis`

Example:

```toml
[[checks]]
kind = "summary_metric_range"
check_id = "synthetic_soma_diameter"
metric_key = "soma_diameter_um"
minimum = 8.9
maximum = 10.3
title = "Synthetic soma diameter matches the target"
criterion = "Target soma diameter is 9.6 plus or minus 0.7 micrometers."
description = "This compares the instantiated surrogate soma diameter against the target range."
acceptable = "The observed soma diameter falls within 8.9 to 10.3 micrometers."
acceptable_basis = "The acceptance range comes from the maintained target constants."
```

## Common optional per-check features

These optional fields are part of the maintained validation-config surface even
though not every rule kind uses every field.

### `validation_design_review_status`, `validation_design_review_reviewer`, `validation_design_review_note`

Definition:

- per-check overrides for validation-design review metadata

Example:

```toml
[[checks]]
kind = "protocol_executed"
check_id = "my_protocol_executed"
title = "My protocol executed"
criterion = "Describe the required protocol."
description = "This confirms the configured protocol actually ran."
acceptable = "At least one metric row is produced."
acceptable_basis = "This is an execution sanity check."
validation_design_review_status = "approved"
validation_design_review_reviewer = "qualified_domain_expert"
validation_design_review_note = "Protocol equivalence manually reviewed."
```

### `enabled_when_arg_truthy`

Definition:

- enables the check only when the named CLI/config arg resolves truthy

Example:

```toml
[[checks]]
kind = "summary_metric_min"
check_id = "candidate_slice_exists"
enabled_when_arg_truthy = "candidate_slice"
metric_key = "candidate_slice_exists_count"
minimum = 1.0
title = "Candidate slice exists"
criterion = "A candidate slice path should be present when candidate-slice mode is requested."
description = "This rule only applies when candidate-slice mode is active."
acceptable = "The observed count is at least one."
acceptable_basis = "The threshold is declared directly in config."
```

### `enabled_when_arg_falsey`

Definition:

- enables the check only when the named CLI/config arg resolves falsey

Example:

```toml
[[checks]]
kind = "summary_metric_range"
check_id = "synthetic_soma_diameter"
enabled_when_arg_falsey = "skip_neuron"
metric_key = "soma_diameter_um"
minimum = 8.9
maximum = 10.3
title = "Synthetic soma diameter matches the target"
criterion = "Target soma diameter is 9.6 plus or minus 0.7 micrometers."
description = "This only applies when the NEURON-backed path actually ran."
acceptable = "The observed diameter falls within 8.9 to 10.3 micrometers."
acceptable_basis = "The accepted interval is declared directly in config."
```

### `enabled_when_arg_in` and `enabled_values`

Definition:

- enables the check only when the named arg value is one of the configured
  allowed values

Example:

```toml
[[checks]]
kind = "note_presence"
check_id = "gc_subtype_fi_caveats"
enabled_when_arg_in = "reference_gc_subtypes"
enabled_values = ["sGC", "dGC"]
title = "Notes / protocol caveats"
criterion = "Subtype current-rate outputs should keep subtype caveats visible."
description = "This rule only applies to the selected subtype targets."
acceptable = "All subtype caveats relevant to the selected subtype are displayed."
acceptable_basis = "The note set is resolved from the selected subtype context."
scope = "fI_validation"
synthetic_contexts = [
  { source = "GC_sGC_dGC_fI_curve.csv", protocol_id = "GERAMITA2016_sGC_dGC_intrinsic_current_clamp", cell_type = "GC", gc_subtype = "sGC" },
  { source = "GC_sGC_dGC_fI_curve.csv", protocol_id = "GERAMITA2016_sGC_dGC_intrinsic_current_clamp", cell_type = "GC", gc_subtype = "dGC" },
]
```

### `filter_field`, `filter_value`, and `filter_values`

Definition:

- filter loaded reference rows down to one field/value subset before the rule
  runs

Example:

```toml
[[checks]]
kind = "reference_curve_match"
check_id = "example_curve_match"
loader = "csv:research_context/EXAMPLE_fI_curve.csv"
filter_field = "protocol_id"
filter_value = "EXAMPLE_PROTOCOL"
protocol_evidence_key = "fi_curve_rows"
reference_current_key = "current_pA"
reference_value_key = "firing_rate_Hz"
model_current_key = "current_pA"
model_value_key = "firing_rate_Hz"
reference_x_unit_text = "pA"
reference_y_unit_text = "Hz"
model_x_unit_text = "pA"
model_y_unit_text = "Hz"
comparison_x_unit_text = "pA"
comparison_y_unit_text = "Hz"
alignment_policy = "exact_transformed_x"
distribution_kind = "empirical_by_x"
score_family = "residual_only"
maximum_mae = 20.0
maximum_rmse = 30.0
minimum_point_count = 5
title = "Model curve stays near the extracted reference curve"
criterion = "Explain the accepted curve mismatch threshold."
description = "Only rows for the configured protocol are compared."
acceptable = "The matched current points satisfy the configured error thresholds."
acceptable_basis = "The current thresholds are declared directly in config."
```

### `filter_values_arg`

Definition:

- pulls filter values from a CLI/config arg instead of hardcoding them

Example:

```toml
[[checks]]
kind = "reference_band_rows"
loader = "gc_subtype_ephys"
filter_field = "gc_subtype"
filter_values_arg = "reference_gc_subtypes"
reference_source = "Geramita et al. (2016)"
group_field = "gc_subtype"
sigma_arg_name = "reference_sigma_multiplier"
property_metric_map = { "Input Resistance" = "input_resistance_MOhm" }
property_band_modes = { "Input Resistance" = "lognormal_sd" }
title = "Reference band placeholder"
criterion = "Unused placeholder text; this rule auto-generates one item per mapped literature row."
description = "Unused placeholder text; this rule auto-generates one item per mapped literature row."
acceptable = "Unused placeholder text; this rule auto-generates one item per mapped literature row."
acceptable_basis = "Unused placeholder text; this rule auto-generates one item per mapped literature row."
```

### `evidence_metric_keys`

Definition:

- appends extra summary metrics to the evidence block for summary rules

Example:

```toml
[[checks]]
kind = "summary_metric_min"
check_id = "baseline_slice_population_counts"
metric_key = "baseline_population_min_count"
minimum = 1.0
title = "Baseline slice contains nonzero populations"
criterion = "The baseline slice should export nonempty key populations."
description = "This is a direct nonzero-count check."
acceptable = "The observed count is at least one."
acceptable_basis = "The threshold is declared directly in config."
evidence_metric_keys = ["baseline_MCs_count", "baseline_TCs_count", "baseline_GCs_count"]
```

### `pass_status` and `fail_status`

Definition:

- override the status emitted when a rule passes or fails

Use it when:

- a comparison is informative but not strong enough to deserve PASS on success

Example:

```toml
[[checks]]
kind = "reference_band_rows"
check_id = "example_soft_reference_band"
pass_status = "WARN"
fail_status = "FAIL"
loader = "csv:research_context/EXAMPLE_ephys.csv"
reference_source = "Example et al. (2026)"
group_field = "cell_type"
sigma_arg_name = "reference_sigma_multiplier"
property_metric_map = { "Skewed Latency" = "latency_ms" }
property_band_modes = { "Skewed Latency" = "quantile_interval" }
title = "Reference band placeholder"
criterion = "Unused placeholder text; this rule auto-generates one item per mapped literature row."
description = "This comparison remains visible as a warning even when inside the accepted interval."
acceptable = "Unused placeholder text; this rule auto-generates one item per mapped literature row."
acceptable_basis = "Unused placeholder text; this rule auto-generates one item per mapped literature row."
```

### `row_contexts`, `as_protocol_context`, and `property_name`

Definition:

- tell `note_presence` how to load rows whose note IDs or protocol IDs should
  generate visible caveats

Example:

```toml
[[checks]]
kind = "note_presence"
check_id = "protocol_caveats"
title = "Notes / protocol caveats"
criterion = "Relevant protocol caveats should remain visible whenever this validation is rendered."
description = "This surfaces protocol caveats instead of burying them in CSV text."
acceptable = "All caveats relevant to the current protocol context are displayed here."
acceptable_basis = "The note set is loaded by matching the configured protocol rows."
scope = "fI_validation"
row_contexts = [
  { loader = "csv:research_context/EXAMPLE_protocols.csv", filter_field = "protocol_id", filter_values = ["EXAMPLE_PROTOCOL"], as_protocol_context = true, property_name = "FI Protocol" },
]
```

### `synthetic_contexts`

Definition:

- inject manually declared note-resolution contexts when the needed context is
  not already present in a loaded CSV

Example:

```toml
[[checks]]
kind = "note_presence"
check_id = "gc_generic_fi_caveats"
title = "Notes / protocol caveats"
criterion = "Generic current-rate outputs should keep extraction-status caveats visible."
description = "This uses a synthetic context because the current-rate point table remains unavailable."
acceptable = "All relevant caveats are displayed here."
acceptable_basis = "The note set is resolved from the synthetic source and protocol context."
scope = "fI_validation"
synthetic_contexts = [
  { source = "GC_fI_curve.csv", protocol_id = "BU2015_GC_intrinsic_current_clamp", cell_type = "GC", gc_subtype = "generic_or_unspecified" },
]
```

### `property_validation_design_review_statuses`, `property_validation_design_review_notes`, and `property_validation_design_review_reviewers`

Definition:

- per-property overrides for `reference_band_rows`, used when one reference-band
  rule expands into many generated items with different review maturity

Example:

```toml
[[checks]]
kind = "reference_band_rows"
...
validation_design_review_reviewer = "qualified_domain_expert"
property_validation_design_review_statuses = {
  "Membrane Resting Voltage" = "approved",
  "AHP Duration" = "provisional",
}
property_validation_design_review_notes = {
  "AHP Duration" = "Temporary stopgap until source-backed quantiles are available.",
}
```

### `property_notes`

Definition:

- optional per-property note text for `reference_band_rows`

Example:

```toml
[[checks]]
kind = "reference_band_rows"
...
property_notes = {
  "Membrane Resting Voltage" = "Reference row comes from a pooled adult cohort.",
}
```

## Skip modes

### `short_circuit`

Definition:

- `--skip-neuron` returns only the configured `[skip_item]`

Use it when:

- the validation is fundamentally simulation-backed
- there is no meaningful cheap partial result

Example:

```toml
skip_neuron_mode = "short_circuit"
```

### `protocol_handles_skip`

Definition:

- `--skip-neuron` still runs cheap checks and lets the protocol runner emit
  status-style items for skipped expensive paths

Use it when:

- some useful structural or source-level checks should still run without the
  expensive NEURON-backed path

Example:

```toml
skip_neuron_mode = "protocol_handles_skip"
```

Current maintained example:

- `epli_correctness`

## Built-in protocols

These are the maintained built-in protocol runners.

### `burton_urban_mctc_current_clamp`

Definition:

- runs maintained mitral-cell and tufted-cell isolated models through the
  Burton and Urban 2014 current-clamp family

Used by:

- `burton_urban_fi`

Example config:

```toml
protocol_runner = "burton_urban_mctc_current_clamp"
```

Example command:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/run_reference_validation.py --validation-id burton_urban_fi --skip-neuron
```

### `epl_fsi_current_clamp`

Definition:

- runs the maintained synthetic EPL fast-spiking interneuron surrogate through
  the Burton, Malyshko, and Urban 2024 current-clamp family

Used by:

- `epl_fsi_intrinsic_validation`

Example config:

```toml
protocol_runner = "epl_fsi_current_clamp"
```

Example command:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/run_reference_validation.py --validation-id epl_fsi_intrinsic_validation --skip-neuron
```

### `epli_correctness_structural`

Definition:

- collects slice-readiness, source-code-default, morphology, and fast-spiking
  scaffold metrics for the provisional EPLI path

Used by:

- `epli_correctness`

Example config:

```toml
protocol_runner = "epli_correctness_structural"
```

Example command:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/run_reference_validation.py --validation-id epli_correctness --skip-neuron
```

### `gc_intrinsic_current_clamp`

Definition:

- runs maintained granule-cell models through the configured intrinsic
  current-clamp protocol and emits the metrics used by GC validations

Used by:

- `gc_intrinsic_validation`

Example config:

```toml
protocol_runner = "gc_intrinsic_current_clamp"
```

Example command:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/run_reference_validation.py --validation-id gc_intrinsic_validation --skip-neuron
```

## Built-in comparison and caveat rule kinds

This section defines every maintained built-in rule kind and shows the minimum
shape of a valid example.

### `protocol_executed`

Definition:

- passes when the protocol runner produced at least one metric row

Use it for:

- top-level execution sanity

Example:

```toml
[[checks]]
kind = "protocol_executed"
check_id = "my_protocol_executed"
title = "My protocol executed"
criterion = "Describe the required protocol."
description = "This confirms the configured protocol actually ran."
acceptable = "At least one metric row is produced."
acceptable_basis = "This is an execution sanity check."
```

### `all_finite_metric`

Definition:

- fails if any metric row has a missing, non-finite, or invalid numeric value
  for the requested metric key

Use it for:

- prerequisite sanity checks

Example:

```toml
[[checks]]
kind = "all_finite_metric"
check_id = "resting_potential_recorded"
metric_key = "resting_potential_mV"
entity_key = "cell_name"
title = "Resting potential was recorded"
criterion = "Every audited model should produce a finite resting membrane potential."
description = "This is a prerequisite sanity check."
acceptable = "Every audited model has a finite resting potential."
acceptable_basis = "A finite resting potential is required before direct literature comparison is meaningful."
```

### `all_exact_metric`

Definition:

- fails if any metric row differs from a fixed expected value by more than the
  configured tolerance

Use it for:

- zero-current quiescence
- exact flag-like metrics

Example:

```toml
[[checks]]
kind = "all_exact_metric"
check_id = "zero_current_quiescence"
metric_key = "zero_step_rate_Hz"
entity_key = "cell_name"
expected = 0.0
tolerance = 1e-9
title = "Cells remain quiescent during the zero-current step"
criterion = "No audited cell should spike during the zero-current step."
description = "This keeps the validation in a positive-rheobase regime."
acceptable = "Every audited model equals zero within tolerance."
acceptable_basis = "This is a protocol-regime sanity check."
```

### `group_ordering`

Definition:

- compares two group means and checks whether the right-hand group is greater
  than or less than the left-hand group

Accepted operators:

- `">"`
- `"<"`

Use it for:

- directional group comparisons from the paper

Example:

```toml
[[checks]]
kind = "group_ordering"
check_id = "tc_fi_gain_higher"
metric_key = "fi_gain_Hz_per_50pA"
left_group = "MC"
right_group = "TC"
operator = ">"
title = "Tufted-cell gain is higher than mitral-cell gain"
criterion = "The paper reports higher gain in tufted cells than in mitral cells."
description = "This is a directional group comparison."
acceptable = "The tufted-cell mean gain is strictly larger than the mitral-cell mean."
acceptable_basis = "The rule checks the direction of the group-mean difference."
```

### `group_abs_diff_max`

Definition:

- compares two group means and requires the absolute difference to stay below a
  configured maximum

Use it for:

- "these groups should stay similar" claims

Example:

```toml
[[checks]]
kind = "group_abs_diff_max"
check_id = "ap_threshold_similarity"
metric_key = "AP_onset_mV"
left_group = "MC"
right_group = "TC"
max_difference = 5.0
title = "Action-potential thresholds remain similar"
criterion = "The paper reports similar thresholds in both groups."
description = "This is a similarity check."
acceptable = "The absolute group-mean difference is no more than five millivolts."
acceptable_basis = "The validation uses a direct maximum-difference threshold."
```

### `group_positive`

Definition:

- requires one or more named group means to be strictly greater than zero

Use it for:

- positive rheobase regime checks

Example:

```toml
[[checks]]
kind = "group_positive"
check_id = "rheobase_positive"
metric_key = "rheobase_pA"
groups = ["MC", "TC"]
title = "Rheobases remain positive"
criterion = "Both groups should require a depolarizing step before spiking."
description = "This is a regime sanity check."
acceptable = "All configured group means are greater than zero."
acceptable_basis = "The rule treats strict positivity as the required condition."
```

### `summary_metric_min`

Definition:

- requires one summary value to be at least the configured minimum

Use it for:

- structural count checks
- nonzero-entry checks

Example:

```toml
[[checks]]
kind = "summary_metric_min"
check_id = "baseline_slice_population_counts"
metric_key = "baseline_population_min_count"
minimum = 1.0
title = "Baseline slice contains nonzero populations"
criterion = "The maintained baseline slice should export nonempty key populations."
description = "This is a direct nonzero-count check."
acceptable = "The observed count is at least one."
acceptable_basis = "The threshold is declared directly in config."
```

### `summary_metric_max`

Definition:

- requires one summary value to be at most the configured maximum

Use it for:

- upper-limit checks

Example:

```toml
[[checks]]
kind = "summary_metric_max"
check_id = "example_latency_ceiling"
metric_key = "latency_ms"
maximum = 25.0
title = "Latency stays below the ceiling"
criterion = "The observed latency should not exceed the accepted ceiling."
description = "This is a direct upper-bound check."
acceptable = "The observed value is at most twenty-five milliseconds."
acceptable_basis = "The threshold is declared directly in config."
```

### `summary_metric_range`

Definition:

- requires one summary value to stay inside a closed interval

Use it for:

- direct numeric range checks

Example:

```toml
[[checks]]
kind = "summary_metric_range"
check_id = "synthetic_soma_diameter"
metric_key = "soma_diameter_um"
minimum = 8.9
maximum = 10.3
title = "Synthetic soma diameter matches the target"
criterion = "Target soma diameter is 9.6 plus or minus 0.7 micrometers."
description = "This compares the observed diameter against the target range."
acceptable = "The observed diameter falls within 8.9 to 10.3 micrometers."
acceptable_basis = "The accepted interval is declared directly in config."
```

### `summary_metric_status_map`

Definition:

- maps a numeric status code to PASS, WARN, or FAIL using configured code sets

Use it for:

- structural/readiness paths that emit explicit status codes

Example:

```toml
[[checks]]
kind = "summary_metric_status_map"
check_id = "epli_reciprocal_architecture"
metric_key = "epli_reciprocal_architecture_code"
pass_values = [2]
warn_values = [1]
fail_values = [0]
title = "Default architecture is reciprocal"
criterion = "Default integration should preserve reciprocal excitatory-inhibitory architecture."
description = "This maps a protocol-emitted status code into audit status."
acceptable = "The configured PASS code indicates the intended architecture."
acceptable_basis = "The code mapping is declared directly in config."
```

### `reference_band_rows`

Definition:

- auto-generates one audit item per mapped literature row and compares the
  observed group mean to an accepted band reconstructed from the reference row

Use it for:

- literature rows with a property name, mean, standard deviation, group label,
  and optional bounds or quantiles

Required concepts:

- `loader`
- `property_metric_map`
- `property_band_modes`

Supported band modes:

- `symmetric_sd`
- `lognormal_sd`
- `beta_sd`
- `binary_indicator`
- `quantile_interval`

Example:

```toml
[[checks]]
kind = "reference_band_rows"
loader = "csv:research_context/EXAMPLE_ephys.csv"
reference_source = "Example et al. (2026)"
group_field = "cell_type"
sigma_arg_name = "reference_sigma_multiplier"
property_metric_map = {
  "Membrane Resting Voltage" = "resting_potential_mV",
  "ISI Coefficient of Variation" = "cv_isi",
}
property_band_modes = {
  "Membrane Resting Voltage" = "symmetric_sd",
  "ISI Coefficient of Variation" = "lognormal_sd",
}
property_lower_bounds = {
  "ISI Coefficient of Variation" = 0.0,
}
title = "Reference band placeholder"
criterion = "Unused placeholder text; this rule auto-generates one item per mapped literature row."
description = "Unused placeholder text; this rule auto-generates one item per mapped literature row."
acceptable = "Unused placeholder text; this rule auto-generates one item per mapped literature row."
acceptable_basis = "Unused placeholder text; this rule auto-generates one item per mapped literature row."
```

### `note_presence`

Definition:

- resolves matching note rows and surfaces them as a warning item so caveats
  stay visible

Use it for:

- protocol caveats
- reference-band caveats
- subtype or maturity caveats

Example:

```toml
[[checks]]
kind = "note_presence"
check_id = "protocol_caveats"
title = "Notes / protocol caveats"
criterion = "Relevant protocol caveats should remain visible whenever this validation is rendered."
description = "This surfaces caveats instead of burying them in raw CSV text."
acceptable = "All caveats relevant to the current protocol context are displayed here."
acceptable_basis = "The note set is loaded by matching the rows in scope."
scope = "fI_validation"
row_contexts = [
  { loader = "csv:research_context/EXAMPLE_protocols.csv", filter_field = "protocol_id", filter_values = ["EXAMPLE_PROTOCOL"], as_protocol_context = true, property_name = "FI Protocol" },
]
```

This rule usually emits:

- `WARN` when matching notes were found
- `PASS` when no matching notes were found

That makes caveats visible without pretending they are hard failures.

### `reference_curve_match`

Definition:

- compares model and reference series as current-conditioned empirical
  distributions over shared transformed x values
- keeps unit metadata explicit for both axes
- still reports practical residual diagnostics such as MAE and RMSE over the
  aligned per-bin mean responses

Use it for:

- current-rate or similar series comparisons where duplicate or repeated x
  values should be treated as a distribution rather than silently averaged
- comparisons where the model and reference axes may need an explicit declared
  transform

Example:

```toml
[[checks]]
kind = "reference_curve_match"
check_id = "reference_curve_match"
loader = "csv:research_context/EXAMPLE_fI_curve.csv"
protocol_evidence_key = "fi_curve_rows"
reference_current_key = "current_pA"
reference_value_key = "firing_rate_Hz"
model_current_key = "current_pA"
model_value_key = "firing_rate_Hz"
reference_x_unit_text = "pA"
reference_y_unit_text = "Hz"
model_x_unit_text = "pA"
model_y_unit_text = "Hz"
comparison_x_unit_text = "pA"
comparison_y_unit_text = "Hz"
x_quantity_name = "Injected current"
y_quantity_name = "Firing rate"
alignment_policy = "exact_transformed_x"
distribution_kind = "empirical_by_x"
score_family = "hybrid_residual_equivalence"
maximum_mae = 25.0
maximum_rmse = 35.0
minimum_point_count = 10
title = "Model curve stays near the reference curve"
criterion = "The model current-rate curve should stay within a moderate error band of the extracted reference points."
description = "This compares the model and reference series over shared transformed x values, treating duplicate x points as empirical distributions."
acceptable = "The shared-current MAE and RMSE stay within the configured limits."
acceptable_basis = "The current thresholds are pragmatic first-pass tolerances for a curve comparison."
```

Important details:

- `reference_x_unit_text`, `reference_y_unit_text`, `model_x_unit_text`,
  `model_y_unit_text`, `comparison_x_unit_text`, and
  `comparison_y_unit_text` are mandatory
- `alignment_policy`, `distribution_kind`, and `score_family` are also
  mandatory; the maintained path does not silently choose the core comparison
  semantics for you
- the current implementation aligns bins using exact shared transformed x
  values after unit conversion and optional explicit affine transforms
- alternatively, `alignment_policy = "nearest_within_tolerance"` matches
  monotone nearest transformed x bins within an explicit
  `x_match_tolerance`
- `alignment_policy = "tolerance_clusters"` pools transformed x values from
  both sides into shared monotone clusters whose span does not exceed
  `x_match_tolerance`, then compares the pooled empirical y distributions per
  cluster
- `alignment_policy = "resampled_grid"` interpolates each per-series path onto
  a shared comparison grid, then compares the resulting empirical y
  distributions per grid point
- the current maintained distribution policy is `empirical_by_x`, which keeps
  duplicate x values as a response distribution instead of collapsing them
  before comparison
- `score_family = "residual_only"` gates on MAE/RMSE thresholds only
- `score_family = "equivalence_only"` gates on per-bin TOST-style equivalence
  tests only
- `score_family = "hybrid_residual_equivalence"` requires both the residual and
  equivalence gates to pass
- equivalence score families use two-sample Welch TOST when both the reference
  and model bins have replication, and one-sample TOST when only one side has
  replication
- if both sides are singleton at a matched x bin, the current maintained path
  does not pretend a statistical equivalence claim is supported there
- `equivalence_margin` is optional for equivalence score families; when it is
  omitted, the maintained path falls back to `maximum_mae`
- `pvalue_aggregation` is optional; `auto` resolves to `max` for equivalence
  score families so every matched bin must satisfy the equivalence gate
- `score_family = "welch_only"` and `score_family = "hybrid_residual_welch"`
  remain available as legacy difference-test diagnostics only
- only legacy Welch-based score families should declare
  `minimum_median_welch_pvalue`
- omitting `pvalue_aggregation` in a legacy Welch family resolves `auto` to
  `median`; the emitted evidence records the resolved choice
- `alignment_policy = "nearest_within_tolerance"` and
  `alignment_policy = "tolerance_clusters"` should also declare
  `x_match_tolerance` explicitly in the comparison x-axis units
- `alignment_policy = "resampled_grid"` defaults `resampling_grid_source` to
  `union_observed_x` and `interpolation_method` to `linear`; the emitted
  evidence records that resolved choice
- if `resampling_grid_source = "explicit_grid"`, also declare
  `resampling_grid_values = [ ... ]`
- if the model x-axis is not expressed in the same physical quantity as the
  reference, declare the mapping explicitly with `model_x_transform` instead of
  pretending the field names are already comparable
- `model_x_transform` and `reference_x_transform` now support:
  - `kind = "identity"`
  - `kind = "affine"`
  - `kind = "piecewise_linear"`
- use `piecewise_linear` when the mapping is monotone but not globally affine;
  declare it with explicit control points such as
  `points = [{input = 0.10, output = 100.0}, {input = 0.20, output = 210.0}]`
  and optional `extrapolation_mode = "forbid" | "constant" | "linear"`
- the emitted evidence now carries both the aligned mean-series arrays used for
  dashboard plotting and compact provenance summaries for the reference and
  model row bundles
- residual/statistical evidence uses unit-neutral keys such as
  `mean_absolute_error`, `root_mean_square_error`, `reference_sd_values`, and
  `equivalence_margin`, with the unit carried separately in `error_unit_text`
  and `comparison_y_unit_text`
- legacy `_Hz` aliases remain only for backward compatibility with older
  current-rate consumers; new code should prefer the unit-neutral keys
- contiguous `reference_curve_match` checks are now compiled into one shared
  series-comparison suite overview plus the per-check detail items
- suite-case score labels and normalized scores now follow the declared
  `score_family` instead of always collapsing to a residual-only label
- reference/model provenance summaries for series comparisons now come from a
  shared typed provenance contract rather than bespoke dict assembly in the
  rule bridge

## Warning, caveat, math, and visualization features

This section covers the features that control how validation items are rendered
once they reach the audit dashboard.

### Warning-state comparisons

There are three common ways to get a warning-state validation item:

1. the rule itself emits `WARN`
2. the rule is configured with `pass_status = "WARN"`
3. the item resolves to `pending` or `provisional`, which adds a
   warning reason when the item status is already `WARN`

Example of a comparison that stays warning-level even when it "passes":

```toml
[[checks]]
kind = "reference_band_rows"
check_id = "example_warning_level_band"
pass_status = "WARN"
fail_status = "FAIL"
loader = "csv:research_context/EXAMPLE_ephys.csv"
reference_source = "Example et al. (2026)"
group_field = "cell_type"
sigma_arg_name = "reference_sigma_multiplier"
property_metric_map = { "Skewed Latency" = "latency_ms" }
property_band_modes = { "Skewed Latency" = "quantile_interval" }
title = "Reference band placeholder"
criterion = "Unused placeholder text; this rule auto-generates one item per mapped literature row."
description = "This comparison remains a warning even when within range because it is informative but not yet strong enough to count as PASS."
acceptable = "Unused placeholder text; this rule auto-generates one item per mapped literature row."
acceptable_basis = "Unused placeholder text; this rule auto-generates one item per mapped literature row."
validation_design_review_status = "provisional"
```

### Notes and caveats

There are two main note/caveat paths:

- `note_presence` to surface caveats as explicit warning items
- `property_notes` or rule-emitted `note` text to attach an item-local caveat

Example of item-local note text on generated reference-band items:

```toml
[[checks]]
kind = "reference_band_rows"
...
property_notes = {
  "AHP Duration" = "Interpret with caution: current band is a temporary positive-only reconstruction.",
}
```

### Math criteria

The dashboard can render:

- `criterion_latex`
- `criterion_formulae`
- `criterion_definitions`

Built-in rules often generate these automatically. Custom rules can set them
explicitly.

Example:

```toml
[[checks]]
kind = "summary_metric_range"
check_id = "example_soma_diameter_range"
metric_key = "soma_diameter_um"
minimum = 8.0
maximum = 12.0
title = "Soma diameter stays inside the accepted range"
criterion = "The observed soma diameter should remain inside the accepted range."
criterion_latex = '\left|\bar{x} - \mu_{\mathrm{ref}}\right| \leq 2\sigma_{\mathrm{ref}}'
criterion_formulae = [
  '\mu_{\mathrm{ref}} = 9.6',
]
criterion_definitions = [
  { symbol = '\bar{x}', definition = 'observed group mean' },
  { symbol = '\mu_{\mathrm{ref}}', definition = 'reference mean' },
  { symbol = '\sigma_{\mathrm{ref}}', definition = 'reference standard deviation' },
]
description = "This uses explicit criterion math for the rendered dashboard."
acceptable = "The observed value stays inside the configured range."
acceptable_basis = "The accepted interval is declared directly in the validation config."
```

### Built-in visual behavior

Several maintained rule paths already render structured visuals:

- `reference_band_rows`
  - persistent reference-interval visual
- `reference_curve_match`
  - explicit current-versus-rate series graph
- `protocol_executed`
  - current-rate series graph when protocol evidence includes `fi_curve_rows`

These do not need extra TOML visualization fields; the rule handlers emit them.

### Custom series visuals

Custom rule kinds can emit `series_visuals` so the dashboard renders a
persistent plotted series in the card body.

Example:

```python
from olfactorybulb.audit import series_visual_spec
from olfactorybulb.audit.reference_validation_rules import register_validation_rule, _rule_item, _rule_status

@register_validation_rule("example_series_rule")
def example_series_rule(rule, context):
    passed = True
    evidence = {
        "currents_pA": [0.0, 50.0, 100.0, 150.0],
        "reference_values_Hz": [0.0, 2.0, 5.0, 8.6],
        "model_values_Hz": [0.0, 1.7, 4.5, 8.1],
    }
    return [
        _rule_item(
            rule,
            status=_rule_status(rule, passed),
            evidence=evidence,
            series_visuals=[
                series_visual_spec(
                    keys=["currents_pA", "reference_values_Hz", "model_values_Hz"],
                    backend="matplotlib",
                    style={"line_width": 1.8, "marker_size": 3.2},
                )
            ],
        )
    ]
```

### Custom companion visuals

Custom rule kinds can emit `companion_visuals` for compact scalar or sequence
visuals such as numeric strips and sparklines.

Example:

```python
from olfactorybulb.audit.reference_validation_rules import register_validation_rule, _rule_item, _rule_status

@register_validation_rule("example_numeric_visual_rule")
def example_numeric_visual_rule(rule, context):
    passed = True
    evidence = {
        "spike_count": 12,
        "response_latency_ms": 18.6,
        "sample_count": 40,
        "trial_values": [0.2, 0.6, 0.9, 1.4, 1.6],
    }
    return [
        _rule_item(
            rule,
            status=_rule_status(rule, passed),
            evidence=evidence,
            companion_visuals=[
                {"kind": "numeric_strip", "keys": ["spike_count", "response_latency_ms", "sample_count"]},
                {"kind": "numeric_sparkline", "key": "trial_values"},
            ],
        )
    ]
```

### `status_reason`

Custom rule kinds can set `status_reason` directly when a warning or failure
needs a short explicit explanation.

Example:

```python
from olfactorybulb.audit.reference_validation_rules import register_validation_rule, _rule_item

@register_validation_rule("example_status_reason_rule")
def example_status_reason_rule(rule, context):
    return [
        _rule_item(
            rule,
            status="WARN",
            evidence={"reason_code": 1},
            status_reason="This item remains a warning because the comparison depends on an intentionally provisional mapping choice.",
        )
    ]
```

## Definitions glossary

### validation

- one declarative literature-backed comparison package: config plus protocol
  plus rules

### reference dataset

- the normalized literature side of the system

### protocol runner

- code that executes the model-side experiment and emits metrics

### metric

- one measured model-side value, such as `resting_potential_mV` or
  `fi_gain_Hz_per_50pA`

### metric row

- one dictionary-like emitted record from the protocol runner

### metric group field

- the metric key used to group rows before group-mean summary rules run

### rule kind

- the comparison logic family named by `kind` inside `[[checks]]`

### reference band

- the accepted interval reconstructed or read from a literature row

### protocol evidence

- extra structured evidence emitted by the protocol runner and surfaced in the
  resulting audit item

### row context

- a configured set of reference rows used to resolve notes or filter
  comparisons

### synthetic context

- a manually declared context row used for note resolution when the needed
  context is not already in a loaded CSV

### caveat

- a protocol, extraction, pooling, or interpretation warning that should stay
  visible in the rendered output

### validation-design review

- human review of the design choice behind a validation item, not of the
  observed result row itself

## Final checklist for adding a paper metric

1. Decide whether the missing piece is dataset, protocol, rule, or config.
2. Normalize the paper data before writing judgment logic.
3. Keep protocol caveats and review metadata explicit.
4. Prefer built-in rules before adding new code.
5. Use `pending` by default for new scientific design choices.
6. Use `provisional` only for acknowledged stopgaps.
7. Use `approved` only after human review.
8. Use `not_applicable` for control-flow items such as skip items.
9. Smoke-test with `--skip-neuron` first when appropriate.
10. Run the full validation.
11. Run `python tools/run_audit.py validation_design_review_status`.
12. Update this tutorial, the validation HOWTO, and the template whenever
    statuses, top-level TOML sections, built-in protocols, or built-in rule
    semantics change.
