# Reference Validation How-To

If you want the shortest explanation first, start with:
[REFERENCE_VALIDATION_SYSTEM_OVERVIEW.md](/home/michael/OlfactoryBulb/notes/REFERENCE_VALIDATION_SYSTEM_OVERVIEW.md)

If you want the full step-by-step tutorial from paper metric to final audit
item, including review-status rules, accepted top-level TOML sections,
built-in rule kinds, built-in protocol runners, warnings/caveats, math, and
visualization features, use:
[REFERENCE_VALIDATION_TUTORIAL.md](/home/michael/OlfactoryBulb/notes/REFERENCE_VALIDATION_TUTORIAL.md)

If you are trying to contribute missing literature values by hand rather than
add new validation logic, use:
[manual_reference_templates/README.md](/home/michael/OlfactoryBulb/research_context/manual_reference_templates/README.md)

If you need the boundary between raw sources, configs, manual intake, and
generated canonical outputs, see:
[research_context/README.md](/home/michael/OlfactoryBulb/research_context/README.md)

This guide explains how to add a simulation-backed, literature-driven
validation audit using the declarative validation layer.

Use this system when you already have normalized reference rows and want to:

- run a specific stimulus protocol against one or more model cells,
- extract model-side measurements from the resulting traces,
- compare those measurements against literature rows, and
- keep protocol caveats visible in the output.

This is the validation-side companion to
[REFERENCE_DATASET_HOWTO.md](/home/michael/OlfactoryBulb/notes/REFERENCE_DATASET_HOWTO.md).

## What is modular now

The validation system is split into four parts:

1. **Reference dataset config**
   - declares and extracts literature rows
   - lives under `research_context/reference_datasets/`

2. **Reference validation config**
   - declares which protocol runner to use
   - declares which rule checks to run
   - lives under `research_context/reference_validations/`

3. **Protocol runners**
   - run the model-side experiment
   - emit metrics and protocol evidence

4. **Rule handlers**
   - consume emitted metrics and normalized reference rows
   - turn comparisons into audit items

The important design point is that `burton_urban_fi` is now just one configured
validation that uses one registered protocol runner:

- validation config:
  [burton_urban_fi.validation.toml](/home/michael/OlfactoryBulb/research_context/reference_validations/burton_urban_fi.validation.toml)
- protocol runner:
  `burton_urban_mctc_current_clamp`

The same framework now also drives:

- [gc_intrinsic_validation.validation.toml](/home/michael/OlfactoryBulb/research_context/reference_validations/gc_intrinsic_validation.validation.toml)
- [epl_fsi_intrinsic_validation.validation.toml](/home/michael/OlfactoryBulb/research_context/reference_validations/epl_fsi_intrinsic_validation.validation.toml)
- [epli_correctness.validation.toml](/home/michael/OlfactoryBulb/research_context/reference_validations/epli_correctness.validation.toml)

## Where the pieces live

- Validation config template:
  [research_context/reference_validations/TEMPLATE.validation.toml](/home/michael/OlfactoryBulb/research_context/reference_validations/TEMPLATE.validation.toml)
- Current built-in validation config:
  [research_context/reference_validations/burton_urban_fi.validation.toml](/home/michael/OlfactoryBulb/research_context/reference_validations/burton_urban_fi.validation.toml)
- Additional built-in validation configs:
  - [gc_intrinsic_validation.validation.toml](/home/michael/OlfactoryBulb/research_context/reference_validations/gc_intrinsic_validation.validation.toml)
  - [epl_fsi_intrinsic_validation.validation.toml](/home/michael/OlfactoryBulb/research_context/reference_validations/epl_fsi_intrinsic_validation.validation.toml)
  - [epli_correctness.validation.toml](/home/michael/OlfactoryBulb/research_context/reference_validations/epli_correctness.validation.toml)
- Validation config loader:
  [olfactorybulb/audit/reference_validation_config.py](/home/michael/OlfactoryBulb/olfactorybulb/audit/reference_validation_config.py)
- Validation engine:
  [olfactorybulb/audit/reference_validation_engine.py](/home/michael/OlfactoryBulb/olfactorybulb/audit/reference_validation_engine.py)
- Built-in protocol registry:
  [olfactorybulb/audit/reference_validation_protocols.py](/home/michael/OlfactoryBulb/olfactorybulb/audit/reference_validation_protocols.py)
- Built-in rule registry:
  [olfactorybulb/audit/reference_validation_rules.py](/home/michael/OlfactoryBulb/olfactorybulb/audit/reference_validation_rules.py)
- Generic CLI:
  [tools/run_reference_validation.py](/home/michael/OlfactoryBulb/tools/run_reference_validation.py)
- Audit HTML dashboard renderer:
  [olfactorybulb/audit/dashboard.py](/home/michael/OlfactoryBulb/olfactorybulb/audit/dashboard.py)
- Unified dashboard shell HOWTO:
  [DASHBOARD_SHELL_HOWTO.md](/home/michael/OlfactoryBulb/notes/DASHBOARD_SHELL_HOWTO.md)

## Quick start

### List available validations

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/run_reference_validation.py --list-validations
```

### List registered protocols

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/run_reference_validation.py --list-protocols
```

### Run the built-in Burton validation

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/run_reference_validation.py --validation-id burton_urban_fi
```

### Smoke-test a validation without running NEURON

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/run_reference_validation.py --validation-id burton_urban_fi --skip-neuron
```

### Run the richer audit wrapper

Use the audit wrapper when you also want Burton-specific slice-context and
registry checks:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/run_audit.py burton_urban_fi
```

### Render audit results as maintained HTML

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m olfactorybulb.audit.dashboard new_sweep --output-dir /tmp/full_audit -- --skip-neuron
```

## Minimal validation config

Copy the template:

```bash
cp \
  research_context/reference_validations/TEMPLATE.validation.toml \
  research_context/reference_validations/my_validation.validation.toml
```

At minimum, fill in:

- `validation_id`
- `title`
- `description`
- `protocol_runner`
- at least one `[[checks]]`

Optional but useful:

- `extensions`
- `[defaults]`
- `[protocol]`
- `[skip_item]`
- `[validation_design_review]`
- `notes_path`
- `skip_neuron_mode`

## How a validation is evaluated

The flow is:

1. load the validation config
2. load any extension modules named in `extensions`
3. resolve the configured `protocol_runner`
4. add the protocol-specific CLI arguments
5. run the protocol runner
6. collect model-side metrics
7. evaluate the configured rule checks
8. render a styled audit report

The protocol runner is responsible for **measurements**.
The rules are responsible for **judgment**.

That boundary matters.

If you need a new measured quantity, add it to the protocol runner output.
If you need a new decision rule, add a new rule kind.

If a result should also render as a chart, declare that in the rule output
instead of letting the dashboard guess from array-shaped evidence. Use the
helper builders in `olfactorybulb.audit` so the visual contract stays explicit:

```python
from olfactorybulb.audit import series_visual_spec

series_visuals = [
    series_visual_spec(
        keys=["currents_pA", "reference_values_Hz", "model_values_Hz"],
        backend="matplotlib",
        style={"line_width": 1.8, "marker_size": 3.2},
    )
]
```

Use `backend="matplotlib"` for the standard plots and reserve `backend="svg"`
for compact bespoke renderers that really need hand-tuned HTML/SVG behavior.

For the SciUnit-backed suite rule families (`reference_band_rows`,
`summary_metric_*`, the grouped comparison rules, and `reference_curve_match`),
the maintained path now emits:

- one compact suite-overview item
- followed by the individual case items

The suite-overview item is intentionally `summary_rollup_exempt`, so the
top-level PASS/WARN/FAIL counts still reflect the detailed validation cases
rather than double-counting the overview card.

The shared suite overview / matrix contract lives in
`olfactorybulb.neuronunit.suite_presentation`. If you add another SciUnit-
backed suite family, reuse that helper layer rather than hand-building a new
overview card or dashboard payload shape inside the specific adapter.

One consequence of that split is that **reference-band assumptions belong in
config**, not hidden in Python defaults. A metric such as membrane resting
voltage can often tolerate a symmetric arithmetic band, while a metric such as
the interspike-interval coefficient of variation is positive-only and usually
needs a different shape.

If a numerical criterion should render compactly in the dashboard, make it
explicit with `criterion_latex`, optional `criterion_formulae`, and
`criterion_definitions`. The definitions list can define functions as well as
variables when a criterion needs them. The audit dashboard uses the bundled
KaTeX runtime in the browser, so direct inequalities and ordinary LaTeX math
render correctly. Keep `criterion` as the plain-language fallback for items
that do not opt into math rendering:

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
criterion_definitions = [
  { symbol = '\bar{x}', definition = 'observed group mean' },
  { symbol = '\mu_{\mathrm{ref}}', definition = 'uploaded reference mean' },
  { symbol = '\sigma_{\mathrm{ref}}', definition = 'uploaded reference standard deviation' },
]
```

For reference-band rows, prefer a compact absolute-residual inequality when the
rule is centered around a mean, and keep explicit endpoint notation only for
modes that really need asymmetric bounds. The rendered headline inequality
substitutes the configured sigma multiplier numerically, so a
`reference_sigma_multiplier = 2` setting will show `2\sigma_{\mathrm{ref}}`
or `2\sigma_{\log}` in the display. For example:

- symmetric bands:
  `\left|\bar{x} - \mu_{\mathrm{ref}}\right| \leq k\sigma_{\mathrm{ref}}`
- lognormal bands:
  `\left|\ln(\bar{x}) - \mu_{\log}\right| \leq k\sigma_{\log}`, with
  `\mu_{\log} = \ln(\mu_{\mathrm{ref}}) - \sigma_{\log}^2 / 2` and
  `\sigma_{\log} = \sqrt{\ln(1 + (\sigma_{\mathrm{ref}} / \mu_{\mathrm{ref}})^2)}`

For mean-plus-SD criteria, prefer these residual forms over standardized
z-score notation or endpoint notation. They avoid redundant function
definitions while keeping the sigma multiplier visible in the headline
criterion. For lognormal reconstructions, `\sigma_{\log}` is the reconstructed
log-space standard deviation; do not put the uploaded arithmetic
`\sigma_{\mathrm{ref}}` on the right-hand side of the log-space inequality.

When a metric has a conventional symbol, prefer it over `\bar{x}`. Examples
include `\bar{R}_{\mathrm{in}}`, `\bar{\tau}_m`, `\bar{I}_{\mathrm{rh}}`,
and `\bar{V}_{\mathrm{rest}}`.

Built-in rule kinds generate their own criterion math through
`olfactorybulb.audit.criterion_math`. Do not duplicate those TeX strings in
validation configs. For custom rule kinds, use the same builder module instead
of hand-writing a separate notation convention inside each rule. Centered
tolerance checks should render as absolute residuals, for example
`\left|x - c\right| \leq \epsilon`, rather than endpoint forms such as
lower-target-to-upper-target inequalities.

The framework now supports two skip behaviors:

- `short_circuit`
  - `--skip-neuron` returns only the configured `[skip_item]`
  - best for purely simulation-backed validations such as current-clamp sweeps
- `protocol_handles_skip`
  - the protocol runner still executes cheap checks and emits warning-state
    metrics for the expensive skipped parts
  - best for mixed audits such as `epli_correctness`, where source-code and
    slice-export checks should still run when NEURON-backed morphology or
    behavior checks are skipped

## Validation-design review status is mandatory

Every declarative validation item should resolve to a validation-design review
state.

It means:

- whether an appropriately qualified expert has reviewed the validation-design choice itself
- whether the chosen reference-band shape is acceptable
- whether a pooling/separation rule is acceptable
- whether a protocol-equivalence assumption is acceptable
- whether a manual extraction or mapping choice is acceptable

Why this field exists:

- LLM-assisted drafting is a supported workflow in this repo
- but LLM agents are not scientifically reliable enough to count as review
- so this metadata exists to distinguish "an LLM or script assembled this
  candidate validation design" from "an appropriately qualified person has
  actually reviewed the scientific validity of that design choice"

It does not mean:

- that someone has manually checked this specific observed audit result
- that someone has read and verified the full stack of code the audit uses
- that the underlying model implementation is therefore endorsed in general
- that every upstream paper/source file was re-reviewed in full for this run

Use the shared vocabulary:

- `approved`
- `provisional`
- `pending`
- `not_applicable`

At minimum, set:

```toml
[validation_design_review]
default_status = "pending"
```

Then override where needed:

- `validation_design_review_status` on a single check
- `validation_design_review_reviewer` and `validation_design_review_note` on a single check
- `property_validation_design_review_statuses` for `reference_band_rows`
- `property_validation_design_review_notes` for per-property caveats

Think of these as review fields for the test design, not for the observed run
result. In practice, they are also the boundary between LLM-assisted drafting
and qualified scientific sign-off.

Example:

```toml
[validation_design_review]
default_status = "pending"

[[checks]]
kind = "reference_band_rows"
...
validation_design_review_reviewer = "qualified_domain_expert"
property_validation_design_review_statuses = {
  "ISI Coefficient of Variation" = "approved",
  "AHP Duration" = "provisional",
}
property_validation_design_review_notes = {
  "AHP Duration" = "Temporary stopgap until source-backed quantiles are available.",
}
```

Run the dedicated coverage audit to check this metadata:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/run_audit.py validation_design_review_status
```

That audit:

- fails if any declarative item resolves to no review status
- fails if a config uses an unknown status string
- warns on `pending`
- warns on `provisional`

The audit title is `Validation design review status audit`.

## Built-in rule kinds

The built-in rule layer already covers common cases:

- `protocol_executed`
- `all_finite_metric`
- `all_exact_metric`
- `group_ordering`
- `group_abs_diff_max`
- `group_positive`
- `summary_metric_min`
- `summary_metric_max`
- `summary_metric_range`
- `summary_metric_status_map`
- `reference_band_rows`
- `reference_curve_match`
- `note_presence`

Use config alone whenever one of these can express the paper cleanly.

For `reference_curve_match`, treat axis metadata as part of the rule contract:

- declare reference/model x and y units explicitly
- declare comparison units explicitly
- use an explicit transform when the model and reference axes are not already
  the same physical quantity
- declare `alignment_policy`, `distribution_kind`, and `score_family`
  explicitly; there is no silent policy fallback for the core comparison
  semantics
- secondary knobs may still use stable ergonomic defaults when the emitted
  evidence records the resolved choice

Current maintained series-comparison policy choices are:

- `alignment_policy = "exact_transformed_x"`
  - compare only shared x bins after unit conversion and any explicit
    transform
- `alignment_policy = "nearest_within_tolerance"`
  - compare monotone nearest transformed x bins within an explicit
    `x_match_tolerance`
- `alignment_policy = "tolerance_clusters"`
  - pool transformed x bins from both sides into shared monotone clusters whose
    span does not exceed `x_match_tolerance`, then compare the pooled empirical
    y distributions per cluster
- `alignment_policy = "resampled_grid"`
  - interpolate each per-series path onto a shared comparison grid, then
    compare the resulting empirical y distributions per grid point
- `distribution_kind = "empirical_by_x"`
  - treat duplicate x values as an empirical response distribution at each x
- `score_family`
  - `residual_only`
  - `equivalence_only`
  - `hybrid_residual_equivalence`
  - `welch_only` (legacy difference-test diagnostic)
  - `hybrid_residual_welch` (legacy difference-test diagnostic)

Use the equivalence families for an actual statistical similarity claim. They
run per-bin TOST-style equivalence tests:

- replicated reference plus replicated model bins -> two-sample Welch TOST
- replicated one side plus singleton other side -> one-sample TOST against the
  singleton point target
- singleton vs singleton bins -> no statistical equivalence claim; the
  statistical gate fails because the data do not support it

For equivalence families:

- `equivalence_margin` is optional; if omitted, the maintained path falls back
  to the configured `maximum_mae`
- `equivalence_alpha` defaults to `0.05`
- `pvalue_aggregation` is optional; `auto` resolves to `max`, which means every
  matched bin must satisfy the equivalence gate

Use `minimum_median_welch_pvalue` only with a legacy Welch-based
`score_family`. Those Welch families remain available for backward-compatible
diagnostics, but a large two-sample Welch p-value is not evidence of
equivalence. When `pvalue_aggregation` is omitted there, `auto` resolves to
`median` and the emitted evidence records that resolved choice.

The emitted residual/statistical evidence now uses unit-neutral keys such as:

- `mean_absolute_error`
- `root_mean_square_error`
- `max_absolute_error`
- `reference_sd_values`
- `model_sd_values`
- `equivalence_margin`

and carries the unit separately in `error_unit_text` plus the existing
`comparison_y_unit_text`. Legacy `_Hz` aliases remain only for backward
compatibility with older current-rate consumers.

Contiguous `reference_curve_match` checks now compile into one shared
series-comparison suite overview plus the per-check detail items, rather than
emitting one one-case suite overview per rule.

The emitted suite-case score labels follow the declared `score_family`:
- `residual_only`: residual diagnostic text such as `MAE 12.4 Hz`
- `equivalence_only`: equivalence diagnostic text such as `TOST p 0.013`
- hybrid families: both residual and statistical text

When normalized case scores are available, the suite overview also emits a
`suite_norm_score_summary` so downstream dashboards or reports can show an
aggregate continuous score summary instead of only binary pass/fail counts.

Series-comparison provenance summaries and compact protocol-context summaries
now come from the shared typed layer in
`olfactorybulb.neuronunit.provenance`. If you need to extend the emitted
reference/model provenance payload, do it there first and let the suite bridge
adapt the typed object back into audit evidence.

Likewise, the declarative `reference_curve_match` builder now compiles the
reference/model axis bundles into typed `SeriesDataSpec` plus
`SeriesVisualContract` objects before it constructs the observation. If you
change how series axes, units, transforms, or visual payload keys are declared,
extend those typed specs rather than reintroducing a long flat field bundle at
the rule-builder callsite. Keep the declarative parsing centralized in the
typed `SeriesComparisonRuleSpec` path so one reader can recover the full
series-rule contract from one place.

When you use `alignment_policy = "nearest_within_tolerance"` or
`alignment_policy = "tolerance_clusters"`, declare `x_match_tolerance`
explicitly in the comparison x-axis units.
When you use `alignment_policy = "resampled_grid"`, the maintained path now
defaults `resampling_grid_source` to `union_observed_x` and
`interpolation_method` to `linear`. If you need a different grid, declare one
of:

- `resampling_grid_source = "reference_observed_x"`
- `resampling_grid_source = "model_observed_x"`
- `resampling_grid_source = "union_observed_x"`
- `resampling_grid_source = "explicit_grid"`
  - and then also declare `resampling_grid_values = [ ... ]`

For transforms, start with:

- `kind = "identity"` when the source axis is already expressed in the desired
  comparison quantity
- `kind = "affine"` for simple scaling/offset conversions such as current flux
  to injected current under a fixed linear mapping
- `kind = "piecewise_linear"` when the literature/model x-axis relationship is
  monotone but not well described by one global scale/offset

For `piecewise_linear`, declare at least two control points and keep the mapping
explicit in the validation config. The current transform object accepts
`points = [{input = ..., output = ...}, ...]` plus an optional
`extrapolation_mode = "forbid" | "constant" | "linear"`.
The current maintained EPL-FSI example-cell comparison uses
`score_family = "residual_only"` because the model side usually exposes one
response trace per current step, so a formal per-bin two-sample test is not
the main decision criterion there.

## Choosing acceptable bands for literature rows

`reference_band_rows` no longer assumes that every metric should use the same
arithmetic `mean +/- sigma * sd` interval.

It also no longer permits a silent fallback. Every property listed in
`property_metric_map` must appear in `property_band_modes`. If one is missing,
the validation fails immediately instead of quietly defaulting to a symmetric
band.

The maintained `reference_band_rows` parser now routes this per-property
contract through typed `ReferenceBandRuleSpec` /
`ReferenceBandPropertyRuleSpec` objects before building the SciUnit-backed
cases. If you change bounds, quantile-field selection, or per-property review
metadata, keep that parser path as the single declarative source of truth.

Use these config knobs deliberately:

- `property_band_modes`
  - per-property band shape
  - built-ins:
    - `symmetric_sd`
    - `lognormal_sd`
    - `beta_sd`
    - `binary_indicator`
  - this choice is required for every property; there is no fallback mode
    - `quantile_interval`
- `property_lower_bounds`
  - clip naturally non-negative metrics at zero
- `property_upper_bounds`
  - clip bounded metrics such as probabilities at one

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
  "Firing Probability" = "firing_probability",
  "Rebound Potential Presence" = "rebound_potential_presence",
  "Skewed Latency" = "skewed_latency_ms",
}
property_band_modes = {
  "Membrane Resting Voltage" = "symmetric_sd",
  "ISI Coefficient of Variation" = "lognormal_sd",
  "Firing Probability" = "beta_sd",
  "Rebound Potential Presence" = "binary_indicator",
  "Skewed Latency" = "quantile_interval",
}
property_lower_bounds = {
  "ISI Coefficient of Variation" = 0.0,
  "Firing Probability" = 0.0,
}
property_upper_bounds = { "Firing Probability" = 1.0 }
default_quantile_low_field = "q_low"
default_quantile_high_field = "q_high"
default_quantile_low_label_field = "q_low_label"
default_quantile_high_label_field = "q_high_label"
title = "Reference band placeholder"
criterion = "Unused placeholder text; this rule auto-generates one item per mapped literature row."
description = "Unused placeholder text; this rule auto-generates one item per mapped literature row."
acceptable = "Unused placeholder text; this rule auto-generates one item per mapped literature row."
acceptable_basis = "Unused placeholder text; this rule auto-generates one item per mapped literature row."
```

Use this decision tree explicitly, one property at a time:

1. If the source row encodes an exact binary class label such as present vs not
   present, use `binary_indicator`.
2. Else if the source or extracted row already provides an explicit empirical
   interval such as 5th-to-95th percentile or Q1-to-Q3, use
   `quantile_interval`.
3. Else if the metric is a true probability or fraction on `[0, 1]`, use
   `beta_sd`.
4. Else if the metric can meaningfully cross zero or is naturally signed, use
   `symmetric_sd`.
5. Else if the metric is positive-only continuous and only mean plus standard
   deviation are available, use `lognormal_sd`.

That is the default manual policy in this repo. If a paper clearly requires a
different assumption, encode it explicitly in config and explain why in the row
notes or validation notes.

Use `lognormal_sd` when all of the following are true:

- the metric is strictly positive,
- the literature value is better treated as right-skewed than symmetric, and
- the source gives only arithmetic mean and standard deviation.

Use `beta_sd` when all of the following are true:

- the metric is a probability, fraction, or other quantity naturally bounded to
  `[0, 1]`,
- the source gives an arithmetic mean and standard deviation,
- and you want a bounded interval that is more defensible than simply clipping a
  symmetric Gaussian-style band.

Use `quantile_interval` when the paper already reports an interval directly,
such as:

- 25th to 75th percentile,
- 5th to 95th percentile,
- lower to upper quartile,
- any other explicit low/high quantile pair.

In that case, add row fields such as:

- `q_low`
- `q_high`
- `q_low_label`
- `q_high_label`

- `binary_indicator` is for categorical rows such as a presence/absence class
  that were historically squeezed into a mean-plus-or-minus-spread column.
  These are not treated as continuous dispersion rows at all. The accepted
  interval becomes the exact binary target.

Use `property_lower_bounds` / `property_upper_bounds` when the metric has a
hard physical or definitional bound even if you still want a symmetric band in
the interior.

Do not describe these bands as formal confidence intervals unless the source
actually reports a confidence interval and you model it that way. In this
framework, the default `reference_sigma_multiplier` is a configurable
**dispersion-band width**, not a statistical confidence level.

Rendered reports now expose the assumption explicitly through:

- `accepted_interval_mode`
- `accepted_interval_standard`

Do not hide those fields. If the standard is questionable for a given metric,
change the config rather than quietly accepting the default.

If the paper needs a genuinely different comparison rule, register a new rule
kind in an extension module.

## Registering a new protocol runner

If a paper has a stimulus protocol that is not already represented, create an
extension module and point the validation config at it through `extensions`.

For example, add:

```toml
validation_id = "smith2026_intrinsic_validation"
title = "Smith 2026 intrinsic validation"
description = "Validate Example Cells against Smith 2026."
protocol_runner = "smith2026_current_clamp"
extensions = ["olfactorybulb.audit.smith2026_validation_extensions:register"]
metric_group_field = "cell_type"
```

Then create an extension module that registers the protocol:

```python
from __future__ import annotations

import argparse

from olfactorybulb.audit.reference_validation_protocols import (
    ProtocolRunResult,
    ValidationProtocolSpec,
    register_validation_protocol,
)


def _add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cell-count", type=int, default=4)
    parser.add_argument("--dt-ms", type=float, default=0.1)


def _run_protocol(args: argparse.Namespace, protocol_config: dict[str, object]) -> ProtocolRunResult:
    metrics = [
        {
            "cell_name": "Example1",
            "cell_type": "Example Cell",
            "input_resistance_MOhm": 123.4,
            "fi_gain_Hz_per_50pA": 9.8,
        }
    ]
    protocol_evidence = {
        "step_duration_ms": protocol_config.get("step_duration_ms", 1000.0),
        "current_start_pA": protocol_config.get("current_start_pA", 0.0),
        "current_stop_pA": protocol_config.get("current_stop_pA", 400.0),
        "current_step_pA": protocol_config.get("current_step_pA", 50.0),
    }
    return ProtocolRunResult(metrics=metrics, protocol_evidence=protocol_evidence, group_field="cell_type")


def register() -> None:
    register_validation_protocol(
        ValidationProtocolSpec(
            protocol_id="smith2026_current_clamp",
            title="Smith 2026 example current clamp",
            description="Example protocol registration for a literature-backed validation.",
            add_cli_args=_add_cli_args,
            run=_run_protocol,
        )
    )
```

Key point:

- the protocol runner can emit any measurement keys you want
- those keys become available to rule checks
- protocol runners are not limited to current clamp

They can just as easily emit:

- slice export counts
- source-code default status codes
- morphology measurements
- network-readiness booleans
- protocol-backed trace measurements

## Adding custom measurements

Suppose the paper cares about `first_spike_latency_ms`, but none of the current
protocol runners emit it yet.

The correct place to add it is the protocol runner:

```python
metrics = [
    {
        "cell_name": "Example1",
        "cell_type": "Example Cell",
        "first_spike_latency_ms": 212.0,
    }
]
```

Once the metric exists, you have two options:

1. express the judgment using a built-in rule, if possible
2. register a new rule kind if the logic is new

When a paper needs a brand-new **protocol family**, write a new registered
protocol runner.

When a paper only needs a brand-new **measurement** inside an already-matching
protocol family, extend that runner’s emitted metrics instead of creating a
second near-duplicate runner.

## Registering a new rule kind

If the built-in checks are not enough, register a new rule kind in the same
extension module:

```python
from olfactorybulb.audit import AuditItem
from olfactorybulb.audit.reference_validation_rules import register_validation_rule


@register_validation_rule("minimum_metric")
def _minimum_metric(rule, context):
    metric_key = str(rule["metric_key"])
    minimum = float(rule["minimum"])
    observed = float(context.metrics[0][metric_key])
    status = "PASS" if observed >= minimum else "FAIL"
    return [
        AuditItem(
            check_id=str(rule["check_id"]),
            status=status,
            title=str(rule["title"]),
            criterion=str(rule["criterion"]),
            description=str(rule["description"]),
            acceptable=str(rule["acceptable"]),
            acceptable_basis=str(rule["acceptable_basis"]),
            evidence={"observed": observed, "minimum": minimum},
        )
    ]
```

Then use it in the validation config:

```toml
[[checks]]
kind = "minimum_metric"
check_id = "example_first_spike_latency"
metric_key = "first_spike_latency_ms"
minimum = 150.0
title = "First-spike latency exceeds the lower bound"
criterion = "Explain the literature requirement here."
description = "Explain why this check matters."
acceptable = "The observed first-spike latency is at least 150 milliseconds."
acceptable_basis = "This threshold comes from the cited paper."
```

## When to add a new protocol runner versus a new rule

Add a **new protocol runner** when the paper changes:

- stimulus family
- current-step schedule
- holding potential normalization
- temperature assumptions
- trace-processing pipeline
- measured quantities

Also add a new protocol runner when the paper is not really a current-clamp
paper at all, for example:

- structural or morphology audits
- slice-export readiness audits
- synaptic latency protocols
- modulation protocols
- mixed audits that combine cheap static checks with optional NEURON-backed
  measurements

Add a **new rule kind** when the paper changes:

- comparison logic
- pass/fail decision style
- grouping semantics
- reference-band construction
- protocol-caveat resolution logic

Do not add a new protocol runner merely because the paper wants a different
ordering check or tolerance band. That belongs in config or a rule.

## Skip behavior

Use `[skip_item]` in the validation config so `--skip-neuron` still produces a
useful report when the validation is `short_circuit`.

Use `skip_neuron_mode = "protocol_handles_skip"` when the validation should
still execute cheap source-code, dataset, or slice checks while reporting the
NEURON-backed parts as skipped warnings.

Example:

```toml
[skip_item]
check_id = "smith2026_validation_skipped"
status = "WARN"
title = "Smith 2026 validation was skipped"
criterion = "The report should say when the expensive protocol was not run."
description = "This keeps the generic CLI informative during smoke tests."
acceptable = "The report explicitly says that no protocol-backed measurements were produced."
acceptable_basis = "This item is generated by command-line control flow."
evidence_arg_keys = ["reference_sigma_multiplier", "cell_count"]
```

## Notes and protocol caveats

If the validation involves protocol-dependent comparisons, add a `note_presence`
check so caveats remain visible.

Important implementation details:

- set `notes_path` when the dataset uses a dataset-local notes table instead of
  the shared default
- use `row_contexts` when the caveat should be resolved from real extracted rows
- use `synthetic_contexts` when the caveat is driven by config state or by an
  intentionally empty extracted file
- use `filter_value_arg` or `filter_values_arg` when the relevant note context
  comes from CLI-selected targets such as `--reference-gc-subtypes`

That is what keeps differences such as:

- MC/TC protocol versus EPL fast-spiking interneuron protocol
- baseline versus modulated condition
- superficial granule cell versus deep granule cell subtype

from being buried in raw CSV text.

## Recommended workflow for a new paper

1. create or update the normalized reference dataset
2. copy `TEMPLATE.validation.toml`
3. point `protocol_runner` at an existing runner if one already matches
4. if no runner matches, write an extension module and register a new protocol
5. add checks using built-in rule kinds first
6. add a new rule kind only when config cannot express the comparison
7. decide whether `--skip-neuron` should short-circuit or whether the protocol
   should keep running cheap checks
8. add a `[skip_item]` so smoke runs stay readable
9. if the paper uses dataset-local caveats, set `notes_path` and add
   `note_presence` checks
10. when one config should cover multiple related targets, gate checks with
   `enabled_when_arg_truthy`, `enabled_when_arg_falsey`, or
   `enabled_when_arg_in`
11. run the generic validation CLI
12. add a dedicated audit wrapper only if you also need repo-specific structural
   or context checks beyond the literature comparison itself

## Practical boundary

The generic validation layer is intended to make new literature-backed tests
cheap to add.

It is not intended to eliminate scientific judgment.

The paper-specific judgment should live in:

- the normalized reference rows,
- the selected protocol runner,
- the configured checks,
- and explicit notes/caveats.

That is the level where new validations stay understandable to other people in
the lab instead of turning back into one-off audit scripts.
