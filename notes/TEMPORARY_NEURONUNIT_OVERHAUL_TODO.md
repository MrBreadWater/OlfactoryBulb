# Temporary NeuronUnit Overhaul TODO

This file is a temporary branch-local working list for the
`neuronunit-overhaul` branch. It exists to keep the in-flight overhaul design
constraints visible between sessions while the migration is incomplete.

Remove this file when the listed items are either:
- implemented and verified in the branch, or
- folded into stable permanent docs such as
  `notes/SCIUNIT_NEURONUNIT_AUDIT_SYSTEM_ASSESSMENT_2026-06-14.md`,
  `notes/REFERENCE_VALIDATION_HOWTO.md`, or `AGENTS.md`.

## Current next slice: unit-aware series comparison

- [x] Replace the current ad hoc `reference_curve_match` rule implementation
      with a SciUnit-backed series-comparison core.
- [x] Treat unit handling as mandatory for series comparison, not optional.
- [x] Make the series observation explicit about what the x-axis quantity is:
      for example point current, current density, current flux, or another
      transformed drive quantity.
- [x] Support explicit x-axis transforms so a comparison can declare how to map
      between model output and literature reference coordinates when they are
      not expressed in the same physical quantity.
- [x] Carry axis-unit metadata through the observation / prediction / score
      path using `quantities`, rather than relying on field-name conventions
      like `current_pA` or `firing_rate_Hz`.

## Distribution-oriented comparison direction

- [x] Do not lock the library abstraction to only “curves”.
- [x] Prefer a more general “series observation” or “current-conditioned
      response distribution” concept that can handle:
      - duplicate x-values
      - unaligned x-values
      - sparse overlap
      - pointwise series as a limiting case
- [x] First-pass approximation should be simple and robust, not overfitted:
      likely a current-conditioned distribution summary with explicit alignment
      policy and residual statistics.
- [x] Keep the door open to standard statistical tests or p-value based scores,
      but do not force the first version to pretend precision we do not have.
- [x] Replace the misleading "large Welch p-value implies similarity" shortcut
      with an actual statistical equivalence path.
- [x] The first design should be able to express both:
      - deterministic example-cell comparisons
      - literature-derived distributions or pooled repeated-current datasets

## Decision points to resolve before or during implementation

- [x] Choose the maintained name for the new abstraction:
      `series comparison`.
- [x] Decide the minimum alignment policies we support in v1:
      - exact shared transformed x only
      - monotone nearest transformed x within explicit tolerance
      - duplicate-x aggregation into empirical per-x distributions
      - explicit transformed-axis comparison
      - interpolation/resampling deferred until a later slice
- [x] Decide the first score family:
      - supported v1 score families:
        - `residual_only`
        - `equivalence_only`
        - `hybrid_residual_equivalence`
        - `welch_only` (legacy difference-test diagnostic)
        - `hybrid_residual_welch` (legacy difference-test diagnostic)
      - current maintained EPL-FSI use:
        - `residual_only`
- [x] Decide the first statistical semantics:
      - per-bin Welch TOST for replicated reference/model bins
      - per-bin one-sample TOST when only one side has replication
      - explicit no-claim / failure when both sides are singleton at a matched
        x bin
      - ergonomic defaults:
        - `equivalence_margin` may fall back to `maximum_mae`
        - `pvalue_aggregation = "auto"` resolves to `max` for equivalence
          families and `median` otherwise
- [x] Decide what provenance belongs on the series observation object:
      source rows, source file/location/url, protocol ids, extraction method,
      sample scope, rate definition, note ids, alignment policy, transform
      used, and compact row/series counts.
- [x] Decide how much of the graph-ready aligned data should be emitted from the
      SciUnit side versus reconstructed in the audit shell.
      The SciUnit side now emits aligned mean-series arrays plus sd/count
      diagnostics and provenance summaries; the audit shell owns rendering.

## Migration targets

- [x] Migrate `reference_curve_match` in
      `epl_fsi_intrinsic_validation.validation.toml` to the new SciUnit-backed
      series-comparison core.
- [x] Add direct bridge tests analogous to the reference-band, summary-rule,
      and comparison-rule suite tests.
- [x] Add at least one maintained audit smoke test that exercises the new
      series-comparison path end-to-end.
- [x] Update the overhaul assessment note once the series-comparison contract is
      concrete enough to describe precisely.

## Next priorities

- [x] Broaden transform policies beyond identity / affine when a maintained
      series comparison really needs them.
      - implemented `piecewise_linear` as a declarative monotone transform with
        explicit control points and explicit extrapolation mode
- [x] Extend alignment policies beyond exact shared transformed x bins and
      monotone nearest-within-tolerance matching when a real validation needs
      something better justified than those first two choices.
      - implemented `tolerance_clusters` to pool nearby transformed x bins into
        shared tolerance-bounded empirical comparison groups
- [x] Improve result presentation for the migrated NeuronUnit-backed suite
      outputs without moving repo-level meta checks out of the audit shell.
      - added one shared suite-overview item per migrated suite family
      - added a compact persistent status-matrix companion visual
      - kept report/group summary counts tied to the detailed cases via
        `summary_rollup_exempt`
- [x] Replace the four parallel suite-overview adapter patterns with one shared
      suite result/presentation contract.
      - added `olfactorybulb.neuronunit.suite_presentation`
      - suite matrices can now carry compact per-case score labels in addition
        to status
- [x] Carry score-family-specific suite-case labels and continuous normalized
      score summaries through the migrated series-suite overview instead of
      collapsing every case to a bare MAE label plus binary PASS/FAIL.
- [x] Replace ad hoc series-provenance dict assembly with a shared typed
      provenance layer and route the series suite through it.
- [x] Factor repeated series axis/visual field bundles into typed core specs
      instead of re-plumbing raw keys and units through every helper call.
- [x] Route the declarative `reference_curve_match` builder through those typed
      series specs instead of assembling the flat observation bundle inline.
- [x] Collapse the remaining declarative series-rule parsing into one typed
      `SeriesComparisonRuleSpec` path instead of a spread of small helpers.
- [x] Collapse the declarative `reference_band_rows` parser into typed
      `ReferenceBandRuleSpec` / `ReferenceBandPropertyRuleSpec` objects instead
      of leaving per-property parsing sprawled through the handler.
- [x] Collapse the grouped summary/comparison rule parsing into typed
      `SummaryRuleSpec` / `ComparisonRuleSpec` paths instead of leaving those
      handlers as raw rule-dict plumbing.
- [x] Extract the typed declarative compiler specs into a dedicated
      `olfactorybulb.audit.reference_validation_specs` module so the rule
      handler file stays orchestration-focused.
- [x] Move graphable protocol-evidence row bundles onto a typed contract so
      `protocol_executed`, `reference_curve_match`, and the dashboard do not
      keep special-casing `fi_curve_rows`.
      - added `olfactorybulb.audit.protocol_evidence`
      - registered typed intrinsic f-I evidence specs in the maintained GC and
        EPL-FSI protocol runners
      - `protocol_executed` now consumes those explicit row-series specs
      - `reference_curve_match` can now use the matching protocol series spec
        for model-side default keys/units/labels while keeping reference-side
        and comparison-space units explicit
      - the dashboard can now render explicit row-source series visuals without
        flattening every row bundle into one guessed synthetic series
- [x] Decide whether the next alignment generalization should be interpolation /
      resampling or whether the current distribution-first clustering policy is
      the right stopping point.
      - implemented `resampled_grid` as a distribution-preserving interpolation
        policy
      - it interpolates each per-series path onto a shared comparison grid
      - the current ergonomic defaults are:
        - `resampling_grid_source = "union_observed_x"`
        - `interpolation_method = "linear"`
      - explicit-grid mode remains available through
        `resampling_grid_source = "explicit_grid"` plus
        `resampling_grid_values = [ ... ]`
- [x] Add a declarative context-dependent affine transform for cases where the
      model/reference axis conversion factor lives in explicit row or protocol
      metadata rather than in a validation-local constant.
      - implemented `kind = "affine_lookup"` with optional
        `scale_lookup_key` and `offset_lookup_key`
      - dotted lookup keys resolve against the current row first and the
        protocol-evidence context second
      - this keeps conversions like point current versus current flux explicit
        in config instead of burying them in runner code
- [x] Stop threading protocol evidence as two parallel values and move the
      runtime onto one typed evidence bundle.
      - added `ProtocolEvidenceBundle`
      - `ProtocolRunResult` now carries one bundled evidence object
      - cache annotation, protocol-executed rendering, and series-rule
        fallback resolution now consume that same bundle instead of a raw
        dict plus separate `evidence_series_specs`
- [x] Separate suite aggregation semantics from suite presentation so the
      overview cards stop computing aggregate status/norm rollups as loose dict
      math.
      - added the typed `olfactorybulb.neuronunit.suite_scores` layer
      - suite overviews now emit `suite_aggregate_score` in addition to
        `suite_norm_score_summary`
      - the dashboard matrix header now prefers that typed aggregate rollup
        instead of reconstructing one from summary dicts
- [x] Replace the repeated suite id / kind / candidate-id / aggregate-policy
      argument bundle with one typed suite descriptor.
      - added `SuiteDescriptor`
      - migrated the SciUnit-backed suite adapters to pass that bundle through
        `suite_presentation`
- [x] Extend the new suite-descriptor / aggregate-policy contract to
      `reference_band_rows` so the declarative scientific suite families share
      one consistent rollup path.
      - `ReferenceBandRuleSpec` now carries a `SuiteDescriptor`
      - `reference_band_rows` accepts the same `suite_aggregate_policy` field
- [x] Move the remaining shell-side rule parsing off raw handler-local dict
      plumbing without pulling those rules into the NeuronUnit core.
      - added typed parser specs for `protocol_executed` and `note_presence`
      - kept those rules in the audit shell as intended
- [x] Move the remaining reference-band and series case construction off the
      handler body and into the typed compiler/spec layer.
      - `ReferenceBandRuleSpec.build_cases(...)` now owns the per-row
        reference-band case assembly
      - `SeriesComparisonRuleSpec.to_case(...)` now owns the
        `SeriesComparisonCase` construction
      - `reference_validation_rules.py` now stays closer to orchestration plus
        suite compilation instead of rebuilding flat case payloads inline
- [x] Move the top-level runtime path off loose validation-config dict access
      and onto one compiled validation plan object.
      - added `olfactorybulb.audit.reference_validation_plan`
      - `ReferenceValidationPlan` now owns resolved protocol spec, defaults,
        skip-item construction, design-review defaults, and runtime rule
        context metadata
      - the engine, maintained wrappers, and generic CLI now consume that
        typed plan instead of passing raw config dicts around at runtime
- [x] Add a typed raw-document layer above TOML loading so static config
      consumers stop walking loose dicts too.
      - added `olfactorybulb.audit.reference_validation_document`
      - `ReferenceValidationDocument` now owns typed title/protocol/defaults/
        skip-item/rule metadata directly above the raw TOML loader
      - the runtime plan now compiles from that typed document
      - `validation_design_review_status` and the generic validation listing
        path now consume the typed document instead of raw config accessors
- [x] Collapse the remaining duplicated raw-config accessor layer so
      `reference_validation_config.py` stays a narrow loader/extension module
      and the typed parsing actually lives in the document layer.
      - `ReferenceValidationDocument` now owns the typed title/protocol/
        defaults/rule/skip parsing directly
      - `load_validation_extensions(...)` accepts extension-spec iterables
        directly, so the runtime plan no longer has to fake a mini config dict
        just to register extensions
      - the validation-engine smoke test now exercises the typed document path
        directly instead of asserting against parallel raw-config helpers
- [x] Remove the last generic runtime config dict from the rule-engine boundary.
      - `ValidationRuleContext` now carries typed runtime fields such as
        `validation_id`, `notes_path`, `default_group`, and typed
        validation-design-review defaults
      - `reference_validation_rules.py` and
        `reference_validation_specs.py` no longer depend on a catch-all
        `context.config` tunnel for those values
- [x] Precompile the rule-dispatch sequence inside `ReferenceValidationPlan`
      instead of regrouping contiguous suite families at runtime on every run.
      - the plan now carries grouped/single dispatch entries
      - the runtime executes that compiled sequence directly
      - contiguous summary/comparison/series family grouping is now a
        validation-plan compilation concern instead of a per-run buffering
        concern
- [x] Remove the duplicate raw-rule copy from the runtime plan now that the
      typed document owns raw rules and the typed plan owns compiled dispatches.
      - `ReferenceValidationDocument` remains the home of the raw `checks`
        list
      - `ReferenceValidationPlan` now carries the compiled dispatch sequence
        instead of both dispatches and a second raw-rule tuple
- [x] Extract the typed protocol contract into its own core layer and formalize
      process-local protocol-result caching.
      - added `olfactorybulb.audit.reference_validation_protocol_core`
      - moved typed `ValidationProtocolSpec`, registry lookup, execution-cache
        policy, and cache-key normalization there
      - `ReferenceValidationPlan.run_protocol(...)` now executes through that
        shared contract instead of calling the raw protocol function directly
      - cacheable maintained protocols declare explicit semantic
        `cache_arg_names`
      - the engine smoke test now asserts same-process cache miss/hit behavior
        and argument-sensitive invalidation
- [x] Centralize the migrated NeuronUnit-to-`AuditItem` bridge behind one
      shared adapter contract instead of hand-copying the same presentation
      fields in every suite module.
      - added `AuditItemAdapterSpec` in
        `olfactorybulb.neuronunit.suite_presentation`
      - migrated the reference-band, summary-rule, comparison-rule, and
        series-comparison adapters onto that shared contract
      - added explicit adapter coverage for criterion math, review metadata,
        notes, visuals, status reasons, and overview-rollup preservation
- [x] Add a typed per-case score payload so migrated suite cases no longer
      collapse their semantics to `score_text` plus `norm_score` only.
      - added the shared `SuiteCaseScorePayload` contract
      - suite cases now emit typed `case_score` payloads with score kind,
        numeric value, units, interpretation, and structured observation /
        prediction / normalization metadata when available
      - kept the compact `score_text` label for glanceable matrix rendering
- [x] Make the scalar SciUnit-backed rule families unit-aware through one typed
      metric-quantity contract instead of raw `metric_key` strings plus
      ad hoc labels.
      - added `olfactorybulb.neuronunit.metric_quantities`
      - summary/comparison rule specs now resolve `metric_key` through one
        typed metric quantity with ergonomic suffix-based defaults plus
        explicit `metric_unit_text` / `metric_quantity_name` /
        `metric_observed_symbol` overrides
      - the summary/comparison suite cases, score payloads, and maintained
        evidence now carry those resolved units, quantity names, and observed
        symbols directly
      - the shared typed scalar-observation metadata now carries the same
        observed symbol so downstream adapters do not have to recompute it
- [x] Grow the suite aggregate layer beyond equal-case mean/min/median when a
      migrated family has a principled case weight.
      - added `weighted_mean` to `SuiteAggregatePolicy`
      - suite cases can now carry typed aggregate weights through the shared
        suite-case contract
      - the maintained series-comparison suite currently uses
        `matched_point_count` as that weight source
- [x] Add explicit typed scalar observation / prediction helpers for the
      migrated scalar scientific rules instead of passing raw dict bundles
      between summary/comparison tests and score builders.
      - added `olfactorybulb.neuronunit.scalar_observations`
      - summary rules now carry typed scalar value predictions
      - comparison rules now carry typed scalar map / group-pair /
        grouped-value predictions
      - the overhaul import gate and dedicated regression coverage now include
        that shared scalar layer
- [x] Move `summary_metric_status_map` onto one typed scalar status-map policy
      instead of threading parallel pass/warn/fail value lists through the
      parser, case, scorer, and adapted evidence.
      - added `ScalarStatusMapPolicy` to
        `olfactorybulb.neuronunit.scalar_observations`
      - `SummaryRuleSpec` now compiles that one typed policy for
        `summary_metric_status_map`
      - the summary suite and adapted maintained evidence now consume the
        typed policy rather than raw tuple/list plumbing
- [x] Lift case-level statistical diagnostics into a typed suite-overview
      summary so migrated series suites no longer flatten the p-value layer to
      norm scores only.
      - added `SuiteStatisticalSummary` to the shared suite-score layer
      - series case-score payloads now carry the threshold/statistical-family
        fields that overview summaries need
      - suite overviews and the status-matrix header now surface a compact
        statistical summary when the detailed cases expose compatible
        equivalence or Welch diagnostics
- [x] Make suite-level statistical rollups configurable through the same typed
      descriptor path instead of hardcoding the overview p-value summary per
      score family.
      - added `SuiteStatisticalPolicy` to the shared suite-score/descriptor
        layer
      - grouped validations may now declare `suite_statistical_policy`
      - explicit rollups such as `median` now surface as `rollup_source =
        "explicit"` in the emitted suite statistical summary
- [x] Bind reference/model series rows through one typed observed-dataset
      contract before the scorer computes bins, paths, or provenance.
      - added `SeriesObservedDataset`
      - `SeriesDistributionObservation` now produces typed reference/model
        datasets instead of making the scorer pair raw `rows + spec + context`
        repeatedly
- [x] Grow the resampled-grid alignment path beyond linear-only interpolation
      without reopening the rule surface into bespoke special cases.
      - the shared series-comparison core now supports the explicit
        interpolation family `linear`, `nearest`, and `step_hold`
      - the maintained HOWTO/tutorial/template contract now records those
        options
      - direct suite and declarative rule coverage now exercises the new
        interpolation modes
- [x] Lift protocol metric rows and grouped numeric summaries onto one typed
      shared contract instead of reintroducing fresh raw `list[dict]` /
      `dict[group][metric]` bundles at the protocol, plan, context, and
      migrated-suite seams.
      - added `olfactorybulb.neuronunit.metric_tables`
      - `ProtocolRunResult`, `ValidationRuleContext`,
        `ReferenceValidationPlan`, `ReferenceValidationModel`, and the
        migrated suite compilers now coerce through that shared metric-table
        layer
      - added focused regression coverage for mapping compatibility,
        grouped-summary generation, and protocol/context boundary coercion
- [x] Move the maintained built-in runtime rule-dispatch layer off loose raw
      dict execution and onto typed dispatch records/specs.
      - `compile_rule_dispatches(...)` now emits typed dispatch objects for
        maintained built-in rule kinds
      - `build_rule_items(...)` now executes those typed dispatch objects
        directly instead of re-discovering built-in handler semantics from raw
        dicts
      - the raw `register_validation_rule(...)` hook remains only as the
        compatibility surface for extension-defined custom rules
- [x] Move the typed rule-record layer above the runtime boundary so the
      document/plan path stops carrying raw `checks` dicts.
      - added shared `ValidationRuleRecord` / `coerce_validation_rule_records`
        in `olfactorybulb.audit.reference_validation_rule_records`
      - `ReferenceValidationDocument` now carries `rule_records`
      - `ReferenceValidationPlan` now compiles dispatches from those typed rule
        records
      - the validation-design-review status audit now walks typed rule records
        instead of reparsing loose raw rule dicts from the document
- [x] Finish the summary-range math cleanup so maintained closed intervals use
      the shared absolute-residual form instead of raw endpoint notation.
      - `criterion_math_for_closed_range` now emits `|x - c| <= r` for finite
        closed intervals, one-sided inequalities for semi-bounded intervals,
        and exact equality for zero-width ranges
      - `epli_correctness` now shows forms such as
        `|\bar{x}_{\mathrm{EPLI}} - 9.6| \leq 0.7` and
        `\bar{x}_{\mathrm{EPLI}} \leq 30` on the live maintained path
      - updated the maintained HOWTO/tutorial/template contract to make that
        rendering rule explicit
- [x] Move the Burton MC/TC legacy summary CSV dependency behind the same
      declarative reference-dataset contract used by the other maintained
      literature bundles.
      - added `burton_mc_tc_principal_cells.dataset.toml`
      - moved the legacy source CSVs under
        `research_context/source_data/burton_mc_tc_principal_cells/`
      - added canonical generated outputs for ephys/protocol/manual/readme
      - the Burton validation path now loads those canonical dataset outputs
        instead of normalizing the source CSVs inline at runtime
