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
- [ ] Decide whether the next alignment generalization should be interpolation /
      resampling or whether the current distribution-first clustering policy is
      the right stopping point.
