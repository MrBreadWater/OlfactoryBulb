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

- [ ] Replace the current ad hoc `reference_curve_match` rule implementation
      with a SciUnit-backed series-comparison core.
- [ ] Treat unit handling as mandatory for series comparison, not optional.
- [ ] Make the series observation explicit about what the x-axis quantity is:
      for example point current, current density, current flux, or another
      transformed drive quantity.
- [ ] Support explicit x-axis transforms so a comparison can declare how to map
      between model output and literature reference coordinates when they are
      not expressed in the same physical quantity.
- [ ] Carry axis-unit metadata through the observation / prediction / score
      path using `quantities`, rather than relying on field-name conventions
      like `current_pA` or `firing_rate_Hz`.

## Distribution-oriented comparison direction

- [ ] Do not lock the library abstraction to only “curves”.
- [ ] Prefer a more general “series observation” or “current-conditioned
      response distribution” concept that can handle:
      - duplicate x-values
      - unaligned x-values
      - sparse overlap
      - pointwise series as a limiting case
- [ ] First-pass approximation should be simple and robust, not overfitted:
      likely a current-conditioned distribution summary with explicit alignment
      policy and residual statistics.
- [ ] Keep the door open to standard statistical tests or p-value based scores,
      but do not force the first version to pretend precision we do not have.
- [ ] The first design should be able to express both:
      - deterministic example-cell comparisons
      - literature-derived distributions or pooled repeated-current datasets

## Decision points to resolve before or during implementation

- [ ] Choose the maintained name for the new abstraction:
      `series comparison`, `series observation`, `curve comparison`, or
      `current-conditioned response distribution`.
- [ ] Decide the minimum alignment policies we support in v1:
      - exact shared x only
      - duplicate-x aggregation
      - interpolation/resampling
      - transformed-axis comparison
- [ ] Decide the first score family:
      - MAE / RMSE with unit awareness
      - distribution-distance metric
      - formal statistical-test score
      - hybrid score with attached diagnostics
- [ ] Decide what provenance belongs on the series observation object:
      source rows, protocol metadata, extraction method, alignment policy,
      transform used, and any caveats about example-cell vs population data.
- [ ] Decide how much of the graph-ready aligned data should be emitted from the
      SciUnit side versus reconstructed in the audit shell.

## Migration targets

- [ ] Migrate `reference_curve_match` in
      `epl_fsi_intrinsic_validation.validation.toml` to the new SciUnit-backed
      series-comparison core.
- [ ] Add direct bridge tests analogous to the reference-band, summary-rule,
      and comparison-rule suite tests.
- [ ] Add at least one maintained audit smoke test that exercises the new
      series-comparison path end-to-end.
- [ ] Update the overhaul assessment note once the series-comparison contract is
      concrete enough to describe precisely.
