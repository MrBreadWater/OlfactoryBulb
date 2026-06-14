# Justas Original Tests vs Maintained Validation Surfaces

Date: 2026-06-14

## Scope

This note compares Justas' original electrophysiology-testing stack with the
current maintained validation stack in this repository. It is narrower than the
broader SciUnit/NeuronUnit architecture assessment in
[`notes/SCIUNIT_NEURONUNIT_AUDIT_SYSTEM_ASSESSMENT_2026-06-14.md`](./SCIUNIT_NEURONUNIT_AUDIT_SYSTEM_ASSESSMENT_2026-06-14.md).

The goal here is practical parity analysis:

1. What the old tests actually did
2. What the maintained validations do now
3. Which old measurements have a maintained analogue
4. Which old measurements or workflows are still missing
5. Where the maintained stack is stronger than the old one

## Primary source files examined

### Legacy Justas test surface

- [`olfactorybulb/neuronunit/tests/tests.py`](../olfactorybulb/neuronunit/tests/tests.py)
- [`olfactorybulb/neuronunit/tests/publications.py`](../olfactorybulb/neuronunit/tests/publications.py)
- [`olfactorybulb/neuronunit/tests/__init__.py`](../olfactorybulb/neuronunit/tests/__init__.py)
- [`olfactorybulb/neuronunit/models/neuron_cell.py`](../olfactorybulb/neuronunit/models/neuron_cell.py)
- [`olfactorybulb/neuronunit/capabilities/__init__.py`](../olfactorybulb/neuronunit/capabilities/__init__.py)
- [`olfactorybulb/neuronunit/tests/utilities.py`](../olfactorybulb/neuronunit/tests/utilities.py)
- [`notebooks/ephyz-validation.ipynb`](../notebooks/ephyz-validation.ipynb)

### Maintained validation surface

- [`olfactorybulb/audit/reference_validation_engine.py`](../olfactorybulb/audit/reference_validation_engine.py)
- [`olfactorybulb/audit/reference_validation_protocols.py`](../olfactorybulb/audit/reference_validation_protocols.py)
- [`olfactorybulb/audit/reference_validation_rules.py`](../olfactorybulb/audit/reference_validation_rules.py)
- [`research_context/reference_validations/burton_urban_fi.validation.toml`](../research_context/reference_validations/burton_urban_fi.validation.toml)
- [`research_context/reference_validations/gc_intrinsic_validation.validation.toml`](../research_context/reference_validations/gc_intrinsic_validation.validation.toml)
- [`research_context/reference_validations/epl_fsi_intrinsic_validation.validation.toml`](../research_context/reference_validations/epl_fsi_intrinsic_validation.validation.toml)
- [`research_context/reference_validations/epli_correctness.validation.toml`](../research_context/reference_validations/epli_correctness.validation.toml)
- [`tests/reference/test_reference_validation_engine.py`](../tests/reference/test_reference_validation_engine.py)
- [`tests/audit/test_audit_burton_urban_fi.py`](../tests/audit/test_audit_burton_urban_fi.py)
- [`tests/audit/test_audit_gc_intrinsic_validation.py`](../tests/audit/test_audit_gc_intrinsic_validation.py)
- [`tests/audit/test_audit_epl_fsi_intrinsic_validation.py`](../tests/audit/test_audit_epl_fsi_intrinsic_validation.py)
- [`tests/audit/test_audit_epli_correctness.py`](../tests/audit/test_audit_epli_correctness.py)

## Bottom line

The maintained stack is not a one-for-one rewrite of Justas' original tests.

It does reproduce most of the old single-cell intrinsic physiology metrics that
matter for MC/TC, GC, and now EPL-FSI work, but it does so through a different
architecture:

- old stack: SciUnit/NeuronUnit generic test classes plus publication mixins
- maintained stack: declarative validation configs plus protocol runners and
  rule builders

The maintained stack is stronger on provenance, caveats, human review metadata,
dashboard rendering, and explicit reference-band construction.

The maintained stack is weaker on genericity and breadth:

- several old electrophysiology measurements still have no maintained analogue
- the old publication mixin surface covered many more papers than the current
  maintained validations
- the new `tests/` modules are mostly engine/CLI/report smoke tests, not a
  generic reusable scientific test battery in the SciUnit sense

## What the old system actually was

The old stack was not just a notebook experiment. It had four real layers.

### 1. Generic electrophysiology test battery

[`olfactorybulb/neuronunit/tests/tests.py`](../olfactorybulb/neuronunit/tests/tests.py)
defines a reusable test battery with 30 generic test/helper classes. The
important ones are:

- passive properties:
  - `RestingVoltageTest`
  - `InputResistanceTest`
  - `MembraneTimeConstantTest`
  - `CellCapacitanceTest`
- spike onset and waveform:
  - `RheobaseTest`
  - `SpikeThresholdTest`
  - `SpikePeakTest`
  - `SpikeAmplitudeTest`
  - `SpikeHalfWidthTest`
  - `AfterHyperpolarizationAmplitudeTest`
  - `AfterHyperpolarizationTimeTest`
  - `AfterDepolarizationDepthTest`
  - `AfterDepolarizationTimeTest`
- subthreshold / rebound:
  - `SagVoltageTest`
  - `ReboundSpikingTest`
- train and f-I behavior:
  - `FISlopeTest`
  - `ISICVTest`
  - `SpikeAccommodationTest`
  - `SpikeAccommodationTimeConstantTest`
  - `SpikesAtCurrentTest`

The old stack therefore had a real property library, not just paper-specific
ad hoc code.

### 2. Publication mixins

[`olfactorybulb/neuronunit/tests/publications.py`](../olfactorybulb/neuronunit/tests/publications.py)
defines publication-specific protocol mixins such as:

- `BurtonUrban2014`
- `BurtonUrban2015`
- `Yu2015`
- `Hu2016`
- `JohnsonDelaney2010`
- `Zibman2011`
- `Stroh2012`
- `Fukunaga2012`

These mixins supplied protocol assumptions like:

- current duration
- temperature
- sag target voltage
- threshold extraction method
- rebound-spiking method
- spike-train generation method
- target rate for CV or accommodation checks

This was effectively a handwritten protocol registry.

### 3. SciUnit-style model/capability interface

[`olfactorybulb/neuronunit/models/neuron_cell.py`](../olfactorybulb/neuronunit/models/neuron_cell.py)
wraps a NEURON segment in a SciUnit/NeuronUnit model implementing:

- square-current injection
- membrane-potential recording
- spike production
- stop-time control
- temperature control
- three-stage voltage clamp

That gave the old tests a reusable simulation interface.

### 4. Notebook-driven suite assembly and optimization

[`notebooks/ephyz-validation.ipynb`](../notebooks/ephyz-validation.ipynb)
did three important things:

1. pulled observation rows from the old `Measurement` / `Property` / `Source`
   database tables
2. dynamically composed specific classes like
   `InputResistanceTestBurtonUrban2014`
3. aggregated those test scores into a model score used inside an optimization
   loop

That last point matters: the old stack was not just for reporting; it was used
as an optimization objective.

## What the maintained system is now

The maintained stack has a different split.

### 1. Declarative validation config

Each maintained scientific validation is defined by a TOML config in
[`research_context/reference_validations/`](../research_context/reference_validations/),
for example:

- `burton_urban_fi`
- `gc_intrinsic_validation`
- `epl_fsi_intrinsic_validation`
- `epli_correctness`

These configs declare:

- protocol runner
- defaults
- skip behavior
- rule list
- reference-band modes
- note/caveat sources
- validation-design review metadata

### 2. Shared protocol runners

[`olfactorybulb/audit/reference_validation_protocols.py`](../olfactorybulb/audit/reference_validation_protocols.py)
computes the metrics and protocol evidence.

This is the real replacement for the old test-class `generate_prediction`
logic.

### 3. Rule engine

[`olfactorybulb/audit/reference_validation_rules.py`](../olfactorybulb/audit/reference_validation_rules.py)
turns those metrics into `AuditItem`s using rule kinds like:

- `protocol_executed`
- `reference_band_rows`
- `reference_curve_match`
- `group_ordering`
- `group_abs_diff_max`
- `all_finite_metric`
- `all_exact_metric`

This is the real replacement for the old SciUnit score layer.

### 4. Maintained report, CLI, and dashboard surface

The new stack is exposed through:

- [`tools/run_reference_validation.py`](../tools/run_reference_validation.py)
- [`tools/run_audit.py`](../tools/run_audit.py)
- the audit dashboard / control-center stack

That is a stronger maintained surface than the old notebook-only workflow.

## Metric parity matrix

The table below is the most important result of this comparison.

### Legacy metrics that are clearly preserved

| Legacy metric/test | Maintained analogue | Status |
| --- | --- | --- |
| `RestingVoltageTest` | `resting_potential_mV` in Burton, GC, EPL-FSI validations | Preserved |
| `InputResistanceTest` | `input_resistance_MOhm` | Preserved |
| `MembraneTimeConstantTest` | `membrane_time_constant_ms` | Preserved |
| `CellCapacitanceTest` | `cell_capacitance_pF` | Preserved |
| `RheobaseTest` | `rheobase_pA` | Preserved |
| `SpikeThresholdTest` | `AP_onset_mV` | Preserved |
| `SpikeAmplitudeTest` | `Amplitude_mV` | Preserved |
| `SpikeHalfWidthTest` | `FWHM_ms` | Preserved |
| `SagVoltageTest` | `sag_amplitude_mV` | Preserved |
| `ReboundSpikingTest` | `rebound_potential_presence` | Preserved for Burton / some protocol paths |
| `AfterHyperpolarizationAmplitudeTest` | `AHP_amplitude_mV` | Preserved |
| `AfterHyperpolarizationTimeTest` | `T_AHP50_ms` | Preserved, though exact operational definition changed |
| `FISlopeTest` | `fi_gain_Hz_per_50pA` or `fi_slope_Hz_per_nA` | Preserved |
| `ISICVTest` | `cv_isi` | Preserved |
| `SpikeAccommodationTest` | `spike_accommodation_hz` | Preserved |
| `SpikeAccommodationTimeConstantTest` | `spike_accommodation_time_constant_ms` | Preserved |

### Legacy metrics partially preserved or changed in meaning

| Legacy metric/test | Maintained analogue | Status | Notes |
| --- | --- | --- | --- |
| `SpikePeakTest` | no direct first-class metric; implied by threshold + amplitude | Partial | Absolute spike peak is no longer a first-class reported comparison |
| `ReboundSpikingTest` | `rebound_potential_presence` | Partial | Present in Burton reference mapping; not carried through all maintained validations |
| `SpikesAtCurrentTest` | protocol evidence such as `fi_curve_rows`, target-rate helpers inside protocol code | Partial | No maintained first-class generic test with the same API |
| `SpikeTrainTest` / target-rate helpers | protocol-side selection of CV/accommodation target rates | Partial | Behavior is embedded in protocol runners instead of exposed as a reusable public test type |

### Legacy metrics currently missing from the maintained scientific surface

| Legacy metric/test | Maintained status | Notes |
| --- | --- | --- |
| `AfterDepolarizationDepthTest` | Missing | No maintained ADP depth metric or rule surfaced |
| `AfterDepolarizationTimeTest` | Missing | No maintained ADP time metric or rule surfaced |
| explicit rheobase spike-trace helper outputs | Missing | No maintained direct equivalent of reusable spike-trace helper tests |
| explicit target-frequency spike-train helper outputs | Missing | Current protocols use target-rate selection internally but do not expose old helper-style tests |

## New metrics or capabilities the maintained system has that the old one did not

The maintained system is not only a loss. It also gained important things.

### Strongly improved scientific/reporting behavior

- explicit note and caveat rendering via `note_presence`
- explicit reference-band mode selection:
  - `symmetric_sd`
  - `lognormal_sd`
  - `quantile_interval`
  - `binary_indicator`
- per-item validation-design review metadata:
  - `approved`
  - `provisional`
  - `pending`
  - `not_applicable`
- audit-visible caveats around reconstructed positive-only bands
- explicit subtype-aware GC separation instead of silent pooling
- explicit reference curve matching for EPL-FSI (`reference_curve_match`)

### New metrics now visible in maintained validations

These are either absent from or not first-class in the old generic battery:

- `Rise_slope_mV_per_ms`
- `Fall_slope_mV_per_ms`
- `first_spike_latency_ms`
- `peak_instantaneous_rate_Hz`
- `max_fi_rate_Hz`
- `spontaneous_firing_rate_Hz`

### New structural audit surface

[`research_context/reference_validations/epli_correctness.validation.toml`](../research_context/reference_validations/epli_correctness.validation.toml)
has no old NeuronUnit equivalent. It mixes:

- baseline slice export sanity checks
- reciprocal architecture checks
- target-pattern checks
- synthetic morphology constraints
- synthetic fast-spiking runtime checks

That is useful, but it is a different category than the old electrophysiology
tests. It should not be mistaken for direct parity with the old battery.

## Major architectural differences

### 1. Old generic battery vs new target-specific configs

The old system had one reusable generic test class per property. The new system
has fewer reusable generic "test objects" and more target-specific validation
configs.

Implication:

- old stack was broader and more composable
- new stack is more explicit and maintainable for the specific validations we
  actually run

### 2. Old score objects vs new audit items

Old stack:

- SciUnit `judge()` style
- `ZScore` / `BooleanScore`
- natural path to score matrices and suite aggregation

New stack:

- `AuditItem` with `PASS` / `WARN` / `FAIL`
- explicit criteria and evidence
- more presentation-ready, less mathematically uniform as a global score layer

Implication:

- the maintained system is better for dashboards and review
- the old system was cleaner for pure "suite score as optimizer objective"

### 3. Old observation DB vs new reference-bundle plus config model

Old stack:

- database query over `Measurement`, `Property`, `Source`
- dynamic class construction from DB rows and publication mixins

New stack:

- explicit reference loaders
- explicit validation config
- explicit notes path
- explicit review metadata

Implication:

- the maintained system is much better at provenance and caveats
- the old system was more compact for quickly instantiating many paper/property
  combinations

### 4. Old tests were the scientific layer; new `tests/` are mostly harness checks

This is an important discrepancy.

The old `olfactorybulb/neuronunit/tests/*.py` files were the actual scientific
test logic.

The new files under [`tests/reference/`](../tests/reference/) and
[`tests/audit/`](../tests/audit/) are mostly:

- engine smoke tests
- CLI coverage
- dashboard/report contract checks
- fixture-based rule checks

They are important, but they are not the same thing as the old scientific test
battery.

Examples:

- [`tests/reference/test_reference_validation_engine.py`](../tests/reference/test_reference_validation_engine.py)
  checks config discovery, CLI listing, extension loading, and skip behavior
- [`tests/audit/test_audit_gc_intrinsic_validation.py`](../tests/audit/test_audit_gc_intrinsic_validation.py)
  is mainly a smoke test that the audit runs and emits expected fields
- [`tests/audit/test_audit_burton_urban_fi.py`](../tests/audit/test_audit_burton_urban_fi.py)
  is the strongest scientific regression of the new set, but it still leans on
  fixture metrics plus output-shape assertions, not a general property-by-
  property reusable test battery

## Concrete discrepancies worth calling out

### 1. The maintained system does not yet cover the full old publication surface

The old publication mixins covered at least:

- `Angelo2012`
- `BurtonUrban2014`
- `BurtonUrban2015`
- `Yu2015`
- `Hu2016`
- `JohnsonDelaney2010`
- `Zibman2011`
- `Stroh2012`
- `Abraham2010`
- `Hovis2010`
- `Shpak2012`
- `Christie2005`
- `Fukunaga2012`

The current maintained validation IDs are only:

- `burton_urban_fi`
- `gc_intrinsic_validation`
- `epl_fsi_intrinsic_validation`
- `epli_correctness`

So the maintained stack is much narrower in literature coverage.

### 2. ADP metrics were dropped

The old stack had:

- `AfterDepolarizationDepthTest`
- `AfterDepolarizationTimeTest`

I did not find a maintained analogue in current validation configs or protocol
metric maps.

### 3. Absolute spike peak is no longer first-class

The old stack had a separate `SpikePeakTest`. The maintained stack compares
threshold and amplitude, but not absolute peak voltage as a first-class
reported target.

### 4. Some protocol-derived metrics are computed but not always surfaced

The maintained protocol runners compute a fairly rich metric set, but the
validation config determines what is actually judged. For example:

- `rebound_potential_presence` is part of the protocol metric vocabulary
- but not every maintained validation maps it into reference-band checks

So "metric exists in the protocol code" is not the same as "metric is part of
the maintained scientific comparison surface."

### 5. The old system had a cleaner optimization hook

The old notebook used the property suite score directly in an optimization
workflow.

The current maintained audit system is much better for reviewed reporting, but
it does not provide the same clean generic suite-score abstraction out of the
box.

That is an actual workflow regression if we care about broad, reusable
single-cell parameter optimization against a large electrophysiology battery.

## Where the maintained system is strictly better

These are real upgrades, not just rebranding.

### 1. Explicit caveats and protocol notes

The maintained validations can render protocol caveats and reference-band
caveats directly in the audit output. The old stack had no comparably explicit
maintained path for that.

### 2. Explicit distribution-shape handling

The maintained system can represent:

- symmetric SD bands
- lognormal reconstructed bands
- quantile intervals
- binary indicators

The old SciUnit layer mostly reduced everything to observation mean/std plus a
`ZScore` or boolean outcome.

### 3. Human review metadata

The maintained system explicitly records whether a validation item is:

- approved
- provisional
- pending
- not applicable

That is critical for scientific presentation and was missing from the old
stack.

### 4. Better separation of incompatible biological targets

The maintained GC validation explicitly separates:

- generic or unspecified GC
- superficial GC
- deep GC
- adult-born GC
- modulated GC

That is a better scientific stance than quietly reusing one generic property
battery across targets that are not truly equivalent.

## Net assessment

If the question is, "Did we already rebuild most of Justas' old intrinsic
electrophysiology battery?" then the answer is:

- **yes, for the core passive, spike-shape, rheobase, sag, rebound, f-I, and
  accommodation metrics**

If the question is, "Did we completely replace the old system?" then the answer
is:

- **no**

The main unresolved gaps are:

1. missing ADP metrics
2. missing generic reusable helper tests for spike-train and fixed-current
   conditions
3. far narrower literature coverage than the old publication mixin surface
4. no equally clean generic suite-score abstraction for optimization

If the question is, "Is the maintained system scientifically better for the
validations it does cover?" then the answer is:

- **yes**

It is better on:

- provenance
- caveats
- explicit review state
- subtype separation
- presentability
- dashboardability
- reference-band transparency

## Practical recommendation

If we want true parity with the old Justas stack without regressing to the old
architecture, the next high-value steps are:

1. add maintained ADP metrics and validation mappings
2. add a maintained "suite score" aggregation layer on top of `AuditItem`s for
   optimization use
3. decide which legacy publication mixins still matter scientifically and
   migrate only those into declarative maintained validations
4. keep treating the current `tests/reference` and `tests/audit` modules as
   harness/contract tests, not as proof that the old scientific test battery
   has been fully replaced

