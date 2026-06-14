# SciUnit / NeuronUnit Assessment for the Current Audit System

Date: 2026-06-14

## Scope

This note answers four practical questions for the current repository state:

1. How much of the current audit and reference-validation system overlaps with
   SciUnit and NeuronUnit?
2. Is the repo already carrying a NeuronUnit-based validation layer?
3. Would reviving SciUnit or NeuronUnit reduce duplication, or would it create
   a second unsupported stack?
4. If we want SciUnit-like benefits, what is the cleanest way to add them to
   the maintained dashboard and optimization workflow?

This note is about the maintained repo architecture as it exists now, not about
what the old stack was ideally supposed to become.

## Executive Summary

- The repo already contains a real legacy SciUnit / NeuronUnit-derived
  validation layer under [`olfactorybulb/neuronunit/`](../olfactorybulb/neuronunit/).
  It is not hypothetical.
- The current maintained
  [reference-validation system](./REFERENCE_VALIDATION_SYSTEM_OVERVIEW.md)
  partially reimplements the old NeuronUnit electrophysiology workflow, but it
  does so in a different architectural style.
- The overlap is real and substantial for intrinsic single-cell physiology:
  resting voltage, input resistance, membrane time constant, capacitance,
  rheobase, spike threshold, spike amplitude, spike half-width, sag, rebound,
  firing-rate-versus-current slope, ISI CV, and accommodation all existed in
  the old NeuronUnit-style layer and now exist again in the maintained
  declarative validation layer.
- The current maintained layer is not redundant with SciUnit in a broader
  sense. It adds things the old stack did not formalize well:
  dataset provenance, explicit protocol caveats, per-item human
  validation-design review state, explicit reference-band selection for each
  property, maintained CLI discovery, and a user-facing dashboard.
- The maintained `OBGPU` workflow does not currently install `sciunit` or
  `neuronunit`, and the maintained import audit explicitly excludes the older
  neuronunit stack. So "just use NeuronUnit again" is not a drop-in move.
- The best path is not to make maintained OBGPU depend directly on the legacy
  NeuronUnit package. The better path is:
  1. keep the current audit/report/dashboard layer as the maintained surface
  2. treat the old SciUnit / NeuronUnit ideas as an architectural reference
  3. selectively add the missing high-value abstractions locally:
     suite matrices, score objects, capability-like contracts where useful, and
     better prediction caching
  4. if legacy NeuronUnit execution is ever revived, expose it through an
     adapter that emits `AuditReport` data instead of making it the new root
     architecture

## The Three Distinct Things That Should Not Be Confused

### 1. Developer test wrapping

The current grouped developer-test surface is
[`olfactorybulb/audit/test_suite_status.py`](../olfactorybulb/audit/test_suite_status.py).
It is a presentation wrapper around selected `tests.*` modules, not a
scientific model-validation framework.

Today it exposes three suite IDs:

- `maintained_core`
- `reference_bundles`
- `audit_surface`

This is useful and maintained, but it is orthogonal to SciUnit.

#### Current audit-visible unit-test inventory

As of this note:

- the repo contains `123` `tests/test_*.py` modules in total
- only `14` of those modules are currently surfaced through the audit system's
  grouped `test_suite_status` entrypoint

The total raw test inventory is currently distributed as:

- `60` under `tests/neuroinfra/`
- `19` under `tests/olfactorybulb/`
- `16` under `tests/audit/`
- `8` under `tests/reference/`
- `7` under `tests/cell_models/`
- `5` under `tests/hfo/`
- `4` under `tests/slice/`
- `2` under `tests/simulation/`
- `2` under `tests/integration/`

The audit-visible grouped suites are currently:

- `maintained_core`:
  - `tests.integration.test_config_helpers`
  - `tests.reference.test_reference_validation_engine`
  - `tests.reference.test_reference_data_sanity`
- `reference_bundles`:
  - `tests.reference.test_reference_dataset_engine`
  - `tests.reference.test_download_epl_fsi_reference_sources`
  - `tests.reference.test_pv_crh_epl_fsi_reference_data`
  - `tests.reference.test_download_gc_reference_sources`
  - `tests.reference.test_gc_reference_data`
- `audit_surface`:
  - `tests.audit.test_repo_health`
  - `tests.audit.test_audit_cli_output`
  - `tests.audit.test_audit_dashboard`
  - `tests.audit.test_audit_style_contracts`
  - `tests.neuroinfra.dashboard.test_neuroinfra_dashboard_shell`
  - `tests.integration.test_control_center_dashboard`

That matters for the redundancy question:

- these audit-visible unit tests are mostly infrastructure, contract,
  dashboard, downloader, and smoke-test modules
- they are generally **not** the sort of scientific model-validation tests that
  SciUnit or NeuronUnit were designed to formalize
- only a small portion of the current audit-visible test surface is even
  conceptually adjacent to SciUnit, mainly the reference-validation-engine
  smoke coverage

So if someone says "a lot of our unit-test work is redundant with SciUnit," the
correct answer is:

- that is **false** for most of the current developer test inventory
- it is only materially true for the biological validation slice discussed
  later in this note

### 2. Maintained literature / audit system

The maintained scientific validation surface is the combination of:

- [`tools/run_audit.py`](../tools/run_audit.py)
- [`tools/run_reference_validation.py`](../tools/run_reference_validation.py)
- [`olfactorybulb/audit/reference_validation_engine.py`](../olfactorybulb/audit/reference_validation_engine.py)
- [`olfactorybulb/audit/reference_validation_rules.py`](../olfactorybulb/audit/reference_validation_rules.py)
- [`olfactorybulb/audit/reference_validation_protocols.py`](../olfactorybulb/audit/reference_validation_protocols.py)
- [`olfactorybulb/audit/core.py`](../olfactorybulb/audit/core.py)
- [`olfactorybulb/audit/dashboard.py`](../olfactorybulb/audit/dashboard.py)

This layer is maintained and user-facing.

### 3. Legacy SciUnit / NeuronUnit workflow

The repo still carries an older model-validation layer built around SciUnit and
NeuronUnit concepts:

- [`olfactorybulb/neuronunit/tests/tests.py`](../olfactorybulb/neuronunit/tests/tests.py)
- [`olfactorybulb/neuronunit/tests/publications.py`](../olfactorybulb/neuronunit/tests/publications.py)
- [`olfactorybulb/neuronunit/tests/__init__.py`](../olfactorybulb/neuronunit/tests/__init__.py)
- [`olfactorybulb/neuronunit/models/neuron_cell.py`](../olfactorybulb/neuronunit/models/neuron_cell.py)
- [`olfactorybulb/neuronunit/capabilities/__init__.py`](../olfactorybulb/neuronunit/capabilities/__init__.py)
- [`notebooks/ephyz-validation.ipynb`](../notebooks/ephyz-validation.ipynb)

This layer is historically important and architecturally informative, but it is
not the current maintained OBGPU runtime surface.

## What SciUnit and NeuronUnit Contribute Conceptually

Based on current upstream documentation:

- SciUnit is organized around `Model`, `Test`, and `Score`, plus
  `Capability`, `TestSuite`, and `ScoreMatrix`.
- A SciUnit test defines:
  - required capabilities
  - a prediction-generation step
  - a score type
  - a score-computation step
- NeuronUnit is a neuroscience-specific library built on SciUnit for ion
  channel, neuron, and network testing.
- The broader SciUnit ecosystem has already been extended to network-scale work
  such as NetworkUnit, so the framework is not intrinsically limited to single
  neurons.

The useful takeaway is not "use the package because it exists." The useful
takeaway is that SciUnit cleanly separates:

- model interface
- protocol / prediction generation
- judgment / scoring
- suite-level aggregation

That separation remains good architecture.

## What the Repo Already Had in Its Older NeuronUnit Layer

The old layer is more substantial than a few experiments.

### Generic test battery

The repo-local file
[`olfactorybulb/neuronunit/tests/tests.py`](../olfactorybulb/neuronunit/tests/tests.py)
currently defines 30 generic test/helper classes, including:

- `RestingVoltageTest`
- `InputResistanceTest`
- `MembraneTimeConstantTest`
- `CellCapacitanceTest`
- `RheobaseTest`
- `SpikeThresholdTest`
- `SpikePeakTest`
- `SpikeAmplitudeTest`
- `SpikeHalfWidthTest`
- `SagVoltageTest`
- `ReboundSpikingTest`
- `AfterHyperpolarizationAmplitudeTest`
- `AfterHyperpolarizationTimeTest`
- `AfterDepolarizationDepthTest`
- `AfterDepolarizationTimeTest`
- `FISlopeTest`
- `ISICVTest`
- `SpikeAccommodationTest`
- `SpikeAccommodationTimeConstantTest`
- `SpikesAtCurrentTest`

Most of these tests use `scores.ZScore`; rebound uses `BooleanScore`.

### Publication-specific protocol mixins

[`olfactorybulb/neuronunit/tests/publications.py`](../olfactorybulb/neuronunit/tests/publications.py)
defines 13 publication classes:

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

These are protocol bundles in everything but name.

### Dynamic generic-test plus publication composition

[`olfactorybulb/neuronunit/tests/__init__.py`](../olfactorybulb/neuronunit/tests/__init__.py)
builds dependent test classes dynamically by combining:

- a generic property test class
- a publication class

That is almost the same conceptual split that the maintained declarative system
now encodes as:

- `protocol_runner`
- rule kinds
- validation config

### Model adapter and capabilities

[`olfactorybulb/neuronunit/models/neuron_cell.py`](../olfactorybulb/neuronunit/models/neuron_cell.py)
defines `NeuronCellModel` as a SciUnit model with capabilities such as:

- `ReceivesSquareCurrent`
- `ProducesMembranePotential`
- `ProducesSpikes`
- local voltage-clamp / temperature / stop-time capabilities

That is a real capability-based execution contract, not an ad hoc helper.

### Database-backed measurement registry

The historical pipeline also used the SQLite database
[`olfactorybulb/model-data.sqlite`](../olfactorybulb/model-data.sqlite):

- `measurement` held extracted values
- `property.test_class_generic` selected the generic test class

The historical explanation is described directly in
[`docs/_sources/recreating.rst.txt`](../docs/_sources/recreating.rst.txt).

## How Much of That the Current Maintained System Reimplements

For intrinsic electrophysiology, the overlap is large.

The current maintained Burton validation maps literature properties to metrics
in
[`research_context/reference_validations/burton_urban_fi.validation.toml`](../research_context/reference_validations/burton_urban_fi.validation.toml).
That map includes:

- `AHP Amplitude`
- `AHP Duration`
- `AP Amplitude`
- `AP Threshold`
- `AP Width at Half-height`
- `AP Half-Width`
- `Capacitance`
- `FI Curve Slope`
- `ISI Coefficient of Variation`
- `Input Resistance`
- `Membrane Resting Voltage`
- `Membrane Time Constant`
- `Rebound Potential Presence`
- `Rheobase Current`
- `Sag Amplitude`
- `Spiking Rate Accommodation`
- `Spiking Rate Accom. Time Constant`

Those correspond closely to older property-to-test mappings stored in the
SQLite `property` table. The historical mappings include examples such as:

- `Mitral Cell Membrane Resting Voltage -> RestingVoltageTest`
- `Mitral Cell Input Resistance -> InputResistanceTest`
- `Mitral Cell Membrane Time Constant -> MembraneTimeConstantTest`
- `Mitral Cell Rheobase Current -> RheobaseTest`
- `Mitral Cell ISI Coefficient of Variation -> ISICVTest`
- `Mitral Cell Spiking Rate Accommodation -> SpikeAccommodationTest`
- `Granule Cell AP Half-Width -> SpikeHalfWidthTest`
- `Granule Cell Rebound Potential Presence -> ReboundSpikingTest`
- `Tufted Cell AP Width at Half-height -> SpikeHalfWidthTest`

So yes: for single-cell intrinsic metrics, the repo has already rebuilt much
of the old NeuronUnit-style measurement battery in new maintained form.

## Where the Current Maintained System Is Actually Better

The new system is not just a duplicate. It solves several problems the older
layer did not cleanly solve.

### 1. Literature provenance and ingestion are first-class

The current system distinguishes:

- raw source files
- normalized dataset configs
- generated canonical CSV outputs
- validation configs
- rule logic

That structure is documented in:

- [`notes/REFERENCE_VALIDATION_SYSTEM_OVERVIEW.md`](./REFERENCE_VALIDATION_SYSTEM_OVERVIEW.md)
- [`notes/REFERENCE_VALIDATION_HOWTO.md`](./REFERENCE_VALIDATION_HOWTO.md)

The old NeuronUnit flow was stronger on protocol formalization than on current
dataset-governance and provenance discipline.

### 2. Explicit reference-band selection is now enforced

The current rule engine requires an explicit band mode for every mapped
property in `reference_band_rows`.

See
[`olfactorybulb/audit/reference_validation_rules.py`](../olfactorybulb/audit/reference_validation_rules.py):

- no `default_band_mode`
- every property must choose its own `property_band_modes` entry

That is a real improvement over silently applying one generic score style.

### 3. Human validation-design review is explicit

The maintained layer now distinguishes the scientific review status of the
validation-design choice itself:

- `approved`
- `provisional`
- `pending`
- `not_applicable`

That metadata is part of the maintained config and audit surface, not a side
note in prose.

### 4. User-facing reporting is much stronger

The maintained layer has:

- `AuditItem`
- `AuditReport`
- CLI rendering
- browser rendering
- item-level math
- item-level visuals
- warning/caveat surfacing
- grouped session history in the control-center flow

This is substantially better for interactive use than the older notebook-led
validation workflow.

### 5. The maintained system covers more than electrophysiology score tests

The current declarative framework also handles things that do not naturally fit
the older single-score NeuronUnit pattern, for example:

- structural or slice-correctness audits such as
  [`epli_correctness.validation.toml`](../research_context/reference_validations/epli_correctness.validation.toml)
- provenance/caveat display
- dataset contract checks
- validation-design review status audits

That broader scope matters.

## Where the Current Maintained System Is Weaker Than SciUnit / NeuronUnit

This is the part worth learning from rather than ignoring.

### 1. No first-class score object model

The current maintained layer mainly outputs PASS / WARN / FAIL audit items with
evidence and visuals. That is useful for end users, but it is not the same as a
typed scoring layer like SciUnit's `Score`, `ZScore`, `BooleanScore`,
`ScoreArray`, or `ScoreMatrix`.

The current system has judgment logic, but not a general score calculus.

### 2. No first-class multi-model suite matrix

SciUnit has `TestSuite` and `ScoreMatrix`.

The current dashboard is strong at item cards, intervals, and series plots, but
it does not yet have a first-class matrix view for:

- models or optimization candidates as rows
- tests as columns
- typed scores in the cells

That is a real missing feature, especially for optimization workflows.

### 3. Capability contracts are weaker and more implicit

SciUnit / NeuronUnit encode what a model must provide to take a test.

The maintained validation layer uses protocol runners and runtime assumptions,
but not a general capability system. That is fine when the repo owns both sides
of the interface, but it becomes limiting if we want to validate multiple model
backends or external simulators under one formal surface.

### 4. Reusable prediction caching is less formalized

The older NeuronUnit layer had explicit dependent-prediction reuse through
`get_dependent_prediction(...)` and a test-result cache.

The maintained protocol layer reuses work inside protocol runners, but it does
not yet expose a repo-wide formal prediction-cache contract.

### 5. The current developer-test layer and the current scientific validation
layer are separate worlds

That separation is not wrong, but it can make the repo feel like it has many
parallel "test" systems:

- raw `tests/`
- grouped developer test suites
- operational audits
- declarative literature validations
- older NeuronUnit biological tests

This is one reason the repo can feel more redundant than it strictly is.

## Important Boundary: The Maintained OBGPU Surface Does Not Currently Support SciUnit / NeuronUnit

This is the most important operational point.

The maintained import audit
[`tools/setup/verify_obgpu_python_imports.py`](../tools/setup/verify_obgpu_python_imports.py)
explicitly says it does **not** validate the older neuronunit stack.

The maintained `OBGPU` environment currently does not import `sciunit` or
`neuronunit` successfully.

There is also evidence of historical version skew:

- the repo environment files still mention `neuronunit==0.19` and
  `sciunit==0.2.0.2`
- the current public PyPI `neuronunit` package is `0.1.8.2` from July 4, 2016
- the current public PyPI `sciunit` package is `0.2.8` from January 3, 2024
- historical notebook traces point at an external fork path under
  `.../neuronunit_justasb/...`

So there are at least four different things that can be confused:

1. upstream SciUnit
2. upstream NeuronUnit
3. repo-local `olfactorybulb.neuronunit`
4. historical external forks and environments used by older notebooks

That is exactly why reintroducing NeuronUnit as a maintained dependency would
need deliberate ownership rather than casual reuse.

## Redundancy Assessment

### Fully or mostly redundant now

These older NeuronUnit ideas have largely been rebuilt in maintained form:

- publication-specific protocol variation
- intrinsic electrophysiology feature extraction
- literature-to-model comparison for many MC / TC / GC properties
- binary and scalar property checks
- f-I curve comparisons

### Partially redundant

The old and new systems overlap, but neither fully dominates:

- score semantics
- caching strategy
- multi-model evaluation
- optimization-facing aggregation

### Not redundant

The maintained layer adds capabilities that the older NeuronUnit path did not
provide in a maintained, user-facing way:

- source-traceable dataset normalization
- explicit caveat surfacing
- required per-property band-selection metadata
- per-item validation-design review state
- current CLI and dashboard presentation
- structural correctness audits beyond classical single-cell score tests

### Specifically not redundant for the audit-visible unit-test suites

For the three `test_suite_status` suites currently visible through the audit
system:

- `maintained_core` is mostly smoke coverage for notebook/config wiring and the
  declarative validation engine itself
- `reference_bundles` is downloader, extraction, and normalization coverage
- `audit_surface` is reporting/dashboard/shell coverage

Those suites are software-quality checks for the maintained platform. They are
not substitutes for SciUnit, and SciUnit is not a substitute for them.

The only meaningful overlap with SciUnit-style concepts in those audit-visible
developer suites is indirect:

- `tests.reference.test_reference_validation_engine` covers a system that plays
  a role similar to a scientific test runner
- some dashboard tests exercise presentation of scientific judgment results

But even there, the tests are about repo behavior and rendering contracts, not
about formal model-vs-data scoring semantics.

## Practical Recommendation

### Recommendation 1: Do not make maintained OBGPU depend directly on legacy NeuronUnit

This would create two problems at once:

- a support problem
- an architecture problem

Support problem:

- the maintained environment does not currently own those dependencies
- the historical stack shows fork / version drift
- the maintained import audit excludes that stack by design

Architecture problem:

- the repo would end up with two competing scientific-validation roots instead
  of one maintained reporting surface

### Recommendation 2: Keep the maintained audit/reference-validation system as the user-facing root

The maintained root should remain:

- protocol runner
- rule registry
- audit item / report
- CLI / dashboard

That part is already aligned with current repo direction.

### Recommendation 3: Re-import the good SciUnit ideas as local abstractions

The repo should selectively add the pieces that are missing, without taking a
hard dependency on the legacy package stack.

High-value candidates:

- a local typed score object for scalar scientific judgments
- a local suite abstraction for "many tests over many candidate models"
- a score-matrix or score-table visualization contract
- explicit capability-like metadata where tests truly need backend-specific
  interfaces
- reusable protocol-result caching keyed by model identity, protocol settings,
  and measurement request

This recommendation applies primarily to:

- declarative literature validations
- protocol-backed scientific audits
- optimization candidate evaluation surfaces

It does **not** imply migrating ordinary downloader tests, shell tests,
dashboard rendering tests, or other software smoke tests into a SciUnit-like
framework.

### Recommendation 4: If legacy NeuronUnit execution is revived, wrap it as an adapter

If we ever want to run the old tests again, the clean architecture is:

1. run them in their own explicitly supported environment
2. capture predictions, observations, and scores
3. convert that result into local `AuditReport` data
4. render it through the maintained dashboard

Do not reverse that layering.

## What This Means for the Dashboard

If the dashboard is going to become more useful for SciUnit-like workflows, the
main missing piece is not "show raw JSON from another library." The main missing
piece is proper suite-level presentation.

### High-value dashboard additions

1. A score-matrix view:
   - rows = models, cells, or optimization candidates
   - columns = tests
   - cells = typed scores or normalized pass distance

2. A per-test score distribution view:
   - histogram or strip across candidates
   - useful in optimization and model-comparison workflows

3. Drill-down from a matrix cell to:
   - observation
   - prediction
   - score transform
   - protocol evidence
   - traces / f-I curves / derived features

4. Cache visibility:
   - whether a result reused a prior protocol execution
   - what cache key defined equivalence

5. Explicit support for suite semantics in optimization:
   - candidate archive rows judged against a fixed suite
   - sortable aggregate suite score
   - per-test failure signatures

### What not to do

- Do not bolt raw SciUnit `Score` stringification into the current cards and
  call that integration.
- Do not let the dashboard become hard-wired to a direct third-party runtime
  dependency if the repo is not prepared to maintain that dependency in OBGPU.
- Do not bypass the current `AuditReport` / dashboard contract for a separate
  second reporting stack.

## Recommended Implementation Direction if We Want SciUnit-Like Features Without Package Coupling

The clean internal shape would be:

### Layer A: measurement execution

Already present in current protocol runners.

Possible improvement:

- formal protocol-result cache

### Layer B: scoring

Add a local score schema, for example:

- `score_kind`
- `score_value`
- `score_units`
- `score_interpretation`
- `observation`
- `prediction`
- `normalization`

This can coexist with PASS / WARN / FAIL.

### Layer C: suite aggregation

Add a local suite schema:

- suite id
- test ids
- model ids or candidate ids
- score matrix
- aggregate score function

### Layer D: presentation

Keep `AuditReport` as the rendering root, but allow audit items or groups to
embed:

- score matrices
- per-test summary tables
- aggregate suite views

This preserves the maintained dashboard shell and avoids a second UI stack.

## Concrete Call on "Is This Redundant?"

The correct answer is:

- yes, partially and materially redundant at the level of intrinsic
  electrophysiology measurement batteries
- no, not redundant at the level of maintained dataset governance,
  review-state tracking, audit presentation, and general repo-facing workflow

So the repo has not "accidentally reinvented SciUnit" in full.
It has rebuilt one important slice of old NeuronUnit functionality while also
building a different maintained layer around it.

## Recommended Near-Term Actions

1. Keep using the current declarative validation and audit system as the
   maintained surface.
2. Do not add `sciunit` / `neuronunit` back into `OBGPU` casually.
3. Treat [`olfactorybulb/neuronunit/`](../olfactorybulb/neuronunit/) and
   [`notebooks/ephyz-validation.ipynb`](../notebooks/ephyz-validation.ipynb)
   as architectural reference material for:
   - score objects
   - test suites
   - capability contracts
   - dependent-prediction caching
4. If optimization work needs a richer scientific testing view, add a local
   suite / score-matrix layer to the maintained dashboard rather than reviving
   a second user-facing validation surface.
5. If legacy NeuronUnit execution becomes scientifically valuable again, revive
   it as an optional adapter-backed subsystem with its own supported
   environment, not as an implicit dependency of the maintained OBGPU path.
6. Keep the distinction sharp between:
   - developer test suites for software correctness
   - scientific validation suites for model-vs-data judgment
   If SciUnit-like abstractions are added, aim them at the second category
   rather than trying to force the entire `tests/` tree into one framework.

## Source Pointers

### Repo-local

- [`olfactorybulb/audit/test_suite_status.py`](../olfactorybulb/audit/test_suite_status.py)
- [`olfactorybulb/audit/core.py`](../olfactorybulb/audit/core.py)
- [`olfactorybulb/audit/reference_validation_engine.py`](../olfactorybulb/audit/reference_validation_engine.py)
- [`olfactorybulb/audit/reference_validation_rules.py`](../olfactorybulb/audit/reference_validation_rules.py)
- [`olfactorybulb/audit/dashboard.py`](../olfactorybulb/audit/dashboard.py)
- [`research_context/reference_validations/burton_urban_fi.validation.toml`](../research_context/reference_validations/burton_urban_fi.validation.toml)
- [`research_context/reference_validations/epli_correctness.validation.toml`](../research_context/reference_validations/epli_correctness.validation.toml)
- [`olfactorybulb/neuronunit/tests/tests.py`](../olfactorybulb/neuronunit/tests/tests.py)
- [`olfactorybulb/neuronunit/tests/publications.py`](../olfactorybulb/neuronunit/tests/publications.py)
- [`olfactorybulb/neuronunit/tests/__init__.py`](../olfactorybulb/neuronunit/tests/__init__.py)
- [`olfactorybulb/neuronunit/models/neuron_cell.py`](../olfactorybulb/neuronunit/models/neuron_cell.py)
- [`docs/_sources/recreating.rst.txt`](../docs/_sources/recreating.rst.txt)
- [`tools/setup/verify_obgpu_python_imports.py`](../tools/setup/verify_obgpu_python_imports.py)
- [`notebooks/ephyz-validation.ipynb`](../notebooks/ephyz-validation.ipynb)

### External references consulted

- SciUnit quick tutorial:
  <https://sciunit.readthedocs.io/en/latest/quick_tutorial.html>
- SciUnit basics:
  <https://sciunit.readthedocs.io/en/latest/basics.html>
- SciUnit GitHub:
  <https://github.com/scidash/sciunit>
- SciUnit PyPI:
  <https://pypi.org/project/sciunit/>
- NeuronUnit GitHub:
  <https://github.com/scidash/neuronunit>
- NeuronUnit PyPI:
  <https://pypi.org/project/neuronunit/>
- NetworkUnit / SciUnit network-validation paper:
  <https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2018.00090/full>
