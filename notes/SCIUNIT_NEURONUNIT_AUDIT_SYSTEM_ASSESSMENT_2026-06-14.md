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

## The Actual Rebase Question

The better question is not:

- "Should we replace everything with SciUnit?"

The better question is:

- "Which layer of the current system would benefit from being rebased onto
  SciUnit-style abstractions, and which layer should stay ours?"

That distinction matters because the current repo has at least two different
concerns:

1. a **scientific validation core**
2. a **user-facing audit/report/dashboard shell**

SciUnit is strongest at the first concern.
The current repo is already strong at the second.

### Where a rebase would genuinely help

It would help most in the scientific-validation core:

- typed observations and predictions
- unit-aware values
- explicit score objects
- explicit capability contracts
- suite-level aggregation

This is exactly where the current maintained system is most bespoke.

### Where a rebase would mostly be churn

It would help much less in the reporting shell:

- audit grouping
- warning and caveat surfacing
- human validation-design review metadata
- literature provenance presentation
- maintained dashboard rendering
- docs and CLI discovery

SciUnit does not replace those repo-specific workflow requirements.

## Would It Fix The "Unit Discrepancy" Class Of Problem?

Yes, or at least it would make that class of bug much harder to write.

That is one of the strongest arguments for adopting more SciUnit-like semantics.

The old NeuronUnit layer is explicitly unit-aware. For example:

- [`olfactorybulb/neuronunit/tests/tests.py`](../olfactorybulb/neuronunit/tests/tests.py)
  defines tests with declared units such as `pq.mV`, `pq.MOhm`, and `pq.ms`
  and returns `quantities` values rather than naked floats
- [`olfactorybulb/neuronunit/models/neuron_cell.py`](../olfactorybulb/neuronunit/models/neuron_cell.py)
  accepts and rescales quantities such as `tstop.rescale(pq.ms)`

By contrast, the current maintained audit layer centers on:

- metric-key naming conventions
- plain floats inside evidence dicts
- optional textual unit labels

[`olfactorybulb/audit/core.py`](../olfactorybulb/audit/core.py) does not have a
typed unit-bearing scalar or score field in `AuditItem`; it is a report object,
not a unit-safe scientific value object.

So on this point your professor is probably right:

- NeuronUnit or a similar quantity-aware test core would catch some classes of
  unit mismatch earlier and more automatically than the current float-heavy
  maintained path

That does **not** imply that the whole current audit/dashboard architecture
should be replaced.

It implies that the scientific-value layer should become more unit-aware.

## What A Full Rebase Would Actually Mean

If "rebase onto SciUnit / NeuronUnit" is taken literally, it would mean
rewriting or adapting all of the following:

### 1. Protocol runners become test classes

Current maintained protocol execution lives in:

- [`olfactorybulb/audit/reference_validation_protocols.py`](../olfactorybulb/audit/reference_validation_protocols.py)

Under a SciUnit-first design, much of this would move into:

- `Test.generate_prediction(...)`
- possibly model backends and capability objects

### 2. Rule handlers become score logic

Current judgment logic lives in:

- [`olfactorybulb/audit/reference_validation_rules.py`](../olfactorybulb/audit/reference_validation_rules.py)

Under a SciUnit-first design, a lot of this would move into:

- `compute_score(...)`
- score types
- converters
- suite aggregators

But note:

- our explicit per-property band-mode rules
- our human-review metadata
- our caveat semantics

would still remain repo-specific extensions.

SciUnit would not remove the need for them.

### 3. Validation configs would change shape

Current validations are declarative TOML configs that name:

- protocol runners
- rule kinds
- per-property band modes
- validation-design review metadata

A serious SciUnit rebase would likely push those configs toward:

- test class registries
- suite construction
- observation payload definitions
- score policies

That would be a significant migration, not a search-and-replace.

### 4. Dashboard result models would need an adapter layer

Current dashboard rendering expects `AuditReport` / `AuditItem` data:

- [`olfactorybulb/audit/core.py`](../olfactorybulb/audit/core.py)
- [`olfactorybulb/audit/dashboard.py`](../olfactorybulb/audit/dashboard.py)

A rebase does not eliminate this. It means either:

- the dashboard is rewritten around SciUnit result objects
  or
- SciUnit results are adapted back into `AuditReport`

The second option is much safer.

### 5. Environment support would have to become maintained

Today, the maintained OBGPU import verifier explicitly excludes the old
neuronunit stack:

- [`tools/setup/verify_obgpu_python_imports.py`](../tools/setup/verify_obgpu_python_imports.py)

And in the current `OBGPU` environment, `import sciunit` and `import neuronunit`
both fail.

So a real rebase means:

- adding those dependencies back to the supported environment
- verifying them in maintained setup audits
- owning compatibility issues going forward

That is a support commitment, not just a design preference.

## Difficulty And Blast Radius

There are really three different projects hiding under the word "rebase."

### Option A: Unit-safe retrofit inside the current system

What it means:

- keep current protocol runners, rule engine, audit reports, dashboard, and
  TOML configs
- add typed unit-bearing values to protocol outputs and rule computations
- normalize arithmetic through `quantities` or a similar unit library
- add unit-aware helper builders for comparisons and evidence rendering

What it would change:

- mostly the scientific data path
- minimal dashboard change
- minimal user workflow change

Rough difficulty:

- moderate
- roughly a few focused days to about one week, depending on how broadly the
  unit semantics are pushed

Why this is attractive:

- it addresses the exact class of problem your professor flagged
- it preserves the current maintained workflow
- it avoids a dependency rebase of the whole audit platform

### Option B: Rebase the scientific test core onto SciUnit-style abstractions, but keep the current audit shell

What it means:

- keep `AuditReport` / dashboard / CLI / review metadata / caveat handling
- introduce a local or direct-SciUnit scientific core with:
  - model wrappers
  - capability contracts
  - test objects
  - score objects
  - suite matrices
- adapt the resulting scores back into the current audit/report layer

What it would change:

- major internal changes to validation execution
- moderate changes to docs, tests, and configs
- limited change to the end-user dashboard workflow if the adapter is done well

Rough difficulty:

- substantial but realistic
- roughly one to three weeks for a careful first slice, likely longer if the
  migration includes multiple maintained validations and dashboard suite views

Why this is the best "serious standards" path:

- it gets the credibility and semantics benefits of a recognized testing model
- it keeps the repo-specific workflow features that SciUnit does not provide
- it avoids rebuilding the UI around a new result schema all at once

### Option C: Full NeuronUnit-first rebase of the maintained validation stack

What it means:

- make NeuronUnit or a close derivative the main scientific-validation root
- rebuild validations and possibly configs around its classes
- make the maintained environment support that stack directly
- either rewrite the dashboard or maintain a large adapter bridge

What it would change:

- protocol execution
- scoring
- config shape
- tests
- docs
- setup/audit contracts
- possibly notebook workflows

Rough difficulty:

- high
- likely multiple weeks of churn, plus compatibility risk

Why I do **not** recommend it:

- NeuronUnit is the more fragile dependency choice here
- the repo's maintained workflow now extends beyond classic NeuronUnit-style
  single-cell ephys tests
- much of the dashboard, provenance, caveat, and review machinery would still
  need to remain custom anyway

## My Actual Recommendation

If the goal is standards, robustness, and fewer scientific footguns, then:

- **yes**, I think rebasing the **scientific core semantics** toward
  SciUnit-style testing would be a good move
- **no**, I do not think rebasing the entire maintained audit system onto
  NeuronUnit would be the best move

More concretely:

### Best near-term move

Do Option A first:

- make the current maintained validation path unit-aware
- add explicit typed observation / prediction helpers
- make rule math operate on quantity-bearing values before display

This gives you the biggest quality gain per unit of churn.

### Best medium-term move

Then do Option B in a limited slice:

- pick one maintained validation family, probably
  `burton_urban_fi` or `gc_intrinsic_validation`
- define a SciUnit-style or directly SciUnit-backed model/test/score layer for
  that family
- adapt its result into `AuditReport`
- add a score-matrix view to the dashboard

That would tell us quickly whether the standards benefit is worth a broader
migration.

### What I would avoid

I would avoid a repo-wide "switch everything to NeuronUnit now" plan.

That is the highest-churn and least certain path, and it still would not remove
our need for:

- dataset ingestion configs
- band-mode policy
- validation-design review metadata
- caveat presentation
- maintained dashboard UX

## What Could Go Wrong During The Overhaul

This section assumes the serious path is pursued:

- fork `neuronunit`
- migrate the scientific validation core toward that fork
- keep the current repo as the provenance-aware application shell

These are the main repo-specific failure modes.

### 1. Environment support can fail before any scientific migration starts

Current fact:

- the maintained import verifier explicitly excludes the old neuronunit stack
- `import sciunit` and `import neuronunit` currently fail in `OBGPU`

Why this matters:

- the migration can stall at packaging and compatibility before any actual
  scientific value is gained
- a forked `neuronunit` that only works in an older side environment would
  recreate the same split-brain support problem we already have

What could go wrong:

- Python-version mismatches
- stale dependency pins
- incompatibility with current `neuron`, `numpy`, `scipy`, or `quantities`
- import-time failures hidden from the maintained setup audits

Repo surfaces affected:

- [`tools/setup/verify_obgpu_python_imports.py`](../tools/setup/verify_obgpu_python_imports.py)
- `OBGPU` activation and setup flow

Mitigation:

- make `sciunit` / `neuronunit` import support part of the maintained
  environment contract before migrating scientific logic
- add them to the maintained import verifier only after the support path is
  actually stable

### 2. We can end up with two scientific sources of truth

Current fact:

- the repo already has a maintained protocol/rule validation engine
- the repo also already has a legacy `olfactorybulb.neuronunit` layer

Why this matters:

- a half-migration can leave the current protocol/rule engine and the new
  NeuronUnit core both claiming authority over the same biological checks

What could go wrong:

- `burton_urban_fi` semantics live partly in rule handlers and partly in
  NeuronUnit tests
- one path gets updated while the other drifts
- the dashboard renders one result while notebooks or researchers cite another

Repo surfaces affected:

- [`olfactorybulb/audit/reference_validation_engine.py`](../olfactorybulb/audit/reference_validation_engine.py)
- [`olfactorybulb/neuronunit/`](../olfactorybulb/neuronunit/)

Mitigation:

- migrate one validation family completely at a time
- define a single authoritative execution path for each migrated family
- make the deprecated path adapter-only or read-only as soon as possible

### 3. The config compiler may become the new complexity sink

Current fact:

- the maintained validation layer is driven by declarative TOML configs
- the config loader and engine have repo-specific semantics for defaults,
  extensions, skip behavior, rule specs, and review metadata

Why this matters:

- if we rebase onto NeuronUnit, these TOMLs do not map 1:1 onto plain
  `Model` / `Test` / `Score` objects

What could go wrong:

- the compiler from TOML to NeuronUnit suites becomes more complex than the
  current system it replaced
- repo-specific semantics such as `skip_neuron_mode`, `extensions`, or
  `validation_design_review` become awkward bolt-ons
- configuration errors become harder to explain because failures happen inside
  a compile step plus a test-execution step

Repo surfaces affected:

- [`olfactorybulb/audit/reference_validation_config.py`](../olfactorybulb/audit/reference_validation_config.py)
- [`olfactorybulb/audit/reference_validation_engine.py`](../olfactorybulb/audit/reference_validation_engine.py)

Mitigation:

- keep the first compiler slice narrow
- target one validation family
- refuse to compile unsupported semantics implicitly; fail loudly when a TOML
  field has no clean NeuronUnit mapping

### 4. The migration can silently destroy the current band-policy guarantees

Current fact:

- the maintained rule engine requires explicit `property_band_modes` for every
  property in `reference_band_rows`
- this was added precisely because one-size-fits-all scoring was scientifically
  unsafe

Why this matters:

- a naive SciUnit / NeuronUnit migration could accidentally collapse back to
  "everything is a z-score around a mean and standard deviation"

What could go wrong:

- lognormal or bounded metrics get remapped to symmetric arithmetic scoring
- positive-only metrics lose their current protections
- the migration appears cleaner architecturally while scientifically regressing

Repo surfaces affected:

- [`olfactorybulb/audit/reference_validation_rules.py`](../olfactorybulb/audit/reference_validation_rules.py)
- current validation TOMLs

Mitigation:

- move band policy into the fork as an explicit first-class abstraction
- do not allow a migration path that erases the per-property band-mode choice

### 5. Review-state and caveat semantics can be lost or downgraded

Current fact:

- the maintained system explicitly carries validation-design review status,
  reviewer, focus, and caveat/warning messaging into the audit outputs

Why this matters:

- these are part of the current scientific trust model, not decorative UI

What could go wrong:

- a migrated NeuronUnit result contains a score but no place for:
  - approved / provisional / pending status
  - caveat text
  - warning reason
- the dashboard loses the distinction between "failed scientifically" and
  "scientifically provisional but still visible"

Repo surfaces affected:

- [`olfactorybulb/audit/reference_validation_engine.py`](../olfactorybulb/audit/reference_validation_engine.py)
- [`olfactorybulb/audit/core.py`](../olfactorybulb/audit/core.py)
- [`olfactorybulb/audit/dashboard.py`](../olfactorybulb/audit/dashboard.py)

Mitigation:

- define review/caveat metadata in the fork before adapting results back into
  `AuditReport`
- treat review-state preservation as a migration gate, not an optional polish step

### 6. Provenance can be weakened even if the new scientific core is better

Current fact:

- the repo has an explicit extraction and normalization engine for literature
  datasets

Why this matters:

- provenance is easy to talk about abstractly, but easy to flatten during a
  migration if the new core only wants "observation values"

What could go wrong:

- normalized rows get reduced to bare observations without source location
- transformation lineage gets lost
- caveat note IDs stop propagating
- paper ingestion becomes "good enough for the test" but no longer auditable

Repo surfaces affected:

- [`olfactorybulb/audit/reference_dataset_engine.py`](../olfactorybulb/audit/reference_dataset_engine.py)
- normalized dataset CSV outputs

Mitigation:

- design provenance-bearing observation/reference objects first
- make row-level source identity and transformation lineage mandatory in the
  migrated scientific-core interface

### 7. `AuditReport` adaptation can become lossy and misleading

Current fact:

- the current user-facing surface depends on `AuditItem` / `AuditReport`
- the dashboard expects fields such as `criterion_latex`, `criterion_formulae`,
  `criterion_definitions`, `series_visuals`, `note`, and `status_reason`

Why this matters:

- if NeuronUnit results are richer in one direction and poorer in another, the
  adapter can quietly throw away meaning

What could go wrong:

- typed score objects get collapsed into simplistic PASS/WARN/FAIL without
  enough explanation
- NeuronUnit observations/predictions exist but the dashboard no longer gets
  the math, visual, or warning structure it expects
- two people reading the same result through two interfaces get different
  interpretations

Repo surfaces affected:

- [`olfactorybulb/audit/core.py`](../olfactorybulb/audit/core.py)
- [`olfactorybulb/audit/dashboard.py`](../olfactorybulb/audit/dashboard.py)

Mitigation:

- define an explicit adapter contract
- test round-tripping of one migrated suite into `AuditReport`
- refuse silent field loss for caveats, criterion math, or visuals

### 8. Dashboard regressions can make the migration look scientifically worse than it is

Current fact:

- the dashboard is already a major maintained product surface
- it expects item cards, interval visuals, series visuals, compact warnings,
  and rich criterion rendering

Why this matters:

- even a scientifically better core will look like a step backward if the UI
  becomes less legible or less informative

What could go wrong:

- migrated tests render as raw JSON or generic score strings
- current compact interval summaries disappear
- f-I curve evidence no longer renders because the new result objects do not map
  to current visual specs

Repo surfaces affected:

- [`olfactorybulb/audit/dashboard.py`](../olfactorybulb/audit/dashboard.py)
- [`olfactorybulb/dashboard/control_center.py`](../olfactorybulb/dashboard/control_center.py)

Mitigation:

- treat dashboard parity as part of the migration acceptance criteria
- add suite-level views only after current single-item evidence remains intact

### 9. Control-center orchestration can break on non-audit-shaped results

Current fact:

- the control center currently assumes audits, docs, and optimization all live
  behind one shell and that audit runs emit report artifacts with the current
  shape

Why this matters:

- a migration that changes the internal execution model can still break control
  center orchestration even if command-line tests appear to pass

What could go wrong:

- state polling and session history no longer reflect migrated scientific runs
- score-matrix or suite results do not fit the current audit-group model
- optimization tab integrations cannot consume the new result shape

Repo surfaces affected:

- [`olfactorybulb/dashboard/control_center.py`](../olfactorybulb/dashboard/control_center.py)

Mitigation:

- keep the outer control-center contract stable
- evolve the inner result schema behind adapters first

### 10. Performance can regress badly if dependent predictions are not preserved

Current fact:

- the old NeuronUnit layer already had dependent-prediction caching
- the maintained protocol runners currently bundle several measurements into one
  execution path

Why this matters:

- a naive one-test-one-simulation migration can explode the runtime cost

What could go wrong:

- each score reruns a full protocol
- optimization candidate evaluation becomes much slower
- dashboard-triggered reruns become unusably expensive

Repo surfaces affected:

- protocol runners
- optimization candidate evaluation paths

Mitigation:

- preserve multi-measurement protocol execution
- require protocol-result caching before broadening migration scope

### 11. Historical comparability can be lost

Current fact:

- the repo already has maintained audits, generated report artifacts, and human
  interpretations of their current outputs

Why this matters:

- changing the scientific core can change not just formatting but actual
  numerical behavior and status outcomes

What could go wrong:

- old PASS/WARN/FAIL judgments no longer match
- it becomes hard to tell whether differences are scientific improvements or
  migration bugs
- previously reviewed validation choices become difficult to compare across versions

Repo surfaces affected:

- maintained audit outputs
- checked-in or archived dashboard/report artifacts

Mitigation:

- keep before/after baselines for one migrated validation family
- compare score-by-score and item-by-item, not just final status counts

### 12. Test coverage can give false confidence

Current fact:

- the repo has many test modules, but only a small subset is surfaced through
  the audit-visible grouped suites
- many existing tests are infrastructure tests, not scientific-core equivalence
  tests

Why this matters:

- "tests passed" may not mean the migration preserved scientific meaning

What could go wrong:

- dashboard and config smoke tests stay green while scientific semantics drift
- migrated suites lack equivalence tests against the old maintained outputs

Repo surfaces affected:

- `tests/`
- `test_suite_status` grouped suites

Mitigation:

- add focused equivalence tests for migrated scientific families
- distinguish "software still runs" from "scientific judgments still mean the
  same thing"

### 13. The fork itself can become a second neglected codebase

Current fact:

- a forked NeuronUnit would become another maintained project surface

Why this matters:

- the migration only improves things if the fork is actively owned

What could go wrong:

- the fork accumulates local patches without release discipline
- this repo depends on unreleased fork behavior
- future agents and collaborators cannot tell what lives upstream, in the fork,
  or only here

Mitigation:

- define release/versioning expectations for the fork up front
- keep the boundary between forked library and application repo explicit

### 14. Scope creep can derail the migration

Current fact:

- much of the current audit system has nothing to do with classical NeuronUnit
  scientific testing

Why this matters:

- if the overhaul tries to move everything at once, it will burn time on the
  least appropriate targets

What could go wrong:

- downloader tests, docs audits, dashboard shell checks, scratch-boundary
  audits, and HFO contract audits get forced into a NeuronUnit-shaped design
- migration energy is spent on non-scientific surfaces instead of the scientific core

Repo surfaces affected:

- [`olfactorybulb/audit/registry.py`](../olfactorybulb/audit/registry.py)
- all non-scientific audits

Mitigation:

- explicitly scope the overhaul to the scientific validation core first
- treat operational and infrastructure audits as separate concerns unless a
  clear NeuronUnit abstraction genuinely helps them

## Practical Risk Ranking

Highest risk:

1. environment and dependency instability
2. split-brain scientific sources of truth
3. loss of explicit band-policy and review/caveat semantics
4. performance regression from broken caching
5. lossy `AuditReport` adaptation

Moderate risk:

1. dashboard regression
2. config compiler complexity
3. historical comparability drift
4. misleading test confidence
5. fork maintenance overhead

Lower risk, but still real:

1. scope creep into non-scientific audits
2. control-center orchestration friction

## The Main Strategic Mistake To Avoid

The biggest strategic mistake would be to interpret "move onto NeuronUnit" as
"replace the current system wholesale."

That path maximizes:

- churn
- ambiguity
- temporary duplicate truths
- UI regressions
- setup breakage

without guaranteeing that the scientifically valuable parts of the current
system survive the transition.

The safer strategy is:

1. stabilize library support
2. migrate one scientific validation family end to end
3. prove parity plus improvement
4. only then widen the migration surface

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
