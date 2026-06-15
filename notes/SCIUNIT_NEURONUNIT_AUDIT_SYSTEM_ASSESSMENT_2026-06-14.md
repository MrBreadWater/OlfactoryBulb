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

## Strategies For Handling The Overhaul Risks

The mitigations above are per-risk. This section turns them into an actual
execution strategy.

The guiding rule should be:

- do not migrate by abstraction alone
- migrate by **one fully verified scientific family at a time**

### Strategy 1: Establish a supported library baseline before any scientific rewrite

Purpose:

- eliminate the packaging/setup risk first

Concrete steps:

1. make the forked `neuronunit` importable inside the maintained `OBGPU`
   environment
2. verify that `sciunit`, the fork, and their unit-handling dependencies work
   under the same maintained path used for current audits
3. add a temporary dedicated setup audit for fork readiness before wiring the
   fork into any maintained scientific validation

Required gate before proceeding:

- maintained environment can import the fork reliably
- import behavior is tested in the same path as the rest of the maintained
  setup checks

Why this works:

- it prevents a scientific migration from being blocked by hidden environment
  instability halfway through

### Strategy 2: Define a hard source-of-truth rule per migrated validation family

Purpose:

- prevent split-brain scientific logic

Concrete steps:

1. choose one validation family, for example `burton_urban_fi`
2. mark one execution path as authoritative for that family
3. keep the old path only as:
   - baseline comparator
   - adapter target
   - rollback path
4. once the migrated path is accepted, stop editing the old scientific logic
   for that family except to preserve baseline comparison fixtures

Required gate before proceeding to the next family:

- there is a documented authoritative path for the migrated family
- the older path is no longer co-evolving semantically

Why this works:

- it avoids the most dangerous state, where two active implementations drift
  while both still look legitimate

### Strategy 3: Build a thin config compiler, not a second framework

Purpose:

- stop the config translation layer from becoming worse than the current engine

Concrete steps:

1. keep the current TOML surface for the first migration slice
2. compile only the minimum supported subset into forked-NeuronUnit tests and
   suites
3. reject unsupported config features explicitly and early
4. do not try to compile every current rule kind into the fork on day one

Compiler design rule:

- the compiler should be a **deterministic projection** from current config
  semantics into forked-NeuronUnit primitives
- it should not become a second hidden policy engine

Required gate before broadening compiler scope:

- one family compiles cleanly
- error messages are clear
- unsupported semantics fail loudly rather than degrading silently

Why this works:

- it keeps the migration visible and understandable instead of replacing one
  bespoke layer with a more obscure bespoke layer

### Strategy 4: Move scientific policy objects into the fork before migrating judgments

Purpose:

- preserve band policies, provenance, and review semantics

Concrete steps:

Before migrating real validations, define explicit fork-side abstractions for:

- provenance-bearing observation/reference records
- band-policy objects:
  - symmetric
  - lognormal
  - beta/bounded
  - quantile
  - binary
- review-state metadata
- caveat / comparability metadata

Required gate before judgment migration:

- no migrated validation family may fall back to a generic z-score-only policy
  unless that is the scientifically intended policy

Why this works:

- it prevents the migration from "succeeding" architecturally while regressing
  scientifically

### Strategy 5: Preserve protocol bundling and dependent-prediction reuse from day one

Purpose:

- avoid catastrophic runtime regression

Concrete steps:

1. identify which current maintained protocol runners bundle multiple
   measurements from one execution
2. model those as shared prediction/protocol-result objects in the forked core
3. preserve or improve the old dependent-prediction cache behavior
4. measure runtime before and after for each migrated family

Current overhaul-branch status:

- implemented a formal process-local protocol-result cache in
  `olfactorybulb.audit.reference_validation_protocol_core`
- moved protocol spec/registry/cache ownership into that typed contract layer
- cacheable maintained protocols now declare explicit semantic
  `cache_arg_names`
- cache hits/misses and resolved cache inputs are emitted in `protocol_cache`
  evidence instead of staying hidden

Required gate before using migrated suites in optimization or dashboard reruns:

- repeated tests do not rerun the same expensive simulation unnecessarily
- candidate evaluation cost stays within an acceptable multiplier of the current path

Why this works:

- it keeps the migration from becoming unusable in the places where scale
  matters most

### Strategy 6: Make `AuditReport` adaptation a first-class contract, not an afterthought

Purpose:

- protect the maintained CLI and dashboard surfaces

Concrete steps:

1. define an adapter spec from forked-NeuronUnit results into `AuditItem` /
   `AuditReport`
2. enumerate every current report feature that must survive:
   - criterion text
   - criterion math
   - definitions
   - warnings
   - caveats
   - review status
   - series visuals
   - compact interval visuals
3. write explicit adapter tests for those fields

Current overhaul-branch status:

- added a shared `AuditItemAdapterSpec` bridge in
  `olfactorybulb.neuronunit.suite_presentation`
- migrated the SciUnit-backed suite adapters onto that shared contract instead
  of hand-copying the same presentation field bundle in each suite module
- added explicit adapter coverage for criterion math, review metadata, notes,
  visuals, status reasons, and suite-overview rollup preservation

Required gate before any migrated family is exposed in the control center:

- the migrated family renders through the same maintained report path
- no critical user-facing field is silently dropped

Why this works:

- it keeps the fork from fragmenting the user-facing surfaces into incompatible
  result models

### Strategy 7: Enforce before/after equivalence reviews per migrated family

Purpose:

- distinguish scientific improvement from migration bug

Concrete steps:

For each migrated family:

1. run the old maintained path on a fixed baseline
2. run the migrated path on the same baseline
3. compare:
   - per-item observations
   - per-item scores or acceptance-band judgments
   - final PASS/WARN/FAIL outcomes
   - caveats and review metadata
4. classify each difference as one of:
   - intended scientific improvement
   - adapter bug
   - migration bug
   - unresolved discrepancy

Required gate before acceptance:

- every meaningful difference is explained and recorded

Why this works:

- it prevents accidental semantic drift from being normalized into the new path

### Strategy 8: Add migration-specific tests instead of trusting the existing test tree

Purpose:

- avoid false confidence from infrastructure-heavy test coverage

Concrete steps:

Add dedicated tests for:

- fork import readiness
- config compilation for migrated families
- provenance preservation
- band-policy preservation
- review/caveat propagation
- `AuditReport` adaptation parity
- runtime caching behavior

Important rule:

- do not count generic dashboard or downloader smoke tests as evidence that the
  scientific migration is correct

Why this works:

- it aligns the tests with the actual migration risks

### Strategy 9: Keep the dashboard stable while expanding it

Purpose:

- stop UI churn from obscuring scientific progress

Concrete steps:

1. preserve the existing item-card path for migrated families first
2. only add score-matrix or suite-level views after the single-item audit view
   remains correct
3. require dashboard parity for:
   - interval displays
   - series graphs
   - warnings/caveats
   - criterion math

Required gate before enabling new suite views by default:

- legacy-style per-item interpretation remains available and correct

Why this works:

- it keeps the new suite semantics additive rather than destructive

### Strategy 10: Separate core fork ownership from application ownership

Purpose:

- stop the fork from turning into another ambiguous local patch pile

Concrete steps:

1. define what belongs in the fork versus this repo
2. version and tag fork releases
3. avoid depending on unreleased fork behavior for long periods
4. record which repo features are:
   - upstream SciUnit concepts
   - fork-specific NeuronUnit additions
   - OlfactoryBulb application behavior

Why this works:

- it preserves maintainability and makes future contributions legible

## Recommended Migration Sequence

The safest overall strategy is:

### Phase 0: Fork readiness

- make the fork installable/importable in `OBGPU`
- verify unit-handling support
- define ownership/versioning expectations

### Phase 1: Scientific primitives

- add provenance-bearing observation/reference objects
- add band-policy abstractions
- add review/caveat metadata carriers
- add shared prediction/protocol-result caching semantics

### Phase 2: One-family pilot

- choose one family, preferably `burton_urban_fi`
- compile the existing config into forked-NeuronUnit structures
- adapt results back into `AuditReport`
- compare old versus new outputs exhaustively

### Phase 3: Dashboard extension

- keep current item-card rendering
- add optional suite / score-matrix views for the migrated family
- confirm control-center compatibility

### Phase 4: Broaden cautiously

- only migrate the next family after the previous one has:
  - environment stability
  - parity or explained improvement
  - adapter stability
  - acceptable runtime

## Rollback Strategy

The overhaul should be run so that rollback is always possible.

Required rollback rules:

1. keep the old maintained path runnable for the current pilot family until the
   migrated path is accepted
2. keep baseline result artifacts for comparison
3. do not delete current config or report surfaces during the pilot
4. if one of the highest-risk failures appears:
   - packaging instability
   - semantic drift without explanation
   - major runtime regression
   - loss of caveat/review/provenance behavior
   then freeze scope expansion and revert to the last accepted family boundary

This matters because a migration that cannot be rolled back is too easy to
continue just because it has already consumed time.

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

Current branch state: the first maintained presentation step is now in place.
The migrated SciUnit-backed suite families emit:

- one compact suite-overview item
- a persistent status-matrix companion visual
- the original detailed case items

The suite-overview item is intentionally rollup-exempt so the maintained audit
summary still reflects the detailed scientific cases rather than double-
counting the overview card.

The branch now also has a shared suite result/presentation helper so those
overview cards are not hand-implemented four different ways. The shared helper
owns:

- suite-overview item construction
- rollup-exempt report semantics
- suite-case matrix payload shape
- optional compact per-case score labels

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

Implemented on the overhaul branch:

- formal process-local protocol-result cache
- explicit per-protocol cache-identity contract through
  `ValidationProtocolSpec.cache_arg_names`
- cache visibility surfaced in protocol evidence through `protocol_cache`

Current limitation:

- cache scope is still one Python process; broader shared-cache policy remains
  future work if optimization-scale measurements show it is needed

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

Current overhaul-branch status:

- added a typed per-case `case_score` payload on the migrated suite bridge
- suite cases now carry explicit `score_kind`, `score_value`, `score_units`,
  `score_interpretation`, plus structured `observation`, `prediction`, and
  `normalization` payloads when that information is available
- the compact `score_text` labels remain for scanning, but they no longer have
  to serve as the only preserved score semantics

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

## Current Overhaul-Branch Progress

The dedicated `neuronunit-overhaul` branch now has a first real migrated slice
instead of note-only planning:

1. `environment-modern.yml` now records the currently required SciUnit-side
   packages for this branch:
   - `quantities==0.16.4`
   - `neo==0.14.4`
   - `sciunit==0.2.8`
   - `neuronunit==0.1.8.2`
2. A dedicated readiness gate exists at
   [`tools/setup/verify_neuronunit_overhaul_imports.py`](../tools/setup/verify_neuronunit_overhaul_imports.py)
   so the branch can verify SciUnit / NeuronUnit / quantities / neo imports
   plus the repo-local NeuronUnit bridge modules without claiming the full main
   OBGPU surface has already migrated.
3. The maintained `reference_band_rows` rule family now executes through a
   SciUnit-backed bridge in:
   - [`olfactorybulb/neuronunit/reference_bands.py`](../olfactorybulb/neuronunit/reference_bands.py)
   - [`olfactorybulb/neuronunit/reference_validation_suite.py`](../olfactorybulb/neuronunit/reference_validation_suite.py)
4. That bridge introduces:
   - provenance-bearing reference observations
   - explicit band-policy objects
   - validation-review metadata attached to observations
   - a SciUnit `Model` / `TestSuite` / `Score` execution path
   - adaptation back into the maintained `AuditItem` / `AuditReport` shell
5. The repo-local legacy `olfactorybulb.neuronunit` imports were also patched
   to tolerate the current upstream package layout instead of assuming the
   older `neuronunit.tests.base` module path.
6. A second bridge now covers the summary/comparison-rule family in:
   - [`olfactorybulb/neuronunit/summary_validation_suite.py`](../olfactorybulb/neuronunit/summary_validation_suite.py)
   - contiguous runs of
     `summary_metric_min` / `summary_metric_max` /
     `summary_metric_range` / `summary_metric_status_map`
     are compiled into one SciUnit-backed suite by
     [`olfactorybulb/audit/reference_validation_rules.py`](../olfactorybulb/audit/reference_validation_rules.py)
7. That second bridge is enough to migrate the maintained
   [`epli_correctness.validation.toml`](../research_context/reference_validations/epli_correctness.validation.toml)
   family end to end, because that validation is entirely composed of those
   summary-rule kinds.
8. A third bridge now covers the maintained exactness/comparison-rule family in:
   - [`olfactorybulb/neuronunit/comparison_validation_suite.py`](../olfactorybulb/neuronunit/comparison_validation_suite.py)
   - contiguous runs of
     `all_finite_metric` / `all_exact_metric` /
     `group_ordering` / `group_abs_diff_max` / `group_positive`
     are compiled into one SciUnit-backed suite by
     [`olfactorybulb/audit/reference_validation_rules.py`](../olfactorybulb/audit/reference_validation_rules.py)
9. That third bridge is enough to migrate the maintained directional and
   exactness portion of
   [`burton_urban_fi.validation.toml`](../research_context/reference_validations/burton_urban_fi.validation.toml)
   into the same SciUnit-backed core instead of leaving those judgments in
   bespoke rule handlers.
10. A fourth bridge now covers unit-aware series/distribution comparison in:
    - [`olfactorybulb/neuronunit/series_validation_suite.py`](../olfactorybulb/neuronunit/series_validation_suite.py)
    - the maintained `reference_curve_match` rule now compiles into that
      SciUnit-backed series core through
      [`olfactorybulb/audit/reference_validation_rules.py`](../olfactorybulb/audit/reference_validation_rules.py)
11. That fourth bridge makes `reference_curve_match` explicitly:
    - unit-aware on both axes
    - transform-aware for mismatched model/reference axis quantities
    - distribution-oriented over duplicate x bins instead of silently reducing
      the data to one point per x before comparison
12. Current verified branch state for `epli_correctness`:
    - `python tools/run_reference_validation.py --validation-id epli_correctness --skip-neuron --json`
    - `python tools/run_reference_validation.py --validation-id epli_correctness --json`
    - `python tools/run_audit.py epli_correctness --json`
    all produce a maintained report surface successfully on this branch, and
    the current full run reports `20` items with summary
    `{'FAIL': 1, 'PASS': 16, 'WARN': 3}`.
13. The Burton MC/TC legacy summary CSVs are now behind a real reference-dataset
   contract instead of a direct special-case file read:
   - source files now live under
     `research_context/source_data/burton_mc_tc_principal_cells/`
   - the maintained dataset config is
     `research_context/reference_datasets/burton_mc_tc_principal_cells.dataset.toml`
   - the maintained canonical outputs are now
     `research_context/BURTON_MC_TC_ephys.csv`,
     `research_context/BURTON_MC_TC_protocols.csv`,
     `research_context/BURTON_MC_TC_needs_manual_extraction.csv`, and
     `research_context/BURTON_MC_TC_extraction_README.md`
   - the Burton validation path now loads those canonical dataset outputs
     instead of normalizing the source CSVs inline at runtime
14. Current verified branch state for the maintained non-skip Burton
    validation path:
    - `python tools/run_reference_validation.py --validation-id burton_urban_fi --cell-count 1 --jobs 1 --json`
    - `python tools/run_audit.py burton_urban_fi --cell-count 1 --jobs 1 --json`
    produces a maintained report surface successfully on this branch, and the
    current one-MC/one-TC validation run reports `46` items with summary
    `{'FAIL': 15, 'PASS': 29, 'WARN': 2}`, while the audit-wrapper path reports
    `52` items with summary `{'FAIL': 15, 'PASS': 35, 'WARN': 2}` because it
    includes the maintained preflight/context checks around the same protocol.
15. Current verified branch state for the maintained EPL-FSI series-comparison
    path:
    - `python tools/run_reference_validation.py --validation-id epl_fsi_intrinsic_validation --cell-models SyntheticEPL2026.PVCRH_FSI1 --jobs 1 --json`
    - `python tools/run_audit.py epl_fsi_intrinsic_validation --cell-models SyntheticEPL2026.PVCRH_FSI1 --jobs 1 --json`
    both produce a maintained report surface successfully on this branch, and
    the current run reports `19` items with summary
    `{'FAIL': 12, 'PASS': 6, 'WARN': 1}`. The migrated
    `epl_fsi_reference_curve_match` item reports `matched_point_count = 12`
    and now carries explicit comparison-axis unit metadata such as
    `comparison_x_unit_text = "pA"`.

This does **not** mean the overhaul is complete. It means one important
scientific-core judgment family is now moving under NeuronUnit/SciUnit-style
semantics while the maintained audit shell stays intact.

## Series-comparison follow-up

The first series-comparison slice is now implemented. Remaining follow-up work
for that area is tracked in:
- [`notes/TEMPORARY_NEURONUNIT_OVERHAUL_TODO.md`](./TEMPORARY_NEURONUNIT_OVERHAUL_TODO.md)

The open design questions are no longer whether to add the abstraction, but how
far to generalize it next. The branch now treats the maintained v1 contract as:

- abstraction name: `series comparison`
- alignment policies:
  - exact shared transformed x bins
  - monotone nearest transformed x bins within explicit tolerance
  - pooled shared tolerance clusters with bounded x-span
  - resampled shared comparison grids built from per-series interpolation
- transform policies:
  - identity
  - affine
  - affine lookup from explicit row/protocol metadata
  - piecewise linear with explicit control points and explicit extrapolation
    mode
- distribution policy: empirical per-x response distributions
- score-family choices:
  - `residual_only`
  - `equivalence_only`
  - `hybrid_residual_equivalence`
  - `welch_only` (legacy difference-test diagnostic)
  - `hybrid_residual_welch` (legacy difference-test diagnostic)
- emitted evidence:
  - aligned mean-series arrays for plotting
  - unit-aware residual diagnostics with unit-neutral keys plus separate unit
    metadata
  - resolved statistical-policy defaults such as `pvalue_aggregation` and
    `equivalence_margin` when those were not declared explicitly
  - resolved alignment-policy defaults such as `resampling_grid_source` when a
    stable default was used for ergonomics
  - equivalence-test diagnostics
  - optional legacy Welch-test diagnostics
  - family-specific suite-case score labels plus aggregate suite norm-score
    summaries when normalized case scores are available
  - shared typed provenance summaries for reference/model series payloads and
    compact protocol-context metadata
  - typed `SeriesDataSpec` / `SeriesVisualContract` structure inside the core
    so the series bridge no longer has to repeat the same axis/unit/transform
    plumbing across bins, paths, provenance, and visual payload assembly
  - the declarative `reference_curve_match` builder now targets those typed
    specs directly instead of spelling the flat series-field bundle inline
  - the declarative parser for that rule now lives in one typed
    `SeriesComparisonRuleSpec` path instead of a growing pile of unrelated
    helper calls
  - graphable protocol-evidence row bundles now have their own typed contract
    in `olfactorybulb.audit.protocol_evidence`, so `protocol_executed`, the
    dashboard, and `reference_curve_match` no longer need to special-case
    `fi_curve_rows`; the model-side defaults can come from the protocol
    contract while reference-side and comparison-space units stay explicit
  - protocol results now carry one typed protocol-evidence bundle instead of a
    raw `protocol_evidence` dict plus a parallel `evidence_series_specs`
    channel, so cache annotation, row lookup, protocol-executed rendering, and
    series-rule fallback resolution all consume the same evidence object
  - the declarative `reference_band_rows` path now also compiles through typed
    `ReferenceBandRuleSpec` / `ReferenceBandPropertyRuleSpec` objects instead
    of keeping bounds, quantile-field selection, and review parsing sprawled
    across the handler body; those compiler specs now live in the dedicated
    `olfactorybulb.audit.reference_validation_specs` module
  - the grouped summary/comparison rule families now also compile through
    typed `SummaryRuleSpec` / `ComparisonRuleSpec` parser objects before the
    SciUnit-backed cases are built, from that same dedicated compiler module
  - the remaining reference-band and series case assembly now also lives in
    that typed compiler layer: `ReferenceBandRuleSpec.build_cases(...)` owns
    the per-row reference-band case construction, and
    `SeriesComparisonRuleSpec.to_case(...)` owns the
    `SeriesComparisonCase` construction, so
    `reference_validation_rules.py` stays closer to orchestration and suite
    compilation instead of rebuilding flat case payloads inline
  - the top-level runtime path now also compiles through a typed
    `ReferenceValidationPlan` in
    `olfactorybulb.audit.reference_validation_plan`, so the engine, generic
    CLI, and maintained audit wrappers no longer pass a loose validation-config
    dict plus parallel accessor calls around at runtime
  - the raw TOML layer now also compiles through a typed
    `ReferenceValidationDocument` in
    `olfactorybulb.audit.reference_validation_document`, so static config
    consumers such as validation-design-review coverage audits no longer need
    to hand-walk loose config dicts either; the runtime plan now composes from
    that typed document instead of bypassing it
  - `olfactorybulb.audit.reference_validation_config` is correspondingly
    shrinking toward the narrow boring layer it should have been all along:
    raw TOML I/O, validation-id discovery, and extension registration, rather
    than a second semi-typed API parallel to the document layer
  - the rule-runtime boundary is now typed too: `ValidationRuleContext` carries
    explicit runtime fields such as validation id, notes path, default group,
    and typed validation-design-review defaults instead of one generic config
    dict leaking through the rule engine
  - the maintained built-in rule-dispatch layer now also compiles into typed
    dispatch records/specs before runtime execution, so the internal engine no
    longer has to keep reparsing loose raw dicts all the way down to
    `protocol_executed`, `note_presence`, and the grouped suite families;
    the raw `register_validation_rule(...)` surface remains only as the
    compatibility hook for extension-defined custom rules
  - `ReferenceValidationPlan` now also carries a precompiled rule-dispatch
    sequence, so the runtime no longer has to rediscover contiguous grouped
    suite families from raw rule dicts after the plan has already been built;
    correspondingly, the raw `checks` list is staying with the typed document
    layer rather than being duplicated yet again inside the runtime plan
  - the typed document boundary has moved up one more step too:
    `ReferenceValidationDocument` now carries typed `ValidationRuleRecord`
    entries rather than a raw `checks` tuple, so both the runtime plan and the
    static validation-design-review audit now consume the same typed rule
    record layer instead of reparsing raw rule dicts independently
  - the protocol-result seam is now typed through the maintained runtime path
    too: `ReferenceValidationPlan`, `reference_validation_engine`, and
    `ValidationRuleContext` now pass `ProtocolRunResult` explicitly instead of
    falling back to `Any` once protocol execution returns from the registry/
    cache layer
  - the remaining rule/spec interface seams are shrinking too: a shared
    `reference_validation_contracts` module now carries the non-cyclic typed
    interfaces for validation-rule context and review payloads, so the typed
    parser/presentation layer no longer has to fall back to `context: Any` or
    `validation_review: Any`
  - the protocol-evidence boundary is now tighter too: once protocol evidence
    has crossed the coercion layer, the public runtime/model seam now stays on
    `ProtocolEvidenceBundle` rather than continuing to advertise raw dicts as
    a first-class protocol-evidence type
  - the remaining adapter-input seams are shrinking too: suite presentation
    now consumes a structural case protocol instead of `case: Any`, and the
    protocol-evidence helper now consumes a structural carrier protocol
    instead of `result: Any`
  - the series-comparison provenance layer is now more coherent too: a typed
    `SeriesObservationProvenance` payload now carries the paired reference and
    model provenance summaries together, so score evidence no longer has to
    scatter separate `reference_provenance` / `model_provenance` siblings and
    parallel top-level count fields
  - compact reference/model provenance summaries
  - model-side protocol context when that metadata is available in the
    protocol-evidence bundle
  - one shared series-suite overview when multiple contiguous
    `reference_curve_match` checks are compiled together
  - a typed suite-aggregation layer in `olfactorybulb.neuronunit.suite_scores`
    so suite-level status/norm rollups are computed once and then adapted into
    `suite_aggregate_score` evidence for the maintained audit shell
  - a typed `SuiteDescriptor` bundle so suite id, suite-kind label,
    candidate/model ids, and aggregate-policy choices do not have to be
    threaded as parallel parameters through every suite adapter
  - the same `suite_aggregate_policy` path now reaches `reference_band_rows`
    as well as the grouped summary/comparison/series suite families, so the
    declarative suite contract is uniform across the migrated scientific rule
    surface
  - the scalar summary/comparison suite families now resolve one typed metric
    quantity from `metric_key` plus optional overrides, so unit-bearing scalar
    rules no longer have to pass raw metric keys, ad hoc unit labels, or
    hand-chosen observed symbols separately through the spec, case, score, and
    maintained presentation layers
  - the migrated scalar scientific suites now also share one typed scalar
    observation/prediction layer in
    `olfactorybulb.neuronunit.scalar_observations`, so summary/comparison
    tests no longer have to shuttle raw dict bundles for metric maps,
    left/right group pairs, or grouped scalar value sets through their score
    builders, and that shared scalar metadata now carries the resolved
    observed symbol as well as the unit/name contract
  - the remaining discrete scalar status-code path in
    `summary_metric_status_map` now also compiles through one typed
    `ScalarStatusMapPolicy`, so the declarative parser, summary-suite case,
    scorer, and maintained evidence no longer have to keep three parallel
    pass/warn/fail value lists in sync by hand
  - protocol metric rows and grouped numeric summaries now also share one
    typed contract in `olfactorybulb.neuronunit.metric_tables`, so the
    protocol result, validation context, runtime plan, and migrated
    NeuronUnit model/suite bridges no longer have to exchange fresh raw
    `list[dict]` / `dict[group][metric]` scientific payloads at every seam
  - finite closed `summary_metric_range` rules now render through the shared
    absolute-residual criterion builder, while semi-bounded ranges collapse to
    one-sided inequalities, so the maintained validation math no longer falls
    back to endpoint notation or `-\infty \leq x \leq U` artifacts
  - suite overviews can now also carry a typed `suite_statistical_summary`
    when detailed cases expose compatible equivalence or Welch diagnostics, so
    the overview layer no longer flattens that statistical evidence down to
    norm scores only
  - the same suite-descriptor layer now also carries an explicit
    `SuiteStatisticalPolicy`, so grouped validations can override the default
    suite-level p-value rollup declaratively instead of baking it into one
    suite adapter or the dashboard
  - that suite-level statistical contract now also supports explicit
    supported-case count/fraction requirements, and the emitted summary
    reports support coverage plus separate support and threshold gates instead
    of letting a rolled-up p-value imply stronger statistical coverage than
    the underlying cases actually supplied
  - the maintained SciUnit wrappers now also delegate their observation maps
    back to the existing typed case/observation objects through small
    `observation_payload()` helpers, so the suite layer no longer duplicates
    field lists inline just to satisfy the SciUnit observation contract
  - the series-suite prediction seam now preserves `ProtocolEvidenceBundle`
    instead of immediately re-splitting it into loose `rows + context`
    payloads, so the typed protocol-evidence contract survives all the way
    into model-side series scoring
  - the model/capability seam now exposes that same typed protocol-evidence
    contract explicitly through `ProvidesProtocolEvidenceBundle`, so migrated
    suites that need the full payload do not have to reconstruct bundles from
    a looser map/row pair
  - grouped suite default ids are now family-scoped consistently across the
    migrated scientific rule families, so series-suite overview items no
    longer reuse the bare validation id while summary/comparison suites use
    family-specific ids
  - suite-level statistical rollups now also consume an explicit typed
    `SuiteCaseStatisticalPayload` from case scores instead of inferring
    p-value meaning from generic `observation` / `prediction` dict fragments,
    so the statistical aggregation contract is explicit at the case level too
  - the remaining non-statistical suite score sections now also sit on a
    shared frozen `SuiteCaseMappingPayload`, but the builders still accept
    plain mappings and coerce them at the boundary so the scientific-core
    contract tightens without making the maintained rule adapters verbose
  - the remaining scalar/reference-band detailed evidence seam is tighter too:
    summary, comparison, and reference-band suites now route their emitted
    item evidence through a shared typed
    `olfactorybulb.neuronunit.validation_evidence` layer instead of
    hand-assembling the same dict shapes inline in each suite adapter
  - the runtime metric-table seam is tighter too: `ProtocolRunResult`,
    `ReferenceValidationPlan`, and `ValidationRuleContext` now treat
    `MetricTable` / `MetricSummaryTable` as the maintained post-protocol
    contract, while outer helpers still coerce raw rows for ergonomic tests
    and legacy call sites at the boundary
  - the typed scalar helper layer is tighter too: `ScalarMetricValueMap` and
    `ScalarGroupValueSet` now keep their stored per-entity / per-group values
    on one shared frozen `ScalarValueMapPayload` instead of advertising fresh
    mutable dict seams after the scalar prediction has crossed into the
    NeuronUnit-side scientific core
  - the runtime per-entity metric-map seam is tighter too: `MetricTable` and
    `ReferenceValidationModel` now expose one shared frozen
    `MetricValueMapPayload` instead of building fresh mutable dicts and then
    immediately re-coercing them again in the scalar comparison/exactness
    helper layer
  - the migrated SciUnit observation seam is tighter too: maintained summary,
    comparison, reference-band, and series cases now export their observation
    payloads through named frozen payload types rather than building one more
    anonymous dict at the `sciunit.Test(...)` boundary
  - the literature-row seam is tighter too: `_load_rows`, `_filter_rows`, and
    the `reference_band_rows` / `reference_curve_match` spec entrypoints now
    pass through a shared typed `olfactorybulb.audit.reference_rows` layer
    instead of teaching each rule/spec helper to expect fresh CSV-style dict
    rows directly
  - the protocol-evidence row seam is tighter too: once a maintained protocol
    bundle exposes row-shaped evidence such as `fi_curve_rows`, that payload
    now moves through a shared typed `ProtocolEvidenceRowTable` wrapper at the
    bundle/model boundary instead of being rediscovered as one more
    `list[dict]` view in each consumer
  - the reference-band case seam is tighter too: once a maintained
    `reference_band_rows` check has matched a typed literature row to a
    declared property spec, the band math, criterion text, observation, and
    review payload now move through one typed `_ReferenceBandRowBinding`
    object instead of being reconstructed from a long run of parallel locals
    inside `ReferenceBandRuleSpec.build_cases(...)`
  - the series row/context seam is tighter too: bound series datasets now keep
    their concrete rows and protocol context on shared typed
    `SeriesRowTable` / `SeriesContextPayload` payloads, so transforms,
    interpolated paths, and provenance summaries no longer have to accept new
    mutable `list[dict]` / `dict` seams once the data has crossed into the
    NeuronUnit-side scientific core
  - the deeper provenance-bearing series observation layer has started too:
    a `SeriesObservedDatasetPair` now binds the typed reference and model
    datasets together before scoring, so paired provenance moves through one
    typed object instead of being reconstructed from separate locals late in
    the score path
  - the same series scorer now keeps its remaining internal scientific support
    metadata typed too: cluster membership, resampling-domain/support
    metadata, and per-bin equivalence-test results no longer have to move
    through anonymous nested dicts once the comparison has entered the
    NeuronUnit-side scoring layer
  - the migrated suite compiler/model seam is tighter too: the
    reference-band, summary, comparison, and series suite compilers now all
    construct `ReferenceValidationModel` through one shared typed
    `ReferenceValidationRuntimeData` bundle, so summary metrics, row metrics,
    and protocol evidence are coerced once at suite-compilation time instead
    of being passed as parallel mutable runtime tables into each suite family
  - the central protocol-evidence seam now follows the same pattern through a
    shared `FrozenMappingPayload` layer, so `ProtocolEvidenceBundle.values`
    and `ProtocolEvidenceSeriesSpec.style` stop carrying mutable raw maps
    internally even though maintained protocol builders and tests can still
    pass plain mappings at the boundary
  - the same wrapper pattern now also covers the remaining top-level
    validation config/cache tables after load: document defaults,
    protocol defaults, skip-item evidence, and protocol-cache arg/config
    payloads no longer fall back to loose mutable dicts once they have
    crossed the load/cache boundary
  - the declarative rule/spec boundary now follows that same contract too:
    `ValidationRuleRecord.raw_rule` stores a typed frozen
    `ValidationRulePayload`, and the rule/spec parsers now accept
    mapping-like payloads instead of treating mutable raw dicts as the
    primary maintained contract
  - the migrated series-comparison score/evidence seam is tighter too:
    computed series diagnostics now live on an explicit
    `SeriesComparisonEvidencePayload` that owns score text, suite-case
    statistical payload derivation, case weighting, and evidence export,
    rather than rebuilding one large ad hoc dict directly inside
    `compute_score()`
  - the same slice forced the parser contract to match the frozen payload
    reality: once nested declarative values have crossed the frozen wrapper
    boundary, rule/spec parsers now accept tuple-backed sequence values
    rather than assuming every config list stayed mutable
  - the bound reference/model series data now also live on a typed
    `SeriesObservedDataset` layer, so bins, interpolated paths, transforms,
    and provenance summaries no longer have to rediscover `rows + spec +
    context` tuples at each scoring step
  - the resampled-grid alignment path now also supports an explicit small
    interpolation family (`linear`, `nearest`, `pchip`, `step_hold`) instead of
    treating straight-line interpolation as the only possible series
    resampling contract
  - the resampled-grid alignment path now also separates grid choice from
    domain choice: the policy can keep partial support on the whole grid or
    clip to the shared/reference/model interpolation domain explicitly via
    `resampling_domain_policy`, and the emitted evidence records both the
    resolved policy and the filtered-out grid points
  - alignment support is now a first-class typed contract too: the
    series-comparison policy can declare minimum reference/model coverage
    fractions in addition to `minimum_point_count`, and the emitted evidence
    now records the aligned-support counts, coverage fractions, coverage gate,
    and alignment-support norm score instead of letting a thin overlap look
    indistinguishable from a fully supported comparison
  - the typed suite-aggregation layer now supports a richer
    `weighted_mean` norm rollup when a migrated suite family emits a
    principled per-case weight; the maintained series-comparison suite
    currently does this via matched-point count
  - the same suite-level statistical contract now also supports weighted
    support gates: grouped validations may declare
    `minimum_available_case_weight` and/or
    `minimum_available_case_weight_fraction`, and the emitted
    `suite_statistical_summary` now records weighted support counts,
    fractions, labels, and gate results when the suite family exposes a real
    per-case weight
  - the suite-level statistical-support contract is now visible in the
    migrated presentation layer too: when a suite matrix has only partial
    statistical support or fails its declared support gate, the dashboard
    header now surfaces that support shortfall instead of hiding it only in
    raw `suite_statistical_summary` evidence
  - the remaining shell-side meta checks such as `protocol_executed` and
    `note_presence` now parse through typed config specs too, while still
    remaining outside the NeuronUnit scientific-core layer

The remaining open questions are now narrower:

- whether the new typed `suite_statistical_summary` plus declarative
  `SuiteStatisticalPolicy` support is enough, or whether future work should
  promote additional suite-level statistical contracts beyond rollup choice
  plus the current supported-case count/fraction/weight gates
- whether transform generalization should stop at explicit piecewise-linear
  mappings plus the current context-dependent affine lookup path, or later
  grow into richer metadata-driven transforms
- whether alignment generalization should stop at the current
  distribution-preserving resampled-grid contract with its current explicit
  interpolation/domain families (`linear`, `nearest`, `pchip`, `step_hold`;
  `allow_partial_support`, `intersection`, `reference`, `model`)
- more explicit provenance-bearing series observation objects if future
  validations need more than the current EPL-FSI example-cell path
- how much NeuronUnit-native result presentation should grow before it starts
  competing with the maintained audit shell instead of feeding it

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
