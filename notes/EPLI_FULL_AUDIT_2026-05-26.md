# EPLI Full Audit - 2026-05-26

This note audits the optional `EPLI` path against the current literature
constraints and against the actual repository state.

The goal is stricter than "looks plausible." The implementation should satisfy:

1. explicit morphology targets,
2. explicit slice-distribution constraints,
3. explicit synapse-class constraints,
4. explicit network-readiness criteria, and
5. explicit behavioral validation gates.

The corresponding machine-checkable script is:

- `tools/audit_epli_correctness.py`

## Source constraints

Primary papers used here:

- Kato et al. 2013, *Parvalbumin-expressing interneurons linearly control olfactory bulb output*
  - dense reciprocal MC<->PV connectivity within 200 um
  - PV cells in EPL are multipolar and typically axonless/anaxonic
  - https://komiyamalab.biosci.ucsd.edu/wp-content/uploads/2021/05/2013-1-s2.0-S0896627313007952-main.pdf
- Huang et al. 2013, *Reciprocal connectivity between mitral cells and external plexiform layer interneurons in the mouse olfactory bulb*
  - CRH+ EPL interneuron morphology: soma diameter 9.6 +/- 0.7 um, 3.5 +/- 0.4 primary processes, neurite span 71 +/- 4.5 um, strongest branching within 30 um
  - https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2013.00032/pdf
- Burton et al. 2024, *Fast-spiking interneuron detonation drives high-fidelity inhibition in the olfactory bulb*
  - PV+ FSIs are anaxonic EPL interneurons
  - they perisomatically inhibit M/T cells with release-competent dendrites
  - sparse M/T synchrony can supralinearly recruit them
  - https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002660
- Lepousez and Lledo 2013, *Odor discrimination requires proper olfactory fast oscillations in awake mice*
  - gamma requires excitatory-output / inhibitory-interneuron interplay
  - E/I changes can enhance synchronization without changing mean firing
  - https://research.pasteur.fr/en/publication/odor-discrimination-requires-proper-olfactory-fast-oscillations-in-awake-mice/

Host-model context:

- Birgiolas dissertation local copy:
  - `research_context/Birgiolas_Dissertation.pdf`

## What the repo currently gets right

### 1. The surrogate morphology is internally consistent with Huang 2013

Current surrogate:

- `prev_ob_models/SyntheticEPL2026/isolated_cells.py`

Encoded targets:

- soma diameter ~9.6 um
- 4 primary dendrites
- compact multipolar dendritic arbor
- planar span ~71 um
- axonless topology

Measured from the instantiated cell in the OBGPU NEURON environment:

- planar span: `70.88 um`
- primary dendrites: `4`
- branch dendrites: `8`
- soma length/diameter: `9.6 um / 9.6 um`

That is a good first-pass geometry donor for a CRH/PV-overlap surrogate.

### 2. The high-level circuit architecture is directionally correct

Current opt-in design:

- `M/T -> EPLI` excitation via `AmpaNmdaSyn`
- `EPLI -> M/T` inhibition via `GabaSyn`
- reciprocal default wiring scaffold

That matches the broad architecture from Kato 2013, Huang 2013, and Burton
2024 better than using a PG or GC proxy would.

### 3. The maintained Birgiolas baseline slice is still internally coherent

Canonical slice:

- `olfactorybulb/slices/DorsalColumnSlice`

Current counts:

- MCs: `10`
- TCs: `24`
- GCs: `159`

Ratios:

- TC / MC = `2.4`
- GC / MC = `15.9`

These are close to the Birgiolas slice-builder defaults:

- `2.36` TCs per MC
- `16.97` GCs per MC

Canonical GC reciprocal connectivity is present and nontrivial:

- `GCs__MCs`: `2238` entries, source coverage `0.975`, target coverage `1.0`,
  median distance `3.10 um`
- `GCs__TCs`: `1751` entries, source coverage `0.327`, target coverage `0.958`,
  median distance `3.28 um`

So the host platform itself is not the weak point.

## What is still not correct enough

### 1. There is no maintained network-ready EPLI slice in the repository

Severity: high

The canonical shipped slice is still:

- MC / TC / GC only

There is no maintained exported slice in the repo with:

- `EPLIs.json`
- `EPLIs__MCs.json`
- `EPLIs__TCs.json`

That means the runtime hooks are ahead of the maintained biological asset.
Today, "EPLI support" means infrastructure exists, not that the biological
population is validated and ready.

### 2. The default EPLI placement distribution is not biologically grounded yet

Severity: high

Current builder defaults:

- `epli_particles_object_name=None`
- fallback particle cloud = TC/OPL particles
- depth filter = relative band between inner and outer OPL surfaces
- default selection strategy = `slice_order`

Relevant code:

- `olfactorybulb/slicebuilder/blender.py`

This is acceptable as a conservative implementation scaffold. It is not
acceptable as a "correct by construction" biological density model.

The repo already contains evidence that:

- there is no dedicated EPL particle cloud in the current Blender scene
- order-based candidate truncation biases small smoke slices
- `principal_proximity` can improve local contact opportunities but is a
  debugging heuristic, not a biological prior

See:

- `notes/EPLI_PLACEMENT_DEBUG_2026-05-26.md`

### 3. The default EPLI synapse geometry is too narrow relative to the literature

Severity: high

Current default EPLI blueprints:

- source pattern: `*dend*`
- dest pattern: `*soma*`
- max distance: `20 um`

Relevant code:

- `olfactorybulb/epli.py`

That does **not** adequately reflect Burton 2024, which argues for
perisomatic inhibition involving soma, proximal apical dendrite, and axon
hillock territory.

The current default therefore encodes a placeholder contact class, not a
validated anatomical targeting rule.

### 4. Slice-level contact evidence is still incomplete

Severity: medium-high

What has been shown so far:

- small smoke slices can support nonzero local `EPLI -> TC` dendritic overlap
- the same smoke slices did **not** produce robust `EPLI -> MC` overlap
  even after:
  - fixing the root-export bug
  - improving soma-candidate ranking
  - widening the dendrite confinement corridor

Interpretation:

- the current synthetic morphology plus current placement heuristic is not yet
  enough to justify a default `EPLI -> MC` connectivity claim
- the next bottleneck is likely morphology reach and/or distribution, not only
  the contact-radius parameter

### 5. Behavioral validation is missing for the new population

Severity: high

The repo contains historical validation artifacts for the maintained cell
classes, including:

- `notes/gc_model_ephyz_validation_results.csv`
- `prev_ob_models/morphology-validation-results.xlsx`

But there is no equivalent validation suite yet for:

- EPLI FI curves
- passive properties
- synaptic response properties
- network ablation signatures
- gamma / HFO effects under explicit perturbations

That means the surrogate is morphologically constrained but not yet
behaviorally validated.

## Current pass/fail judgment

### PASS

- surrogate morphology matches first-pass literature targets well
- high-level reciprocal circuit architecture is directionally correct
- host Birgiolas slice and GC connectivity remain coherent

### WARN

- EPLI placement distribution is heuristic
- default candidate ranking is order-sensitive unless explicitly overridden
- default contact radius is heuristic rather than evidence-backed

### FAIL

- no maintained network-ready EPLI slice asset is shipped
- default EPLI target pattern is too narrow relative to the literature
- no cell-level or network-level validation suite exists yet for the new class

## What "guarantee correctness" can and cannot mean here

What can be guaranteed:

1. the code matches explicitly stated assumptions
2. exported slices satisfy layer and connectivity invariants
3. every default has a cited justification or is labeled as heuristic
4. failed assumptions are caught automatically by audit scripts/tests

What cannot be guaranteed:

1. that the biology is true before it is tested
2. that one synthetic cell is the final correct PV-EPL model
3. that a rhythm match alone implies the right mechanism

So the correct target is not "guarantee truth." It is:

- guarantee implementation integrity,
- guarantee auditability,
- and refuse to overclaim beyond validated evidence.

## Required gates before claiming biological readiness

1. **Morphology gate**
   - surrogate must remain inside cited soma/process/span constraints
   - status: mostly passed

2. **Distribution gate**
   - EPLI placement must come from an explicit EPL density prior or a defended
     proxy distribution, not raw OPL fallback order
   - status: not passed

3. **Synapse-geometry gate**
   - default contact classes must include literature-supported perisomatic
     targeting, not soma-only placeholder rules
   - status: not passed

4. **Export gate**
   - at least one maintained slice must ship with nonzero `EPLIs__MCs` and/or
     `EPLIs__TCs` if network-ready support is claimed
   - status: not passed

5. **Cell-physiology gate**
   - passive and spiking responses must be benchmarked against cited EPL-IN
     measurements
   - status: not passed

6. **Network-behavior gate**
   - ablations must show that the new class changes synchrony/timing in the
     predicted way, not just total excitation
   - status: not passed

## Immediate next actions implied by this audit

1. Replace the default `*dend* -> *soma*` EPLI rule with an auditable
   perisomatic target family plan.
2. Build and retain one maintained opt-in EPLI slice export for automated
   geometry checks.
3. Add a cell-level physiology validation harness for the surrogate.
4. Add explicit network ablation criteria before interpreting any HFO match as
   mechanism support.
