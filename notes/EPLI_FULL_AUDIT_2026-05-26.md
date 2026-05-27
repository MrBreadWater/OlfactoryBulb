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
- generic audit entrypoint: `tools/run_audit.py`
- reusable package: `olfactorybulb/audit/`

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

Follow-up export evidence is now more specific than that:

- reduced headless smoke exports do write valid `MCs.json`, `TCs.json`,
  `GCs.json`, `EPLIs.json`, and the corresponding synapse-set files
- however, those reduced exports still write `0` explicit entries in:
  - `GCs__MCs`
  - `GCs__TCs`
  - `EPLIs__MCs`
  - `EPLIs__TCs`
- canonical-density headless exports still hit repeated Blender background
  errors of the form:
  - `Error: Object '1 OPL-Inner' has no evaluated mesh data`
  - `Error: Object '1 OPL-Outer' has no evaluated mesh data`
  and did not complete to a maintained asset

So the remaining hard failure is now specifically twofold:

- **maintained canonical-density EPLI slice asset export is not reliable yet**
- **the current reduced deterministic exports are too sparse to certify
  network-ready connectivity**

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

### 3. The default EPLI synapse geometry is still heuristic, but no longer soma-only

Severity: medium

Current default EPLI blueprints:

- source pattern: `*dend*`
- dest pattern: `@principal_perisomatic`
- max distance: `20 um`

Relevant code:

- `olfactorybulb/epli.py`
- selector implementation: `olfactorybulb/slicebuilder/blender.py`

This is materially better than the earlier `*soma*` placeholder. The selector
now encodes a broader perisomatic principal-cell territory spanning:

- soma
- proximal apical dendrite
- axon hillock / axon-root territory

with a local point-level distance gate around the soma.

That is directionally consistent with Burton 2024.

However, it is still **heuristic**:

- the local perisomatic radius is a coded scaffold, not a published exact
  anatomical threshold
- the selector has not yet been validated on a maintained exported EPLI slice

### 4. Slice-level contact evidence is still incomplete

Severity: medium-high

What has been shown so far:

- reduced deterministic slices now provide a usable local evidence base
- under the **exported default** perisomatic rule, all tested reduced slices
  still produce `0` explicit `EPLI -> MC` and `EPLI -> TC` entries
- offline latent-contact search on those same exports shows:
  - reproducible nonzero `EPLI -> TC` dendritic overlap
  - persistent `EPLI -> MC` zero overlap across all tested reduced scans,
    including MC-favoring count/depth sweeps

Observed examples from the reduced scans:

- `DorsalColumnSliceEPLI_mcscan_d`
  - export: `10 MC / 4 TC / 27 GC / 8 EPLI`
  - latent best `EPLI -> TCs`: `24` entries, `src_cov=0.875`,
    `dst_cov=0.5`, `dist50=5.98 um`
  - latent best `EPLI -> MCs`: `0` entries
- `DorsalColumnSliceEPLI_mcscan_f`
  - export: `10 MC / 10 TC / 42 GC / 10 EPLI`
  - latent best `EPLI -> TCs`: `99` entries, `src_cov=1.0`,
    `dst_cov=0.4`, `dist50=12.63 um`
  - latent best `EPLI -> MCs`: `0` entries
- `DorsalColumnSliceEPLI_mcscan_g`
  - export: `12 MC / 6 TC / 58 GC / 12 EPLI`
  - latent best `EPLI -> TCs`: `91` entries, `src_cov=0.833`,
    `dst_cov=0.833`, `dist50=15.70 um`
  - latent best `EPLI -> MCs`: `0` entries

Interpretation:

- the current synthetic morphology plus current placement heuristic is not yet
  enough to justify a default **PV-symmetric MC/TC** connectivity claim
- the strongest current mismatch is **TC-biased latent geometry**
- the next bottleneck is likely morphology reach and/or soma-depth
  distribution, not only the contact-radius parameter

### 5. Behavioral validation is now partial rather than absent

Severity: medium-high

The repo now has a machine-runnable isolated-cell behavior gate for the
synthetic surrogate via:

- `olfactorybulb/audit/neuron_protocols.py`
- `tools/run_audit.py epli_correctness`
- `test_synthetic_epl_fsi.py`

Current observed fixed-step behavior after removing the destabilizing `Ih`
mechanism from the surrogate soma:

- stable at rest with no spontaneous spikes
- fires repetitively under modest current injection
- reaches:
  - `30 Hz` at `0.2 nA`
  - `53.3 Hz` at `1.0 nA`
  - `66.7 Hz` at `1.5 nA`
  - `76.7 Hz` at `2.0 nA`
  - `113.3 Hz` at `3.0 nA`

This is now enough to say the surrogate has a validated isolated
fast-spiking regime in the Huang range.

What is still missing:

- passive-property comparison against reported electrophysiology
- synaptic response validation
- slice-distribution validation
- network ablation signatures
- gamma / HFO effects under explicit perturbations

So the surrogate is no longer "behaviorally unvalidated" in the narrow
single-cell sense, but it is still far from network-validated.

## Current pass/fail judgment

### PASS

- surrogate morphology matches first-pass literature targets well
- high-level reciprocal circuit architecture is directionally correct
- host Birgiolas slice and GC connectivity remain coherent

### WARN

- EPLI placement distribution is heuristic
- default candidate ranking is order-sensitive unless explicitly overridden
- default contact radius is heuristic rather than evidence-backed
- isolated behavior is now stable and fast-spiking, but MC/TC balance is not
  yet validated

### FAIL

- no maintained network-ready EPLI slice asset is shipped
- reduced deterministic slices remain too sparse to certify network-ready
  connectivity
- current reduced-slice evidence is TC-biased and does not support the intended
  default MC/TC symmetry

### Remaining structural WARN

- default EPLI perisomatic selector is still heuristic rather than directly
  measured

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
