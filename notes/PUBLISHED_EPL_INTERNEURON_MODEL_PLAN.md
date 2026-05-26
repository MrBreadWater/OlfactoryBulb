# Published EPL Interneuron Candidate Plan

Status as of 2026-05-26: this is still a planning/reference document. The
maintained full network still uses the Birgiolas2020 MC/TC/GC slice; published
candidate cells remain single-cell/proxy assets until a real EPL population path
is added.

## What is implemented now

The repo now has a registry of locally available published cell templates:

- `prev_ob_models.cell_registry`
- `prev_ob_models.Short2016.isolated_cells_obgpu`
- `prev_ob_models.LiCleland2013.isolated_cells_obgpu`

Stable model keys:

- `Short2016.PGC`
- `Short2016.ETC`
- `LiCleland2013.PGC`
- `LiCleland2013.GC`
- `Birgiolas2020.MC1` ... `Birgiolas2020.TC5`

The Li/Cleland mitral template is intentionally not in the public registry yet.
Its direct template wrapper still needs separate modern-NEURON debugging, and
it is not relevant to the first EPL-interneuron increment.

Python usage:

```python
from prev_ob_models.cell_registry import instantiate_cell, list_fast_inhibitory_proxy_models

cell = instantiate_cell("Short2016.PGC")
proxy_keys = [spec.key for spec in list_fast_inhibitory_proxy_models()]
```

Family-or-role resolution:

```python
from prev_ob_models.cell_registry import resolve_cell_choice

spec = resolve_cell_choice(family="Short2016", role="PGC")
cell = spec.instantiate()
```

Mechanism build helper:

```bash
conda run -n OBGPU tools/build_published_candidate_mechs.sh Short2016 /tmp/Short2016_mechs
conda run -n OBGPU tools/build_published_candidate_mechs.sh LiCleland2013 /tmp/LiCleland2013_mechs
```

Then point the wrappers at the compiled directory for that session:

```bash
export OBGPU_MECHANISM_ROOT=/tmp/Short2016_mechs
```

## Important limitation

The current full network cannot honestly hot-swap foreign morphologies into the
existing `MCs.json` / `GCs.json` / `TCs.json` slice at runtime. Those JSON files
encode morphology-specific section trees and segment locations generated from
the Birgiolas 2020 family.

So:

- published candidate cells are now available for single-cell work,
  side-by-side comparisons, and future population integration;
- they are **not** automatically interchangeable with the current live
  `MC/TC/GC` network just by setting a param today.

To make a new family truly swappable in the full network, we will need either:

1. slice-builder metadata plus new slice exports for that family, or
2. a new population added alongside the existing slice-driven cells.

## Biological target vs currently available proxies

### Target inhibitory population

What we actually want for the ketamine/HFO work is an `EPL` fast inhibitory
population that is closer to the parvalbumin-positive / CRH-positive
interneurons described experimentally.

Relevant biology:

- Miyamichi et al. 2013 showed that OB `PV` cells are primarily in the `EPL`,
  are densely connected with mitral cells, and anatomically reconstructed cells
  had `multipolar dendrites localized within the EPL` and `lacked an obvious
  axon`.
- Huang et al. 2013 showed that `CRH+ EPL interneurons` are also reciprocal
  inhibitory partners of mitral cells and their neurites lacked the axonal
  marker `βIV-spectrin`, consistent with an axonless EPL interneuron class.

These are the right biological targets for a future true `PV/CRH EPL`
population.

### What is locally available now

The repo does **not** currently contain a reconstructed PV/CRH EPL cell model.
The nearest published local candidates are glomerular / fast local proxy cells:

| Key | Paper | Morphology style | Why it is useful | Limitation |
| --- | --- | --- | --- | --- |
| `Short2016.PGC` | Short et al. (2016) | stylized multicompartment | fast inhibitory local-cell proxy; used in respiration-gating OB model | not EPL, not PV-specific |
| `LiCleland2013.PGC` | Li and Cleland (2013) | stylized multicompartment | published PG template; easy inhibitory comparison point | not EPL, not PV-specific |
| `Short2016.ETC` | Short et al. (2016) | stylized multicompartment | feedforward excitatory control for glomerular microcircuit hypotheses | excitatory, not inhibitory |

So the current published candidates should be treated as:

- `fast local inhibitory proxies`, not literal PV cells;
- useful for wiring and sensitivity work before a true EPL interneuron family
  is imported.

## Build note

The full legacy `Short2016/` mod tree is not fully ported to current NEURON 9:
the unrelated `thetastim.mod` path still fails C++ compilation. The helper
script above deliberately compiles only the mechanism subset required by the
published proxy cells we are exposing here.

## Suggested parameter convention for future swapability

When we add the actual new population, keep the configuration interface simple:

```python
epl_interneuron_model = "Short2016.PGC"
# or
epl_interneuron_family = "Short2016"
```

and resolve it through `prev_ob_models.cell_registry.resolve_cell_choice(...)`.

Those placeholder fields now exist in `olfactorybulb.paramsets.base.SilentNetwork`
so notebooks and future population code can start using the same names now.

That gives us:

- exact-model selection when we want full control;
- family-level selection when we want a sensible default model for a role.

## Next implementation step

The next honest step is:

1. add a new `EPLI` / `PVI` population path to the network,
2. use the registry to select an initial published proxy model,
3. keep the existing `MC/TC/GC` slice intact,
4. then replace the proxy with a true EPL morphology-backed family once we
   import one.

## Citations

- Short et al. (2016) *Respiration Gates Sensory Input Responses in the Mitral Cell Layer of the Olfactory Bulb*
- Li and Cleland (2013) *A two-layer biophysical model of cholinergic neuromodulation in olfactory bulb*
- Miyamichi et al. (2013) *Parvalbumin-Expressing Interneurons Linearly Control Olfactory Bulb Output*
- Huang et al. (2013) *Reciprocal connectivity between mitral cells and external plexiform layer interneurons in the mouse olfactory bulb*
