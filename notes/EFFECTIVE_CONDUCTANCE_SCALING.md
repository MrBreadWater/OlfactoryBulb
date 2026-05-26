# Effective conductance scaling for reduced olfactory bulb runs

Status as of 2026-05-26: this is a modeling note for interpreting and
choosing notebook-facing cell and circuit parameters in
`notebooks/obgpu-working-experiment.ipynb`. It is not a claim that the
effective parameters below are unitary synaptic conductances. The new
paramset that implements the nominal recommendation is
`GammaSignature_ContactScaled`.

## Main conclusion

There are two different ranges that should not be mixed:

1. Strict biophysical ranges describe one synapse, one electrical junction, or
   one fitted intrinsic channel density. These are the values to use when the
   model has the same number and pattern of inputs as the biological circuit.
2. Effective reduced-model ranges describe a coarse model contact that stands
   in for many biological contacts, plus missing input diversity, missing
   population size, and simplified activity statistics. These values can be
   much larger than a unitary conductance without meaning that one real synapse
   has that conductance.

For this repository's current dorsal-column slice, the contact-compression
calculation supports a `GabaSyn.gmax` near `0.3..0.4 uS` and an
`AmpaNmdaSyn.gmax` near `0.8..2.0`. Going above that can still make sense for
phenomenological spectrogram matching, but it should be labeled as an
additional network-gain parameter, not a bioplausible synapse/contact-count
parameter. In particular, `gaba_gmax = 0.1..0.4` is defensible as contact
compression; `1..2` is a strong effective gain; `10` is diagnostic/exploratory
only.

## Sources used

- Davison, A. P. 2001. [Mathematical modelling of information processing in the
  olfactory bulb](https://andrewdavison.info/files/davison_thesis.pdf).
  Useful here for dendrodendritic contact-count estimates, granule-cell spine
  counts, and the older OB network synaptic constants.
- Aghvami, Kubota, and Egger 2022.
  [Anatomical and Functional Connectivity at the Dendrodendritic Reciprocal
  Mitral Cell-Granule Cell Synapse](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2022.933201/full).
  Useful here for the rat reciprocal-synapse discussion, a unitary inhibitory
  conductance on the order of `0.2 nS`, and synapse density around
  `0.65..0.83 per um` on MC lateral dendrites.
- De Almeida, Antunes, and Roque 2014.
  [Electrical responses of three classes of granule cells of the olfactory bulb
  to synaptic inputs in different dendritic locations](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2014.00128/full).
  Useful here for OB granule-spine counts and AMPA/NMDA model constants:
  `g_AMPA = 1 nS`, `g_NMDA = 0.593 nS`, `tau_AMPA = 2/5.5 ms`,
  `tau_NMDA = 52/343 ms`.
- Gire et al. 2012.
  [Mitral cells in the olfactory bulb are mainly excited through a multistep
  signaling path](https://pubmed.ncbi.nlm.nih.gov/22378870/).
  Useful here for the point that MC direct OSN input is strongly shunted by
  connexin-36 electrical coupling, and for a model using `1.1 nS` MC-MC gap
  junction conductance.
- Chen, Lin, and Schild 2009.
  [Odor coding by modules of coherent mitral/tufted cells in the vertebrate
  olfactory bulb](https://meds371s.uchc.edu/olfactory-chenlinschild2009.pdf).
  Useful here as additional support that M/T cells belonging to the same
  glomerulus can be electrically coupled and produce coherent activity.
- Blakemore, Corthell, and Trombley 2018.
  [Kainate receptors play a role in modulating synaptic transmission in the
  olfactory bulb](https://pmc.ncbi.nlm.nih.gov/articles/PMC6267532/).
  Useful here for treating KARs as modulatory OB receptors rather than as a
  well-constrained unitary conductance in this model.
- Schoppa and Westbrook 1999.
  [Regulation of synaptic timing in the olfactory bulb by an A-type potassium
  current](https://www.nature.com/articles/nn1299_1106.pdf).
  Useful here for interpreting `gc_ka_gbar_scale`: GC A-type potassium current
  regulates the timing of dendrodendritic inhibition, so scaling it is an
  intrinsic-channel intervention, not a contact-count correction.
- Duchamp-Viret, Duchamp, and Chaput 2000.
  [Peripheral Odor Coding in the Rat and Frog](https://pubmed.ncbi.nlm.nih.gov/10704512/).
  Useful here for the order of magnitude of receptor-neuron firing used by the
  repo's `max_firing_rate = 150 Hz` default.

## Repository units and conversion rules

The notebook-facing names come through `obgpu_experiment_helpers.py` and then
override fields in `olfactorybulb.paramsets`. The important mechanism-level
units are:

| Parameter | Mechanism meaning | Conversion used here |
|---|---|---|
| `gaba_gmax` | `GabaSyn.gmax`, in `uS`; normalized event peak is `gmax * weight` | `1 uS = 1000 nS` |
| `ampa_nmda_gmax` | `AmpaNmdaSyn.gmax`; AMPA event adds `gmax * 0.001 uS` | numeric `gmax` is approximately AMPA peak `nS` for weight 1 |
| `ampa_nmda_nmdafactor` | multiplier in `gnmda = mgblock(v) * R * gmax * nmdafactor` | keep at the mechanism default `0.0035` unless testing receptor balance |
| `gap_mc`, `gap_tc` | `GapJunction.g`, in `uS`, direct pairwise electrical conductance | `0.001 uS = 1 nS` |
| `kar_mt_gmax`, `kar_gc_gmax` | `KainateSyn.gmax`, in `uS`; peak kernel fraction is `amp1 + amp2 + amp3` | `g_peak_nS = 90.731603114 * gmax_uS * weight * block` |

The KAR peak factor is from the current local mechanism:

```text
amp_sum = 0.06942183802 + 0.008503803144 + 0.01280596195
        = 0.090731603114
g_peak_nS = 1000 * gmax_uS * weight * block * amp_sum
          = 90.731603114 * gmax_uS * weight * block
```

## Current reduced slice counts

The contact denominator comes from the actual slice JSON files in
`olfactorybulb/slices/DorsalColumnSlice`:

```text
GCs__MCs.json:
  total reciprocal entries       = 2238
  target MC cells with entries   = 10
  mean entries per target MC     = 2238 / 10 = 223.8
  median/min/max per target MC   = 250 / 70 / 306

GCs__TCs.json:
  total reciprocal entries       = 1751
  target TC cells with entries   = 23
  mean entries per target TC     = 1751 / 23 = 76.13
  median/min/max per target TC   = 56 / 11 / 272

All reciprocal entries:
  total reciprocal entries       = 2238 + 1751 = 3989
  modeled GC roots               = 159
  mean M/T inputs per modeled GC = 3989 / 159 = 25.09
```

Davison estimates about `17000` dendrodendritic synapses per mitral cell and
`4600` per tufted cell, based on dendritic length and synapse-density
calculations. The compression factors for this reduced slice are therefore:

```text
MC dendrodendritic compression = 17000 / 223.8 = 75.96
TC dendrodendritic compression =  4600 /  76.13 = 60.42
```

For parameters that act on the GC side of reciprocal excitation, use granule
spine counts. Davison reports `144..297` peripheral dendrite spines in mice,
and De Almeida et al. model three GC classes with `194`, `118`, and `114`
observed pedunculated spines.

```text
GC reciprocal-input compression, low  = 144 / 25.09 = 5.74
GC reciprocal-input compression, mid  = 200 / 25.09 = 7.97
GC reciprocal-input compression, high = 297 / 25.09 = 11.84
```

## Parameter-by-parameter ranges

| Notebook parameter | Strict biophysical range | Contact-scaled/effective range | Nominal value in `GammaSignature_ContactScaled` |
|---|---:|---:|---:|
| `gap_mc` | `0.00015..0.0011 uS` (`0.15..1.1 nS`) | no contact multiplier recommended; sweep up to `0.005 uS` only as synchrony gain | `0.0011` |
| `gap_tc` | `0.00015..0.0011 uS` if enabled | no contact multiplier recommended; same caveat as MC | `0.0011` |
| `ampa_nmda_gmax` | about `0.6..1.2` as a one-spine AMPA/NMDA scale | `0.6..2.0` nominal; `2..8` strong recruitment gain; current `64` is not contact-count explained | `2.0` |
| `ampa_nmda_nmdafactor` | mechanism default `0.0035` | no contact multiplier; this changes receptor balance, not connection count | inherited default |
| `ampa_block` | `1` unblocked, `0..1` pharmacology | no contact multiplier | inherited default |
| `ketamine_block` | `1` unblocked, `0..1` pharmacology | no contact multiplier | inherited default |
| `gaba_gmax` | `0.0002..0.0006 uS` from unitary `0.2..0.6 nS`; repo coarse-contact default is `0.005 uS` | `0.30..0.38 uS` from missing M/T dendrodendritic contacts; `0.4..2` extra fit gain; `10` diagnostic only | `0.35` |
| `gaba_tau2_ms` | Davison model used `18 ms`; repo saved coarse contact uses `100 ms` | `16..100 ms` is reasonable sensitivity range | `36` |
| `kar_mt_gmax` | not tightly constrained as a unitary OB synaptic KAR | use effective peak nS formula; `0.01..0.05 uS` gives useful nS-scale slow drive with current weights | `0.03` |
| `enable_gc_kar` | false if testing baseline receptor set | true for the contact-scaled KAR hypothesis | `True` |
| `kar_gc_gmax` | no firm OB unitary value; use small modulatory component | `0.006..0.012 uS` if scaling a `0.001 uS` slow component by the GC input-compression factor | `0.008` |
| `kar_tau1_ms`, `kar_tau2_ms`, `kar_tau3_ms` | local fitted KAR kernel constants | keep fixed unless refitting the kernel | inherited default |
| `kar_amp1`, `kar_amp2`, `kar_amp3` | local fitted KAR kernel amplitudes | keep fixed unless refitting the kernel | inherited default |
| `kar_kd` | `0` is calibrated linear single-event kernel | positive values are saturation sensitivity tests | inherited default |
| `kar_block` | `1` unblocked, `0..1` pharmacology | no contact multiplier | inherited default |
| `kar_osn_weight_scale` | relative OSN event multiplier | tune as stimulus calibration, not synaptic anatomy | inherited default |
| `kar_gc_weight_scale` | relative reciprocal event multiplier | can be swept instead of `kar_gc_gmax` for missing event-count sensitivity | inherited default |
| `gc_ka_gbar_scale` | `1` preserves fitted GC intrinsic conductance; `0` mimics strong removal/blockade | `0.5..2` sensitivity range; no contact multiplier | inherited default |
| `max_firing_rate_hz` | repo default `150 Hz` is already a population-input cap | do not inflate to compensate for missing contacts without labeling it as stimulus gain | inherited default |
| `inhale_duration_ms` | repo default `125 ms`, in the sniff-like range used locally | not a contact-scaled parameter | inherited default |
| `input_syn_tau1_ms`, `input_syn_tau2_ms` | repo defaults `6/12 ms` | stimulus-filter parameters, not contact-scaled | inherited default |
| `mc_input_weight`, `tc_input_weight` | dimensionless odor-input event weights | stimulus and pathway calibration; use with Gire et al. MC-shunt/TC-pathway context | inherited `0.2`, `0.8` |
| `mc_input_delay_ms`, `tc_input_delay_ms` | pathway timing offsets | tune timing, not conductance | inherited `0`, `0` |

## GABA calculation

The saved slice generator uses a coarse reciprocal GABA contact value:

```text
saved dest_syn_params for GabaSyn = {'gmax': 0.005, 'tau1': 1, 'tau2': 100}
0.005 uS = 5 nS per modeled reciprocal entry
```

Using the missing-contact factors above:

```text
MC-scaled GABA gmax = 0.005 uS * 75.96 = 0.3798 uS
TC-scaled GABA gmax = 0.005 uS * 60.42 = 0.3021 uS
nominal compromise   = 0.35 uS
```

Common sweep values interpreted as effective contact counts:

| `gaba_gmax` | Multiplier over saved `0.005 uS` | Effective MC entries | Effective TC entries | Interpretation |
|---:|---:|---:|---:|---|
| `0.1` | `20x` | `223.8 * 20 = 4476` | `76.13 * 20 = 1523` | below Davison contact estimate, still plausible compression |
| `0.35` | `70x` | `223.8 * 70 = 15666` | `76.13 * 70 = 5329` | close to MC/TC anatomical estimates |
| `1.0` | `200x` | `44760` | `15226` | beyond contact-count explanation; effective gain |
| `10.0` | `2000x` | `447600` | `152260` | far beyond anatomy; exploratory diagnostic gain |

This is why the spectrogram cleaning up only around `0.1..10` is not
surprising. The low end of that range is still contact-compression plausible,
but the high end is compensating for additional missing activity structure.

## AMPA/NMDA calculation

The saved reciprocal excitation contact value is:

```text
saved source_syn_params for AmpaNmdaSyn = {'gmax': 0.1}
AmpaNmdaSyn AMPA peak for weight 1 ~= gmax nS
saved AMPA peak ~= 0.1 nS per modeled reciprocal entry
```

For GC-side contact compression:

```text
low  effective gmax = 0.1 *  5.74 = 0.574
mid  effective gmax = 0.1 *  7.97 = 0.797
high effective gmax = 0.1 * 11.84 = 1.184
```

The literature model values also put a one-spine AMPA scale around `1 nS` and
NMDA around `0.593 nS`. A nominal `ampa_nmda_gmax = 2.0` therefore keeps the
new paramset near the contact-scaled range while adding some recruitment gain
for missing correlated excitatory drive. It is intentionally far below the
older `GammaSignature` value of `64`, which should be interpreted as a very
large effective network-drive gain rather than a contact-count correction.

Do not multiply `ampa_nmda_nmdafactor` by the contact factor. That parameter
changes AMPA/NMDA balance and voltage-dependent calcium/NMDA drive to GABA
release; it is not a count of missing synapses.

## Gap junction calculation

Gap junction values are direct electrical conductances, not chemical contacts
that can be corrected by multiplying by missing synapse count.

```text
Gire et al. model value        = 1.1 nS = 0.0011 uS
current GammaSignature value   = 32 uS = 32000 nS
ratio to 1.1 nS reference      = 32000 / 1.1 = 29091x
```

The current `32 uS` gap setting is therefore not biophysical as a pairwise
gap-junction conductance. It can still be useful as a synchrony forcing
parameter in a reduced model, but the contact-scaled paramset uses `0.0011 uS`
for both MC and TC gap conductance so that gap currents do not dominate the
interpretation of the chemical conductance sweeps.

## KAR calculation

The current KAR mechanism is a fitted slow conductance kernel. There is no
well-constrained OB unitary KAR conductance to use the same way as GABA or
AMPA. The safest interpretation is therefore effect-size based.

With current `GammaSignature` input weights:

```text
kar_mt_gmax = 0.03 uS
MC input weight = 0.2
TC input weight = 0.8

MC KAR peak = 90.731603114 * 0.03 * 0.2 = 0.544 nS
TC KAR peak = 90.731603114 * 0.03 * 0.8 = 2.178 nS
```

Those are plausible nS-scale slow conductances for an effective OSN drive
component, especially because the odor-input generator is already a population
input abstraction. I would not apply the dendrodendritic contact multiplier to
`kar_mt_gmax` unless the OSN input generator is also changed to represent
explicit individual OSN contacts.

For optional MC/TC-to-GC KARs at reciprocal excitation sites, it is reasonable
to apply the GC input-compression factor to a small slow component:

```text
baseline slow GC KAR component = 0.001 uS
mid GC compression factor      = 7.97
contact-scaled kar_gc_gmax     = 0.001 * 7.97 = 0.00797 uS
nominal rounded value          = 0.008 uS
peak conductance at weight 1   = 90.731603114 * 0.008 = 0.726 nS
```

This is still a modulatory value. If it starts driving GC activity by itself,
prefer reducing `kar_gc_gmax` or `kar_gc_weight_scale` before changing the
kernel time constants.

## Intrinsic and stimulus parameters

`gc_ka_gbar_scale` is not a contact-count parameter. Schoppa and Westbrook
showed that GC A-type potassium current regulates the timing of
dendrodendritic inhibition by filtering fast AMPA-driven excitation relative
to slower NMDA-driven excitation. Therefore:

```text
gc_ka_gbar_scale = 1.0  preserves the fitted cell models
gc_ka_gbar_scale = 0.0  is a strong blockade/removal intervention
0.5..2.0                is a reasonable sensitivity range
```

The odor-input parameters are also not contact-count parameters:

```text
max_firing_rate_hz = 150
inhale_duration_ms = 125
input_syn_tau1_ms = 6
input_syn_tau2_ms = 12
mc_input_weight = 0.2
tc_input_weight = 0.8
mc_input_delay_ms = 0
tc_input_delay_ms = 0
```

These remain inherited from `GammaSignature` and `SilentNetwork` in the new
paramset. If the simulated activity is still too weak after using
contact-scaled dendrodendritic conductances, tune these explicitly as stimulus
and pathway gain/timing parameters rather than folding the correction into
GABA or AMPA/NMDA conductance.

## Recommended sweep interpretation

Use three labels in run metadata and figures:

| Label | Conductance meaning | Example |
|---|---|---|
| `unitary/strict` | one real synapse/junction scale | `gaba_gmax <= 0.005`, `gap <= 0.0011` |
| `contact_scaled` | one modeled entry stands in for many real contacts | `gaba_gmax = 0.1..0.4`, `ampa_nmda_gmax = 0.8..2.0` |
| `phenomenological_gain` | compensates for missing inputs, synchrony, receptor state, or activity statistics | `gaba_gmax > 0.4`, `ampa_nmda_gmax > 2`, old `gap = 32` |

It is acceptable to go beyond the contact-scaled range for this simulation if
the goal is matching a recorded LFP/spectrogram signature. The important
constraint is interpretability: once beyond contact scaling, report those
parameters as effective gains of a reduced model, not as bioplausible cell
conductances.
