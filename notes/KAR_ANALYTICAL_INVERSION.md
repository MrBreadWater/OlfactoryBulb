# KAR analytical inversion for the olfactory bulb model

This note specifies how to convert the Frerking and Ohliger-Frerking KAR EPSP
voltage fit into a conductance waveform for the current olfactory bulb model.
The main conclusion is that the original paper's 20 ms passive membrane time
constant should not be treated as a universal olfactory bulb constant. In this
project, the inversion should be parameterized by the target cell or target
segment passive properties.

## Source waveform

Frerking and Ohliger-Frerking (2002) fit KAR-mediated EPSPs with a lognormal
voltage waveform:

```text
u_K(t) = a * exp(-0.5 * (ln(t / b) / c)^2), t > 0
```

where `u_K(t)` is the voltage deflection above rest. Their reported mean KAR
fit parameters are:

| Parameter | Value | Interpretation here |
|---|---:|---|
| `a` | `1.19 mV` | Peak voltage deflection, not conductance |
| `b` | `32 ms` | Time of the voltage peak |
| `c` | `1.68` | Dimensionless lognormal width |

The same paper also states that these fitting functions were chosen for
goodness of fit to the voltage response and do not define receptor kinetics.
That matters here: the voltage waveform is the response of a membrane to a
hidden synaptic conductance. It should not be inserted directly as a voltage
source in the network.

The paper used a passive membrane model with `tau_m = 20 ms` to illustrate how
EPSCs are filtered into EPSPs. That value is a model assumption in hippocampal
interneurons, not an olfactory bulb constant.

## Passive inversion

For a one-compartment passive membrane receiving a KAR excitatory conductance:

```text
C * dV/dt = -g_L * (V - E_L) + g_K(t) * (E_exc - V)
```

Let:

```text
u(t) = V(t) - E_L
V(t) = E_L + u(t)
tau_m = C / g_L
```

Then:

```text
g_K(t) / g_L = (tau_m * du/dt + u) / (E_exc - E_L - u)
```

For the lognormal KAR voltage fit:

```text
u(t) = a * exp(-0.5 * (ln(t / b) / c)^2)
du/dt = -u(t) * ln(t / b) / (c^2 * t)
```

So the exact passive conductance inversion is:

```text
g_K(t) / g_L =
    (u(t) * (1 - tau_m * ln(t / b) / (c^2 * t))) /
    (E_exc - E_L - u(t))
```

Use `t` and `tau_m` in the same time units, preferably ms. Set `g_K(t) = 0`
for `t <= 0`.

For NEURON point-process implementation with conductance in `uS`:

```text
area_cm2 = h.area(x, sec=sec) * 1e-8
g_L_uS = seg.g_pas * area_cm2 * 1e6
g_K_uS(t) = g_L_uS * g_K_over_g_L(t) * kar_scale
```

where `seg.g_pas` is in `S/cm2`, `h.area()` returns `um2`, and `kar_scale` is
the sweepable fitted/sensitivity multiplier. This keeps the paper-derived
waveform shape separate from the unknown absolute KAR density.

If using only a cell-class value instead of segment area, compute:

```text
tau_m_ms = 0.001 * cm / g_pas
```

with `cm` in `uF/cm2` and `g_pas` in `S/cm2`.

## Olfactory bulb constants to use in this project

Use the target cell or segment passive constant instead of a fixed 20 ms.

The current Birgiolas2020 cell classes set `cm`, `e_pas`, and `g_pas` globally
on each model cell. Derived passive time constants from
`prev_ob_models/Birgiolas2020/isolated_cells.py` are:

| Cell group | Classes | `tau_m = 0.001 * cm / g_pas` |
|---|---|---:|
| MC, all available classes | `MC1` to `MC5` | `20.25..47.12 ms`, mean `32.53 ms` |
| MC, slice-used classes | `MC4`, `MC5` | `22.94..42.28 ms`, mean `32.61 ms` |
| TC, all available classes | `TC1` to `TC5` | `0.96..28.11 ms`, mean `9.06 ms` |
| TC, slice-used classes | `TC3` to `TC5` | `0.96..28.11 ms`, mean `14.22 ms` |
| GC, all/slice-used classes | `GC1` to `GC5` | `25.75..52.41 ms`, mean `37.20 ms` |

The TC range is unusually broad and includes very short passive constants from
high fitted leak values. Treat those as model-specific segment constants, not
as a claim that every tufted cell whole-cell time constant is below 10 ms.

The observation tables already in this repo give these olfactory bulb
whole-cell membrane time constants:

| Cell group | Source table | Reported membrane time constants |
|---|---|---|
| MC validation observations | `notes/mc_model_ephyz_observations.csv` | Burton and Urban 2014: `21.3 +/- 9.4 ms`; Hovis 2010: `14.0 +/- 4.5 ms`; Yu et al. 2015: `28.1 +/- 16.5 ms`; Zibman et al. 2011: `42.5 +/- 16.1 ms` |
| GC | `notes/gc_model_ephyz_observations.csv` | Burton and Urban 2015: `27.3 +/- 13.2 ms` |

There is no separate TC observation table in the current notes. For TCs, use
the fitted TC model constants above and treat the MC validation observations as
only a broad principal-cell sanity range.

Recommended default policy:

1. For OSN to M/T KARs, compute `tau_m` from the target MC/TC segment when the
   target has `pas`, otherwise use the target cell class passive value.
2. For optional M/T to GC KARs, compute `tau_m` from the target GC segment.
3. Use `E_L = e_pas` from the target segment or class, and keep
   `E_exc = 0 mV`, consistent with `AmpaNmdaSyn.E` and the current
   `KainateSyn.e`.
4. Keep `a`, `b`, and `c` at the Frerking and Ohliger-Frerking KAR values for
   the first implementation, but sweep them through the reported SEM ranges as
   sensitivity checks.

This means the analytical inversion has no single "correct" 20 ms constant.
The correct constant for this model is the target membrane's own passive
constant, with the observed olfactory bulb values above used as sanity bounds.

## Consistency with the current implementation

The current implementation is already conductance-based:

| Mechanism | Current project behavior |
|---|---|
| `AmpaNmdaSyn.mod` | Glutamate event drives AMPA and NMDA conductances. `ketamine_block` multiplies NMDA current only. `ampa_block` multiplies AMPA current. |
| `KainateSyn.mod` | Glutamate event drives a slow excitatory KAR conductance point process with reversal `e = 0 mV`. Current is `i = g * (v - e)`, using NEURON's outward-current sign convention. |
| `GabaSyn.mod` | Fast inhibitory reset loop remains separate from KAR gain. |

The analytical inversion should not change the ketamine mechanism. Ketamine
still acts through `AmpaNmdaSyn.ketamine_block`; KAR conductance is changed only
by explicit KAR sensitivity/blockade parameters.

The current `KainateSyn.mod` defaults `tau1 = 8 ms` and `tau2 = 80 ms` are a
qualitative slow-gain placeholder. They are not the Frerking and
Ohliger-Frerking lognormal EPSP fit. The analytical inversion can replace or
calibrate that placeholder while preserving the event-driven point-process
architecture.

## Implementation options

### Preferred network path

Keep the existing event-driven point-process architecture, but calibrate its
conductance kernel from the analytical inversion:

1. For a chosen target class or target segment, compute `tau_m` and `E_L`.
2. Generate `g_K_over_g_L(t)` from the formula above over a finite window, for
   example `0.1..500 ms`.
3. Fit the generated positive conductance trace to the cheapest kernel that is
   stable in NEURON:

```text
g_fit(t) = A_fit * (exp(-t / tau_decay) - exp(-t / tau_rise))
```

or, if the one-pair fit is poor:

```text
g_fit(t) =
    A1 * (exp(-t / tau_d1) - exp(-t / tau_r1)) +
    A2 * (exp(-t / tau_d2) - exp(-t / tau_r2))
```

4. Put the fitted `tau_rise`, `tau_decay`, and amplitude scaling into
   `KainateSyn.mod` or a new `KainateInvertedSyn.mod`.
5. Keep `kar_scale`/`gmax` sweepable. The inversion constrains the first-pass
   waveform shape, not the receptor density.

This is preferable to summing lognormal histories directly in a large network
because it keeps the synapse event-driven and cheap.

### Direct table path

For small validation runs, implement a table-driven point process:

```text
on NET_RECEIVE:
    store event time

on BREAKPOINT:
    g = sum_events(g_table[t - event_time]) * g_L_uS * kar_scale
```

Use this only for validation or small cell counts. It is less attractive for
the working GPU experiment because every active event remains in the synapse
history until the tail cutoff.

### Minimal helper for calibration

```python
import math


def kar_g_over_gl(t_ms, tau_m_ms, e_l_mv, e_exc_mv=0.0,
                  a_mv=1.19, b_ms=32.0, c=1.68):
    if t_ms <= 0.0:
        return 0.0
    u = a_mv * math.exp(-0.5 * (math.log(t_ms / b_ms) / c) ** 2)
    du_dt = -u * math.log(t_ms / b_ms) / (c * c * t_ms)
    denom = e_exc_mv - e_l_mv - u
    if denom <= 0.0:
        return 0.0
    return max(0.0, (tau_m_ms * du_dt + u) / denom)
```

This function returns conductance in units of leak conductance. Convert to `uS`
with the segment's absolute leak conductance.

## Validation checks

Before using the inverted kernel in the full ketamine/HFO experiment:

1. In a passive one-compartment test, inject the inverted `g_K(t)` and confirm
   the voltage reproduces the lognormal KAR EPSP with peak near `32 ms` and
   amplitude near `1.19 mV` for the chosen `tau_m`.
2. Repeat on an isolated MC/TC and GC morphology with active conductances
   disabled or held near rest. The exact match will degrade with morphology,
   but peak time and amplitude should remain close after `kar_scale` adjustment.
3. Re-enable active currents and verify that KAR changes excitability/gain
   rather than setting the HFO period directly.
4. In the network, keep the existing controls:
   `ketamine_block`, `kar_block`, `kar_mt_gmax`, `kar_gc_gmax`,
   `ampa_block`, GABA-A block, and `gc_ka_gbar_scale`.

## Sensitivity sweeps

Run at least these sweeps to avoid over-interpreting one constant:

| Sweep | Values |
|---|---|
| `tau_m` policy | target segment, target class, fixed `20 ms`, fixed observed MC/GC means |
| KAR voltage fit | `a = 1.19 +/- 0.12 mV`, `b = 32 +/- 6.4 ms`, `c = 1.68 +/- 0.23` |
| KAR scale | `0`, low, baseline, high |
| Ketamine | `ketamine_block = 1`, partial block, near-zero |
| Fast loop controls | AMPA block, GABA-A block |
| GC excitability gate | baseline `gbar_KA`, reduced `gbar_KA`, no KAR-to-IA interaction |

The hypothesis is supported only if KAR increases gain/synchronizability under
NMDA block while AMPA and GABA-A remain necessary for the fast rhythm.

## References

- Frerking M, Ohliger-Frerking P. 2002. "AMPA receptors and kainate receptors
  encode different features of afferent activity." Journal of Neuroscience
  22(17):7434-7443. DOI: `10.1523/JNEUROSCI.22-17-07434.2002`.
  PMCID: `PMC2967721`. https://doi.org/10.1523/JNEUROSCI.22-17-07434.2002
  and https://pmc.ncbi.nlm.nih.gov/articles/PMC2967721/
- Burton SD, Urban NN. 2014. "Greater excitability and firing irregularity of
  tufted cells underlies distinct afferent-evoked activity of olfactory bulb
  mitral and tufted cells." Journal of Physiology 592:2097-2118.
  DOI: `10.1113/jphysiol.2013.269886`.
  https://doi.org/10.1113/jphysiol.2013.269886
- Burton SD, Urban NN. 2015. "Rapid Feedforward Inhibition and Asynchronous
  Excitation Regulate Granule Cell Activity in the Mammalian Main Olfactory
  Bulb." Journal of Neuroscience 35(42):14103-14122.
  DOI: `10.1523/JNEUROSCI.0746-15.2015`.
  https://doi.org/10.1523/JNEUROSCI.0746-15.2015
- Burton SD, Urban NN. 2021. "Cell and circuit origins of fast network
  oscillations in the mammalian main olfactory bulb." eLife 10:e74213.
  DOI: `10.7554/eLife.74213`. https://doi.org/10.7554/eLife.74213
- Project sources used for consistency: `prev_ob_models/Birgiolas2020/isolated_cells.py`,
  `prev_ob_models/Birgiolas2020/Mechanisms/KainateSyn.mod`,
  `prev_ob_models/Birgiolas2020/Mechanisms/AmpaNmdaSyn.mod`,
  `notes/mc_model_ephyz_observations.csv`,
  `notes/gc_model_ephyz_observations.csv`, and
  `notes/porting/NETWORK_AND_CELL_PARAMETER_CATALOG.md`.
