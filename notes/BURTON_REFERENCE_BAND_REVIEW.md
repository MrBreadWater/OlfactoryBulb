# Burton MC/TC Reference-Band Review

Date: 2026-06-02

## Scope

This note reviews the **Burton & Urban 2014 MC/TC reference rows** currently
used by the audit layer, with a narrow focus on the **band-shape choice** for
each positive-only metric that is currently configured as `lognormal_sd`.

This is **not** a GC unit audit. The GC bundle still needs a separate unit and
extraction review before its band choices should be treated as reliable.

## Key distinction

For the Burton MC/TC rows:

- the legacy CSV values are intended to preserve the paper's reported
  `mean ± standard deviation`
- a `lognormal_sd` band is an **audit-side reconstruction choice**
- it is **not** a claim that Burton & Urban reported lognormal statistics

The paper itself states that Table 4 and Table 5 values are reported as
**mean ± standard deviation**.

## What lognormal reconstruction does

The current `lognormal_sd` implementation is **moment-matched**:

- it takes the reported arithmetic mean and standard deviation
- reconstructs a lognormal distribution with those same arithmetic moments
- uses that reconstructed shape only for the acceptance interval

So if the input row is:

- `mean = m`
- `sd = s`

the reconstructed lognormal has arithmetic mean `m` and arithmetic standard
deviation `s` again, up to floating-point error.

This means the main question is **not** whether the moments are preserved.
They are.

The real question is whether a **right-skewed positive-only distribution**
is a defensible shape assumption for the metric.

## Manual decision rule used here

For Burton MC/TC rows, use this order:

1. If the row is really categorical, use `binary_indicator`.
2. If the metric is signed or naturally crosses zero, use `symmetric_sd`.
3. If the metric is positive-only and plausibly right-skewed, `lognormal_sd`
   is a reasonable fallback when only `mean ± SD` are available.
4. If the metric is positive-only but looks more like a compact shape/size
   descriptor than a skewed population variable, prefer `symmetric_sd` or
   `symmetric_sd` with clipping over `lognormal_sd`.
5. If variance is extremely large and likely reflects mixture structure or
   outliers rather than ordinary skew, treat `lognormal_sd` as only a
   stopgap. Prefer raw-cell quantiles if they ever become available.

## Summary recommendation table

| Metric | Current mode | Recommendation | Confidence | Short reason |
| --- | --- | --- | --- | --- |
| ISI Coefficient of Variation | `lognormal_sd` | Keep | High | Positive-only irregularity ratio; symmetric bands produce biologically silly lower tails |
| Rheobase Current | `lognormal_sd` | Keep | Medium-high | Positive threshold quantity with plausible right tail |
| Input Resistance | `lognormal_sd` | Keep | Medium | Positive heterogeneous cell property; skew is plausible |
| Membrane Time Constant | `lognormal_sd` | Keep | Medium | Positive, often right-skewed through heterogeneity in passive properties |
| FI Curve Slope | `lognormal_sd` | Keep | Medium | Positive gain metric; right-skew is plausible, though not proved |
| Capacitance | `lognormal_sd` | Review later | Medium-low | Positive, but often closer to compact shape/size variation than strong skew |
| AHP Duration | `lognormal_sd` | Keep only as stopgap | Low-medium | Positive-only, but variance is extremely large and may reflect mixture/outlier structure |
| Spiking Rate Accom. Time Constant | `lognormal_sd` | Keep only as stopgap | Low-medium | Positive-only, but spread is huge relative to mean |
| AHP Amplitude | `lognormal_sd` | Prefer symmetric | Medium-high | Positive in this paper, but behaves more like a compact waveform magnitude |
| AP Amplitude | `lognormal_sd` | Prefer symmetric | High | Tight, compact waveform metric with low CV |
| AP Half-Width / Width at Half-height | `lognormal_sd` | Prefer symmetric | High | Tight positive waveform width; lognormal adds complexity with little gain |

## Per-metric review

### 1. ISI Coefficient of Variation

Burton rows:

- MC: `0.45 ± 0.29`
- TC: `0.80 ± 0.43`

### Recommendation

Keep `lognormal_sd`.

### Why

- the metric is strictly nonnegative
- it is a ratio-like irregularity measure
- a symmetric band can produce meaningless negative acceptance bounds
- skew is biologically plausible

### Best alternative

- `quantile_interval` from raw per-cell values, if those ever become available

### Pros / cons

**Pros of current lognormal choice**

- respects positivity
- avoids fake negative irregularity
- still preserves Burton's arithmetic mean and sd

**Cons**

- assumes a particular skew family that Burton did not report

## 2. Rheobase Current

Burton rows:

- MC: `111.4 ± 55.7 pA`
- TC: `94.6 ± 49.7 pA`

### Recommendation

Keep `lognormal_sd`.

### Why

- strictly positive threshold quantity
- often heterogeneous across cells with a plausible right tail
- symmetric bands are not disastrous here, but lognormal is more defensible

### Best alternative

- `quantile_interval` from raw cell values

### Pros / cons

**Pros**

- respects positivity
- plausible biological skew

**Cons**

- not directly source-reported
- if the true distribution is closer to truncated normal than lognormal, the
  upper tail may be overstated

## 3. Input Resistance

Burton rows:

- MC: `94.3 ± 40.5 MOhm`
- TC: `111.8 ± 51.6 MOhm`

### Recommendation

Keep `lognormal_sd` for now.

### Why

- positive-only
- cross-cell heterogeneity makes skew plausible
- symmetric bands are usable, but lognormal is still a defensible fallback

### Best alternative

- `quantile_interval` from raw recordings

### Pros / cons

**Pros**

- avoids impossible negative lower tails
- fits heterogeneous positive cell-scale parameters reasonably well

**Cons**

- the evidence for a specifically lognormal shape is only indirect

## 4. Membrane Time Constant

Burton rows:

- MC: `21.3 ± 9.4 ms`
- TC: `18.8 ± 8.6 ms`

### Recommendation

Keep `lognormal_sd` for now.

### Why

- strictly positive
- passive-property heterogeneity makes right-skew plausible

### Best alternative

- `quantile_interval` if raw values are available

### Pros / cons

**Pros**

- positive-only interval
- plausible skew for a derived membrane timescale

**Cons**

- not directly evidenced by Burton's reporting format

## 5. FI Curve Slope

Burton rows:

- MC: `196 ± 76 Hz/nA`
- TC: `406 ± 144 Hz/nA`

### Recommendation

Keep `lognormal_sd` for now.

### Why

- positive-only gain parameter
- population heterogeneity likely yields a right tail

### Best alternative

- `quantile_interval` from cell-level gain values

### Pros / cons

**Pros**

- respects positivity
- plausible for gain-like derived metrics

**Cons**

- gain is a summary of an underlying curve-fit process, so the shape may depend
  as much on protocol and fit method as on cell biology

## 6. Capacitance

Burton rows:

- MC: `236.4 ± 94.6 pF`
- TC: `188.8 ± 110.0 pF`

### Recommendation

Treat `lognormal_sd` as acceptable but not preferred.

Preferred future mode: `symmetric_sd` unless stronger evidence of skew emerges.

### Why

- capacitance is positive-only
- but it is also a compact cell-size descriptor, not obviously a strong
  multiplicative or ratio-like skew variable
- lognormal is plausible, but weakly justified

### Better alternative today

- `symmetric_sd` with a zero lower clip, if you want a simpler interpretation

### Pros / cons

**Pros of current lognormal choice**

- preserves positivity

**Cons**

- likely over-interprets skew
- adds model complexity with limited evidence

## 7. AHP Duration

Burton rows:

- MC: `58.2 ± 77.5 ms`
- TC: `20.5 ± 20.1 ms`

### Recommendation

Keep `lognormal_sd` only as a **stopgap**.

### Why

- the metric is positive-only
- symmetric bands would be very awkward
- but the spread is so large that this may reflect mixture structure, outliers,
  or a non-Gaussian heavy-tail process more than a simple lognormal family

### Better alternatives

1. `quantile_interval` from raw-cell values
2. robust interval from reported median/IQR, if those data ever surface
3. as a fallback, `symmetric_sd` with zero clipping is simpler but less honest
   about skew

### Pros / cons

**Pros of current lognormal choice**

- avoids impossible negative lower tails
- at least acknowledges strong asymmetry

**Cons**

- large-variance regime makes the exact family assumption fragile
- likely too much confidence in a single skew model

## 8. Spiking Rate Accom. Time Constant

Burton rows:

- MC: `398 ± 562 ms`
- TC: `585 ± 664 ms`

### Recommendation

Keep `lognormal_sd` only as a **stopgap**.

### Why

- positive-only
- enormous variance relative to mean
- likely sensitive to fitting instability, protocol effects, and cell
  heterogeneity

### Better alternatives

1. `quantile_interval` from raw-cell fits
2. robust interval or explicit uncertainty on the fitting procedure

### Pros / cons

**Pros of current lognormal choice**

- more reasonable than a naive symmetric band

**Cons**

- this may not be a simple skew problem at all
- could be dominated by unstable fit estimates

## 9. AHP Amplitude

Burton rows:

- MC: `14.8 ± 3.2 mV`
- TC: `16.8 ± 3.3 mV`

### Recommendation

Prefer `symmetric_sd`.

### Why

- positive in this table, but behaves like a compact waveform-magnitude
  measurement
- low relative spread
- no strong evidence that a lognormal family is buying us anything

### Better alternative today

- `symmetric_sd` with zero lower clipping if you want positivity enforced

### Pros / cons

**Pros of switching to symmetric**

- simpler interpretation
- closer to what the paper literally reported

**Cons**

- does not encode positivity as strongly as lognormal unless clipped

## 10. AP Amplitude

Burton rows:

- MC: `76.2 ± 5.4 mV`
- TC: `72.1 ± 5.5 mV`

### Recommendation

Prefer `symmetric_sd`.

### Why

- compact waveform measurement
- low coefficient of variation
- little evidence for strong right-skew

### Better alternative today

- `symmetric_sd`

### Pros / cons

**Pros of switching**

- more literal interpretation of the reported summary
- little practical downside because the spread is already tight

**Cons**

- loses the positivity guarantee, though that is not practically important here

## 11. AP Half-Width / Width at Half-height

Burton rows:

- MC: `1.06 ± 0.20 ms`
- TC: `0.87 ± 0.10 ms`

### Recommendation

Prefer `symmetric_sd`.

### Why

- positive, but compact
- low spread
- waveform width is exactly the kind of metric where a symmetric summary is
  usually adequate unless there is direct evidence otherwise

### Better alternative today

- `symmetric_sd`

### Pros / cons

**Pros of switching**

- simpler and more source-faithful
- very small practical loss in positivity handling

**Cons**

- lower bound can drift slightly downward unless clipped

## Bottom line

### Strong keepers for `lognormal_sd`

- `ISI Coefficient of Variation`
- `Rheobase Current`
- `Input Resistance`
- `Membrane Time Constant`
- `FI Curve Slope`

### Keep only as stopgaps

- `AHP Duration`
- `Spiking Rate Accom. Time Constant`

### Better as `symmetric_sd`

- `AHP Amplitude`
- `AP Amplitude`
- `AP Half-Width / AP Width at Half-height`
- probably `Capacitance`, unless we want positivity-first handling more than
  literal interpretability

## Practical next step

If we want to tighten the Burton audit further without waiting on new source
data, the next clean change is:

1. switch these Burton rows from `lognormal_sd` to `symmetric_sd`:
   - `AHP Amplitude`
   - `AP Amplitude`
   - `AP Half-Width / AP Width at Half-height`
   - optionally `Capacitance`
2. leave these as `lognormal_sd`:
   - `ISI Coefficient of Variation`
   - `Rheobase Current`
   - `Input Resistance`
   - `Membrane Time Constant`
   - `FI Curve Slope`
3. leave these as explicitly provisional:
   - `AHP Duration`
   - `Spiking Rate Accom. Time Constant`

That would move the Burton path closer to a defensible metric-by-metric policy
without pretending the legacy CSV has more distribution information than the
paper actually gave us.
