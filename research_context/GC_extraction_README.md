# Olfactory bulb granule-cell reference-data extraction

This directory contains protocol-aware, subtype-aware reference tables for olfactory-bulb granule-cell validation targets. Generic mature granule cells, superficial granule cells, deep granule cells, adult-born granule cells, and modulated granule-cell states are tracked separately.

## Source summary

- Burton & Urban 2015 contributes the primary machine-readable generic granule-cell intrinsic and spike-train summary tables through the PMC article tables.
- Geramita, Burton & Urban 2016 contributes the primary superficial-versus-deep granule-cell subtype source data through Figure 4 supplementary tables in DOCX form.
- Hu et al. 2016 contributes secondary adult-born granule-cell sag and resonance constraints.
- Labarrera, London & Angelo 2013 contributes tonic-inhibition pharmacology constraints.
- Egger, Svoboda & Mainen 2005 and Giridhar & Urban 2012 contribute identity and latency-coding constraints.

## Suitable now

- Burton 2015 generic granule-cell passive, action-potential, and spike-train summary rows.
- Geramita 2016 superficial-versus-deep granule-cell subtype rows for morphology, passive properties, action-potential properties, and spike-train summary metrics.
- Hu 2016 adult-born granule-cell sag and resonance reference rows.
- Labarrera 2013 tonic-inhibition pharmacology rows.
- Identity, morphology, and long-latency reference rows with visible protocol caveats.

## Caveats

- No exact generic-GC or sGC/dGC current-vs-rate point table has been extracted yet. The f-I curve CSVs remain header-only until a machine-readable point table or manually curated point table is added.
- The pipeline does not perform built-in figure digitization. Figure-only f-I curve gaps remain in GC_needs_manual_extraction.csv.
- GC subtype, maturity, and modulation conditions are intentionally separated. Baseline validation should not silently include adult-born or pharmacology rows.

## Extraction status

- `ephys` rows: 13
- `fi_curve` rows: 0
- `subtype_ephys` rows: 20
- `subtype_fi_curve` rows: 0
- `protocols` rows: 11
- `identity` rows: 15
- `synaptic_latency` rows: 2
- `modulation` rows: 6
- `notes` rows: 8
- `manual` rows: 8
- `readme` rows: 0
- Missing required sources after acquisition: none
