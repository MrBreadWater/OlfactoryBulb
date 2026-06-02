# Manual Reference Extraction Templates

This directory contains **raw intake templates** for every currently unresolved
manual-extraction item in the literature reference system.

These files are meant to be filled in by a human when the paper does not expose
clean machine-readable values for a target we still want.

They are **not** the final canonical validation CSVs.

The canonical files are generated later by the dataset pipeline.

## Fastest Way To Use These

1. open the template file for the paper/task you care about
2. fill in only the values you can support
3. leave unknown fields blank
4. do not guess
5. keep `source_location` specific
6. preserve the exact wording in `reported_text`

If you only have a figure-derived value, say so clearly in `notes`.

## Why These Templates Exist

They solve three problems:

- they keep manual work separate from the canonical outputs
- they preserve provenance for hard-to-extract rows
- they let the pipeline stay reproducible instead of silently absorbing guesses

## Required General Standards

For any manual intake file:

- one row per metric for summary tables
- one row per current step for f-I point tables
- one row per protocol variant for protocol templates
- keep subtype and condition separated
- do not pool different cell classes or protocols in one row

## Files In This Directory

### EPL-FSI

- [epl_fsi/huang2013_crh_epl_intrinsic_summary.template.csv](/home/michael/OlfactoryBulb/research_context/manual_reference_templates/epl_fsi/huang2013_crh_epl_intrinsic_summary.template.csv)
- [epl_fsi/huang2013_crh_epl_fi_points.template.csv](/home/michael/OlfactoryBulb/research_context/manual_reference_templates/epl_fsi/huang2013_crh_epl_fi_points.template.csv)
- [epl_fsi/kato2013_pv_epl_fi_points.template.csv](/home/michael/OlfactoryBulb/research_context/manual_reference_templates/epl_fsi/kato2013_pv_epl_fi_points.template.csv)
- [epl_fsi/liu2019_epl_network_constraints.template.csv](/home/michael/OlfactoryBulb/research_context/manual_reference_templates/epl_fsi/liu2019_epl_network_constraints.template.csv)

### Granule cells

- [granule_cells/burton_urban_2015_gc_protocol_details.template.csv](/home/michael/OlfactoryBulb/research_context/manual_reference_templates/granule_cells/burton_urban_2015_gc_protocol_details.template.csv)
- [granule_cells/burton_urban_2015_gc_fi_points.template.csv](/home/michael/OlfactoryBulb/research_context/manual_reference_templates/granule_cells/burton_urban_2015_gc_fi_points.template.csv)
- [granule_cells/geramita2016_sgc_dgc_fi_points.template.csv](/home/michael/OlfactoryBulb/research_context/manual_reference_templates/granule_cells/geramita2016_sgc_dgc_fi_points.template.csv)
- [granule_cells/geramita2016_subtype_spontaneous_synaptic_summary.template.csv](/home/michael/OlfactoryBulb/research_context/manual_reference_templates/granule_cells/geramita2016_subtype_spontaneous_synaptic_summary.template.csv)
- [granule_cells/stroh2012_gc_trpc_nmda_summary.template.csv](/home/michael/OlfactoryBulb/research_context/manual_reference_templates/granule_cells/stroh2012_gc_trpc_nmda_summary.template.csv)
- [granule_cells/balu2007_gc_excitation_mode_summary.template.csv](/home/michael/OlfactoryBulb/research_context/manual_reference_templates/granule_cells/balu2007_gc_excitation_mode_summary.template.csv)
- [granule_cells/pressler2007_gc_muscarinic_summary.template.csv](/home/michael/OlfactoryBulb/research_context/manual_reference_templates/granule_cells/pressler2007_gc_muscarinic_summary.template.csv)
- [granule_cells/dong_heinbockel_2007_gc_mglur_summary.template.csv](/home/michael/OlfactoryBulb/research_context/manual_reference_templates/granule_cells/dong_heinbockel_2007_gc_mglur_summary.template.csv)

## Template-Specific Instructions

### `epl_fsi/huang2013_crh_epl_intrinsic_summary.template.csv`

Purpose:
- fill the missing CRH+ EPL-IN intrinsic summary values from Huang 2013 Figure 4A-D

Target deliverable:
- one row per metric
- metrics of interest:
  - Input Resistance
  - Capacitance
  - Membrane Resting Voltage
  - AP Threshold

### `epl_fsi/huang2013_crh_epl_fi_points.template.csv`

Purpose:
- fill real current-vs-rate points for Huang 2013 Figure 4F

Target deliverable:
- one row per current point
- do not fill this from max-rate summary only

### `epl_fsi/kato2013_pv_epl_fi_points.template.csv`

Purpose:
- fill PV+ EPL-IN current-rate points and protocol details from Kato 2013 Figure 1D if recoverable

Target deliverable:
- one row per current point
- if only step increment or max rate is recoverable, leave this unresolved

### `epl_fsi/liu2019_epl_network_constraints.template.csv`

Purpose:
- capture only numeric identity/network constraints from Liu 2019 if they are truly extractable

Target deliverable:
- one row per constraint
- low priority unless the paper exposes useful quantitative rows

### `granule_cells/burton_urban_2015_gc_protocol_details.template.csv`

Purpose:
- recover the exact generic GC current-clamp protocol details

Target deliverable:
- usually one row
- highest-value fields:
  - step duration
  - current range
  - increment
  - rate definition

### `granule_cells/burton_urban_2015_gc_fi_points.template.csv`

Purpose:
- capture generic GC current-vs-rate points

Target deliverable:
- one row per current point
- high priority

### `granule_cells/geramita2016_sgc_dgc_fi_points.template.csv`

Purpose:
- capture sGC and dGC current-vs-rate points separately

Target deliverable:
- one row per current point
- keep `sGC` and `dGC` separate
- this is one of the highest-value missing deliverables

### `granule_cells/geramita2016_subtype_spontaneous_synaptic_summary.template.csv`

Purpose:
- capture subtype-specific spontaneous EPSP/EPSC/IPSC summary rows

Target deliverable:
- one row per metric
- keep subtype separated

### `granule_cells/stroh2012_gc_trpc_nmda_summary.template.csv`

Purpose:
- capture numeric slow depolarization / NMDA-TRPC effect sizes

Target deliverable:
- one row per quantitative effect

### `granule_cells/balu2007_gc_excitation_mode_summary.template.csv`

Purpose:
- capture quantitative excitation-mode distributions if needed

Target deliverable:
- one row per mode/condition summary

### `granule_cells/pressler2007_gc_muscarinic_summary.template.csv`

Purpose:
- capture muscarinic modulation effect sizes

Target deliverable:
- one row per metric and condition
- keep control and modulated conditions separate

### `granule_cells/dong_heinbockel_2007_gc_mglur_summary.template.csv`

Purpose:
- capture group-I-mGluR modulation effect sizes

Target deliverable:
- one row per metric and condition
- keep control and modulated conditions separate

## Notes On Provenance

Every row should make it easy to answer:

- which paper?
- which local file?
- which figure/table/sheet?
- what exact wording or numbers did the paper show?

That is why every template has:

- `source`
- `source_file`
- `source_url`
- `source_location`
- `reported_text`

## Minimum Acceptable Fill Quality

If time is limited, a row is still useful when it includes:

- the value itself
- the metric name
- the figure/table location
- the unit
- the exact paper wording in `reported_text`

Anything beyond that is helpful, but those fields are the minimum needed for a
clean second-pass normalization.
