# PV/CRH-overlap EPL fast-spiking interneuron reference-data extraction

This directory contains a protocol-aware reference-data set for a PV/CRH-overlap, axonless, external-plexiform-layer fast-spiking interneuron target.

## Source summary

- Burton, Malyshko & Urban 2024 contributes the primary fast-spiking-interneuron intrinsic, firing-rate-current, and morphology sources through S8, S15, and S16 supporting files.
- Huang et al. 2013 contributes CRH/PV-overlap identity constraints, axonless morphology constraints, spontaneous firing summary, and maximum current-evoked firing summary.
- Kato et al. 2013 contributes PV+ external-plexiform-layer interneuron identity constraints and summary intrinsic electrophysiology.
- Liu et al. 2019 remains identity/network-only here unless explicit intrinsic or current-rate numeric data are added.

## Suitable now

- Burton 2024 S15-derived intrinsic-property rows
- Burton 2024 S8-derived example-cell current-rate rows
- Burton 2024 S16-derived morphology rows
- Huang 2013 spontaneous-firing and maximum current-evoked firing summary rows
- Kato 2013 intrinsic-property summary rows
- Protocol metadata and protocol caveat notes

## Caveats

- Burton 2024 S8 rows are tagged sample_scope = example_cell; they are not population-average firing-rate curves.
- Burton 2014 MC/TC and Burton/Malyshko/Urban 2024 EPL-FSI firing-rate validation remain protocol-non-equivalent. N_FI_PROTOCOL_DIFFERENCE must remain visible in combined outputs.
- The pipeline does not perform built-in figure digitization. Figure-only gaps stay in needs_manual_extraction.csv.

## Extraction status

- `ephys` rows: 21
- `fi_curve` rows: 24
- `protocols` rows: 4
- `identity` rows: 22
- `notes` rows: 3
- `manual` rows: 4
- `readme` rows: 0
- Missing required sources after acquisition: none
