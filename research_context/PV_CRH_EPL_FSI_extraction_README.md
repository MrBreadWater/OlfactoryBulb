# PV/CRH-overlap EPL fast-spiking interneuron reference-data extraction

This directory contains a protocol-aware reference-data set for a PV/CRH-overlap, axonless, external-plexiform-layer fast-spiking interneuron target.

## Source summary

- **Burton, Malyshko & Urban 2024, PLOS Biology**
  - Contributed: remote-acquired article PDF, S1 Table DOCX, S2 Table DOCX, S8 Data XLS, S15 Data XLS, and S16 Data XLS.
  - f-I contribution: S8-derived example-cell current-rate rows for example fast-spiking interneurons only.
  - intrinsic contribution: S15-derived FSI intrinsic-property summary rows computed directly from the per-cell workbook.
  - identity contribution: S16-derived morphology rows plus article-text axonless and marker constraints.

- **Huang et al. 2013, Frontiers in Neural Circuits**
  - Contributed: CRH+/PV-overlap identity constraints, axonless morphology constraints, spontaneous firing summary, and maximum current-evoked firing summary.
  - Did **not** contribute: protocol-equivalent current-rate points.

- **Kato et al. 2013, Neuron**
  - Contributed: PV+ EPL interneuron identity constraints, axonless morphology constraints, input resistance, membrane time constant, action-potential half-width, and maximum high-frequency spiking summary.
  - Did **not** contribute: protocol-equivalent current-rate points.

- **Liu et al. 2019, Nature Communications**
  - Used only as a future identity/network-only source unless numeric intrinsic or current-rate data are extracted.

## File guide

- `PV_CRH_EPL_FSI_ephys.csv`
  - Legacy-compatible summary table with explicit provenance, stable source URLs, and protocol tags.
  - Includes S15-derived Burton 2024 intrinsic-property rows plus Huang/Kato summary rows.

- `PV_CRH_EPL_FSI_fI_curve.csv`
  - Current-vs-firing-rate points only.
  - S8-derived example-cell current-rate points were extracted for the primary EPL-FSI target population.

- `PV_CRH_EPL_FSI_protocols.csv`
  - One row per stimulation protocol, including the legacy Burton 2014 MC/TC protocol and the Burton/Malyshko/Urban 2024 EPL-FSI protocol.

- `PV_CRH_EPL_FSI_identity.csv`
  - Marker overlap, morphology, axonless constraints, and population-identity rows.

- `validation_notes.csv`
  - Reusable note sidecar for downstream renderers. This is where protocol caveats live.

- `needs_manual_extraction.csv`
  - Structured backlog of source items that still require supplemental import or figure digitization.

## Validation suitability

### Suitable now

- Burton 2024 S15-derived intrinsic-property rows
- Burton 2024 S8-derived example-cell current-rate rows
- Burton 2024 S16-derived morphology rows
- Huang 2013 spontaneous-firing and maximum current-evoked firing summary rows
- Kato 2013 intrinsic-property summary rows
- Huang/Kato/Burton identity and morphology constraints
- Explicit protocol metadata and caveat rendering

### Caveats

- Burton 2024 S8 rows are tagged `sample_scope = example_cell`; they are not population-average firing-rate curves.
- Burton 2014 MC/TC and Burton/Malyshko/Urban 2024 EPL-FSI firing-rate validation remain protocol-non-equivalent. The downstream renderer must keep showing `N_FI_PROTOCOL_DIFFERENCE`.
- Remaining missing required Burton sources after acquisition attempt: none

## Extraction mechanics

- Required Burton files are fetched from stable PLOS source URLs through `tools/download_epl_fsi_reference_sources.py`.
- Redirected storage URLs are followed at download time, but `source_url` fields in the generated CSVs always preserve the stable manifest URL, never the transient signed redirect target.
- Exact f-I current-rate points are taken only from S8 workbook sheets that correspond to fast-spiking example cells. Summary metrics from S15 remain in `PV_CRH_EPL_FSI_ephys.csv` and are **not** back-projected into synthetic current-rate rows.
