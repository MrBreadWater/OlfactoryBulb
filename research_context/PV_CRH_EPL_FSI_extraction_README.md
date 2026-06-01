# PV/CRH-overlap EPL fast-spiking interneuron reference-data extraction

This directory contains a protocol-aware reference-data scaffold for a PV/CRH-overlap, axonless, external-plexiform-layer fast-spiking interneuron target.

## Source summary

- **Burton, Malyshko & Urban 2024, PLOS Biology**
  - Contributed: explicit EPL-FSI identity constraints and the canonical EPL-FSI current-injection protocol definition.
  - Missing locally: `S8 Data` and `S15 Data`, which are the preferred sources for actual current-rate points and the full intrinsic-property table.
  - Consequence: this extraction does **not** yet contain Burton-2024 current-rate points or the full FSI intrinsic summary table.

- **Huang et al. 2013, Frontiers in Neural Circuits**
  - Contributed: CRH+/PV-overlap identity constraints, axonless morphology constraints, spontaneous firing summary, and maximum current-evoked firing summary.
  - Did **not** contribute: protocol-equivalent current-rate points.

- **Kato et al. 2013, Neuron**
  - Contributed: PV+ EPL interneuron identity constraints, axonless morphology constraints, input resistance, membrane time constant, action-potential half-width, and a maximum high-frequency spiking summary.
  - Did **not** contribute: protocol-equivalent current-rate points.

- **Liu et al. 2019, Nature Communications**
  - No local asset was found in this checkout.
  - No rows were extracted.

## File guide

- `PV_CRH_EPL_FSI_ephys.csv`
  - Legacy-compatible summary table with explicit provenance and protocol tags.
  - Suitable for intrinsic-property validation and summary firing-rate checks.
  - F-I summary rows are tagged but not treated as exact protocol-equivalent targets.

- `PV_CRH_EPL_FSI_fI_curve.csv`
  - Current-vs-firing-rate points only.
  - Intentionally empty in the current local state because no validated point set was recoverable without missing supplements or a committed digitization manifest.

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

- Huang 2013 spontaneous-firing and maximum current-evoked firing summary rows
- Kato 2013 intrinsic-property summary rows
- Huang/Kato/Burton identity and morphology constraints
- Explicit protocol metadata and caveat rendering

### Not yet suitable for exact f-I curve validation

- Burton 2024 EPL-FSI current-rate points
- Burton 2024 firing-irregularity-current points
- Huang 2013 and Kato 2013 current-rate point sets

Until those point sets are added, compare models only to rows with matching `protocol_id`, and treat summary-rate rows as context rather than exact protocol-equivalent f-I targets.
