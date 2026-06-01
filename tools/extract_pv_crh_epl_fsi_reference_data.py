"""Generate PV/CRH-overlap EPL fast-spiking interneuron reference-data assets."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from olfactorybulb.audit.reference_data import (
    BMU2024_EPL_FSI_PROTOCOL_ID,
    BU2014_MC_TC_PROTOCOL_ID,
    FI_PROTOCOL_DIFFERENCE_NOTE_ID,
    HUANG2013_CRH_EPL_PROTOCOL_ID,
    KATO2013_PV_EPL_PROTOCOL_ID,
    NEEDS_MANUAL_EXTRACTION_COLUMNS,
    NEEDS_MANUAL_EXTRACTION_FILENAME,
    PV_CRH_EPL_FSI_EPHYS_COLUMNS,
    PV_CRH_EPL_FSI_EPHYS_FILENAME,
    PV_CRH_EPL_FSI_EXTRACTION_README_FILENAME,
    PV_CRH_EPL_FSI_FI_CURVE_COLUMNS,
    PV_CRH_EPL_FSI_FI_CURVE_FILENAME,
    PV_CRH_EPL_FSI_IDENTITY_COLUMNS,
    PV_CRH_EPL_FSI_IDENTITY_FILENAME,
    PV_CRH_EPL_FSI_PROTOCOLS_COLUMNS,
    PV_CRH_EPL_FSI_PROTOCOLS_FILENAME,
    VALIDATION_NOTES_COLUMNS,
    VALIDATION_NOTES_FILENAME,
)


REFERENCE_DATA_DIR = REPO_ROOT / "research_context"


def _bool_csv(value: bool) -> str:
    return "true" if value else "false"


def _format_mean_plus_minus(mean: float | None, spread: float | None) -> str:
    if mean is None or spread is None:
        return ""
    return f"{mean:g} +/- {spread:g}"


def _sd_from_sem(sem: float, n: int) -> float:
    return float(sem) * math.sqrt(float(n))


def _ephys_row(
    *,
    property_name: str,
    source: str,
    notes: str,
    cell_type: str,
    marker_profile: str,
    protocol_id: str = "",
    mean: float | None = None,
    sd: float | None = None,
    sem: float | None = None,
    n: int | None = None,
    stat_type: str = "",
    unit: str = "",
    source_file: str,
    source_location: str,
    data_kind: str,
    extraction_method: str,
    include_in_validation: bool,
    include_in_fi_validation: bool,
    confidence: str,
    note_ids: str = "",
    reported_value_raw: str = "",
) -> dict[str, object]:
    return {
        "Property": property_name,
        "mean +/- sd": _format_mean_plus_minus(mean, sd if sd is not None else sem),
        "n": n if n is not None else "",
        "Source": source,
        "Notes": notes,
        "cell_type": cell_type,
        "marker_profile": marker_profile,
        "protocol_id": protocol_id,
        "mean": mean if mean is not None else "",
        "sd": sd if sd is not None else "",
        "sem": sem if sem is not None else "",
        "stat_type": stat_type,
        "unit": unit,
        "source_file": source_file,
        "source_location": source_location,
        "data_kind": data_kind,
        "extraction_method": extraction_method,
        "include_in_validation": _bool_csv(include_in_validation),
        "include_in_fi_validation": _bool_csv(include_in_fi_validation),
        "confidence": confidence,
        "note_ids": note_ids,
        "reported_value_raw": reported_value_raw,
    }


def _identity_row(
    *,
    source: str,
    source_file: str,
    source_location: str,
    cell_type: str,
    marker_profile: str,
    identity_kind: str,
    property_name: str,
    mean: float | None = None,
    sd: float | None = None,
    sem: float | None = None,
    stat_type: str = "",
    unit: str = "",
    n: int | None = None,
    data_kind: str,
    extraction_method: str,
    include_in_validation: bool,
    confidence: str,
    note_ids: str = "",
    notes: str = "",
    reported_value_raw: str = "",
) -> dict[str, object]:
    return {
        "source": source,
        "source_file": source_file,
        "source_location": source_location,
        "cell_type": cell_type,
        "marker_profile": marker_profile,
        "identity_kind": identity_kind,
        "Property": property_name,
        "mean": mean if mean is not None else "",
        "sd": sd if sd is not None else "",
        "sem": sem if sem is not None else "",
        "stat_type": stat_type,
        "unit": unit,
        "n": n if n is not None else "",
        "data_kind": data_kind,
        "extraction_method": extraction_method,
        "include_in_validation": _bool_csv(include_in_validation),
        "confidence": confidence,
        "note_ids": note_ids,
        "notes": notes,
        "reported_value_raw": reported_value_raw,
    }


def _protocol_row(
    *,
    protocol_id: str,
    source: str,
    cell_type: str,
    marker_profile: str,
    stimulus_type: str,
    step_duration_ms: float | None,
    current_start_pA: float | None,
    current_stop_pA: float | None,
    current_step_pA: float | None,
    current_values_pA: str,
    rate_definition: str,
    spike_detection_rule: str,
    baseline_or_holding_vm_mV: float | str | None,
    synaptic_blockers: str,
    temperature_C: float | str | None,
    compatible_group: str,
    notes: str,
) -> dict[str, object]:
    return {
        "protocol_id": protocol_id,
        "source": source,
        "cell_type": cell_type,
        "marker_profile": marker_profile,
        "stimulus_type": stimulus_type,
        "step_duration_ms": step_duration_ms if step_duration_ms is not None else "",
        "current_start_pA": current_start_pA if current_start_pA is not None else "",
        "current_stop_pA": current_stop_pA if current_stop_pA is not None else "",
        "current_step_pA": current_step_pA if current_step_pA is not None else "",
        "current_values_pA": current_values_pA,
        "rate_definition": rate_definition,
        "spike_detection_rule": spike_detection_rule,
        "baseline_or_holding_vm_mV": baseline_or_holding_vm_mV if baseline_or_holding_vm_mV is not None else "",
        "synaptic_blockers": synaptic_blockers,
        "temperature_C": temperature_C if temperature_C is not None else "",
        "compatible_group": compatible_group,
        "notes": notes,
    }


def _note_row(
    *,
    note_id: str,
    severity: str,
    scope: str,
    target_type: str,
    target: str,
    message: str,
    display_order: int,
    source: str,
    source_location: str,
) -> dict[str, object]:
    return {
        "note_id": note_id,
        "severity": severity,
        "scope": scope,
        "target_type": target_type,
        "target": target,
        "message": message,
        "display_order": display_order,
        "source": source,
        "source_location": source_location,
    }


def _manual_row(
    *,
    source: str,
    source_file: str,
    figure_or_table: str,
    target_metric: str,
    reason: str,
    suggested_action: str,
) -> dict[str, str]:
    return {
        "source": source,
        "source_file": source_file,
        "figure_or_table": figure_or_table,
        "target_metric": target_metric,
        "reason": reason,
        "suggested_action": suggested_action,
    }


def build_ephys_rows() -> list[dict[str, object]]:
    huang_spont_sd = _sd_from_sem(0.09, 14)
    huang_max_rate_sd = _sd_from_sem(6.14, 10)
    kato_input_resistance_sd = _sd_from_sem(5.6, 28)
    kato_tau_sd = _sd_from_sem(0.4, 28)
    kato_half_width_sd = _sd_from_sem(0.027, 28)
    kato_max_rate_sd = _sd_from_sem(12.0, 14)

    return [
        _ephys_row(
            property_name="Spontaneous Firing Rate",
            source="Huang et al. (2013)",
            notes="CRH+ EPL interneurons; baseline firing rate reported directly in text near Figure 4E.",
            cell_type="CRH+ EPL-IN",
            marker_profile="CRH+; PV_overlap_population",
            mean=0.20,
            sd=huang_spont_sd,
            sem=0.09,
            n=14,
            stat_type="sd_from_sem",
            unit="Hz",
            source_file="Huang EPLI.pdf",
            source_location="Figure 4E; Results text describing baseline firing rates",
            data_kind="intrinsic_property",
            extraction_method="pdf_text",
            include_in_validation=True,
            include_in_fi_validation=False,
            confidence="high",
            reported_value_raw="0.20 +/- 0.09 Hz (n = 14, reported as mean +/- SEM)",
        ),
        _ephys_row(
            property_name="Max FI Rate",
            source="Huang et al. (2013)",
            notes=(
                "Maximum current-evoked firing rate only. The local PDF text does not expose the full current-step series, "
                "so this row is not used as a protocol-equivalent f-I target."
            ),
            cell_type="CRH+ EPL-IN",
            marker_profile="CRH+; PV_overlap_population",
            protocol_id=HUANG2013_CRH_EPL_PROTOCOL_ID,
            mean=77.20,
            sd=huang_max_rate_sd,
            sem=6.14,
            n=10,
            stat_type="sd_from_sem",
            unit="Hz",
            source_file="Huang EPLI.pdf",
            source_location="Figure 4F; Results text describing current-evoked firing",
            data_kind="fI_summary_metric",
            extraction_method="pdf_text",
            include_in_validation=True,
            include_in_fi_validation=False,
            confidence="high",
            note_ids=FI_PROTOCOL_DIFFERENCE_NOTE_ID,
            reported_value_raw="77.20 +/- 6.14 Hz (n = 10, reported as mean +/- SEM)",
        ),
        _ephys_row(
            property_name="Input Resistance",
            source="Kato et al. (2013)",
            notes="PV+ EPL interneurons; electrophysiology summary from text near Figure 1D.",
            cell_type="PV+ EPL-IN",
            marker_profile="PV+; CRH_unknown",
            mean=90.5,
            sd=kato_input_resistance_sd,
            sem=5.6,
            n=28,
            stat_type="sd_from_sem",
            unit="MOhm",
            source_file="kato2013 EPLI.pdf",
            source_location="Results text describing electrophysiological properties near Figure 1D",
            data_kind="intrinsic_property",
            extraction_method="pdf_text",
            include_in_validation=True,
            include_in_fi_validation=False,
            confidence="high",
            reported_value_raw="90.5 +/- 5.6 MOhm (n = 28, reported as mean +/- SEM)",
        ),
        _ephys_row(
            property_name="Membrane Time Constant",
            source="Kato et al. (2013)",
            notes="PV+ EPL interneurons; electrophysiology summary from text near Figure 1D.",
            cell_type="PV+ EPL-IN",
            marker_profile="PV+; CRH_unknown",
            mean=5.9,
            sd=kato_tau_sd,
            sem=0.4,
            n=28,
            stat_type="sd_from_sem",
            unit="ms",
            source_file="kato2013 EPLI.pdf",
            source_location="Results text describing electrophysiological properties near Figure 1D",
            data_kind="intrinsic_property",
            extraction_method="pdf_text",
            include_in_validation=True,
            include_in_fi_validation=False,
            confidence="high",
            reported_value_raw="5.9 +/- 0.4 ms (n = 28, reported as mean +/- SEM)",
        ),
        _ephys_row(
            property_name="AP Half-Width",
            source="Kato et al. (2013)",
            notes=(
                "Converted from the paper's microsecond-scale half-width summary to milliseconds. "
                "The local pdftotext stream drops the micro sign, so reported_value_raw preserves the intended source unit."
            ),
            cell_type="PV+ EPL-IN",
            marker_profile="PV+; CRH_unknown",
            mean=0.530,
            sd=kato_half_width_sd,
            sem=0.027,
            n=28,
            stat_type="sd_from_sem",
            unit="ms",
            source_file="kato2013 EPLI.pdf",
            source_location="Figure 1D; Results text describing fast action potentials",
            data_kind="intrinsic_property",
            extraction_method="pdf_text",
            include_in_validation=True,
            include_in_fi_validation=False,
            confidence="high",
            reported_value_raw="530 +/- 27 μs (n = 28, reported as mean +/- SEM)",
        ),
        _ephys_row(
            property_name="Max FI Rate",
            source="Kato et al. (2013)",
            notes=(
                "Peak high-frequency spiking summary only. The local PDF exposes the 100 pA step increment but not a recoverable current-rate point table, "
                "so this row remains outside exact f-I validation."
            ),
            cell_type="PV+ EPL-IN",
            marker_profile="PV+; CRH_unknown",
            protocol_id=KATO2013_PV_EPL_PROTOCOL_ID,
            mean=171.0,
            sd=kato_max_rate_sd,
            sem=12.0,
            n=14,
            stat_type="sd_from_sem",
            unit="Hz",
            source_file="kato2013 EPLI.pdf",
            source_location="Figure 1D; Results text describing high-frequency spikes",
            data_kind="fI_summary_metric",
            extraction_method="pdf_text",
            include_in_validation=True,
            include_in_fi_validation=False,
            confidence="high",
            note_ids=FI_PROTOCOL_DIFFERENCE_NOTE_ID,
            reported_value_raw="171 +/- 12 Hz (n = 14, reported as mean +/- SEM)",
        ),
    ]


def build_fi_curve_rows() -> list[dict[str, object]]:
    return []


def build_protocol_rows() -> list[dict[str, object]]:
    return [
        _protocol_row(
            protocol_id=BU2014_MC_TC_PROTOCOL_ID,
            source="Burton & Urban (2014)",
            cell_type="MC;TC",
            marker_profile="principal_cell",
            stimulus_type="somatic depolarizing current step",
            step_duration_ms=2000.0,
            current_start_pA=0.0,
            current_stop_pA=300.0,
            current_step_pA=50.0,
            current_values_pA="0;50;100;150;200;250;300",
            rate_definition="Legacy MC/TC summary metrics derived from a two-second firing-rate-versus-current protocol.",
            spike_detection_rule="Not explicitly preserved in the legacy CSVs; see Burton & Urban 2014 methods.",
            baseline_or_holding_vm_mV=-58.0,
            synaptic_blockers="",
            temperature_C="",
            compatible_group="MC_TC_fI",
            notes="Legacy mitral-cell / tufted-cell protocol retained so downstream renderers can detect non-equivalence to EPL-FSI protocols.",
        ),
        _protocol_row(
            protocol_id=BMU2024_EPL_FSI_PROTOCOL_ID,
            source="Burton, Malyshko & Urban (2024)",
            cell_type="EPL-FSI",
            marker_profile="PV+; CRH_not_assayed_or_not_reported",
            stimulus_type="somatic depolarizing current step",
            step_duration_ms=500.0,
            current_start_pA=50.0,
            current_stop_pA=600.0,
            current_step_pA=50.0,
            current_values_pA="50;100;150;200;250;300;350;400;450;500;550;600",
            rate_definition="Median inverse interspike interval evoked by each step current.",
            spike_detection_rule="Voltage derivative threshold of 15 mV/ms.",
            baseline_or_holding_vm_mV="held at resting membrane potential (0 pA holding current)",
            synaptic_blockers="",
            temperature_C="",
            compatible_group="EPL_FSI_fI",
            notes="Protocol recovered from the local PDF methods text. Numeric current-rate points require S8 Data or a digitization manifest and are not present locally.",
        ),
        _protocol_row(
            protocol_id=HUANG2013_CRH_EPL_PROTOCOL_ID,
            source="Huang et al. (2013)",
            cell_type="CRH+ EPL-IN",
            marker_profile="CRH+; PV_overlap_population",
            stimulus_type="somatic current injection",
            step_duration_ms=None,
            current_start_pA=None,
            current_stop_pA=None,
            current_step_pA=None,
            current_values_pA="",
            rate_definition="Maximum current-evoked firing rate only; full current-rate series not recoverable from local PDF text.",
            spike_detection_rule="",
            baseline_or_holding_vm_mV="",
            synaptic_blockers="",
            temperature_C="",
            compatible_group="EPL_identity_control",
            notes="Use only as a tagged CRH+ current-clamp summary unless figure-digitized current-rate points are added with provenance.",
        ),
        _protocol_row(
            protocol_id=KATO2013_PV_EPL_PROTOCOL_ID,
            source="Kato et al. (2013)",
            cell_type="PV+ EPL-IN",
            marker_profile="PV+; CRH_unknown",
            stimulus_type="somatic current clamp",
            step_duration_ms=None,
            current_start_pA=None,
            current_stop_pA=None,
            current_step_pA=100.0,
            current_values_pA="",
            rate_definition="High-frequency spike summary only; no extracted current-rate point table.",
            spike_detection_rule="",
            baseline_or_holding_vm_mV="",
            synaptic_blockers="",
            temperature_C="",
            compatible_group="PV_EPL_identity_control",
            notes="The local PDF text confirms 100 pA current-step increments but does not expose a recoverable current-rate point table.",
        ),
    ]


def build_identity_rows() -> list[dict[str, object]]:
    huang_pv_overlap_sd = _sd_from_sem(3.0, 3)
    huang_sst_overlap_sd = _sd_from_sem(0.8, 3)
    huang_crh_fraction_sd = _sd_from_sem(1.31, 3)
    huang_calretinin_fraction_sd = _sd_from_sem(2.70, 3)
    huang_neun_sd = _sd_from_sem(1.7, 3)

    return [
        _identity_row(
            source="Burton, Malyshko & Urban (2024)",
            source_file="burton and urban 2024 (EPLI).pdf",
            source_location="Results text describing FSI neurochemical profile and lack of visible axons",
            cell_type="EPL-FSI",
            marker_profile="PV+; CRH_not_assayed_or_not_reported",
            identity_kind="marker_overlap",
            property_name="PV Positive Fraction",
            mean=100.0,
            stat_type="qualitative",
            unit="%",
            data_kind="marker_overlap",
            extraction_method="pdf_text",
            include_in_validation=True,
            confidence="medium",
            notes="All fast-spiking interneurons were reported as PV-positive.",
            reported_value_raw="100% of FSIs expressed PV",
        ),
        _identity_row(
            source="Burton, Malyshko & Urban (2024)",
            source_file="burton and urban 2024 (EPLI).pdf",
            source_location="Results text describing FSI neurochemical profile",
            cell_type="EPL-FSI",
            marker_profile="PV+; CRH_not_assayed_or_not_reported",
            identity_kind="marker_overlap",
            property_name="TH Overlap Fraction",
            mean=0.0,
            stat_type="qualitative",
            unit="%",
            data_kind="marker_overlap",
            extraction_method="pdf_text",
            include_in_validation=True,
            confidence="medium",
            notes="Neither FSIs nor RSIs expressed tyrosine hydroxylase.",
            reported_value_raw="Neither FSIs nor RSIs expressed TH",
        ),
        _identity_row(
            source="Burton, Malyshko & Urban (2024)",
            source_file="burton and urban 2024 (EPLI).pdf",
            source_location="Results text describing weak VIP expression in a subset of FSIs",
            cell_type="EPL-FSI",
            marker_profile="PV+; CRH_not_assayed_or_not_reported",
            identity_kind="marker_overlap",
            property_name="VIP Positive Fraction",
            mean=27.0,
            stat_type="qualitative",
            unit="%",
            data_kind="marker_overlap",
            extraction_method="pdf_text",
            include_in_validation=True,
            confidence="medium",
            notes="Weak VIP expression was reported in a minority of FSIs.",
            reported_value_raw="27% of FSIs expressed weak VIP",
        ),
        _identity_row(
            source="Burton, Malyshko & Urban (2024)",
            source_file="burton and urban 2024 (EPLI).pdf",
            source_location="Results text describing FSI morphology",
            cell_type="EPL-FSI",
            marker_profile="PV+; CRH_not_assayed_or_not_reported",
            identity_kind="morphology_constraint",
            property_name="Axonless Morphology",
            stat_type="qualitative",
            data_kind="morphology_constraint",
            extraction_method="pdf_text",
            include_in_validation=True,
            confidence="high",
            notes="The paper states that FSIs did not extend visible axons.",
            reported_value_raw="Neither FSIs nor RSIs extended visible axons",
        ),
        _identity_row(
            source="Burton, Malyshko & Urban (2024)",
            source_file="burton and urban 2024 (EPLI).pdf",
            source_location="Discussion text describing compact FSI dendrites",
            cell_type="EPL-FSI",
            marker_profile="PV+; CRH_not_assayed_or_not_reported",
            identity_kind="morphology_constraint",
            property_name="Planar Span",
            mean=100.0,
            stat_type="qualitative",
            unit="um",
            data_kind="morphology_constraint",
            extraction_method="pdf_text",
            include_in_validation=True,
            confidence="medium",
            notes="Compact FSI dendrites were described as extending approximately 100 um radially from the soma.",
            reported_value_raw="Compact FSI dendrites radially extended approximately 100 um from the soma",
        ),
        _identity_row(
            source="Huang et al. (2013)",
            source_file="Huang EPLI.pdf",
            source_location="Figure 2J; CRH-Cre overlap analysis in the EPL",
            cell_type="CRH+ EPL-IN",
            marker_profile="CRH+; PV_overlap_population",
            identity_kind="population_fraction",
            property_name="CRH Positive Fraction Within Calretinin EPL Interneurons",
            mean=25.69,
            sd=huang_crh_fraction_sd,
            sem=1.31,
            stat_type="sd_from_sem",
            unit="%",
            n=3,
            data_kind="marker_overlap",
            extraction_method="pdf_text",
            include_in_validation=True,
            confidence="high",
            notes="Fraction of calretinin-positive EPL interneurons labeled by CRH-Cre.",
            reported_value_raw="25.69 +/- 1.31% (N = 3, reported as mean +/- SEM)",
        ),
        _identity_row(
            source="Huang et al. (2013)",
            source_file="Huang EPLI.pdf",
            source_location="Figure 2J; calretinin-positive EPL fraction text",
            cell_type="CRH+ EPL-IN",
            marker_profile="CRH+; PV_overlap_population",
            identity_kind="population_fraction",
            property_name="Calretinin Positive Fraction Within EPL Cells",
            mean=68.75,
            sd=huang_calretinin_fraction_sd,
            sem=2.70,
            stat_type="sd_from_sem",
            unit="%",
            n=3,
            data_kind="marker_overlap",
            extraction_method="pdf_text",
            include_in_validation=True,
            confidence="high",
            notes="Calretinin-positive interneurons as a fraction of all EPL cells.",
            reported_value_raw="68.75 +/- 2.70% (N = 3, reported as mean +/- SEM)",
        ),
        _identity_row(
            source="Huang et al. (2013)",
            source_file="Huang EPLI.pdf",
            source_location="Figure 2J; marker overlap text",
            cell_type="CRH+ EPL-IN",
            marker_profile="CRH+; PV_overlap_population",
            identity_kind="marker_overlap",
            property_name="PV Overlap Fraction",
            mean=81.5,
            sd=huang_pv_overlap_sd,
            sem=3.0,
            stat_type="sd_from_sem",
            unit="%",
            n=3,
            data_kind="marker_overlap",
            extraction_method="pdf_text",
            include_in_validation=True,
            confidence="high",
            notes="Parvalbumin overlap within the CRH-positive EPL interneuron population.",
            reported_value_raw="81.5 +/- 3% (N = 3, reported as mean +/- SEM)",
        ),
        _identity_row(
            source="Huang et al. (2013)",
            source_file="Huang EPLI.pdf",
            source_location="Figure 2J; marker overlap text",
            cell_type="CRH+ EPL-IN",
            marker_profile="CRH+; PV_overlap_population",
            identity_kind="marker_overlap",
            property_name="SST Overlap Fraction",
            mean=24.8,
            sd=huang_sst_overlap_sd,
            sem=0.8,
            stat_type="sd_from_sem",
            unit="%",
            n=3,
            data_kind="marker_overlap",
            extraction_method="pdf_text",
            include_in_validation=True,
            confidence="high",
            notes="Somatostatin overlap within the CRH-positive EPL interneuron population.",
            reported_value_raw="24.8 +/- 0.8% (N = 3, reported as mean +/- SEM)",
        ),
        _identity_row(
            source="Huang et al. (2013)",
            source_file="Huang EPLI.pdf",
            source_location="Figure 2J; marker overlap text",
            cell_type="CRH+ EPL-IN",
            marker_profile="CRH+; PV_overlap_population",
            identity_kind="marker_overlap",
            property_name="TH Overlap Fraction",
            mean=0.0,
            stat_type="qualitative",
            unit="%",
            data_kind="marker_overlap",
            extraction_method="pdf_text",
            include_in_validation=True,
            confidence="medium",
            notes="The source states that tyrosine hydroxylase labeling did not overlap with CRH-positive EPL interneurons.",
            reported_value_raw="TH did not overlap with CRH+ EPL interneurons",
        ),
        _identity_row(
            source="Huang et al. (2013)",
            source_file="Huang EPLI.pdf",
            source_location="Figure 2I / Figure 2J; NeuN labeling text",
            cell_type="CRH+ EPL-IN",
            marker_profile="CRH+; PV_overlap_population",
            identity_kind="marker_overlap",
            property_name="NeuN Low Or Sparse Fraction",
            mean=44.2,
            sd=huang_neun_sd,
            sem=1.7,
            stat_type="sd_from_sem",
            unit="%",
            n=3,
            data_kind="marker_overlap",
            extraction_method="pdf_text",
            include_in_validation=True,
            confidence="high",
            notes="Low or sparse NeuN labeling fraction for CRH-positive EPL interneurons.",
            reported_value_raw="44.2 +/- 1.7% (N = 3, reported as mean +/- SEM)",
        ),
        _identity_row(
            source="Huang et al. (2013)",
            source_file="Huang EPLI.pdf",
            source_location="Figure 3 morphology text",
            cell_type="CRH+ EPL-IN",
            marker_profile="CRH+; PV_overlap_population",
            identity_kind="morphology_constraint",
            property_name="Primary Process Count",
            mean=3.5,
            sem=0.4,
            stat_type="mean_sem",
            unit="count",
            data_kind="morphology_constraint",
            extraction_method="pdf_text",
            include_in_validation=True,
            confidence="high",
            notes="Reported directly in the figure text, but the local PDF text snippet does not expose the sample count needed to convert SEM to SD.",
            reported_value_raw="3.5 +/- 0.4 primary processes",
        ),
        _identity_row(
            source="Huang et al. (2013)",
            source_file="Huang EPLI.pdf",
            source_location="Figure 3 morphology text",
            cell_type="CRH+ EPL-IN",
            marker_profile="CRH+; PV_overlap_population",
            identity_kind="morphology_constraint",
            property_name="Soma Diameter",
            mean=9.6,
            sem=0.7,
            stat_type="mean_sem",
            unit="um",
            data_kind="morphology_constraint",
            extraction_method="pdf_text",
            include_in_validation=True,
            confidence="high",
            notes="Reported directly in the figure text, but the local PDF text snippet does not expose the sample count needed to convert SEM to SD.",
            reported_value_raw="9.6 +/- 0.7 um",
        ),
        _identity_row(
            source="Huang et al. (2013)",
            source_file="Huang EPLI.pdf",
            source_location="Figure 3 morphology text",
            cell_type="CRH+ EPL-IN",
            marker_profile="CRH+; PV_overlap_population",
            identity_kind="morphology_constraint",
            property_name="Planar Span",
            mean=71.0,
            sem=4.5,
            stat_type="mean_sem",
            unit="um",
            data_kind="morphology_constraint",
            extraction_method="pdf_text",
            include_in_validation=True,
            confidence="high",
            notes="Neurites were reported to span up to 71 +/- 4.5 um from the soma.",
            reported_value_raw="71 +/- 4.5 um",
        ),
        _identity_row(
            source="Huang et al. (2013)",
            source_file="Huang EPLI.pdf",
            source_location="Figure 3 morphology text",
            cell_type="CRH+ EPL-IN",
            marker_profile="CRH+; PV_overlap_population",
            identity_kind="morphology_constraint",
            property_name="Branching Zone Maximum",
            mean=30.0,
            stat_type="qualitative",
            unit="um",
            data_kind="morphology_constraint",
            extraction_method="pdf_text",
            include_in_validation=True,
            confidence="medium",
            notes="Highest branching was reported within 30 um of the soma rather than as a mean +/- spread statistic.",
            reported_value_raw="highest branching occurred within 30 um from the cell body",
        ),
        _identity_row(
            source="Huang et al. (2013)",
            source_file="Huang EPLI.pdf",
            source_location="Figure 3 axon-initial-segment text",
            cell_type="CRH+ EPL-IN",
            marker_profile="CRH+; PV_overlap_population",
            identity_kind="morphology_constraint",
            property_name="Axonless Morphology",
            stat_type="qualitative",
            data_kind="morphology_constraint",
            extraction_method="pdf_text",
            include_in_validation=True,
            confidence="high",
            notes="The source reports no obvious axon initial segment or betaIV-spectrin-defined axon.",
            reported_value_raw="CRH+ EPL interneurons were axonless",
        ),
        _identity_row(
            source="Kato et al. (2013)",
            source_file="kato2013 EPLI.pdf",
            source_location="Results text describing PV-cell laminar distribution",
            cell_type="PV+ EPL-IN",
            marker_profile="PV+; CRH_unknown",
            identity_kind="population_fraction",
            property_name="PV Cell Fraction In EPL",
            mean=91.4,
            stat_type="qualitative",
            unit="%",
            n=1883,
            data_kind="marker_overlap",
            extraction_method="pdf_text",
            include_in_validation=True,
            confidence="high",
            notes="The numerator and denominator are preserved in the raw value text; the mouse count remains in the notes.",
            reported_value_raw="91.4% (1722/1883 cells, n = 5 mice)",
        ),
        _identity_row(
            source="Kato et al. (2013)",
            source_file="kato2013 EPLI.pdf",
            source_location="Figure 1C; anatomical reconstruction text",
            cell_type="PV+ EPL-IN",
            marker_profile="PV+; CRH_unknown",
            identity_kind="morphology_constraint",
            property_name="Axonless Morphology",
            stat_type="qualitative",
            data_kind="morphology_constraint",
            extraction_method="pdf_text",
            include_in_validation=True,
            confidence="high",
            notes="All anatomically reconstructed PV cells were reported to lack an obvious axon.",
            reported_value_raw="All reconstructed PV cells lacked an obvious axon",
        ),
        _identity_row(
            source="Kato et al. (2013)",
            source_file="kato2013 EPLI.pdf",
            source_location="Figure 1C; anatomical reconstruction text",
            cell_type="PV+ EPL-IN",
            marker_profile="PV+; CRH_unknown",
            identity_kind="morphology_constraint",
            property_name="Multipolar Dendrites In EPL",
            stat_type="qualitative",
            data_kind="morphology_constraint",
            extraction_method="pdf_text",
            include_in_validation=True,
            confidence="high",
            notes="The reconstructed PV cells had multipolar dendrites localized within the external plexiform layer.",
            reported_value_raw="PV cells had multipolar dendrites localized within the EPL",
        ),
    ]


def build_validation_notes_rows() -> list[dict[str, object]]:
    return [
        _note_row(
            note_id=FI_PROTOCOL_DIFFERENCE_NOTE_ID,
            severity="warning",
            scope="fI_validation",
            target_type="protocol",
            target=f"{BU2014_MC_TC_PROTOCOL_ID};{BMU2024_EPL_FSI_PROTOCOL_ID}",
            message=(
                "MC/TC and EPL-FSI f-I validation targets use different current-injection protocols. Burton & Urban 2014 MC/TC values use 2 s depolarizing steps "
                "from 0 to 300 pA in 50 pA increments. Burton, Malyshko & Urban 2024 EPL-FSI values use 500 ms depolarizing steps from 50 to 600 pA in 50 pA increments, "
                "with firing rate computed from median inverse ISI. Compare model outputs only against rows with matching protocol_id unless an explicit normalization/reanalysis step is implemented."
            ),
            display_order=10,
            source="Burton & Urban (2014); Burton, Malyshko & Urban (2024)",
            source_location="Protocol metadata rows",
        ),
        _note_row(
            note_id="N_BMU2024_SUPPLEMENT_MISSING",
            severity="warning",
            scope="extraction",
            target_type="source",
            target="Burton, Malyshko & Urban (2024)",
            message="The local checkout contains the main Burton, Malyshko & Urban 2024 PDF but not S8 Data or S15 Data. Current-rate points and full FSI intrinsic property tables remain pending manual source-data import or documented figure digitization.",
            display_order=20,
            source="Burton, Malyshko & Urban (2024)",
            source_location="Supporting-information references to S8 Data and S15 Data in the main PDF",
        ),
        _note_row(
            note_id="N_EPL_FI_POINTS_UNAVAILABLE",
            severity="warning",
            scope="fI_validation",
            target_type="source",
            target=PV_CRH_EPL_FSI_FI_CURVE_FILENAME,
            message="The local source set does not currently provide validated current-rate point tables for the EPL fast-spiking interneuron target population. The f-I curve CSV is intentionally empty until S8 Data is added or a figure-digitization manifest is committed.",
            display_order=30,
            source="Burton, Malyshko & Urban (2024); Huang et al. (2013); Kato et al. (2013)",
            source_location="Local PDFs and missing supplemental-source inventory",
        ),
        _note_row(
            note_id="N_MARKER_PROFILE_SEPARATION",
            severity="warning",
            scope="identity_validation",
            target_type="source",
            target="PV+ EPL-IN;CRH+ EPL-IN;EPL-FSI",
            message="PV+, CRH+, and PV/CRH-overlap-adjacent EPL interneuron rows are intentionally kept separate by marker_profile and protocol_id. Do not pool them unless the source explicitly identifies the same population under a compatible protocol.",
            display_order=40,
            source="Huang et al. (2013); Kato et al. (2013); Burton, Malyshko & Urban (2024)",
            source_location="Cross-source integration rule for this extraction pipeline",
        ),
    ]


def build_manual_extraction_rows() -> list[dict[str, str]]:
    return [
        _manual_row(
            source="Burton, Malyshko & Urban (2024)",
            source_file="burton and urban 2024 (EPLI).pdf",
            figure_or_table="S15 Data / S1 Table",
            target_metric="FSI intrinsic biophysical property summary table",
            reason="Supporting-information spreadsheet is referenced in the PDF but not present in the local checkout.",
            suggested_action="Import S15 Data locally, then regenerate PV_CRH_EPL_FSI_ephys.csv with direct table rows for resting potential, input resistance, capacitance, threshold, rheobase, firing irregularity, and related FSI metrics.",
        ),
        _manual_row(
            source="Burton, Malyshko & Urban (2024)",
            source_file="burton and urban 2024 (EPLI).pdf",
            figure_or_table="S8 Data / S2 Fig",
            target_metric="EPL-FSI firing-rate-current points and firing-irregularity-current points",
            reason="Supporting-information spreadsheet is referenced in the PDF but not present in the local checkout.",
            suggested_action="Import S8 Data locally or add a documented figure-digitization manifest before populating PV_CRH_EPL_FSI_fI_curve.csv.",
        ),
        _manual_row(
            source="Huang et al. (2013)",
            source_file="Huang EPLI.pdf",
            figure_or_table="Figure 4A-D",
            target_metric="CRH+ EPL-IN resistance, capacitance, resting membrane potential, and action-potential threshold numeric values",
            reason="The local PDF text states directionality but does not expose the bar values numerically.",
            suggested_action="Digitize Figure 4A-D with a committed manifest or locate a machine-readable source table before adding these rows to PV_CRH_EPL_FSI_ephys.csv.",
        ),
        _manual_row(
            source="Huang et al. (2013)",
            source_file="Huang EPLI.pdf",
            figure_or_table="Figure 4F",
            target_metric="CRH+ EPL-IN current-rate points",
            reason="The local PDF text exposes only the maximum current-evoked firing rate, not the full current-rate series or exact step protocol values.",
            suggested_action="Digitize Figure 4F with provenance or add a supplemental numeric source before populating PV_CRH_EPL_FSI_fI_curve.csv.",
        ),
        _manual_row(
            source="Kato et al. (2013)",
            source_file="kato2013 EPLI.pdf",
            figure_or_table="Figure 1D",
            target_metric="PV+ EPL-IN current-rate points and exact current range",
            reason="The local PDF text exposes 100 pA step increments and a high-frequency spike summary but not a recoverable current-rate point table.",
            suggested_action="Digitize Figure 1D or locate a supplemental numeric source before treating Kato 2013 as an f-I point-set reference.",
        ),
        _manual_row(
            source="Liu et al. (2019)",
            source_file="not locally available",
            figure_or_table="all",
            target_metric="EPL interneuron identity/network constraints from Liu et al. 2019",
            reason="No local PDF or supplemental asset for Liu et al. 2019 was found in the checkout.",
            suggested_action="Add the local Liu et al. 2019 PDF or supplementary materials, then extract only identity/network constraints unless numeric intrinsic or current-rate data are present.",
        ),
    ]


def build_readme_text() -> str:
    return """# PV/CRH-overlap EPL fast-spiking interneuron reference-data extraction

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
"""


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    REFERENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    _write_csv(REFERENCE_DATA_DIR / PV_CRH_EPL_FSI_EPHYS_FILENAME, PV_CRH_EPL_FSI_EPHYS_COLUMNS, build_ephys_rows())
    _write_csv(REFERENCE_DATA_DIR / PV_CRH_EPL_FSI_FI_CURVE_FILENAME, PV_CRH_EPL_FSI_FI_CURVE_COLUMNS, build_fi_curve_rows())
    _write_csv(REFERENCE_DATA_DIR / PV_CRH_EPL_FSI_PROTOCOLS_FILENAME, PV_CRH_EPL_FSI_PROTOCOLS_COLUMNS, build_protocol_rows())
    _write_csv(REFERENCE_DATA_DIR / PV_CRH_EPL_FSI_IDENTITY_FILENAME, PV_CRH_EPL_FSI_IDENTITY_COLUMNS, build_identity_rows())
    _write_csv(REFERENCE_DATA_DIR / VALIDATION_NOTES_FILENAME, VALIDATION_NOTES_COLUMNS, build_validation_notes_rows())
    _write_csv(REFERENCE_DATA_DIR / NEEDS_MANUAL_EXTRACTION_FILENAME, NEEDS_MANUAL_EXTRACTION_COLUMNS, build_manual_extraction_rows())
    (REFERENCE_DATA_DIR / PV_CRH_EPL_FSI_EXTRACTION_README_FILENAME).write_text(build_readme_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
