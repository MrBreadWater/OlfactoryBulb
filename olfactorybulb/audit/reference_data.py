"""Shared reference-data helpers for electrophysiology validation assets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DATA_DIR = REPO_ROOT / "research_context"

LEGACY_MC_TC_EPHYS_FILENAMES = {
    "MC": "MC_TC_spike_frequency_references - 4_mitral_cell_ephys.csv",
    "TC": "MC_TC_spike_frequency_references - 3_tufted_cell_ephys.csv",
}

PV_CRH_EPL_FSI_EPHYS_FILENAME = "PV_CRH_EPL_FSI_ephys.csv"
PV_CRH_EPL_FSI_FI_CURVE_FILENAME = "PV_CRH_EPL_FSI_fI_curve.csv"
PV_CRH_EPL_FSI_PROTOCOLS_FILENAME = "PV_CRH_EPL_FSI_protocols.csv"
PV_CRH_EPL_FSI_IDENTITY_FILENAME = "PV_CRH_EPL_FSI_identity.csv"
VALIDATION_NOTES_FILENAME = "validation_notes.csv"
PV_CRH_EPL_FSI_EXTRACTION_README_FILENAME = "PV_CRH_EPL_FSI_extraction_README.md"
NEEDS_MANUAL_EXTRACTION_FILENAME = "needs_manual_extraction.csv"

GC_DATASET_ID = "granule_cells"
GC_EPHYS_FILENAME = "GC_ephys.csv"
GC_FI_CURVE_FILENAME = "GC_fI_curve.csv"
GC_SGC_DGC_EPHYS_FILENAME = "GC_sGC_dGC_ephys.csv"
GC_SGC_DGC_FI_CURVE_FILENAME = "GC_sGC_dGC_fI_curve.csv"
GC_PROTOCOLS_FILENAME = "GC_protocols.csv"
GC_IDENTITY_FILENAME = "GC_identity_morphology.csv"
GC_SYNAPTIC_LATENCY_FILENAME = "GC_synaptic_latency_references.csv"
GC_MODULATION_FILENAME = "GC_modulation_references.csv"
GC_VALIDATION_NOTES_FILENAME = "GC_validation_notes.csv"
GC_EXTRACTION_README_FILENAME = "GC_extraction_README.md"
GC_NEEDS_MANUAL_EXTRACTION_FILENAME = "GC_needs_manual_extraction.csv"

BU2014_MC_TC_PROTOCOL_ID = "BU2014_MC_TC_2s_0_300pA_50pA"
BMU2024_EPL_FSI_PROTOCOL_ID = "BMU2024_EPL_FSI_500ms_50_600pA_50pA"
HUANG2013_CRH_EPL_PROTOCOL_ID = "HUANG2013_CRH_EPL_current_injection"
KATO2013_PV_EPL_PROTOCOL_ID = "KATO2013_PV_EPL_current_clamp"

FI_PROTOCOL_DIFFERENCE_NOTE_ID = "N_FI_PROTOCOL_DIFFERENCE"

PROPERTY_ALIASES = {
    "AP Width at Half-height": "AP Half-Width",
    "FWHM": "AP Half-Width",
    "FI gain": "FI Curve Slope",
    "max FI gain": "FI Curve Slope",
    "f-I slope": "FI Curve Slope",
}

PROPERTY_UNITS = {
    "Membrane Resting Voltage": "mV",
    "Input Resistance": "MOhm",
    "Membrane Time Constant": "ms",
    "Capacitance": "pF",
    "Sag Amplitude": "mV",
    "AP Threshold": "mV",
    "AP Amplitude": "mV",
    "AP Half-Width": "ms",
    "AP Rising Slope": "mV/ms",
    "AP Falling Slope": "mV/ms",
    "AHP Amplitude": "mV",
    "AHP Duration": "ms",
    "Rheobase Current": "pA",
    "FI Curve Slope": "Hz/nA",
    "Max FI Rate": "Hz",
    "Spontaneous Firing Rate": "Hz",
    "ISI Coefficient of Variation": "",
    "Spiking Rate Accommodation": "Hz",
    "Resonance Frequency": "Hz",
    "Peak Instantaneous Rate": "Hz",
    "First Spike Latency": "ms",
    "Firing Probability": "",
    "Tonic Inhibitory Current": "pA",
    "Slow Depolarization Amplitude": "mV",
}

FI_RELATED_PROPERTIES = {
    "Rheobase Current",
    "FI Curve Slope",
    "Max FI Rate",
    "ISI Coefficient of Variation",
    "Spiking Rate Accommodation",
}

PV_CRH_EPL_FSI_EPHYS_COLUMNS = [
    "Property",
    "mean +/- sd",
    "n",
    "Source",
    "Notes",
    "cell_type",
    "marker_profile",
    "protocol_id",
    "mean",
    "sd",
    "sem",
    "q_low",
    "q_high",
    "q_low_label",
    "q_high_label",
    "stat_type",
    "unit",
    "source_file",
    "source_location",
    "source_url",
    "data_kind",
    "extraction_method",
    "include_in_validation",
    "include_in_fi_validation",
    "confidence",
    "note_ids",
    "reported_value_raw",
]

PV_CRH_EPL_FSI_FI_CURVE_COLUMNS = [
    "source",
    "source_file",
    "source_location",
    "cell_type",
    "cell_id",
    "marker_profile",
    "protocol_id",
    "source_url",
    "current_pA",
    "firing_rate_Hz",
    "rate_definition",
    "step_duration_ms",
    "current_start_pA",
    "current_stop_pA",
    "current_step_pA",
    "baseline_or_holding_vm_mV",
    "synaptic_blockers",
    "temperature_C",
    "sample_scope",
    "extraction_method",
    "include_in_validation",
    "confidence",
    "note_ids",
    "notes",
]

PV_CRH_EPL_FSI_PROTOCOLS_COLUMNS = [
    "protocol_id",
    "source",
    "cell_type",
    "marker_profile",
    "stimulus_type",
    "step_duration_ms",
    "current_start_pA",
    "current_stop_pA",
    "current_step_pA",
    "current_values_pA",
    "rate_definition",
    "spike_detection_rule",
    "baseline_or_holding_vm_mV",
    "synaptic_blockers",
    "temperature_C",
    "compatible_group",
    "notes",
]

PV_CRH_EPL_FSI_IDENTITY_COLUMNS = [
    "source",
    "source_file",
    "source_location",
    "cell_type",
    "marker_profile",
    "identity_kind",
    "Property",
    "source_url",
    "mean",
    "sd",
    "sem",
    "q_low",
    "q_high",
    "q_low_label",
    "q_high_label",
    "stat_type",
    "unit",
    "n",
    "data_kind",
    "extraction_method",
    "include_in_validation",
    "confidence",
    "note_ids",
    "notes",
    "reported_value_raw",
]

VALIDATION_NOTES_COLUMNS = [
    "note_id",
    "severity",
    "scope",
    "target_type",
    "target",
    "message",
    "display_order",
    "source",
    "source_location",
]

NEEDS_MANUAL_EXTRACTION_COLUMNS = [
    "source",
    "source_url",
    "source_file",
    "figure_or_table",
    "target_metric",
    "reason",
    "suggested_action",
]

GC_EPHYS_COLUMNS = [
    "Property",
    "mean +/- sd",
    "n",
    "Source",
    "Notes",
    "cell_type",
    "gc_subtype",
    "species",
    "age",
    "maturity",
    "layer_or_location",
    "recording_temperature_C",
    "protocol_id",
    "mean",
    "sd",
    "sem",
    "q_low",
    "q_high",
    "q_low_label",
    "q_high_label",
    "stat_type",
    "unit",
    "source_file",
    "source_url",
    "source_location",
    "data_kind",
    "extraction_method",
    "include_in_validation",
    "include_in_fi_validation",
    "confidence",
    "note_ids",
    "reported_value_raw",
]

GC_FI_CURVE_COLUMNS = [
    "source",
    "source_url",
    "source_file",
    "source_location",
    "cell_type",
    "gc_subtype",
    "cell_id",
    "species",
    "age",
    "maturity",
    "protocol_id",
    "current_pA",
    "firing_rate_Hz",
    "rate_definition",
    "step_duration_ms",
    "current_start_pA",
    "current_stop_pA",
    "current_step_pA",
    "baseline_or_holding_vm_mV",
    "synaptic_blockers",
    "temperature_C",
    "sample_scope",
    "extraction_method",
    "include_in_validation",
    "confidence",
    "note_ids",
    "notes",
]

GC_PROTOCOLS_COLUMNS = [
    "protocol_id",
    "source",
    "cell_type",
    "gc_subtype",
    "species",
    "age",
    "stimulus_type",
    "step_duration_ms",
    "current_start_pA",
    "current_stop_pA",
    "current_step_pA",
    "current_values_pA",
    "rate_definition",
    "spike_detection_rule",
    "baseline_or_holding_vm_mV",
    "synaptic_blockers",
    "temperature_C",
    "compatible_group",
    "notes",
]

GC_IDENTITY_COLUMNS = [
    "source",
    "source_url",
    "source_file",
    "source_location",
    "cell_type",
    "gc_subtype",
    "species",
    "age",
    "maturity",
    "layer_or_location",
    "identity_kind",
    "Property",
    "mean",
    "sd",
    "sem",
    "q_low",
    "q_high",
    "q_low_label",
    "q_high_label",
    "stat_type",
    "unit",
    "n",
    "data_kind",
    "extraction_method",
    "include_in_validation",
    "confidence",
    "note_ids",
    "notes",
    "reported_value_raw",
]

REFERENCE_OUTPUT_SCHEMA_PRESETS = {
    "ephys": {"row_type": "ephys", "columns": PV_CRH_EPL_FSI_EPHYS_COLUMNS},
    "fi_curve": {"row_type": "fi_curve", "columns": PV_CRH_EPL_FSI_FI_CURVE_COLUMNS},
    "protocols": {"row_type": "protocols", "columns": PV_CRH_EPL_FSI_PROTOCOLS_COLUMNS},
    "identity": {"row_type": "identity", "columns": PV_CRH_EPL_FSI_IDENTITY_COLUMNS},
    "notes": {"row_type": "notes", "columns": VALIDATION_NOTES_COLUMNS},
    "manual": {"row_type": "manual", "columns": NEEDS_MANUAL_EXTRACTION_COLUMNS},
    "readme": {"row_type": "readme", "columns": []},
    "gc_ephys": {"row_type": "ephys", "columns": GC_EPHYS_COLUMNS},
    "gc_fi_curve": {"row_type": "fi_curve", "columns": GC_FI_CURVE_COLUMNS},
    "gc_protocols": {"row_type": "protocols", "columns": GC_PROTOCOLS_COLUMNS},
    "gc_identity": {"row_type": "identity", "columns": GC_IDENTITY_COLUMNS},
}


def reference_data_path(filename: str) -> Path:
    return REFERENCE_DATA_DIR / filename


def canonical_property_name(property_name: str) -> str:
    text = str(property_name).strip()
    return PROPERTY_ALIASES.get(text, text)


def is_fi_related_property(property_name: str) -> bool:
    return canonical_property_name(property_name) in FI_RELATED_PROPERTIES


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_mean_plus_minus_sd(text: str) -> tuple[float | None, float | None]:
    raw = str(text or "").strip()
    if not raw or "+/-" not in raw:
        return None, None
    mean_text, sd_text = [part.strip() for part in raw.split("+/-", 1)]
    try:
        return float(mean_text), float(sd_text)
    except ValueError:
        return None, None


def _bool_csv(value: bool) -> str:
    return "true" if value else "false"


def _normalized_legacy_row(cell_type: str, csv_path: Path, row: dict[str, str]) -> dict[str, Any]:
    property_name = canonical_property_name(row.get("Property", ""))
    mean_value, sd_value = parse_mean_plus_minus_sd(row.get("mean +/- sd", ""))
    fi_related = is_fi_related_property(property_name)
    notes = str(row.get("Notes", "") or "").strip()
    return {
        "Property": property_name,
        "mean +/- sd": str(row.get("mean +/- sd", "") or "").strip(),
        "n": str(row.get("n", "") or "").strip(),
        "Source": str(row.get("Source", "") or "").strip(),
        "Notes": notes,
        "cell_type": cell_type,
        "marker_profile": "principal_cell",
        "protocol_id": BU2014_MC_TC_PROTOCOL_ID if fi_related else "",
        "mean": mean_value if mean_value is not None else "",
        "sd": sd_value if sd_value is not None else "",
        "sem": "",
        "stat_type": "mean_sd" if mean_value is not None and sd_value is not None else "",
        "unit": PROPERTY_UNITS.get(property_name, ""),
        "source_file": csv_path.name,
        "source_location": "legacy MC/TC electrophysiology reference CSV",
        "source_url": "",
        "data_kind": "fI_summary_metric" if fi_related else "intrinsic_property",
        "extraction_method": "legacy_csv",
        "include_in_validation": _bool_csv(True),
        "include_in_fi_validation": _bool_csv(fi_related),
        "confidence": "high",
        "note_ids": FI_PROTOCOL_DIFFERENCE_NOTE_ID if fi_related else "",
        "reported_value_raw": str(row.get("mean +/- sd", "") or "").strip(),
    }


def load_normalized_legacy_mc_tc_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell_type, filename in LEGACY_MC_TC_EPHYS_FILENAMES.items():
        csv_path = reference_data_path(filename)
        for row in csv_rows(csv_path):
            rows.append(_normalized_legacy_row(cell_type, csv_path, row))
    return rows


def load_pv_crh_epl_fsi_ephys_rows() -> list[dict[str, str]]:
    return csv_rows(reference_data_path(PV_CRH_EPL_FSI_EPHYS_FILENAME))


def load_pv_crh_epl_fsi_fi_curve_rows() -> list[dict[str, str]]:
    return csv_rows(reference_data_path(PV_CRH_EPL_FSI_FI_CURVE_FILENAME))


def load_pv_crh_epl_fsi_protocol_rows() -> list[dict[str, str]]:
    return csv_rows(reference_data_path(PV_CRH_EPL_FSI_PROTOCOLS_FILENAME))


def load_pv_crh_epl_fsi_identity_rows() -> list[dict[str, str]]:
    return csv_rows(reference_data_path(PV_CRH_EPL_FSI_IDENTITY_FILENAME))


def load_gc_ephys_rows() -> list[dict[str, str]]:
    return csv_rows(reference_data_path(GC_EPHYS_FILENAME))


def load_gc_fi_curve_rows() -> list[dict[str, str]]:
    return csv_rows(reference_data_path(GC_FI_CURVE_FILENAME))


def load_gc_sgc_dgc_ephys_rows() -> list[dict[str, str]]:
    return csv_rows(reference_data_path(GC_SGC_DGC_EPHYS_FILENAME))


def load_gc_sgc_dgc_fi_curve_rows() -> list[dict[str, str]]:
    return csv_rows(reference_data_path(GC_SGC_DGC_FI_CURVE_FILENAME))


def load_gc_protocol_rows() -> list[dict[str, str]]:
    return csv_rows(reference_data_path(GC_PROTOCOLS_FILENAME))


def load_gc_identity_rows() -> list[dict[str, str]]:
    return csv_rows(reference_data_path(GC_IDENTITY_FILENAME))


def load_gc_synaptic_latency_rows() -> list[dict[str, str]]:
    return csv_rows(reference_data_path(GC_SYNAPTIC_LATENCY_FILENAME))


def load_gc_modulation_rows() -> list[dict[str, str]]:
    return csv_rows(reference_data_path(GC_MODULATION_FILENAME))


def _matches(value: str, criterion: str | Iterable[str] | None) -> bool:
    if criterion is None:
        return True
    if isinstance(criterion, str):
        return str(value) == criterion
    acceptable = {str(item) for item in criterion}
    return str(value) in acceptable


def _parse_bool_filter(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _row_flag(row: dict[str, Any], key: str) -> bool | None:
    return _parse_bool_filter(row.get(key))


def filter_reference_rows(
    rows: Iterable[dict[str, Any]],
    *,
    cell_type: str | Iterable[str] | None = None,
    marker_profile: str | Iterable[str] | None = None,
    protocol_id: str | Iterable[str] | None = None,
    property_name: str | Iterable[str] | None = None,
    include_in_validation: bool | None = None,
    include_in_fi_validation: bool | None = None,
) -> list[dict[str, Any]]:
    canonical_property = None
    if property_name is not None:
        if isinstance(property_name, str):
            canonical_property = canonical_property_name(property_name)
        else:
            canonical_property = [canonical_property_name(item) for item in property_name]

    filtered: list[dict[str, Any]] = []
    for row in rows:
        if not _matches(str(row.get("cell_type", "")), cell_type):
            continue
        if not _matches(str(row.get("marker_profile", "")), marker_profile):
            continue
        if not _matches(str(row.get("protocol_id", "")), protocol_id):
            continue
        if not _matches(canonical_property_name(str(row.get("Property", ""))), canonical_property):
            continue
        if include_in_validation is not None and _row_flag(row, "include_in_validation") is not include_in_validation:
            continue
        if include_in_fi_validation is not None and _row_flag(row, "include_in_fi_validation") is not include_in_fi_validation:
            continue
        filtered.append(dict(row))
    return filtered
