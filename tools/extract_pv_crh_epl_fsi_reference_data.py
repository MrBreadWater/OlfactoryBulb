"""Generate PV/CRH-overlap EPL fast-spiking interneuron reference-data assets."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
import sys
from typing import Any

try:
    import xlrd
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in real env
    raise RuntimeError(
        "The EPL-FSI reference extractor requires xlrd in the OBGPU environment. "
        "Install it with conda, not pip."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from olfactorybulb.audit.reference_data import (  # noqa: E402
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
    REFERENCE_DATA_DIR,
    VALIDATION_NOTES_COLUMNS,
    VALIDATION_NOTES_FILENAME,
)
from olfactorybulb.audit.reference_sources import (  # noqa: E402
    BURTON2024_ARTICLE_PDF_SOURCE_ID,
    BURTON2024_S15_DATA_SOURCE_ID,
    BURTON2024_S16_DATA_SOURCE_ID,
    BURTON2024_S1_TABLE_SOURCE_ID,
    BURTON2024_S2_TABLE_SOURCE_ID,
    BURTON2024_S8_DATA_SOURCE_ID,
    REQUIRED_BURTON2024_SOURCE_IDS,
    ensure_reference_sources,
    local_source_path,
    source_entry,
    stable_source_url,
)


BURTON2024_SOURCE = "Burton, Malyshko & Urban (2024)"
HUANG2013_SOURCE = "Huang et al. (2013)"
KATO2013_SOURCE = "Kato et al. (2013)"
LIU2019_SOURCE = "Liu et al. (2019)"

BURTON2024_EXAMPLE_CELL_NOTE_ID = "N_BMU2024_EXAMPLE_CELL_SCOPE"
BURTON2024_SUPPLEMENT_MISSING_NOTE_ID = "N_BMU2024_SUPPLEMENT_MISSING"
EPL_FI_POINTS_UNAVAILABLE_NOTE_ID = "N_EPL_FI_POINTS_UNAVAILABLE"
MARKER_PROFILE_SEPARATION_NOTE_ID = "N_MARKER_PROFILE_SEPARATION"

HUANG2013_PDF_URL = "https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2013.00032/pdf"
KATO2013_PDF_URL = "https://komiyamalab.biosci.ucsd.edu/wp-content/uploads/2021/05/2013-1-s2.0-S0896627313007952-main.pdf"

HUANG2013_SOURCE_FILE = "Huang EPLI.pdf"
KATO2013_SOURCE_FILE = "kato2013 EPLI.pdf"

S8_PANEL_CELL_SPECS = {
    "Panels D,E": {
        "cell_id": "BMU2024_FSI_example_Fig1B",
        "cell_type": "EPL-FSI",
        "marker_profile": "PV+; CRH_not_assayed_or_not_reported",
        "sample_scope": "example_cell",
        "source_location": "S8 Data, sheet 'Panels D,E' (example fast-spiking interneuron corresponding to S2 Fig D,E / Fig 1B)",
    },
    "Panels I,J": {
        "cell_id": "BMU2024_FSI_example_Fig1C",
        "cell_type": "EPL-FSI",
        "marker_profile": "PV+; CRH_not_assayed_or_not_reported",
        "sample_scope": "example_cell",
        "source_location": "S8 Data, sheet 'Panels I,J' (example fast-spiking interneuron corresponding to S2 Fig I,J / Fig 1C)",
    },
}

S15_EPHYS_SPECS = (
    {
        "column": "resting potential (mV)",
        "property_name": "Membrane Resting Voltage",
        "unit": "mV",
        "data_kind": "intrinsic_property",
        "notes": "Computed directly from FSI rows in S15 Data.",
        "reported_definition": "resting potential",
    },
    {
        "column": "input resistance\n(MΩ)",
        "property_name": "Input Resistance",
        "unit": "MOhm",
        "data_kind": "intrinsic_property",
        "notes": "Computed directly from FSI rows in S15 Data.",
        "reported_definition": "input resistance",
    },
    {
        "column": "membrane time constant (ms)",
        "property_name": "Membrane Time Constant",
        "unit": "ms",
        "data_kind": "intrinsic_property",
        "notes": "Computed directly from FSI rows in S15 Data.",
        "reported_definition": "membrane time constant",
    },
    {
        "column": "membrane capacitance (pF)",
        "property_name": "Capacitance",
        "unit": "pF",
        "data_kind": "intrinsic_property",
        "notes": "Computed directly from FSI rows in S15 Data.",
        "reported_definition": "membrane capacitance",
    },
    {
        "column": "spontaneous firing rate (Hz)",
        "property_name": "Spontaneous Firing Rate",
        "unit": "Hz",
        "data_kind": "intrinsic_property",
        "notes": "Computed directly from FSI rows in S15 Data.",
        "reported_definition": "spontaneous firing rate",
    },
    {
        "column": "spike amp.\n(mV)",
        "property_name": "AP Amplitude",
        "unit": "mV",
        "data_kind": "intrinsic_property",
        "notes": "Computed directly from FSI rows in S15 Data.",
        "reported_definition": "spike amplitude",
    },
    {
        "column": "spike width\n(ms)",
        "property_name": "AP Half-Width",
        "unit": "ms",
        "data_kind": "intrinsic_property",
        "notes": "The source spreadsheet reports spike width in milliseconds; this row preserves that definition directly.",
        "reported_definition": "spike width",
    },
    {
        "column": "spike threshold\n(mV)",
        "property_name": "AP Threshold",
        "unit": "mV",
        "data_kind": "intrinsic_property",
        "notes": "Computed directly from FSI rows in S15 Data.",
        "reported_definition": "spike threshold",
    },
    {
        "column": "afterhyperpolarization amp. (mV)",
        "property_name": "AHP Amplitude",
        "unit": "mV",
        "data_kind": "intrinsic_property",
        "notes": "Computed directly from FSI rows in S15 Data.",
        "reported_definition": "afterhyperpolarization amplitude",
    },
    {
        "column": "afterhyperpolarization 50% decay (ms)",
        "property_name": "AHP Duration",
        "unit": "ms",
        "data_kind": "intrinsic_property",
        "notes": "This is the afterhyperpolarization fifty-percent decay time reported in S15 Data.",
        "reported_definition": "afterhyperpolarization 50 percent decay",
    },
    {
        "column": "rheobase\n(pA)",
        "property_name": "Rheobase Current",
        "unit": "pA",
        "data_kind": "fI_summary_metric",
        "notes": "Computed directly from FSI rows in S15 Data under the Burton, Malyshko & Urban 2024 EPL-FSI current-clamp protocol.",
        "reported_definition": "rheobase",
    },
    {
        "column": "max. gain \n(Hz/pA)",
        "property_name": "FI Curve Slope",
        "unit": "Hz/nA",
        "data_kind": "fI_summary_metric",
        "transform": lambda value: float(value) * 1000.0,
        "notes": "The source defines this as maximum gain in hertz per picoampere. It is converted here to hertz per nanoampere by multiplying by one thousand.",
        "reported_definition": "maximum gain",
    },
    {
        "column": "max. instantaneous rate (Hz)",
        "property_name": "Max FI Rate",
        "unit": "Hz",
        "data_kind": "fI_summary_metric",
        "notes": "The source defines this as maximum instantaneous rate rather than a full current-rate point set.",
        "reported_definition": "maximum instantaneous rate",
    },
    {
        "column": "max. interspike interval (ISI) C.V. ",
        "property_name": "ISI Coefficient of Variation",
        "unit": "",
        "data_kind": "fI_summary_metric",
        "notes": "The source defines this as the maximum interspike-interval coefficient of variation across the protocol, not as a baseline irregularity measure.",
        "reported_definition": "maximum interspike interval coefficient of variation",
    },
    {
        "column": "relative adaptation (ISIfirst/ISIlast; %)",
        "property_name": "Spiking Rate Accommodation",
        "unit": "%",
        "data_kind": "fI_summary_metric",
        "notes": (
            "The source defines this row as relative adaptation using first-interval over last-interval ratio in percent. "
            "The companion absolute-adaptation column remains available in the raw S15 workbook but is not emitted as a separate canonical row here."
        ),
        "reported_definition": "relative adaptation (ISI first / ISI last; percent)",
    },
)

S16_IDENTITY_SPECS = (
    {
        "column": "soma max. diameter\n(μm)",
        "property_name": "Soma Diameter",
        "unit": "um",
        "identity_kind": "morphology_constraint",
        "notes": "Computed directly from FSI rows in S16 Data somatodendritic morphology sheet.",
    },
    {
        "column": "# primary dendritic branches",
        "property_name": "Primary Process Count",
        "unit": "count",
        "identity_kind": "morphology_constraint",
        "notes": "Computed directly from FSI rows in S16 Data somatodendritic morphology sheet.",
    },
    {
        "column": "total dendritic length (mm)",
        "property_name": "Total Dendritic Length",
        "unit": "mm",
        "identity_kind": "morphology_constraint",
        "notes": "Computed directly from FSI rows in S16 Data somatodendritic morphology sheet.",
    },
)


def _bool_csv(value: bool) -> str:
    return "true" if value else "false"


def _format_mean_plus_minus(mean: float | None, spread: float | None) -> str:
    if mean is None or spread is None:
        return ""
    return f"{mean:g} +/- {spread:g}"


def _sd_from_sem(sem: float, n: int) -> float:
    return float(sem) * math.sqrt(float(n))


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "nan":
            return None
        try:
            return float(text)
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    return None


def _column_index_map(sheet: xlrd.sheet.Sheet) -> dict[str, int]:
    return {str(sheet.cell_value(0, index)): index for index in range(sheet.ncols)}


def _sheet_records(sheet: xlrd.sheet.Sheet) -> list[dict[str, object]]:
    headers = [str(sheet.cell_value(0, index)) for index in range(sheet.ncols)]
    records: list[dict[str, object]] = []
    for row_index in range(1, sheet.nrows):
        values = sheet.row_values(row_index)
        records.append({header: values[column_index] for column_index, header in enumerate(headers)})
    return records


def _stats(values: list[float]) -> tuple[float, float, float, int]:
    if not values:
        raise ValueError("Cannot compute statistics for an empty value list")
    mean_value = statistics.fmean(values)
    sd_value = statistics.stdev(values) if len(values) > 1 else 0.0
    sem_value = sd_value / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean_value, sd_value, sem_value, len(values)


def _clean_label(text: object) -> str:
    return " ".join(str(text).split())


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
    source_url: str,
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
        "source_url": source_url,
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
    source_url: str,
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
        "source_url": source_url,
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


def _burton_source_status() -> dict[str, Any]:
    available, download_errors = ensure_reference_sources(
        source_ids=list(REQUIRED_BURTON2024_SOURCE_IDS),
        include_optional=False,
        strict=False,
    )
    missing_ids = [source_id for source_id in REQUIRED_BURTON2024_SOURCE_IDS if source_id not in available]
    return {"available": available, "download_errors": download_errors, "missing_ids": missing_ids}


def _s15_fsi_ephys_rows(status: dict[str, Any]) -> list[dict[str, object]]:
    if BURTON2024_S15_DATA_SOURCE_ID in status["missing_ids"]:
        return []
    workbook = xlrd.open_workbook(str(local_source_path(BURTON2024_S15_DATA_SOURCE_ID)))
    sheet = workbook.sheet_by_name("S1 Table")
    records = [record for record in _sheet_records(sheet) if str(record.get("EPL-IN type", "")).strip() == "FSI"]

    source_file = source_entry(BURTON2024_S15_DATA_SOURCE_ID)["filename"]
    source_url = stable_source_url(BURTON2024_S15_DATA_SOURCE_ID)
    rows: list[dict[str, object]] = []
    for spec in S15_EPHYS_SPECS:
        raw_values = [_float_or_none(record.get(spec["column"])) for record in records]
        values = [float(spec.get("transform", lambda value: value)(value)) for value in raw_values if value is not None]
        mean_value, sd_value, sem_value, count = _stats(values)
        fi_related = spec["data_kind"] == "fI_summary_metric"
        notes = str(spec["notes"])
        if fi_related:
            notes = (
                f"{notes} Step duration 500 ms, current range 50 to 600 pA in 50 pA increments, "
                "firing rate defined as median inverse interspike interval, spike detection rule = voltage derivative threshold 15 mV/ms."
            )
        rows.append(
            _ephys_row(
                property_name=str(spec["property_name"]),
                source=BURTON2024_SOURCE,
                notes=notes,
                cell_type="EPL-FSI",
                marker_profile="PV+; CRH_not_assayed_or_not_reported",
                protocol_id=BMU2024_EPL_FSI_PROTOCOL_ID if fi_related else "",
                mean=mean_value,
                sd=sd_value,
                sem=sem_value,
                n=count,
                stat_type="mean_sd",
                unit=str(spec["unit"]),
                source_file=source_file,
                source_location=f"S15 Data, sheet 'S1 Table', column '{_clean_label(spec['column'])}', FSI rows",
                source_url=source_url,
                data_kind=str(spec["data_kind"]),
                extraction_method="source_spreadsheet",
                include_in_validation=True,
                include_in_fi_validation=fi_related,
                confidence="high",
                note_ids=FI_PROTOCOL_DIFFERENCE_NOTE_ID if fi_related else "",
                reported_value_raw=(
                    f"Computed from S15 Data FSI rows for {spec['reported_definition']} "
                    f"(n = {count}, mean = {mean_value:.6g}, sd = {sd_value:.6g}, sem = {sem_value:.6g})"
                ),
            )
        )
    return rows


def _s8_fi_curve_rows(status: dict[str, Any]) -> list[dict[str, object]]:
    if BURTON2024_S8_DATA_SOURCE_ID in status["missing_ids"]:
        return []
    workbook = xlrd.open_workbook(str(local_source_path(BURTON2024_S8_DATA_SOURCE_ID)))
    source_file = source_entry(BURTON2024_S8_DATA_SOURCE_ID)["filename"]
    source_url = stable_source_url(BURTON2024_S8_DATA_SOURCE_ID)

    rows: list[dict[str, object]] = []
    for sheet_name, cell_spec in S8_PANEL_CELL_SPECS.items():
        sheet = workbook.sheet_by_name(sheet_name)
        column_map = _column_index_map(sheet)
        for row_index in range(1, sheet.nrows):
            current_value = _float_or_none(sheet.cell_value(row_index, column_map["current (pA)"]))
            firing_rate = _float_or_none(sheet.cell_value(row_index, column_map["firing rate (Hz)"]))
            if current_value is None or firing_rate is None:
                continue
            if current_value < 50.0 or current_value > 600.0:
                continue
            rows.append(
                {
                    "source": BURTON2024_SOURCE,
                    "source_file": source_file,
                    "source_location": cell_spec["source_location"],
                    "cell_type": cell_spec["cell_type"],
                    "cell_id": cell_spec["cell_id"],
                    "marker_profile": cell_spec["marker_profile"],
                    "protocol_id": BMU2024_EPL_FSI_PROTOCOL_ID,
                    "source_url": source_url,
                    "current_pA": current_value,
                    "firing_rate_Hz": firing_rate,
                    "rate_definition": "Median inverse interspike interval.",
                    "step_duration_ms": 500.0,
                    "current_start_pA": 50.0,
                    "current_stop_pA": 600.0,
                    "current_step_pA": 50.0,
                    "baseline_or_holding_vm_mV": "held at resting membrane potential (0 pA holding current)",
                    "synaptic_blockers": "",
                    "temperature_C": "",
                    "sample_scope": cell_spec["sample_scope"],
                    "extraction_method": "source_spreadsheet",
                    "include_in_validation": _bool_csv(True),
                    "confidence": "high",
                    "note_ids": f"{FI_PROTOCOL_DIFFERENCE_NOTE_ID};{BURTON2024_EXAMPLE_CELL_NOTE_ID}",
                    "notes": (
                        "Actual current-rate point from Burton 2024 S8 Data. This row is from an example fast-spiking interneuron, "
                        "not a population average. The companion CV(ISI) column remains in the source workbook and is not emitted here."
                    ),
                }
            )
    return rows


def _s16_identity_rows(status: dict[str, Any]) -> list[dict[str, object]]:
    if BURTON2024_S16_DATA_SOURCE_ID in status["missing_ids"]:
        return []

    workbook = xlrd.open_workbook(str(local_source_path(BURTON2024_S16_DATA_SOURCE_ID)))
    morphology_sheet = workbook.sheet_by_name("somatodendritic morphology")
    morphology_records = [
        record for record in _sheet_records(morphology_sheet) if str(record.get("EPL-IN type", "")).strip() == "FSI"
    ]
    relative_depth_sheet = workbook.sheet_by_name("relative EPL depth")
    relative_depth_records = [
        record for record in _sheet_records(relative_depth_sheet) if str(record.get("EPL-IN type", "")).strip() == "FSI"
    ]

    source_file = source_entry(BURTON2024_S16_DATA_SOURCE_ID)["filename"]
    source_url = stable_source_url(BURTON2024_S16_DATA_SOURCE_ID)
    rows: list[dict[str, object]] = []

    for spec in S16_IDENTITY_SPECS:
        values = [
            value
            for value in (_float_or_none(record.get(spec["column"])) for record in morphology_records)
            if value is not None
        ]
        mean_value, sd_value, sem_value, count = _stats(values)
        rows.append(
            _identity_row(
                source=BURTON2024_SOURCE,
                source_file=source_file,
                source_location=f"S16 Data, sheet 'somatodendritic morphology', column '{_clean_label(spec['column'])}', FSI rows",
                source_url=source_url,
                cell_type="EPL-FSI",
                marker_profile="PV+; CRH_not_assayed_or_not_reported",
                identity_kind=str(spec["identity_kind"]),
                property_name=str(spec["property_name"]),
                mean=mean_value,
                sd=sd_value,
                sem=sem_value,
                stat_type="mean_sd",
                unit=str(spec["unit"]),
                n=count,
                data_kind="morphology_constraint",
                extraction_method="source_spreadsheet",
                include_in_validation=True,
                confidence="high",
                notes=str(spec["notes"]),
                reported_value_raw=(
                    f"Computed from S16 Data FSI rows for {spec['property_name']} "
                    f"(n = {count}, mean = {mean_value:.6g}, sd = {sd_value:.6g}, sem = {sem_value:.6g})"
                ),
            )
        )

    depth_values = [
        value
        for record in relative_depth_records
        for value in [_float_or_none(record.get("relative EPL depth \n(0=MCL, 1=GL)\n[sorted by type and depth]"))]
        if value is not None
    ]
    if depth_values:
        mean_value, sd_value, sem_value, count = _stats(depth_values)
        rows.append(
            _identity_row(
                source=BURTON2024_SOURCE,
                source_file=source_file,
                source_location="S16 Data, sheet 'relative EPL depth', FSI rows",
                source_url=source_url,
                cell_type="EPL-FSI",
                marker_profile="PV+; CRH_not_assayed_or_not_reported",
                identity_kind="morphology_constraint",
                property_name="Relative EPL Depth",
                mean=mean_value,
                sd=sd_value,
                sem=sem_value,
                stat_type="mean_sd",
                unit="normalized depth",
                n=count,
                data_kind="morphology_constraint",
                extraction_method="source_spreadsheet",
                include_in_validation=True,
                confidence="high",
                notes="Computed directly from FSI rows in S16 Data relative EPL depth sheet.",
                reported_value_raw=(
                    f"Computed from S16 Data relative EPL depth FSI rows "
                    f"(n = {count}, mean = {mean_value:.6g}, sd = {sd_value:.6g}, sem = {sem_value:.6g})"
                ),
            )
        )

    return rows


def build_ephys_rows(status: dict[str, Any]) -> list[dict[str, object]]:
    huang_spont_sd = _sd_from_sem(0.09, 14)
    huang_max_rate_sd = _sd_from_sem(6.14, 10)
    kato_input_resistance_sd = _sd_from_sem(5.6, 28)
    kato_tau_sd = _sd_from_sem(0.4, 28)
    kato_half_width_sd = _sd_from_sem(0.027, 28)
    kato_max_rate_sd = _sd_from_sem(12.0, 14)

    rows = _s15_fsi_ephys_rows(status)
    rows.extend(
        [
            _ephys_row(
                property_name="Spontaneous Firing Rate",
                source=HUANG2013_SOURCE,
                notes="CRH+ external-plexiform-layer interneurons; baseline firing rate reported directly in text near Figure 4E.",
                cell_type="CRH+ EPL-IN",
                marker_profile="CRH+; PV_overlap_population",
                mean=0.20,
                sd=huang_spont_sd,
                sem=0.09,
                n=14,
                stat_type="sd_from_sem",
                unit="Hz",
                source_file=HUANG2013_SOURCE_FILE,
                source_location="Figure 4E; Results text describing baseline firing rates",
                source_url=HUANG2013_PDF_URL,
                data_kind="intrinsic_property",
                extraction_method="pdf_text",
                include_in_validation=True,
                include_in_fi_validation=False,
                confidence="high",
                reported_value_raw="0.20 +/- 0.09 Hz (n = 14, reported as mean +/- SEM)",
            ),
            _ephys_row(
                property_name="Max FI Rate",
                source=HUANG2013_SOURCE,
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
                source_file=HUANG2013_SOURCE_FILE,
                source_location="Figure 4F; Results text describing current-evoked firing",
                source_url=HUANG2013_PDF_URL,
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
                source=KATO2013_SOURCE,
                notes="PV+ external-plexiform-layer interneurons; electrophysiology summary from text near Figure 1D.",
                cell_type="PV+ EPL-IN",
                marker_profile="PV+; CRH_unknown",
                mean=90.5,
                sd=kato_input_resistance_sd,
                sem=5.6,
                n=28,
                stat_type="sd_from_sem",
                unit="MOhm",
                source_file=KATO2013_SOURCE_FILE,
                source_location="Results text describing electrophysiological properties near Figure 1D",
                source_url=KATO2013_PDF_URL,
                data_kind="intrinsic_property",
                extraction_method="pdf_text",
                include_in_validation=True,
                include_in_fi_validation=False,
                confidence="high",
                reported_value_raw="90.5 +/- 5.6 MOhm (n = 28, reported as mean +/- SEM)",
            ),
            _ephys_row(
                property_name="Membrane Time Constant",
                source=KATO2013_SOURCE,
                notes="PV+ external-plexiform-layer interneurons; electrophysiology summary from text near Figure 1D.",
                cell_type="PV+ EPL-IN",
                marker_profile="PV+; CRH_unknown",
                mean=5.9,
                sd=kato_tau_sd,
                sem=0.4,
                n=28,
                stat_type="sd_from_sem",
                unit="ms",
                source_file=KATO2013_SOURCE_FILE,
                source_location="Results text describing electrophysiological properties near Figure 1D",
                source_url=KATO2013_PDF_URL,
                data_kind="intrinsic_property",
                extraction_method="pdf_text",
                include_in_validation=True,
                include_in_fi_validation=False,
                confidence="high",
                reported_value_raw="5.9 +/- 0.4 ms (n = 28, reported as mean +/- SEM)",
            ),
            _ephys_row(
                property_name="AP Half-Width",
                source=KATO2013_SOURCE,
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
                source_file=KATO2013_SOURCE_FILE,
                source_location="Figure 1D; Results text describing fast action potentials",
                source_url=KATO2013_PDF_URL,
                data_kind="intrinsic_property",
                extraction_method="pdf_text",
                include_in_validation=True,
                include_in_fi_validation=False,
                confidence="high",
                reported_value_raw="530 +/- 27 μs (n = 28, reported as mean +/- SEM)",
            ),
            _ephys_row(
                property_name="Max FI Rate",
                source=KATO2013_SOURCE,
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
                source_file=KATO2013_SOURCE_FILE,
                source_location="Figure 1D; Results text describing high-frequency spikes",
                source_url=KATO2013_PDF_URL,
                data_kind="fI_summary_metric",
                extraction_method="pdf_text",
                include_in_validation=True,
                include_in_fi_validation=False,
                confidence="high",
                note_ids=FI_PROTOCOL_DIFFERENCE_NOTE_ID,
                reported_value_raw="171 +/- 12 Hz (n = 14, reported as mean +/- SEM)",
            ),
        ]
    )
    return rows


def build_fi_curve_rows(status: dict[str, Any]) -> list[dict[str, object]]:
    return _s8_fi_curve_rows(status)


def build_protocol_rows(status: dict[str, Any]) -> list[dict[str, object]]:
    burton_protocol_note = (
        "Actual current-rate points were extracted from S8 Data example fast-spiking interneuron sheets. "
        "This remains a different protocol family from the legacy Burton and Urban 2014 mitral-cell / tufted-cell protocol."
        if BURTON2024_S8_DATA_SOURCE_ID not in status["missing_ids"]
        else "Protocol recovered from the local PDF methods text. Numeric current-rate points require S8 Data or a digitization manifest."
    )
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
            source=BURTON2024_SOURCE,
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
            notes=burton_protocol_note,
        ),
        _protocol_row(
            protocol_id=HUANG2013_CRH_EPL_PROTOCOL_ID,
            source=HUANG2013_SOURCE,
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
            source=KATO2013_SOURCE,
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


def build_identity_rows(status: dict[str, Any]) -> list[dict[str, object]]:
    huang_pv_overlap_sd = _sd_from_sem(3.0, 3)
    huang_sst_overlap_sd = _sd_from_sem(0.8, 3)
    huang_crh_fraction_sd = _sd_from_sem(1.31, 3)
    huang_calretinin_fraction_sd = _sd_from_sem(2.70, 3)
    huang_neun_sd = _sd_from_sem(1.7, 3)

    burton_pdf_file = source_entry(BURTON2024_ARTICLE_PDF_SOURCE_ID)["filename"]
    burton_pdf_url = stable_source_url(BURTON2024_ARTICLE_PDF_SOURCE_ID)

    rows = _s16_identity_rows(status)
    rows.extend(
        [
            _identity_row(
                source=BURTON2024_SOURCE,
                source_file=burton_pdf_file,
                source_location="Results text describing FSI neurochemical profile and lack of visible axons",
                source_url=burton_pdf_url,
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
                source=BURTON2024_SOURCE,
                source_file=burton_pdf_file,
                source_location="Results text describing FSI neurochemical profile",
                source_url=burton_pdf_url,
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
                notes="Neither fast-spiking nor regular-spiking interneurons expressed tyrosine hydroxylase.",
                reported_value_raw="Neither FSIs nor RSIs expressed TH",
            ),
            _identity_row(
                source=BURTON2024_SOURCE,
                source_file=burton_pdf_file,
                source_location="Results text describing weak VIP expression in a subset of FSIs",
                source_url=burton_pdf_url,
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
                notes="Weak vasoactive-intestinal-peptide expression was reported in a minority of FSIs.",
                reported_value_raw="27% of FSIs expressed weak VIP",
            ),
            _identity_row(
                source=BURTON2024_SOURCE,
                source_file=burton_pdf_file,
                source_location="Results text describing FSI morphology",
                source_url=burton_pdf_url,
                cell_type="EPL-FSI",
                marker_profile="PV+; CRH_not_assayed_or_not_reported",
                identity_kind="morphology_constraint",
                property_name="Axonless Morphology",
                stat_type="qualitative",
                data_kind="morphology_constraint",
                extraction_method="pdf_text",
                include_in_validation=True,
                confidence="high",
                notes="The paper states that fast-spiking interneurons did not extend visible axons.",
                reported_value_raw="Neither FSIs nor RSIs extended visible axons",
            ),
            _identity_row(
                source=HUANG2013_SOURCE,
                source_file=HUANG2013_SOURCE_FILE,
                source_location="Figure 2J; CRH-Cre overlap analysis in the EPL",
                source_url=HUANG2013_PDF_URL,
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
                source=HUANG2013_SOURCE,
                source_file=HUANG2013_SOURCE_FILE,
                source_location="Figure 2J; calretinin-positive EPL fraction text",
                source_url=HUANG2013_PDF_URL,
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
                source=HUANG2013_SOURCE,
                source_file=HUANG2013_SOURCE_FILE,
                source_location="Figure 2J; marker overlap text",
                source_url=HUANG2013_PDF_URL,
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
                source=HUANG2013_SOURCE,
                source_file=HUANG2013_SOURCE_FILE,
                source_location="Figure 2J; marker overlap text",
                source_url=HUANG2013_PDF_URL,
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
                source=HUANG2013_SOURCE,
                source_file=HUANG2013_SOURCE_FILE,
                source_location="Figure 2J; marker overlap text",
                source_url=HUANG2013_PDF_URL,
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
                source=HUANG2013_SOURCE,
                source_file=HUANG2013_SOURCE_FILE,
                source_location="Figure 2I / Figure 2J; NeuN labeling text",
                source_url=HUANG2013_PDF_URL,
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
                source=HUANG2013_SOURCE,
                source_file=HUANG2013_SOURCE_FILE,
                source_location="Figure 3 morphology text",
                source_url=HUANG2013_PDF_URL,
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
                source=HUANG2013_SOURCE,
                source_file=HUANG2013_SOURCE_FILE,
                source_location="Figure 3 morphology text",
                source_url=HUANG2013_PDF_URL,
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
                source=HUANG2013_SOURCE,
                source_file=HUANG2013_SOURCE_FILE,
                source_location="Figure 3 morphology text",
                source_url=HUANG2013_PDF_URL,
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
                source=HUANG2013_SOURCE,
                source_file=HUANG2013_SOURCE_FILE,
                source_location="Figure 3 morphology text",
                source_url=HUANG2013_PDF_URL,
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
                source=HUANG2013_SOURCE,
                source_file=HUANG2013_SOURCE_FILE,
                source_location="Figure 3 axon-initial-segment text",
                source_url=HUANG2013_PDF_URL,
                cell_type="CRH+ EPL-IN",
                marker_profile="CRH+; PV_overlap_population",
                identity_kind="morphology_constraint",
                property_name="Axonless Morphology",
                stat_type="qualitative",
                data_kind="morphology_constraint",
                extraction_method="pdf_text",
                include_in_validation=True,
                confidence="high",
                notes="The source reports no obvious axon initial segment or beta-IV-spectrin-defined axon.",
                reported_value_raw="CRH+ EPL interneurons were axonless",
            ),
            _identity_row(
                source=KATO2013_SOURCE,
                source_file=KATO2013_SOURCE_FILE,
                source_location="Results text describing PV-cell laminar distribution",
                source_url=KATO2013_PDF_URL,
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
                source=KATO2013_SOURCE,
                source_file=KATO2013_SOURCE_FILE,
                source_location="Figure 1C; anatomical reconstruction text",
                source_url=KATO2013_PDF_URL,
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
                source=KATO2013_SOURCE,
                source_file=KATO2013_SOURCE_FILE,
                source_location="Figure 1C; anatomical reconstruction text",
                source_url=KATO2013_PDF_URL,
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
    )
    return rows


def build_validation_notes_rows(status: dict[str, Any], fi_curve_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = [
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
            note_id=BURTON2024_EXAMPLE_CELL_NOTE_ID,
            severity="warning",
            scope="fI_validation",
            target_type="source",
            target=BURTON2024_SOURCE,
            message=(
                "Burton, Malyshko & Urban 2024 S8 current-rate rows are example-cell traces rather than population-level averages. "
                "Use them as tagged validation targets with sample_scope = example_cell."
            ),
            display_order=20,
            source=BURTON2024_SOURCE,
            source_location="S8 Data example FSI sheets",
        ),
        _note_row(
            note_id=MARKER_PROFILE_SEPARATION_NOTE_ID,
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

    if status["missing_ids"]:
        rows.append(
            _note_row(
                note_id=BURTON2024_SUPPLEMENT_MISSING_NOTE_ID,
                severity="warning",
                scope="extraction",
                target_type="source",
                target=BURTON2024_SOURCE,
                message=(
                    "One or more Burton, Malyshko & Urban 2024 supporting files could not be acquired automatically. "
                    "The extractor fell back to partial outputs and tracked the remaining gaps in needs_manual_extraction.csv."
                ),
                display_order=30,
                source=BURTON2024_SOURCE,
                source_location="Remote acquisition step for required supporting files",
            )
        )

    if not fi_curve_rows:
        rows.append(
            _note_row(
                note_id=EPL_FI_POINTS_UNAVAILABLE_NOTE_ID,
                severity="warning",
                scope="fI_validation",
                target_type="source",
                target=PV_CRH_EPL_FSI_FI_CURVE_FILENAME,
                message="No validated EPL fast-spiking interneuron current-rate point set was extracted. The f-I curve CSV remains empty until S8 Data is available or a digitization manifest is committed.",
                display_order=35,
                source=BURTON2024_SOURCE,
                source_location="S8 Data extraction status",
            )
        )

    return rows


def build_manual_extraction_rows(status: dict[str, Any], fi_curve_rows: list[dict[str, object]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if BURTON2024_S15_DATA_SOURCE_ID in status["missing_ids"]:
        rows.append(
            _manual_row(
                source=BURTON2024_SOURCE,
                source_file=source_entry(BURTON2024_S15_DATA_SOURCE_ID)["filename"],
                figure_or_table="S15 Data / S1 Table",
                target_metric="FSI intrinsic biophysical property summary table",
                reason=status["download_errors"].get(BURTON2024_S15_DATA_SOURCE_ID, "Supporting spreadsheet missing after download step."),
                suggested_action="Retry source download, then regenerate PV_CRH_EPL_FSI_ephys.csv from S15 Data.",
            )
        )
    if BURTON2024_S8_DATA_SOURCE_ID in status["missing_ids"] or not fi_curve_rows:
        rows.append(
            _manual_row(
                source=BURTON2024_SOURCE,
                source_file=source_entry(BURTON2024_S8_DATA_SOURCE_ID)["filename"],
                figure_or_table="S8 Data / S2 Fig",
                target_metric="EPL-FSI firing-rate-current points and firing-irregularity-current points",
                reason=status["download_errors"].get(
                    BURTON2024_S8_DATA_SOURCE_ID,
                    "S8 Data did not yield usable fast-spiking-interneuron current-rate rows after the download step.",
                ),
                suggested_action="Inspect the downloaded S8 workbook or add a committed digitization manifest before populating PV_CRH_EPL_FSI_fI_curve.csv.",
            )
        )

    rows.extend(
        [
            _manual_row(
                source=HUANG2013_SOURCE,
                source_file=HUANG2013_SOURCE_FILE,
                figure_or_table="Figure 4A-D",
                target_metric="CRH+ EPL-IN resistance, capacitance, resting membrane potential, and action-potential threshold numeric values",
                reason="The local PDF text states directionality but does not expose the bar values numerically.",
                suggested_action="Digitize Figure 4A-D with a committed manifest or locate a machine-readable source table before adding these rows to PV_CRH_EPL_FSI_ephys.csv.",
            ),
            _manual_row(
                source=HUANG2013_SOURCE,
                source_file=HUANG2013_SOURCE_FILE,
                figure_or_table="Figure 4F",
                target_metric="CRH+ EPL-IN current-rate points",
                reason="The local PDF text exposes only the maximum current-evoked firing rate, not the full current-rate series or exact step protocol values.",
                suggested_action="Digitize Figure 4F with provenance or add a supplemental numeric source before populating PV_CRH_EPL_FSI_fI_curve.csv.",
            ),
            _manual_row(
                source=KATO2013_SOURCE,
                source_file=KATO2013_SOURCE_FILE,
                figure_or_table="Figure 1D",
                target_metric="PV+ EPL-IN current-rate points and exact current range",
                reason="The local PDF text exposes 100 pA step increments and a high-frequency spike summary but not a recoverable current-rate point table.",
                suggested_action="Digitize Figure 1D or locate a supplemental numeric source before treating Kato 2013 as an f-I point-set reference.",
            ),
            _manual_row(
                source=LIU2019_SOURCE,
                source_file="liu_et_al_2019_article.pdf",
                figure_or_table="all",
                target_metric="EPL interneuron identity/network constraints from Liu et al. 2019",
                reason="Liu 2019 remains an identity/network-only source here unless explicit intrinsic or current-rate numeric data are extracted.",
                suggested_action="Add the local Liu et al. 2019 PDF or supplementary materials, then extract only identity/network constraints unless numeric intrinsic or current-rate data are present.",
            ),
        ]
    )
    return rows


def build_readme_text(status: dict[str, Any], fi_curve_rows: list[dict[str, object]]) -> str:
    missing_text = ", ".join(status["missing_ids"]) if status["missing_ids"] else "none"
    fi_curve_summary = (
        "S8-derived example-cell current-rate points were extracted for the primary EPL-FSI target population."
        if fi_curve_rows
        else "No validated current-rate point set was extracted; see needs_manual_extraction.csv."
    )
    return f"""# PV/CRH-overlap EPL fast-spiking interneuron reference-data extraction

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
  - {fi_curve_summary}

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
- Remaining missing required Burton sources after acquisition attempt: {missing_text}

## Extraction mechanics

- Required Burton files are fetched from stable PLOS source URLs through `tools/download_epl_fsi_reference_sources.py`.
- Redirected storage URLs are followed at download time, but `source_url` fields in the generated CSVs always preserve the stable manifest URL, never the transient signed redirect target.
- Exact f-I current-rate points are taken only from S8 workbook sheets that correspond to fast-spiking example cells. Summary metrics from S15 remain in `PV_CRH_EPL_FSI_ephys.csv` and are **not** back-projected into synthetic current-rate rows.
"""


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    REFERENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    status = _burton_source_status()
    fi_curve_rows = build_fi_curve_rows(status)
    ephys_rows = build_ephys_rows(status)
    protocol_rows = build_protocol_rows(status)
    identity_rows = build_identity_rows(status)
    note_rows = build_validation_notes_rows(status, fi_curve_rows)
    manual_rows = build_manual_extraction_rows(status, fi_curve_rows)

    _write_csv(REFERENCE_DATA_DIR / PV_CRH_EPL_FSI_EPHYS_FILENAME, PV_CRH_EPL_FSI_EPHYS_COLUMNS, ephys_rows)
    _write_csv(REFERENCE_DATA_DIR / PV_CRH_EPL_FSI_FI_CURVE_FILENAME, PV_CRH_EPL_FSI_FI_CURVE_COLUMNS, fi_curve_rows)
    _write_csv(REFERENCE_DATA_DIR / PV_CRH_EPL_FSI_PROTOCOLS_FILENAME, PV_CRH_EPL_FSI_PROTOCOLS_COLUMNS, protocol_rows)
    _write_csv(REFERENCE_DATA_DIR / PV_CRH_EPL_FSI_IDENTITY_FILENAME, PV_CRH_EPL_FSI_IDENTITY_COLUMNS, identity_rows)
    _write_csv(REFERENCE_DATA_DIR / VALIDATION_NOTES_FILENAME, VALIDATION_NOTES_COLUMNS, note_rows)
    _write_csv(REFERENCE_DATA_DIR / NEEDS_MANUAL_EXTRACTION_FILENAME, NEEDS_MANUAL_EXTRACTION_COLUMNS, manual_rows)
    (REFERENCE_DATA_DIR / PV_CRH_EPL_FSI_EXTRACTION_README_FILENAME).write_text(build_readme_text(status, fi_curve_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
