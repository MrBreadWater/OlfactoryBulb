"""Verification checks for the PV/CRH-overlap EPL fast-spiking reference-data pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from olfactorybulb.audit.burton_urban_fi import BurtonUrbanProtocol, build_validation_items
from olfactorybulb.audit.core import AuditReport, format_report
from olfactorybulb.audit.reference_data import (
    BMU2024_EPL_FSI_PROTOCOL_ID,
    BU2014_MC_TC_PROTOCOL_ID,
    FI_PROTOCOL_DIFFERENCE_NOTE_ID,
    LEGACY_MC_TC_EPHYS_FILENAMES,
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
    load_normalized_legacy_mc_tc_rows,
    load_pv_crh_epl_fsi_protocol_rows,
)
from olfactorybulb.audit.reference_notes import notes_for_rows, render_notes
from olfactorybulb.audit.reference_sources import (
    BURTON2024_S15_DATA_SOURCE_ID,
    BURTON2024_S8_DATA_SOURCE_ID,
    local_source_path,
    stable_source_url,
)
from olfactorybulb.audit.reference_dataset_engine import write_reference_dataset_outputs


write_reference_dataset_outputs(dataset_id="pv_crh_epl_fsi")

paths = {
    "ephys": REFERENCE_DATA_DIR / PV_CRH_EPL_FSI_EPHYS_FILENAME,
    "fi_curve": REFERENCE_DATA_DIR / PV_CRH_EPL_FSI_FI_CURVE_FILENAME,
    "protocols": REFERENCE_DATA_DIR / PV_CRH_EPL_FSI_PROTOCOLS_FILENAME,
    "identity": REFERENCE_DATA_DIR / PV_CRH_EPL_FSI_IDENTITY_FILENAME,
    "notes": REFERENCE_DATA_DIR / VALIDATION_NOTES_FILENAME,
    "manual": REFERENCE_DATA_DIR / NEEDS_MANUAL_EXTRACTION_FILENAME,
    "readme": REFERENCE_DATA_DIR / PV_CRH_EPL_FSI_EXTRACTION_README_FILENAME,
}
for path in paths.values():
    assert path.exists(), path

ephys_df = pd.read_csv(paths["ephys"])
fi_curve_df = pd.read_csv(paths["fi_curve"])
protocols_df = pd.read_csv(paths["protocols"])
identity_df = pd.read_csv(paths["identity"])
notes_df = pd.read_csv(paths["notes"])
manual_df = pd.read_csv(paths["manual"])

assert list(ephys_df.columns) == PV_CRH_EPL_FSI_EPHYS_COLUMNS
assert list(fi_curve_df.columns) == PV_CRH_EPL_FSI_FI_CURVE_COLUMNS
assert list(protocols_df.columns) == PV_CRH_EPL_FSI_PROTOCOLS_COLUMNS
assert list(identity_df.columns) == PV_CRH_EPL_FSI_IDENTITY_COLUMNS
assert list(notes_df.columns) == VALIDATION_NOTES_COLUMNS

assert not ephys_df.empty
assert not fi_curve_df.empty
assert not protocols_df.empty
assert not identity_df.empty
assert not notes_df.empty
assert not manual_df.empty

fi_summary_rows = ephys_df[ephys_df["data_kind"] == "fI_summary_metric"]
assert not fi_summary_rows.empty
assert fi_summary_rows["protocol_id"].fillna("").ne("").all()
assert fi_summary_rows["source_file"].fillna("").ne("").all()
assert fi_summary_rows["source_location"].fillna("").ne("").all()
assert fi_summary_rows["source_url"].fillna("").ne("").all()

for df, name in ((ephys_df, "ephys"), (fi_curve_df, "fi_curve"), (identity_df, "identity")):
    assert df["source_file"].fillna("").ne("").all(), name
    assert df["source_location"].fillna("").ne("").all(), name
    assert df["source_url"].fillna("").ne("").all(), name

numeric_columns = ("mean", "sd", "sem", "n")
for df in (ephys_df, identity_df):
    for column in numeric_columns:
        populated = df[column].dropna()
        if len(populated):
            converted = pd.to_numeric(populated, errors="coerce")
            assert converted.notna().all(), (column, populated.tolist())

for column in ("q_low", "q_high"):
    populated = ephys_df[column].dropna()
    if len(populated):
        converted = pd.to_numeric(populated, errors="coerce")
        converted = converted[populated.astype(str).str.strip().ne("")]
        assert converted.notna().all(), (column, populated.tolist())

digitized_rows = pd.concat(
    [
        ephys_df[ephys_df["extraction_method"] == "figure_digitized"],
        fi_curve_df[fi_curve_df["extraction_method"] == "figure_digitized"],
        identity_df[identity_df["extraction_method"] == "figure_digitized"],
    ],
    ignore_index=True,
)
if len(digitized_rows):
    assert (digitized_rows["extraction_method"] == "figure_digitized").all()

assert all(";" not in str(source) for source in ephys_df["Source"])
assert all(";" not in str(cell_type) for cell_type in ephys_df["cell_type"])

legacy_columns = ["Property", "mean +/- sd", "n", "Source", "Notes"]
for filename in LEGACY_MC_TC_EPHYS_FILENAMES.values():
    legacy_df = pd.read_csv(REFERENCE_DATA_DIR / filename)
    assert list(legacy_df.columns) == legacy_columns

s8_path = local_source_path(BURTON2024_S8_DATA_SOURCE_ID)
assert s8_path.exists() and s8_path.stat().st_size > 0
s8_rows = fi_curve_df[fi_curve_df["source_file"] == s8_path.name]
assert not s8_rows.empty
assert (s8_rows["protocol_id"] == BMU2024_EPL_FSI_PROTOCOL_ID).all()
assert (s8_rows["source_url"] == stable_source_url(BURTON2024_S8_DATA_SOURCE_ID)).all()
assert (s8_rows["sample_scope"] == "example_cell").all()
assert s8_rows["current_pA"].between(50.0, 600.0).all()

s15_path = local_source_path(BURTON2024_S15_DATA_SOURCE_ID)
assert s15_path.exists() and s15_path.stat().st_size > 0
s15_rows = ephys_df[ephys_df["source_file"] == s15_path.name]
assert not s15_rows.empty
assert (s15_rows["Source"] == "Burton, Malyshko & Urban (2024)").all()
assert (s15_rows["source_url"] == stable_source_url(BURTON2024_S15_DATA_SOURCE_ID)).all()

quantile_properties = {
    "Membrane Resting Voltage",
    "Input Resistance",
    "Membrane Time Constant",
    "Capacitance",
    "Spontaneous Firing Rate",
    "AP Amplitude",
    "AP Half-Width",
    "AP Threshold",
    "AHP Amplitude",
    "AHP Duration",
    "Rheobase Current",
    "FI Curve Slope",
    "Max FI Rate",
    "ISI Coefficient of Variation",
    "Spiking Rate Accommodation",
}
quantile_rows = s15_rows[s15_rows["Property"].isin(quantile_properties)]
assert not quantile_rows.empty
assert quantile_rows["q_low"].fillna("").astype(str).str.strip().ne("").all()
assert quantile_rows["q_high"].fillna("").astype(str).str.strip().ne("").all()
assert (quantile_rows["q_low_label"] == "5th percentile").all()
assert (quantile_rows["q_high_label"] == "95th percentile").all()

normalized_legacy_rows = [
    row for row in load_normalized_legacy_mc_tc_rows() if row["protocol_id"] == BU2014_MC_TC_PROTOCOL_ID
]
bmw_protocol_context = [
    {
        "protocol_id": row["protocol_id"],
        "note_ids": "",
        "Property": "FI Protocol",
        "source": row["source"],
    }
    for row in load_pv_crh_epl_fsi_protocol_rows()
    if row["protocol_id"] == BMU2024_EPL_FSI_PROTOCOL_ID
]
matched_notes = notes_for_rows(normalized_legacy_rows + bmw_protocol_context, scope="fI_validation")
assert FI_PROTOCOL_DIFFERENCE_NOTE_ID in {note.note_id for note in matched_notes}
rendered_notes = render_notes(matched_notes, format="plain")
assert "Notes / protocol caveats" in rendered_notes
assert "MC/TC and EPL-FSI f-I validation targets use different current-injection protocols." in rendered_notes

fixture_metrics = [
    {
        "cell_name": "MC1",
        "cell_type": "MC",
        "resting_potential_mV": -66.0,
        "bias_current_pA": 100.0,
        "zero_step_rate_Hz": 0.0,
        "membrane_time_constant_ms": 21.3,
        "cell_capacitance_pF": 236.4,
        "sag_amplitude_mV": -2.0,
        "rebound_potential_presence": 0.0,
        "rheobase_pA": 100.0,
        "spike_latency_ms": 500.0,
        "peak_rate_Hz": 60.0,
        "fi_gain_Hz_per_50pA": 10.0,
        "spike_accommodation_hz": -9.0,
        "spike_accommodation_time_constant_ms": 398.0,
        "cv_isi": 0.4,
        "cv_isi_step_pA": 150.0,
        "cv_isi_mean_rate_Hz": 20.0,
        "input_resistance_MOhm": 100.0,
        "AP_onset_mV": -42.0,
        "Amplitude_mV": 76.0,
        "FWHM_ms": 1.1,
        "Rise_slope_mV_per_ms": 240.0,
        "Fall_slope_mV_per_ms": -70.0,
        "AHP_amplitude_mV": 15.0,
        "T_AHP50_ms": 60.0,
    },
    {
        "cell_name": "TC1",
        "cell_type": "TC",
        "resting_potential_mV": -64.0,
        "bias_current_pA": 150.0,
        "zero_step_rate_Hz": 0.0,
        "membrane_time_constant_ms": 18.8,
        "cell_capacitance_pF": 188.8,
        "sag_amplitude_mV": -4.4,
        "rebound_potential_presence": 1.0,
        "rheobase_pA": 90.0,
        "spike_latency_ms": 400.0,
        "peak_rate_Hz": 120.0,
        "fi_gain_Hz_per_50pA": 20.0,
        "spike_accommodation_hz": -20.0,
        "spike_accommodation_time_constant_ms": 585.0,
        "cv_isi": 0.8,
        "cv_isi_step_pA": 100.0,
        "cv_isi_mean_rate_Hz": 20.0,
        "input_resistance_MOhm": 110.0,
        "AP_onset_mV": -42.5,
        "Amplitude_mV": 72.0,
        "FWHM_ms": 0.9,
        "Rise_slope_mV_per_ms": 200.0,
        "Fall_slope_mV_per_ms": -90.0,
        "AHP_amplitude_mV": 17.0,
        "T_AHP50_ms": 20.0,
    },
]

validation_items = build_validation_items(fixture_metrics, BurtonUrbanProtocol())
item_by_id = {item.check_id: item for item in validation_items}
assert item_by_id["fi_protocol_caveats"].status == "WARN"
assert FI_PROTOCOL_DIFFERENCE_NOTE_ID in item_by_id["fi_protocol_caveats"].evidence["note_ids"]
rendered_report = format_report(
    AuditReport(audit_id="burton_urban_fi", title="Burton & Urban f-I validation audit", items=validation_items),
    color=False,
)
assert "Notes / protocol caveats" in rendered_report
assert "MC/TC and EPL-FSI f-I validation targets use different current-injection protocols." in rendered_report

readme_text = paths["readme"].read_text()
assert "current-rate" in readme_text
assert "fi_curve" in readme_text

print("pv_crh_epl_fsi_reference_data: OK")
