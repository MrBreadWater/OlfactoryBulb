"""Verification checks for the granule-cell reference-data pipeline."""

from __future__ import annotations

import pandas as pd

from olfactorybulb.audit.reference_data import (
    BU2014_MC_TC_PROTOCOL_ID,
    GC_EPHYS_COLUMNS,
    GC_EPHYS_FILENAME,
    GC_EXTRACTION_README_FILENAME,
    GC_FI_CURVE_COLUMNS,
    GC_FI_CURVE_FILENAME,
    GC_IDENTITY_COLUMNS,
    GC_IDENTITY_FILENAME,
    GC_MODULATION_FILENAME,
    GC_NEEDS_MANUAL_EXTRACTION_FILENAME,
    GC_PROTOCOLS_COLUMNS,
    GC_PROTOCOLS_FILENAME,
    GC_SGC_DGC_EPHYS_FILENAME,
    GC_SGC_DGC_FI_CURVE_FILENAME,
    GC_SYNAPTIC_LATENCY_FILENAME,
    GC_VALIDATION_NOTES_FILENAME,
    LEGACY_MC_TC_EPHYS_FILENAMES,
    REFERENCE_DATA_DIR,
    VALIDATION_NOTES_COLUMNS,
    csv_rows,
    load_gc_ephys_rows,
    load_gc_fi_curve_rows,
    load_gc_identity_rows,
    load_gc_modulation_rows,
    load_gc_protocol_rows,
    load_gc_sgc_dgc_ephys_rows,
    load_gc_sgc_dgc_fi_curve_rows,
    load_gc_synaptic_latency_rows,
    load_normalized_legacy_mc_tc_rows,
)
from olfactorybulb.audit.reference_notes import load_notes, notes_for_rows, render_notes
from tools.extract_gc_reference_data import main as extract_reference_data


extract_reference_data()

paths = {
    "ephys": REFERENCE_DATA_DIR / GC_EPHYS_FILENAME,
    "fi_curve": REFERENCE_DATA_DIR / GC_FI_CURVE_FILENAME,
    "subtype_ephys": REFERENCE_DATA_DIR / GC_SGC_DGC_EPHYS_FILENAME,
    "subtype_fi_curve": REFERENCE_DATA_DIR / GC_SGC_DGC_FI_CURVE_FILENAME,
    "protocols": REFERENCE_DATA_DIR / GC_PROTOCOLS_FILENAME,
    "identity": REFERENCE_DATA_DIR / GC_IDENTITY_FILENAME,
    "latency": REFERENCE_DATA_DIR / GC_SYNAPTIC_LATENCY_FILENAME,
    "modulation": REFERENCE_DATA_DIR / GC_MODULATION_FILENAME,
    "notes": REFERENCE_DATA_DIR / GC_VALIDATION_NOTES_FILENAME,
    "manual": REFERENCE_DATA_DIR / GC_NEEDS_MANUAL_EXTRACTION_FILENAME,
    "readme": REFERENCE_DATA_DIR / GC_EXTRACTION_README_FILENAME,
}
for path in paths.values():
    assert path.exists(), path

ephys_df = pd.read_csv(paths["ephys"])
fi_curve_df = pd.read_csv(paths["fi_curve"])
subtype_ephys_df = pd.read_csv(paths["subtype_ephys"])
subtype_fi_curve_df = pd.read_csv(paths["subtype_fi_curve"])
protocols_df = pd.read_csv(paths["protocols"])
identity_df = pd.read_csv(paths["identity"])
latency_df = pd.read_csv(paths["latency"])
modulation_df = pd.read_csv(paths["modulation"])
notes_df = pd.read_csv(paths["notes"])
manual_df = pd.read_csv(paths["manual"])

assert list(ephys_df.columns) == GC_EPHYS_COLUMNS
assert list(fi_curve_df.columns) == GC_FI_CURVE_COLUMNS
assert list(subtype_ephys_df.columns) == GC_EPHYS_COLUMNS
assert list(subtype_fi_curve_df.columns) == GC_FI_CURVE_COLUMNS
assert list(protocols_df.columns) == GC_PROTOCOLS_COLUMNS
assert list(identity_df.columns) == GC_IDENTITY_COLUMNS
assert list(latency_df.columns) == GC_EPHYS_COLUMNS
assert list(modulation_df.columns) == GC_EPHYS_COLUMNS
assert list(notes_df.columns) == VALIDATION_NOTES_COLUMNS

assert not ephys_df.empty
assert not subtype_ephys_df.empty
assert not protocols_df.empty
assert not identity_df.empty
assert not latency_df.empty
assert not modulation_df.empty
assert not notes_df.empty
assert not manual_df.empty

for df_name, df in (
    ("ephys", ephys_df),
    ("subtype_ephys", subtype_ephys_df),
    ("identity", identity_df),
    ("latency", latency_df),
    ("modulation", modulation_df),
):
    assert df["source_file"].fillna("").ne("").all(), df_name
    assert df["source_location"].fillna("").ne("").all(), df_name
    assert df["source_url"].fillna("").ne("").all(), df_name

for df_name, df in (("fi_curve", fi_curve_df), ("subtype_fi_curve", subtype_fi_curve_df)):
    if len(df):
        assert df["protocol_id"].fillna("").ne("").all(), df_name
        assert df["note_ids"].fillna("").ne("").all(), df_name
        assert df["source_file"].fillna("").ne("").all(), df_name
        assert df["source_location"].fillna("").ne("").all(), df_name
        assert df["source_url"].fillna("").ne("").all(), df_name

fi_summary_rows = pd.concat(
    [
        ephys_df[ephys_df["include_in_fi_validation"].astype(str).str.lower() == "true"],
        subtype_ephys_df[subtype_ephys_df["include_in_fi_validation"].astype(str).str.lower() == "true"],
    ],
    ignore_index=True,
)
assert not fi_summary_rows.empty
assert fi_summary_rows["protocol_id"].fillna("").ne("").all()
assert fi_summary_rows["note_ids"].fillna("").ne("").all()

for df in (ephys_df, subtype_ephys_df, identity_df, latency_df, modulation_df):
    for column in ("mean", "sd", "sem", "q_low", "q_high", "n"):
        populated = df[column].dropna()
        if len(populated):
            converted = pd.to_numeric(populated, errors="coerce")
            # allow blanks from qualitative rows that survived dropna as empty strings
            converted = converted[populated.astype(str).str.strip().ne("")]
            assert converted.notna().all(), (column, populated.tolist())

notes = load_notes(paths["notes"])
legacy_fi_rows = [
    row for row in load_normalized_legacy_mc_tc_rows() if str(row.get("protocol_id", "")).strip() == BU2014_MC_TC_PROTOCOL_ID
]
gc_protocol_context = [
    {"protocol_id": row["protocol_id"], "note_ids": "", "Property": "FI Protocol", "source": row["source"]}
    for row in load_gc_protocol_rows()
    if row["protocol_id"] in {"BU2015_GC_intrinsic_current_clamp", "GERAMITA2016_sGC_dGC_intrinsic_current_clamp"}
]
combined_fi_notes = notes_for_rows(legacy_fi_rows + gc_protocol_context, scope="fI_validation", notes=notes)
combined_fi_note_ids = {note.note_id for note in combined_fi_notes}
assert "N_GC_PROTOCOL_DIFFERENCE" in combined_fi_note_ids
assert "Notes / protocol caveats" in render_notes(combined_fi_notes, format="plain")

sgc_row = next(row for row in load_gc_sgc_dgc_ephys_rows() if row["gc_subtype"] == "sGC")
dgc_row = next(row for row in load_gc_sgc_dgc_ephys_rows() if row["gc_subtype"] == "dGC")
subtype_notes = notes_for_rows([sgc_row, dgc_row], scope="granule_cell_validation", notes=notes)
assert "N_GC_SUBTYPE_DO_NOT_POOL" in {note.note_id for note in subtype_notes}

baseline_row = next(row for row in load_gc_ephys_rows() if row["gc_subtype"] == "generic_or_unspecified")
modulated_row = load_gc_modulation_rows()[0]
modulation_notes = notes_for_rows([baseline_row, modulated_row], scope="granule_cell_validation", notes=notes)
assert "N_GC_MODULATION_STATE" in {note.note_id for note in modulation_notes}

assert set(subtype_ephys_df["gc_subtype"]) <= {"sGC", "dGC"}
assert (modulation_df["include_in_validation"].astype(str).str.lower() == "false").all()

digitized_rows = pd.concat(
    [
        fi_curve_df[fi_curve_df["extraction_method"] == "figure_digitized"],
        subtype_fi_curve_df[subtype_fi_curve_df["extraction_method"] == "figure_digitized"],
        ephys_df[ephys_df["extraction_method"] == "figure_digitized"],
        subtype_ephys_df[subtype_ephys_df["extraction_method"] == "figure_digitized"],
    ],
    ignore_index=True,
)
if len(digitized_rows):
    assert digitized_rows["source_location"].fillna("").ne("").all()
    assert digitized_rows["note_ids"].fillna("").str.contains("N_GC_FI_DIGITIZATION").all()

for filename in LEGACY_MC_TC_EPHYS_FILENAMES.values():
    legacy_rows = csv_rows(REFERENCE_DATA_DIR / filename)
    assert legacy_rows, filename

readme_text = paths["readme"].read_text()
assert "Burton & Urban 2015 contributes" in readme_text
assert "Geramita, Burton & Urban 2016 contributes" in readme_text
assert "No exact generic-GC or sGC/dGC current-vs-rate point table has been extracted yet." in readme_text

print("gc_reference_data: OK")
