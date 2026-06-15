"""Verification checks for the Burton 2014 MC/TC reference-data bundle."""

from __future__ import annotations

import pandas as pd

from olfactorybulb.audit.reference_data import (
    BURTON_MC_TC_DATASET_ID,
    BURTON_MC_TC_EPHYS_FILENAME,
    BURTON_MC_TC_EXTRACTION_README_FILENAME,
    BURTON_MC_TC_NEEDS_MANUAL_EXTRACTION_FILENAME,
    BURTON_MC_TC_PROTOCOLS_FILENAME,
    BU2014_MC_TC_PROTOCOL_ID,
    PV_CRH_EPL_FSI_EPHYS_COLUMNS,
    PV_CRH_EPL_FSI_PROTOCOLS_COLUMNS,
    REFERENCE_DATA_DIR,
    load_burton_mc_tc_ephys_rows,
    load_burton_mc_tc_protocol_rows,
)
from olfactorybulb.audit.reference_dataset_engine import write_reference_dataset_outputs
from olfactorybulb.audit.reference_sources import (
    REQUIRED_BURTON2014_MC_TC_SOURCE_IDS,
    local_source_path,
)


write_reference_dataset_outputs(dataset_id=BURTON_MC_TC_DATASET_ID)

for source_id in REQUIRED_BURTON2014_MC_TC_SOURCE_IDS:
    path = local_source_path(source_id, dataset_id=BURTON_MC_TC_DATASET_ID)
    assert path.exists(), source_id
    assert path.stat().st_size > 0, source_id
    assert path.suffix == ".csv", source_id

paths = {
    "ephys": REFERENCE_DATA_DIR / BURTON_MC_TC_EPHYS_FILENAME,
    "protocols": REFERENCE_DATA_DIR / BURTON_MC_TC_PROTOCOLS_FILENAME,
    "manual": REFERENCE_DATA_DIR / BURTON_MC_TC_NEEDS_MANUAL_EXTRACTION_FILENAME,
    "readme": REFERENCE_DATA_DIR / BURTON_MC_TC_EXTRACTION_README_FILENAME,
}
for path in paths.values():
    assert path.exists(), path

ephys_df = pd.read_csv(paths["ephys"])
protocols_df = pd.read_csv(paths["protocols"])
manual_df = pd.read_csv(paths["manual"])

assert list(ephys_df.columns) == PV_CRH_EPL_FSI_EPHYS_COLUMNS
assert list(protocols_df.columns) == PV_CRH_EPL_FSI_PROTOCOLS_COLUMNS
assert not ephys_df.empty
assert not protocols_df.empty
assert not manual_df.empty

assert set(ephys_df["cell_type"]) == {"MC", "TC"}
assert (ephys_df["marker_profile"] == "principal_cell").all()
assert ephys_df["source_file"].fillna("").ne("").all()
assert ephys_df["source_location"].fillna("").ne("").all()
assert (ephys_df["extraction_method"] == "legacy_csv").all()

fi_rows = ephys_df[ephys_df["include_in_fi_validation"].astype(str).str.lower() == "true"]
assert not fi_rows.empty
assert (fi_rows["protocol_id"] == BU2014_MC_TC_PROTOCOL_ID).all()
assert fi_rows["note_ids"].fillna("").str.contains("N_FI_PROTOCOL_DIFFERENCE").all()

non_fi_rows = ephys_df[ephys_df["include_in_fi_validation"].astype(str).str.lower() != "true"]
if len(non_fi_rows):
    assert non_fi_rows["protocol_id"].fillna("").eq("").all()

mc_tau = ephys_df[
    (ephys_df["cell_type"] == "MC")
    & (ephys_df["Property"] == "Membrane Time Constant")
    & (ephys_df["Source"] == "Burton & Urban (2014)")
].iloc[0]
assert float(mc_tau["mean"]) == 21.3
assert float(mc_tau["sd"]) == 9.4

tc_width = ephys_df[
    (ephys_df["cell_type"] == "TC")
    & (ephys_df["Property"] == "AP Half-Width")
    & (ephys_df["Source"] == "Burton & Urban (2014)")
].iloc[0]
assert float(tc_width["mean"]) == 0.87
assert float(tc_width["sd"]) == 0.1

protocol_row = protocols_df.iloc[0]
assert protocol_row["protocol_id"] == BU2014_MC_TC_PROTOCOL_ID
assert float(protocol_row["step_duration_ms"]) == 2000.0
assert float(protocol_row["current_start_pA"]) == 0.0
assert float(protocol_row["current_stop_pA"]) == 300.0

loaded_rows = load_burton_mc_tc_ephys_rows()
loaded_protocol_rows = load_burton_mc_tc_protocol_rows()
assert len(loaded_rows) == len(ephys_df)
assert len(loaded_protocol_rows) == len(protocols_df)

readme_text = paths["readme"].read_text()
assert "committed Burton & Urban 2014 mitral-cell and tufted-cell summary CSVs" in readme_text
assert "summary metrics only" in readme_text

print("burton_mc_tc_reference_data: OK")
