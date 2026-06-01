"""Human-readable verification summary for the PV/CRH-overlap EPL-FSI reference bundle."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from olfactorybulb.audit.reference_data import (  # noqa: E402
    BMU2024_EPL_FSI_PROTOCOL_ID,
    BU2014_MC_TC_PROTOCOL_ID,
    PV_CRH_EPL_FSI_EPHYS_FILENAME,
    PV_CRH_EPL_FSI_FI_CURVE_FILENAME,
    PV_CRH_EPL_FSI_IDENTITY_FILENAME,
    PV_CRH_EPL_FSI_PROTOCOLS_FILENAME,
    REFERENCE_DATA_DIR,
    VALIDATION_NOTES_FILENAME,
    load_normalized_legacy_mc_tc_rows,
    load_pv_crh_epl_fsi_ephys_rows,
    load_pv_crh_epl_fsi_fi_curve_rows,
    load_pv_crh_epl_fsi_identity_rows,
    load_pv_crh_epl_fsi_protocol_rows,
)
from olfactorybulb.audit.reference_notes import notes_for_rows, render_notes  # noqa: E402


def main() -> int:
    ephys_rows = load_pv_crh_epl_fsi_ephys_rows()
    fi_rows = load_pv_crh_epl_fsi_fi_curve_rows()
    identity_rows = load_pv_crh_epl_fsi_identity_rows()
    protocol_rows = load_pv_crh_epl_fsi_protocol_rows()
    legacy_fi_rows = [
        row for row in load_normalized_legacy_mc_tc_rows() if str(row.get("protocol_id", "")).strip() == BU2014_MC_TC_PROTOCOL_ID
    ]
    bmu_protocol_context = [
        {"protocol_id": row["protocol_id"], "note_ids": "", "Property": "FI Protocol", "source": row["source"]}
        for row in protocol_rows
        if row["protocol_id"] == BMU2024_EPL_FSI_PROTOCOL_ID
    ]
    notes_text = render_notes(notes_for_rows(legacy_fi_rows + bmu_protocol_context, scope="fI_validation"), format="plain")

    print("PV/CRH-overlap EPL-FSI reference bundle")
    print("========================================")
    print(f"Reference data directory: {REFERENCE_DATA_DIR}")
    print(f"- {PV_CRH_EPL_FSI_EPHYS_FILENAME}: {len(ephys_rows)} rows")
    print(f"- {PV_CRH_EPL_FSI_FI_CURVE_FILENAME}: {len(fi_rows)} rows")
    print(f"- {PV_CRH_EPL_FSI_PROTOCOLS_FILENAME}: {len(protocol_rows)} rows")
    print(f"- {PV_CRH_EPL_FSI_IDENTITY_FILENAME}: {len(identity_rows)} rows")
    print(f"- {VALIDATION_NOTES_FILENAME}: loaded")
    print("")
    if notes_text:
        print(notes_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
