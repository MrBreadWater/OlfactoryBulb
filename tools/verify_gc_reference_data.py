"""Human-readable verification summary for the granule-cell reference bundle."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from olfactorybulb.audit.reference_data import (  # noqa: E402
    BU2014_MC_TC_PROTOCOL_ID,
    GC_EPHYS_FILENAME,
    GC_FI_CURVE_FILENAME,
    GC_IDENTITY_FILENAME,
    GC_MODULATION_FILENAME,
    GC_PROTOCOLS_FILENAME,
    GC_SYNAPTIC_LATENCY_FILENAME,
    GC_VALIDATION_NOTES_FILENAME,
    REFERENCE_DATA_DIR,
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
from olfactorybulb.audit.reference_notes import load_notes, notes_for_rows, render_notes  # noqa: E402


def main() -> int:
    notes = load_notes(REFERENCE_DATA_DIR / GC_VALIDATION_NOTES_FILENAME)
    protocol_rows = load_gc_protocol_rows()
    generic_rows = load_gc_ephys_rows()
    subtype_rows = load_gc_sgc_dgc_ephys_rows()
    fi_rows = load_gc_fi_curve_rows()
    subtype_fi_rows = load_gc_sgc_dgc_fi_curve_rows()
    identity_rows = load_gc_identity_rows()
    latency_rows = load_gc_synaptic_latency_rows()
    modulation_rows = load_gc_modulation_rows()
    legacy_fi_rows = [
        row for row in load_normalized_legacy_mc_tc_rows() if str(row.get("protocol_id", "")).strip() == BU2014_MC_TC_PROTOCOL_ID
    ]
    gc_protocol_context = [
        {"protocol_id": row["protocol_id"], "note_ids": "", "Property": "FI Protocol", "source": row["source"]}
        for row in protocol_rows
        if row["protocol_id"] in {"BU2015_GC_intrinsic_current_clamp", "GERAMITA2016_sGC_dGC_intrinsic_current_clamp"}
    ]
    notes_text = render_notes(
        notes_for_rows(legacy_fi_rows + gc_protocol_context + subtype_rows + modulation_rows, scope=None, notes=notes),
        format="plain",
    )

    print("Granule-cell reference bundle")
    print("=============================")
    print(f"Reference data directory: {REFERENCE_DATA_DIR}")
    print(f"- {GC_EPHYS_FILENAME}: {len(generic_rows)} rows")
    print(f"- {GC_FI_CURVE_FILENAME}: {len(fi_rows)} rows")
    print(f"- GC_sGC_dGC_ephys.csv: {len(subtype_rows)} rows")
    print(f"- GC_sGC_dGC_fI_curve.csv: {len(subtype_fi_rows)} rows")
    print(f"- {GC_PROTOCOLS_FILENAME}: {len(protocol_rows)} rows")
    print(f"- {GC_IDENTITY_FILENAME}: {len(identity_rows)} rows")
    print(f"- {GC_SYNAPTIC_LATENCY_FILENAME}: {len(latency_rows)} rows")
    print(f"- {GC_MODULATION_FILENAME}: {len(modulation_rows)} rows")
    print(f"- {GC_VALIDATION_NOTES_FILENAME}: loaded")
    print("")
    if notes_text:
        print(notes_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
