"""Smoke test for the generic reference-dataset status audit."""

from __future__ import annotations

import json
import subprocess
import sys


for dataset_id in ("pv_crh_epl_fsi", "granule_cells"):
    completed = subprocess.run(
        [sys.executable, "tools/run_audit.py", "reference_dataset_status", "--dataset-id", dataset_id, "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed
    payload = json.loads(completed.stdout)
    assert payload["audit_id"] == "reference_dataset_status"
    assert payload["summary"]["FAIL"] == 0
    assert any(item["check_id"] == "readme_loadable" for item in payload["items"])
    assert any(item["check_id"] == "notes_loadable" for item in payload["items"])

listed = subprocess.run(
    [sys.executable, "tools/run_audit.py", "--list"],
    capture_output=True,
    text=True,
    check=False,
)
assert listed.returncode == 0, listed
assert "reference_dataset_status" in listed.stdout

print("audit_reference_dataset_status: OK")
