"""Smoke tests for the reference-dataset contract audit."""

from __future__ import annotations

import json
import subprocess
import sys


for dataset_id in ("pv_crh_epl_fsi", "granule_cells"):
    completed = subprocess.run(
        [sys.executable, "tools/run_audit.py", "reference_dataset_contracts", "--dataset-id", dataset_id, "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed
    payload = json.loads(completed.stdout)
    assert payload["audit_id"] == "reference_dataset_contracts"
    assert payload["summary"]["FAIL"] == 0, payload
    check_ids = {item["check_id"] for item in payload["items"]}
    assert any(check_id.endswith("_schema_columns_match") for check_id in check_ids)
    assert any(check_id.endswith("_numeric_columns_parse") for check_id in check_ids)


listed = subprocess.run(
    [sys.executable, "tools/run_audit.py", "--list", "--no-color"],
    capture_output=True,
    text=True,
    check=False,
)
assert listed.returncode == 0, listed
assert "reference_dataset_contracts" in listed.stdout

print("audit_reference_dataset_contracts: OK")
