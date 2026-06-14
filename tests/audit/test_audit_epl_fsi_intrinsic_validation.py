"""Smoke tests for the declarative EPL fast-spiking interneuron intrinsic validation audit."""

from __future__ import annotations

import json
import subprocess
import sys


completed = subprocess.run(
    [
        sys.executable,
        "tools/run_audit.py",
        "epl_fsi_intrinsic_validation",
        "--cell-models",
        "SyntheticEPL2026.PVCRH_FSI1",
        "--jobs",
        "1",
        "--json",
    ],
    capture_output=True,
    text=True,
    check=False,
)

assert completed.returncode in {0, 1}, completed
payload = json.loads(completed.stdout)
protocol_item = next(item for item in payload["items"] if item["check_id"] == "epl_fsi_protocol_executed")

assert payload["audit_id"] == "epl_fsi_intrinsic_validation"
assert any(item["check_id"] == "epl_fsi_protocol_executed" for item in payload["items"])
assert protocol_item["series_visuals"][0]["title"] == "Model f-I curves"
assert protocol_item["series_visuals"][0]["row_sources"][0]["key"] == "fi_curve_rows"
assert protocol_item["series_visuals"][0]["row_sources"][0]["group_by"] == ["cell_type", "cell_name"]
assert "fi_curve_rows" in protocol_item["evidence"]
assert protocol_item["evidence"]["adp_enabled"] is False
assert "adp_metric_rows" in protocol_item["evidence"]
assert len(protocol_item["evidence"]["adp_metric_rows"]) == 1
assert protocol_item["evidence"]["adp_metric_rows"][0]["adp_duration_ms"] is None
assert protocol_item["evidence"]["adp_metric_rows"][0]["adp_depth_mV"] is None
assert any(item["check_id"] == "epl_fsi_protocol_caveats" for item in payload["items"])
reference_curve_item = next(item for item in payload["items"] if item["check_id"] == "epl_fsi_reference_curve_match")
assert reference_curve_item["series_visuals"][0]["title"] == "Model vs target f-I curves"
assert [source["key"] for source in reference_curve_item["series_visuals"][0]["row_sources"]] == [
    "reference_fi_curve_rows",
    "model_fi_curve_rows",
]

listed = subprocess.run([sys.executable, "tools/run_audit.py", "--list"], capture_output=True, text=True, check=False)
assert listed.returncode == 0, listed
assert "epl_fsi_intrinsic_validation" in listed.stdout

generic = subprocess.run(
    [
        sys.executable,
        "tools/run_audit.py",
        "epl_fsi_intrinsic_validation",
        "--cell-models",
        "SyntheticEPL2026.PVCRH_FSI1",
        "--jobs",
        "1",
        "--json",
    ],
    capture_output=True,
    text=True,
    check=False,
)
assert generic.returncode in {0, 1}, generic
assert json.loads(generic.stdout)["audit_id"] == "epl_fsi_intrinsic_validation"

print("audit_epl_fsi_intrinsic_validation smoke test: OK")
