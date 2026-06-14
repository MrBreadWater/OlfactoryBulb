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
curve_item = next(item for item in payload["items"] if item["check_id"] == "epl_fsi_reference_curve_match")

assert payload["audit_id"] == "epl_fsi_intrinsic_validation"
assert any(item["check_id"] == "epl_fsi_protocol_executed" for item in payload["items"])
assert protocol_item["series_visuals"][0]["keys"] == ["fi_curve_rows"]
assert "fi_curve_rows" in protocol_item["evidence"]
assert any(item["check_id"] == "epl_fsi_protocol_caveats" for item in payload["items"])
assert any(item["check_id"] == "epl_fsi_reference_curve_match" for item in payload["items"])
assert curve_item["evidence"]["score_family"] == "residual_only"
assert curve_item["evidence"]["alignment_policy"] == "exact_transformed_x"
assert curve_item["evidence"]["distribution_kind"] == "empirical_by_x"
assert curve_item["evidence"]["reference_provenance"]["sources"] == ["Burton, Malyshko & Urban (2024)"]
assert curve_item["evidence"]["model_provenance"]["protocol_context"]["cell_models"] == ["PVCRH_FSI1"]

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
