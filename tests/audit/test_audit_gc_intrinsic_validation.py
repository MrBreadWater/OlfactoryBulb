"""Smoke tests for the declarative granule-cell intrinsic validation audit."""

from __future__ import annotations

import json
import subprocess
import sys


completed = subprocess.run(
    [
        sys.executable,
        "tools/run_audit.py",
        "gc_intrinsic_validation",
        "--cell-models",
        "GC1",
        "--jobs",
        "1",
        "--reference-gc-subtypes",
        "generic_or_unspecified",
        "--json",
    ],
    capture_output=True,
    text=True,
    check=False,
)

assert completed.returncode in {0, 1}, completed
payload = json.loads(completed.stdout)
protocol_item = next(item for item in payload["items"] if item["check_id"] == "gc_intrinsic_protocol_executed")

assert payload["audit_id"] == "gc_intrinsic_validation"
assert any(item["check_id"] == "gc_intrinsic_protocol_executed" for item in payload["items"])
assert protocol_item["series_visuals"][0]["title"] == "Model f-I curves"
assert protocol_item["series_visuals"][0]["row_sources"][0]["key"] == "fi_curve_rows"
assert protocol_item["series_visuals"][0]["row_sources"][0]["group_by"] == ["gc_subtype", "cell_name"]
assert "fi_curve_rows" in protocol_item["evidence"]
assert protocol_item["evidence"]["adp_enabled"] is True
assert protocol_item["evidence"]["adp_current_duration_ms"] == 1.0
assert protocol_item["evidence"]["adp_current_amplitude_nA"] == 1.0
assert protocol_item["evidence"]["adp_duration_mean_ms"] is not None
assert "adp_metric_rows" in protocol_item["evidence"]
assert len(protocol_item["evidence"]["adp_metric_rows"]) == 1
assert protocol_item["evidence"]["adp_metric_rows"][0]["cell_name"] == "GC1"
assert any(item["check_id"] == "gc_generic_fi_caveats" for item in payload["items"])
warn_items = [item for item in payload["items"] if item["status"] == "WARN"]
assert warn_items
assert all(str(item.get("status_reason", "")).strip() for item in warn_items)
pending_warn_items = [
    item for item in warn_items if item.get("validation_design_review_status") == "pending"
]
assert not pending_warn_items or any(str(item.get("status_reason", "")).strip() for item in pending_warn_items)

listed = subprocess.run([sys.executable, "tools/run_audit.py", "--list"], capture_output=True, text=True, check=False)
assert listed.returncode == 0, listed
assert "gc_intrinsic_validation" in listed.stdout

generic = subprocess.run(
    [
        sys.executable,
        "tools/run_audit.py",
        "gc_intrinsic_validation",
        "--cell-models",
        "GC1",
        "--jobs",
        "1",
        "--reference-gc-subtypes",
        "generic_or_unspecified",
        "--json",
    ],
    capture_output=True,
    text=True,
    check=False,
)
assert generic.returncode in {0, 1}, generic
assert json.loads(generic.stdout)["audit_id"] == "gc_intrinsic_validation"

print("audit_gc_intrinsic_validation smoke test: OK")
