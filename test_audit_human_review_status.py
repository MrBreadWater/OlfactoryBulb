"""Smoke tests for the declarative human-review coverage audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from olfactorybulb.audit.human_review_status import run as run_human_review_status


report = run_human_review_status(argparse.Namespace())
item_by_id = {item.check_id: item for item in report.items}

assert report.audit_id == "human_review_status"
assert item_by_id["reference_validation_review_status_coverage"].status == "PASS"
assert item_by_id["reference_validation_review_status_values"].status == "PASS"
assert item_by_id["reference_validation_pending_review_items"].status == "WARN"
assert item_by_id["reference_validation_provisional_items"].status == "WARN"
provisional_items = item_by_id["reference_validation_provisional_items"].evidence["provisional_items"]
assert any(entry.startswith("burton_urban_fi:") and entry.endswith(":AHP Duration") for entry in provisional_items)
assert any(
    entry.startswith("burton_urban_fi:") and entry.endswith(":Spiking Rate Accom. Time Constant")
    for entry in provisional_items
)

cli = subprocess.run(
    [sys.executable, "tools/run_audit.py", "human_review_status", "--json"],
    capture_output=True,
    text=True,
    check=False,
)
assert cli.returncode == 0, cli
payload = json.loads(cli.stdout)
assert payload["audit_id"] == "human_review_status"
assert payload["summary"]["FAIL"] == 0
assert payload["summary"]["WARN"] >= 1

print("audit_human_review_status: OK")
