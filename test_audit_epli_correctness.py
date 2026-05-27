"""Smoke test for the EPLI correctness audit script.

Run with:
    python test_audit_epli_correctness.py
"""

from __future__ import annotations

import json
import subprocess
import sys


completed = subprocess.run(
    [sys.executable, "tools/audit_epli_correctness.py", "--skip-neuron", "--json"],
    capture_output=True,
    text=True,
    check=False,
)

assert completed.returncode == 1, completed
payload = json.loads(completed.stdout)

assert payload["summary"]["PASS"] >= 1
assert payload["summary"]["WARN"] >= 1
assert payload["summary"]["FAIL"] >= 1
assert any(item["check_id"] == "canonical_epli_assets_present" for item in payload["items"])
assert any(item["check_id"] == "epli_target_pattern_specificity" for item in payload["items"])

print("audit_epli_correctness smoke test: OK")
