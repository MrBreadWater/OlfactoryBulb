"""Smoke test for the audit submodule and EPLI audit entrypoint.

Run with:
    python test_audit_epli_correctness.py
"""

from __future__ import annotations

import json
import subprocess
import sys


completed = subprocess.run([sys.executable, "tools/run_audit.py", "epli_correctness", "--skip-neuron", "--json"], capture_output=True, text=True, check=False)

assert completed.returncode == 1, completed
payload = json.loads(completed.stdout)

assert payload["summary"]["PASS"] >= 1
assert payload["summary"]["WARN"] >= 1
assert payload["summary"]["FAIL"] >= 1
assert payload["audit_id"] == "epli_correctness"
assert any(item["check_id"] == "canonical_epli_assets_present" for item in payload["items"])
assert any(item["check_id"] == "epli_target_pattern_specificity" for item in payload["items"])

listed = subprocess.run([sys.executable, "tools/run_audit.py", "--list"], capture_output=True, text=True, check=False)
assert listed.returncode == 0, listed
assert "epli_correctness" in listed.stdout

generic = subprocess.run([sys.executable, "tools/run_audit.py", "epli_correctness", "--skip-neuron", "--json"], capture_output=True, text=True, check=False)
assert generic.returncode == 1, generic
assert json.loads(generic.stdout)["audit_id"] == "epli_correctness"

print("audit_epli_correctness smoke test: OK")
