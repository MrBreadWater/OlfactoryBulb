"""Smoke tests for the maintained docs integrity audit."""

from __future__ import annotations

import json
import subprocess
import sys


completed = subprocess.run(
    [sys.executable, "tools/run_audit.py", "maintained_docs_integrity", "--json"],
    capture_output=True,
    text=True,
    check=False,
)
assert completed.returncode == 0, completed
payload = json.loads(completed.stdout)
assert payload["audit_id"] == "maintained_docs_integrity"
assert payload["summary"]["FAIL"] == 0, payload
check_ids = {item["check_id"] for item in payload["items"]}
assert "maintained_docs_avoid_removed_entrypoints" in check_ids
assert "docs_portal_exposes_maintained_markdown" in check_ids

print("audit_maintained_docs_integrity: OK")
