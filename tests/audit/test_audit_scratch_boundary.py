"""Smoke tests for the scratch-boundary audit."""

from __future__ import annotations

import json
import subprocess
import sys


completed = subprocess.run(
    [sys.executable, "tools/run_audit.py", "scratch_boundary", "--json"],
    capture_output=True,
    text=True,
    check=False,
)
assert completed.returncode == 0, completed
payload = json.loads(completed.stdout)
assert payload["audit_id"] == "scratch_boundary"
assert payload["summary"]["FAIL"] == 0, payload
check_ids = {item["check_id"] for item in payload["items"]}
assert "gitignore_contains_required_scratch_patterns" in check_ids
assert "scratch_paths_not_tracked" in check_ids
assert "root_test_files_moved_under_tests_package" in check_ids

print("audit_scratch_boundary: OK")
