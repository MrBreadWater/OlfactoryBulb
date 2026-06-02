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

assert payload["audit_id"] == "gc_intrinsic_validation"
assert any(item["check_id"] == "gc_intrinsic_protocol_executed" for item in payload["items"])
assert any(item["check_id"] == "gc_generic_fi_caveats" for item in payload["items"])

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
