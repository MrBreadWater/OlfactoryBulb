"""Smoke tests for the environment/install audit.

Run with:
    python test_audit_env_install.py
"""

from __future__ import annotations

import json
import subprocess
import sys

from olfactorybulb.audit.env_install import (
    _decode_leading_json,
    audit_mechanism_outputs,
    audit_nvhpc_transient_dependencies,
    audit_repo_layout,
)


repo_items = {item.check_id: item for item in audit_repo_layout()}
assert repo_items["repo_layout"].status == "PASS"

mechanism_items = {item.check_id: item for item in audit_mechanism_outputs()}
assert "mechanism_build_outputs" in mechanism_items

nvhpc_items = {item.check_id: item for item in audit_nvhpc_transient_dependencies()}
assert "nvhpc_transient_dependencies" in nvhpc_items

assert _decode_leading_json('{"ok": true}\nnumprocs=1\n') == {"ok": True}


def _assert_exit_matches_summary(run: subprocess.CompletedProcess[str]) -> dict:
    assert run.returncode in (0, 1), run
    payload = json.loads(run.stdout)
    expected_code = 1 if payload["summary"]["FAIL"] else 0
    assert run.returncode == expected_code
    return payload


skip_imports = subprocess.run(
    [sys.executable, "tools/audit_env_install.py", "--skip-imports", "--json"],
    capture_output=True,
    text=True,
    check=False,
)
payload = _assert_exit_matches_summary(skip_imports)
assert payload["audit_id"] == "env_install"
check_ids = {item["check_id"] for item in payload["items"]}
assert {
    "repo_layout",
    "python_environment",
    "activation_runtime_hooks",
    "command_line_tools",
    "mechanism_build_outputs",
    "nvhpc_transient_dependencies",
    "legacy_nrn_nmodl_path",
    "python_import_surface_skipped",
    "nrniv_launcher_smoke_skipped",
}.issubset(check_ids)

generic = subprocess.run(
    [sys.executable, "tools/run_audit.py", "env_install", "--skip-imports", "--json"],
    capture_output=True,
    text=True,
    check=False,
)
assert _assert_exit_matches_summary(generic)["audit_id"] == "env_install"

full = subprocess.run(
    [sys.executable, "tools/run_audit.py", "env_install", "--json"],
    capture_output=True,
    text=True,
    check=False,
)
payload = _assert_exit_matches_summary(full)
assert payload["audit_id"] == "env_install"
assert any(item["check_id"] == "python_import_surface" for item in payload["items"])

listed = subprocess.run(
    [sys.executable, "tools/run_audit.py", "--list"],
    capture_output=True,
    text=True,
    check=False,
)
assert listed.returncode == 0, listed
assert "env_install" in listed.stdout

print("audit_env_install smoke test: OK")
