"""Regression checks for human-readable audit CLI output."""

from __future__ import annotations

import subprocess
import sys

from olfactorybulb.audit.core import AuditItem, AuditReport, format_report


sample_report = AuditReport(
    audit_id="demo",
    title="Demo audit",
    items=[
        AuditItem(
            check_id="demo_pass",
            status="PASS",
            title="Pass item",
            criterion="A passing criterion should render cleanly.",
            evidence={"count": 3, "names": ["a", "b"]},
        ),
        AuditItem(
            check_id="demo_warn",
            status="WARN",
            title="Warn item",
            criterion="A warning criterion should render cleanly.",
            note="This is only a note.",
        ),
    ],
)

plain = format_report(sample_report, color=False)
assert "\033[" not in plain
assert "Summary" in plain
assert "[PASS] demo_pass" in plain
assert "Evidence" in plain
assert "\"count\": 3" in plain

colored = format_report(sample_report, color=True)
assert "\033[" in colored

listed = subprocess.run(
    [sys.executable, "tools/run_audit.py", "--list", "--no-color"],
    capture_output=True,
    text=True,
    check=False,
)
assert listed.returncode == 0, listed
assert "\033[" not in listed.stdout
assert "Available audits" in listed.stdout
assert "burton_urban_fi" in listed.stdout

text_report = subprocess.run(
    [sys.executable, "tools/run_audit.py", "burton_urban_fi", "--skip-neuron", "--no-color"],
    capture_output=True,
    text=True,
    check=False,
)
assert text_report.returncode == 0, text_report
assert "\033[" not in text_report.stdout
assert "Summary" in text_report.stdout
assert "[WARN] burton_urban_fi_skipped" in text_report.stdout

print("audit_cli_output: OK")
