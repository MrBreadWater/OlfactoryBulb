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
            evidence={
                "count": 3,
                "names": ["a", "b"],
                "__reference_annotations__": {"count": "reference: 4 +/- 1 counts from Demo Source (n=8)"},
            },
        ),
        AuditItem(
            check_id="demo_warn",
            status="WARN",
            title="TC CV_ISI item",
            criterion="TC CV_ISI should render cleanly.",
            description="CV_ISI should be expanded so the reader does not have to infer it.",
            acceptable="The tufted-cell value must exceed the mitral-cell value.",
            acceptable_basis="This simplified sample uses an ordering rule instead of a numeric range.",
            note="This is only a note.",
        ),
    ],
)

plain = format_report(sample_report, color=False)
assert "\033[" not in plain
assert "Summary" in plain
assert "[PASS] demo_pass" in plain
assert "Evidence" in plain
assert "Description" in plain
assert "Acceptable result" in plain
assert "How Acceptable Result Was Determined" in plain
assert "count: 3 (reference: 4 +/- 1 counts from Demo Source (n=8))" in plain
assert "Tufted cell coefficient of variation of interspike intervals item" in plain
assert "coefficient of variation of interspike intervals" in plain
assert "ordering rule instead of a numeric range" in plain

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
assert "How Acceptable Result Was Determined" in text_report.stdout

sample_fi_report = format_report(sample_report, color=False)
assert "Acceptable result" in sample_fi_report

print("audit_cli_output: OK")
