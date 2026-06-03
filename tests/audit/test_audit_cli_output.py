"""Regression checks for human-readable audit CLI output."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import patch

from olfactorybulb.audit.cli import run_audit_by_id
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
            criterion_latex=r"\left|\ln\!\left(\bar{x}\right) - \mu_{\log}\right| \leq 2\sigma_{\log}",
            criterion_formulae=[
                r"\mu_{\log} = \ln(\mu_{\mathrm{ref}}) - \frac{1}{2}\sigma_{\log}^2",
                r"\sigma_{\log} = \sqrt{\ln\!\left(1 + \left(\frac{\sigma_{\mathrm{ref}}}{\mu_{\mathrm{ref}}}\right)^2\right)}",
            ],
            criterion_definitions=[
                {"symbol": r"\bar{x}", "definition": "observed group mean"},
                {"symbol": r"\mu_{\mathrm{ref}}", "definition": "uploaded arithmetic reference mean"},
                {"symbol": r"\sigma_{\mathrm{ref}}", "definition": "uploaded arithmetic reference standard deviation"},
                {"symbol": r"\mu_{\log}", "definition": "reconstructed log-space mean"},
                {"symbol": r"\sigma_{\log}", "definition": "reconstructed log-space standard deviation"},
            ],
            description="CV_ISI should be expanded so the reader does not have to infer it.",
            acceptable="The tufted-cell value must exceed the mitral-cell value.",
            acceptable_basis="This simplified sample uses an ordering rule instead of a numeric range.",
            note="This is only a note.",
            human_review_status="accepted",
            human_review_note="Manually reviewed and accepted.",
            human_review_reviewer="human",
        ),
    ],
)

grouped_report = AuditReport(
    audit_id="new_sweep",
    title="New sweep",
    items=[
        AuditItem(
            check_id="audit_alpha.alpha_pass",
            status="PASS",
            title="Alpha pass",
            criterion="Alpha should pass.",
            description="Grouped pass item.",
            acceptable="Pass.",
            acceptable_basis="Configured.",
            group_id="audit_alpha",
            group_title="Audit alpha",
        ),
        AuditItem(
            check_id="audit_beta.beta_warn",
            status="WARN",
            title="Beta warn",
            criterion="Beta should warn.",
            description="Grouped warning item.",
            acceptable="Warn.",
            acceptable_basis="Configured.",
            group_id="audit_beta",
            group_title="Audit beta",
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
assert "Warning" in plain
assert "Warning surfaced for an unresolved caveat or condition" in plain
assert r"\left|\ln\!\left(\bar{x}\right) - \mu_{\log}\right| \leq 2\sigma_{\log}" in plain
assert "Formulae" in plain
assert r"\mu_{\log} = \ln(\mu_{\mathrm{ref}}) - \frac{1}{2}\sigma_{\log}^2" in plain
assert r"\sigma_{\log} = \sqrt{\ln\!\left(1 + \left(\frac{\sigma_{\mathrm{ref}}}{\mu_{\mathrm{ref}}}\right)^2\right)}" in plain
assert "Definitions" in plain
assert "Observed group mean" in plain
assert "Reconstructed log-space standard deviation" in plain
assert "Why This Is A Warning" not in plain
assert "Human Review" not in plain
assert "Accepted | reviewer: human | Manually reviewed and accepted." not in plain

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
assert "default" in listed.stdout
assert "all" in listed.stdout
assert "burton_urban_fi" in listed.stdout

with patch("olfactorybulb.audit.cli._run_one_audit", return_value=sample_report) as run_one_mock:
    _ = run_audit_by_id("default", [])
    assert run_one_mock.call_args[0][0].audit_id == "repo_health"
    assert run_one_mock.call_args[0][1] == ["--profile", "maintained"]

with patch("olfactorybulb.audit.cli.run_new_sweep", return_value=grouped_report) as sweep_mock:
    report = run_audit_by_id("all", ["--skip-neuron"])
    sweep_mock.assert_called_once_with(["--skip-neuron"], progress_callback=None)
    assert report.audit_id == "new_sweep"

progress_updates: list[dict[str, object]] = []


def _capture_progress(update: dict[str, object]) -> None:
    progress_updates.append(dict(update))


with patch("olfactorybulb.audit.cli._run_one_audit", return_value=sample_report) as run_one_mock, patch(
    "olfactorybulb.audit.cli.iter_new_sweep_audit_specs",
    return_value=[
        type("Spec", (), {"audit_id": "alpha", "title": "Alpha audit", "description": "", "module_path": ""})(),
        type("Spec", (), {"audit_id": "beta", "title": "Beta audit", "description": "", "module_path": ""})(),
    ],
):
    progress_updates.clear()
    _ = run_audit_by_id("all", ["--skip-neuron"], progress_callback=_capture_progress)
    assert run_one_mock.call_count == 2
    assert progress_updates[0]["total"] == 2
    assert progress_updates[0]["current"] == 0
    assert progress_updates[1]["current_audit_id"] == "alpha"
    assert progress_updates[2]["current"] == 1
    assert progress_updates[-1]["phase"] == "done"
    assert progress_updates[-1]["current"] == 2

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

collapsed_grouped = format_report(grouped_report, color=False)
assert "Audit Groups" in collapsed_grouped
assert "[PASS] audit_alpha" in collapsed_grouped
assert "[WARN] audit_beta" in collapsed_grouped
assert "audit_alpha.alpha_pass" not in collapsed_grouped
assert "audit_beta.beta_warn" in collapsed_grouped

expanded_grouped = format_report(grouped_report, color=False, expand=True)
assert "audit_alpha.alpha_pass" in expanded_grouped
assert "audit_beta.beta_warn" in expanded_grouped

print("audit_cli_output: OK")
