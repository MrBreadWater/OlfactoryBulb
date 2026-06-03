"""Smoke tests for the maintained audit HTML dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from olfactorybulb.audit.core import AuditItem, AuditReport
from olfactorybulb.audit.dashboard import export_audit_dashboard


sample_report = AuditReport(
    audit_id="new_sweep",
    title="New sweep",
    items=[
        AuditItem(
            check_id="audit_alpha.alpha_pass",
            status="PASS",
            title="Alpha pass",
            criterion="Alpha should pass.",
            description="Description",
            acceptable="Acceptable",
            acceptable_basis="Configured",
            evidence={
                "MC_mean": 0.21,
                "reference_mean": 0.45,
                "reference_unit": "Hz",
                "accepted_low": 0.12,
                "accepted_high": 1.03,
                "accepted_interval_mode": "lognormal_sd",
                "accepted_interval_standard": "lognormal reference interval",
            },
            group_id="audit_alpha",
            group_title="Audit alpha",
        ),
        AuditItem(
            check_id="audit_beta.beta_warn",
            status="WARN",
            title="Beta warn",
            criterion="Beta should warn.",
            description="Description",
            acceptable="Acceptable",
            acceptable_basis="Configured",
            evidence={"count": 0},
            human_review_status="accepted",
            human_review_note="Manually reviewed and accepted.",
            human_review_reviewer="human",
            group_id="audit_beta",
            group_title="Audit beta",
        ),
    ],
)

with TemporaryDirectory() as tmp:
    output_dir = Path(tmp)
    manifest = export_audit_dashboard(sample_report, output_dir, refresh_endpoint="/__audit_refresh__")
    assert manifest["audit_id"] == "new_sweep"
    assert "manifest_revision" in manifest
    report_payload = json.loads((output_dir / "report.json").read_text())
    assert report_payload["audit_id"] == "new_sweep"
    assert len(report_payload["groups"]) == 2
    assert report_payload["groups"][0]["group_id"] == "audit_alpha"
    html = (output_dir / "index.html").read_text()
    assert "Audit groups" in html
    assert "Audit alpha" in html
    assert "Audit beta" in html
    assert "/__audit_refresh__" in html
    assert "Display controls" in html
    assert "Failures and warnings only" in html
    assert "Collapse all groups" in html
    assert "aria-pressed=\"false\"" in html
    assert "group-collapsed" in html
    assert ">Expand<" in html
    assert "Show detail items" not in html
    assert "group-link-label" in html
    assert "data-item-card" in html
    assert "data-item-toggle" in html
    assert "data-item-detail-body" in html
    assert "item-collapsed" in html
    assert "data-interval-visual" in html
    assert "item-compact-interval" in html
    assert "item-body-summary" in html
    assert "Reference interval" in html
    assert "Details" in html
    assert "evidence-lines" not in html
    assert "Why this is a warning" in html
    assert "Human review" not in html
    assert "reviewer: human" not in html

summary_report = AuditReport(
    audit_id="summary_toggle",
    title="Summary toggle",
    items=[
        AuditItem(
            check_id="summary_toggle.summary_item",
            status="PASS",
            title="Summary item",
            criterion="Criterion",
            description="Description",
            acceptable="Acceptable",
            acceptable_basis="Configured",
            evidence={"count": 1},
            group_id="summary_toggle",
            group_title="Summary toggle",
            detail_level="summary",
        ),
        AuditItem(
            check_id="summary_toggle.detail_item",
            status="PASS",
            title="Detail item",
            criterion="Criterion",
            description="Description",
            acceptable="Acceptable",
            acceptable_basis="Configured",
            evidence={"count": 2},
            group_id="summary_toggle",
            group_title="Summary toggle",
            detail_level="detail",
        ),
    ],
)

with TemporaryDirectory() as tmp:
    output_dir = Path(tmp)
    export_audit_dashboard(summary_report, output_dir)
    html = (output_dir / "index.html").read_text()
    assert "Show detail items" in html
    assert "data-item-toggle" in html

print("audit_dashboard: OK")
