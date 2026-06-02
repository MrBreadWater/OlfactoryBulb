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
            evidence={"count": 3},
            group_id="audit_alpha",
            group_title="Audit alpha",
        ),
        AuditItem(
            check_id="audit_beta.beta_fail",
            status="FAIL",
            title="Beta fail",
            criterion="Beta should fail.",
            description="Description",
            acceptable="Acceptable",
            acceptable_basis="Configured",
            evidence={"count": 0},
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
    assert "group-link-label" in html
    assert "data-item-card" in html

print("audit_dashboard: OK")
