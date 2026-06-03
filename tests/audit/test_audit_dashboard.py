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
            evidence={"count": 0, "notes": ["Evidence caveat from protocol matching."]},
            human_review_status="accepted",
            human_review_note="Manually reviewed and accepted.",
            human_review_reviewer="human",
            note="Protocol caveat exists in this item.",
            group_id="audit_beta",
            group_title="Audit beta",
        ),
        AuditItem(
            check_id="audit_gamma.gamma_curve",
            status="PASS",
            title="Gamma curve",
            criterion="Gamma firing should be shown as a current-versus-rate series.",
            description="Description",
            acceptable="Acceptable",
            acceptable_basis="Configured",
            evidence={
                "currents_pA": [0.0, 50.0, 100.0, 150.0],
                "firing_rates_by_step_Hz": [0.0, 2.4, 5.8, 9.6],
                "reference_values_Hz": [0.0, 2.0, 5.0, 9.0],
                "model_values_Hz": [0.0, 1.8, 4.6, 8.7],
                "label": "Observed sweep",
            },
            group_id="audit_gamma",
            group_title="Audit gamma",
        ),
        AuditItem(
            check_id="audit_gamma.gamma_numeric",
            status="PASS",
            title="Gamma numeric",
            criterion="Gamma numeric evidence should render as a compact scalar comparison.",
            description="Description",
            acceptable="Acceptable",
            acceptable_basis="Configured",
            evidence={
                "spike_count": 12,
                "response_latency_ms": 18.6,
                "sample_count": 40,
                "label": "Observed scalar metrics",
            },
            group_id="audit_gamma",
            group_title="Audit gamma",
        ),
        AuditItem(
            check_id="audit_gamma.gamma_sequence",
            status="PASS",
            title="Gamma sequence",
            criterion="Gamma sequence evidence should render as a compact sparkline.",
            description="Description",
            acceptable="Acceptable",
            acceptable_basis="Configured",
            evidence={
                "trial_values": [0.2, 0.6, 0.9, 1.4, 1.6],
                "label": "Observed sequence",
            },
            group_id="audit_gamma",
            group_title="Audit gamma",
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
    assert len(report_payload["groups"]) == 3
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
    assert "item-body-warning" in html
    assert html.count("<div class='item-body-warning'>") == 1
    assert "Details" in html
    assert "evidence-lines" not in html
    assert "Warning" in html
    assert "warning-summary-text" in html
    assert html.count("data-interval-visual") == 1
    assert html.count("<div class='interval-legend'>") == 1
    assert "<div class='interval-metric-grid'>" not in html
    assert "<div class='interval-range-labels'>" not in html
    assert "Protocol caveat exists in this item." in html
    assert "Evidence caveat from protocol matching." in html
    assert "f-I curve" in html
    assert "data-series-graph" in html
    assert "Current (pA)" in html
    assert "Firing rate (Hz)" in html
    assert html.count("series-axis-tick") >= 4
    assert html.count("series-tick-label") >= 4
    assert html.count("<circle class='series-point'") >= 4
    assert "Observed sweep" in html
    assert "data-numeric-strip" in html
    assert "data-numeric-sparkline" in html
    assert "Numeric summary" in html
    assert "Numeric sequence" in html
    assert "Observed scalar metrics" in html
    assert "Observed sequence" in html
    gamma_index = html.index("audit_gamma.gamma_curve")
    assert html.index("series-graph-block", gamma_index) < html.index("data-item-detail-body", gamma_index)
    strip_index = html.index("data-numeric-strip")
    sparkline_index = html.index("data-numeric-sparkline")
    assert strip_index < sparkline_index
    summary_index = html.index("<div class='item-body-summary'>")
    legend_index = html.index("<div class='interval-legend'>", summary_index)
    track_index = html.index("<div class='interval-track'>", summary_index)
    detail_index = html.index("<div class='item-body item-detail-body'")
    assert track_index < legend_index
    assert legend_index < detail_index
    notes_index = html.index("<div class='item-body-notes'>")
    warning_index = html.index("<div class='item-body-warning'>")
    assert notes_index < warning_index
    assert "item-body-notes" in html
    assert "<ul class='item-notes-list'>" in html
    assert html.count("<li class='item-note'>") == 2
    assert "<span class='item-summary-text'>" not in html
    assert "status-reason-block" not in html
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
