"""Smoke tests for the maintained audit HTML dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from olfactorybulb.audit import series_visual_spec
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
            validation_design_review_status="approved",
            validation_design_review_note="Protocol equivalence reviewed and approved.",
            validation_design_review_reviewer="qualified_domain_expert",
            validation_design_review_required_expertise="cellular_electrophysiology",
            validation_design_review_focus="protocol_equivalence",
            note="Protocol caveat exists in this item.",
            group_id="audit_beta",
            group_title="Audit beta",
        ),
        AuditItem(
            check_id="audit_alpha.alpha_math",
            status="PASS",
            title="Alpha math",
            criterion="The observed mean should stay within the accepted interval.",
            criterion_latex=r"\left|\bar{x} - \mu_{\mathrm{ref}}\right| \leq 2\sigma_{\mathrm{ref}}",
            criterion_definitions=[
                {"symbol": r"\bar{x}", "definition": "observed group mean"},
                {"symbol": r"\mu_{\mathrm{ref}}", "definition": "uploaded reference mean"},
                {"symbol": r"\sigma_{\mathrm{ref}}", "definition": "uploaded reference standard deviation"},
            ],
            description="Description",
            acceptable="Acceptable",
            acceptable_basis="Configured",
            evidence={"count": 4},
            group_id="audit_alpha",
            group_title="Audit alpha",
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
            series_visuals=[
                series_visual_spec(keys=["currents_pA", "reference_values_Hz", "model_values_Hz"]),
            ],
            group_id="audit_gamma",
            group_title="Audit gamma",
        ),
        AuditItem(
            check_id="audit_gamma.gamma_nonfunction_curve",
            status="PASS",
            title="Gamma nonfunction curve",
            criterion="Repeated current values with multiple rates should render as a scatter plot.",
            description="Description",
            acceptable="Acceptable",
            acceptable_basis="Configured",
            evidence={
                "currents_pA": [0.0, 50.0, 50.0, 100.0],
                "reference_values_Hz": [0.0, 2.0, 3.1, 9.0],
                "model_values_Hz": [0.0, 1.8, 2.6, 8.7],
                "label": "Repeated current values",
            },
            series_visuals=[
                series_visual_spec(keys=["currents_pA", "reference_values_Hz", "model_values_Hz"]),
            ],
            group_id="audit_gamma",
            group_title="Audit gamma",
        ),
        AuditItem(
            check_id="audit_gamma.gamma_grouped_fi_rows",
            status="PASS",
            title="Gamma grouped f-I rows",
            criterion="Multiple model f-I curves should render as separate labeled series when fi_curve_rows include more than one cell.",
            description="Description",
            acceptable="Acceptable",
            acceptable_basis="Configured",
            evidence={
                "fi_curve_rows": [
                    {"cell_name": "GC1", "current_pA": 0.0, "firing_rate_Hz": 0.0},
                    {"cell_name": "GC1", "current_pA": 50.0, "firing_rate_Hz": 2.0},
                    {"cell_name": "GC1", "current_pA": 100.0, "firing_rate_Hz": 6.0},
                    {"cell_name": "GC2", "current_pA": 0.0, "firing_rate_Hz": 0.0},
                    {"cell_name": "GC2", "current_pA": 50.0, "firing_rate_Hz": 3.0},
                    {"cell_name": "GC2", "current_pA": 100.0, "firing_rate_Hz": 7.0},
                ],
            },
            series_visuals=[
                series_visual_spec(
                    title="Model f-I curves",
                    row_sources=[{"key": "fi_curve_rows", "group_by": ["cell_name"], "role": "model"}],
                )
            ],
            group_id="audit_gamma",
            group_title="Audit gamma",
        ),
        AuditItem(
            check_id="audit_gamma.gamma_target_curve",
            status="PASS",
            title="Gamma target curve",
            criterion="Target and model f-I curves should render together when both row sets are present.",
            description="Description",
            acceptable="Acceptable",
            acceptable_basis="Configured",
            evidence={
                "reference_fi_curve_rows": [
                    {"series_label": "Target", "current_pA": 50.0, "firing_rate_Hz": 4.0},
                    {"series_label": "Target", "current_pA": 100.0, "firing_rate_Hz": 9.0},
                    {"series_label": "Target", "current_pA": 150.0, "firing_rate_Hz": 14.0},
                ],
                "model_fi_curve_rows": [
                    {"cell_name": "SyntheticEPL2026.PVCRH_FSI1", "current_pA": 50.0, "firing_rate_Hz": 3.2},
                    {"cell_name": "SyntheticEPL2026.PVCRH_FSI1", "current_pA": 100.0, "firing_rate_Hz": 8.2},
                    {"cell_name": "SyntheticEPL2026.PVCRH_FSI1", "current_pA": 150.0, "firing_rate_Hz": 12.6},
                ],
            },
            series_visuals=[
                series_visual_spec(
                    title="Model vs target f-I curves",
                    row_sources=[
                        {"key": "reference_fi_curve_rows", "label": "Target", "role": "reference"},
                        {"key": "model_fi_curve_rows", "group_by": ["cell_name"], "role": "model"},
                    ],
                )
            ],
            group_id="audit_gamma",
            group_title="Audit gamma",
        ),
        AuditItem(
            check_id="audit_delta.delta_formulae_only",
            status="PASS",
            title="Delta formulae only",
            criterion="Delta formulae should still render when the rule only provides supporting math rows.",
            criterion_formulae=[r"x \leq y"],
            criterion_definitions=[
                {"symbol": "x", "definition": "lower quantity"},
                {"symbol": "y", "definition": "upper quantity"},
            ],
            description="Description",
            acceptable="Acceptable",
            acceptable_basis="Configured",
            evidence={"label": "Formula-only math"},
            group_id="audit_delta",
            group_title="Audit delta",
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
            companion_visuals=[
                {
                    "kind": "numeric_strip",
                    "keys": ["spike_count", "response_latency_ms", "sample_count"],
                }
            ],
            group_id="audit_gamma",
            group_title="Audit gamma",
        ),
        AuditItem(
            check_id="audit_gamma.gamma_plain_numeric",
            status="PASS",
            title="Gamma plain numeric",
            criterion="Gamma numeric evidence should stay plain unless a companion visual is explicitly requested.",
            description="Description",
            acceptable="Acceptable",
            acceptable_basis="Configured",
            evidence={
                "spike_count": 7,
                "response_latency_ms": 11.2,
                "sample_count": 18,
                "label": "Plain scalar metrics",
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
            companion_visuals=[
                {
                    "kind": "numeric_sparkline",
                    "key": "trial_values",
                }
            ],
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
    assert (output_dir / "assets" / "katex" / "katex.min.js").exists()
    report_payload = json.loads((output_dir / "report.json").read_text())
    assert report_payload["audit_id"] == "new_sweep"
    assert len(report_payload["groups"]) == 4
    assert report_payload["groups"][0]["group_id"] == "audit_alpha"
    html = (output_dir / "index.html").read_text()
    assert "Audit groups" in html
    assert "Audit alpha" in html
    assert "Audit beta" in html
    assert "Audit delta" in html
    assert "/__audit_refresh__" in html
    assert "Display controls" in html
    assert "overall-status" in html
    assert "Failures and warnings only" in html
    assert "Collapse all groups" in html
    assert "aria-pressed=\"false\"" in html
    assert "group-collapsed" in html
    assert ">Expand<" in html
    assert "Show detail items" not in html
    assert "audit-empty-state-run" in html
    assert "audit-empty-state-title" in html
    assert "audit-empty-state-subtitle" in html
    assert "control-center-run-audit" in html
    assert "group-link-label" in html
    assert "group-link-count" in html
    assert "group-link-main" in html
    assert "aria-current" not in html
    assert "status-text" in html
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
    assert "interval-value-labels" in html
    assert html.count("interval-value-label") >= 4
    assert "interval-bound-label" not in html
    assert "interval-marker-label" not in html
    assert "rgba(17, 24, 39, 0.9)" in html
    assert "<div class='interval-metric-grid'>" not in html
    assert "<div class='interval-range-labels'>" not in html
    assert "0.12 Hz" in html
    assert "0.21 Hz" in html
    assert "0.45 Hz" in html
    assert "1.03 Hz" in html
    assert "Protocol caveat exists in this item." in html
    assert "Evidence caveat from protocol matching." in html
    assert "criterion-math" in html
    assert "criterion-katex-display" in html
    assert "criterion-formulae" in html
    assert "criterion-formula" in html
    assert "criterion-katex-inline" in html
    assert "criterion-definitions" in html
    assert "criterion-definition-symbol" in html
    assert "Delta formulae only" in html
    assert r"x \leq y" in html
    assert "criterion-variables" not in html
    assert "criterion-variable-chip" not in html
    assert "./assets/katex/katex.min.css" in html
    assert "./assets/katex/katex.min.js" in html
    assert "./assets/katex/contrib/auto-render.min.js" in html
    assert "cdn.jsdelivr.net" not in html
    assert r"\lvert" not in html
    assert r"\rvert" not in html
    assert r"\left|\bar{x} - \mu_{\mathrm{ref}}\right| \leq 2\sigma_{\mathrm{ref}}" in html
    assert "justify-content: center" in html
    assert "color: #000" in html
    assert "observed group mean" in html
    assert "f-I curve" in html
    assert "data-series-graph" in html
    assert "data-visual-backend='matplotlib'" in html
    assert "Current (pA)" in html
    assert "Firing rate (Hz)" in html
    assert html.count("series-graph-meta") >= 1
    assert html.count("<text") >= 8
    assert html.count("<path") >= 1
    assert "Observed sweep" in html
    assert "Model f-I curves" in html
    assert "Model vs target f-I curves" in html
    assert "GC1" in html
    assert "GC2" in html
    assert "Target" in html
    assert "SyntheticEPL2026.PVCRH_FSI1" in html
    assert "data-numeric-strip" in html
    assert "data-numeric-sparkline" in html
    assert "Numeric summary" in html
    assert "Numeric sequence" in html
    assert "Observed scalar metrics" in html
    assert "Observed sequence" in html
    assert "Plain scalar metrics" in html
    gamma_index = html.index("audit_gamma.gamma_curve")
    assert html.index("series-graph-block", gamma_index) < html.index("data-item-detail-body", gamma_index)
    nonfunction_index = html.index("audit_gamma.gamma_nonfunction_curve")
    nonfunction_segment = html[nonfunction_index:html.index("</article>", nonfunction_index)]
    assert "data-visual-kind='scatter'" in nonfunction_segment
    assert "series-line" not in nonfunction_segment
    plain_index = html.index("audit_gamma.gamma_plain_numeric")
    strip_item_index = html.index("audit_gamma.gamma_numeric")
    plain_segment = html[plain_index:strip_item_index]
    assert "data-series-graph" not in plain_segment
    assert "series-graph-block" not in plain_segment
    assert "data-numeric-strip" not in plain_segment
    assert "data-numeric-sparkline" not in plain_segment
    strip_index = html.index("data-numeric-strip")
    sparkline_index = html.index("data-numeric-sparkline")
    assert strip_index < sparkline_index
    assert html.count("data-numeric-strip") == 1
    assert html.count("data-numeric-sparkline") == 1
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
