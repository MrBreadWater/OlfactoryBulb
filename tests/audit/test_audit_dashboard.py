"""Smoke tests for the maintained audit HTML dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from olfactorybulb.audit import companion_visual_spec, series_visual_spec
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
            check_id="audit_gamma.gamma_row_curve",
            status="PASS",
            title="Gamma row curve",
            criterion="Protocol row collections should render through an explicit row-source contract.",
            description="Description",
            acceptable="Acceptable",
            acceptable_basis="Configured",
            evidence={
                "protocol_rows": [
                    {"cell_name": "Cell A", "current_pA": 0.0, "firing_rate_Hz": 0.0},
                    {"cell_name": "Cell A", "current_pA": 50.0, "firing_rate_Hz": 2.0},
                    {"cell_name": "Cell A", "current_pA": 100.0, "firing_rate_Hz": 5.0},
                    {"cell_name": "Cell B", "current_pA": 0.0, "firing_rate_Hz": 0.0},
                    {"cell_name": "Cell B", "current_pA": 50.0, "firing_rate_Hz": 2.6},
                    {"cell_name": "Cell B", "current_pA": 100.0, "firing_rate_Hz": 5.8},
                ],
            },
            series_visuals=[
                series_visual_spec(
                    keys=["protocol_rows"],
                    row_source_key="protocol_rows",
                    x_key="current_pA",
                    y_keys=["firing_rate_Hz"],
                    series_id_key="cell_name",
                    style={
                        "x_label": "Injected current (pA)",
                        "y_label": "Firing rate (Hz)",
                        "title": "Protocol f-I curve",
                    },
                ),
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
            check_id="audit_gamma.gamma_generic_curve",
            status="PASS",
            title="Gamma generic curve",
            criterion="Generic series labels should respect explicit quantity and unit metadata.",
            description="Description",
            acceptable="Acceptable",
            acceptable_basis="Configured",
            evidence={
                "currents_pA": [0.0, 50.0, 100.0, 150.0],
                "reference_values": [-62.0, -58.0, -54.0, -50.0],
                "model_values": [-61.5, -57.5, -53.5, -49.5],
                "x_quantity_name": "Injected current",
                "y_quantity_name": "Membrane voltage",
                "comparison_x_unit_text": "pA",
                "comparison_y_unit_text": "mV",
                "label": "Voltage response",
            },
            series_visuals=[
                series_visual_spec(keys=["currents_pA", "reference_values", "model_values"]),
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
    assert "Injected current (pA)" in html
    assert "Membrane voltage (mV)" in html
    assert "Protocol f-I curve" in html
    assert "Cell A" in html
    assert "Cell B" in html
    assert html.count("series-graph-meta") >= 1
    assert html.count("<text") >= 8
    assert html.count("<path") >= 1
    assert "Observed sweep" in html
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

suite_summary_report = AuditReport(
    audit_id="suite_summary_demo",
    title="Suite summary demo",
    items=[
        AuditItem(
            check_id="suite_summary_demo.synthetic_suite.overview",
            status="WARN",
            title="Synthetic suite overview",
            criterion="Every compiled suite case should satisfy its declared criterion.",
            description="Synthetic suite summary item.",
            acceptable="The detailed suite cases pass.",
            acceptable_basis="Synthetic basis.",
            evidence={
                "suite_name": "suite_summary_demo.synthetic_suite",
                "suite_kind": "Synthetic suite",
                "suite_case_count": 2,
                "suite_status_summary": {"PASS": 1, "WARN": 1, "FAIL": 0},
                "suite_candidate_ids": ["SyntheticA", "SyntheticB"],
                "suite_norm_score_summary": {"count": 2.0, "mean": 0.75, "median": 0.75, "min": 0.5, "max": 1.0},
                "suite_aggregate_score": {
                    "status_rollup": "worst_case",
                    "norm_rollup": "minimum",
                    "score_kind": "worst_case_minimum_norm",
                    "score_interpretation": "Aggregate suite score derived from the worst detailed-case status plus the minimum normalized case score.",
                    "status": "WARN",
                    "score_text": "worst WARN, min norm 0.5",
                    "score_value": 0.5,
                },
                "suite_statistical_summary": {
                    "score_family_category": "equivalence",
                    "statistical_test_family": "equivalence_tost",
                    "rollup_method": "max",
                    "rollup_source": "auto_default",
                    "rollup_pvalue": 0.03,
                    "available_case_count": 1,
                    "total_case_count": 2,
                    "available_case_fraction": 0.5,
                    "available_case_weight": 4.0,
                    "total_case_weight": 12.0,
                    "available_case_weight_fraction": 0.333,
                    "weight_label": "matched points",
                    "score_text": "max TOST p 0.03",
                    "score_interpretation": "Synthetic suite statistical summary.",
                    "threshold": 0.05,
                    "threshold_key": "equivalence_alpha",
                    "threshold_direction": "le",
                    "support_gate_passed": False,
                    "weight_support_gate_passed": False,
                    "threshold_gate_passed": True,
                    "gate_passed": False,
                    "case_pvalues": [0.03],
                    "case_check_ids": ["suite_summary_demo.warn_detail"],
                },
                "warning_cases": ["Suite warning detail"],
                "failed_cases": [],
                "suite_cases": [
                    {
                        "check_id": "suite_summary_demo.pass_detail",
                        "title": "Suite pass detail",
                        "status": "PASS",
                        "score_text": "observed 1",
                        "norm_score": 1.0,
                    },
                    {
                        "check_id": "suite_summary_demo.warn_detail",
                        "title": "Suite warning detail",
                        "status": "WARN",
                        "case_score": {
                            "score_kind": "observed_value",
                            "score_value": 2.0,
                            "score_units": "Hz",
                        },
                        "norm_score": 0.5,
                    },
                ],
            },
            companion_visuals=[
                companion_visual_spec(kind="status_matrix", key="suite_cases", title="Suite case summary")
            ],
            group_id="suite_summary_demo",
            group_title="Suite summary demo",
            detail_level="summary",
            summary_rollup_exempt=True,
        ),
        AuditItem(
            check_id="suite_summary_demo.pass_detail",
            status="PASS",
            title="Suite pass detail",
            criterion="Criterion",
            description="Description",
            acceptable="Acceptable",
            acceptable_basis="Configured",
            evidence={"count": 1},
            group_id="suite_summary_demo",
            group_title="Suite summary demo",
        ),
        AuditItem(
            check_id="suite_summary_demo.warn_detail",
            status="WARN",
            title="Suite warning detail",
            criterion="Criterion",
            description="Description",
            acceptable="Acceptable",
            acceptable_basis="Configured",
            evidence={"count": 2},
            status_reason="Synthetic suite warning.",
            group_id="suite_summary_demo",
            group_title="Suite summary demo",
        ),
    ],
)

with TemporaryDirectory() as tmp:
    output_dir = Path(tmp)
    export_audit_dashboard(suite_summary_report, output_dir)
    html = (output_dir / "index.html").read_text()
    payload = json.loads((output_dir / "report.json").read_text())
    group_payload = payload["groups"][0]
    assert payload["summary"] == {"FAIL": 0, "WARN": 1, "PASS": 1}
    assert group_payload["summary"] == {"FAIL": 0, "WARN": 1, "PASS": 1}
    assert group_payload["item_count"] == 2
    assert "Show detail items" in html
    assert "data-suite-status-matrix" in html
    assert "data-visual-kind='status_matrix'" in html
    assert "Suite case summary" in html
    assert "Suite pass detail" in html
    assert "Suite warning detail" in html
    assert "observed 1" in html
    assert "2 Hz" in html
    assert "norm 1" in html
    assert "norm 0.5" in html
    assert html.count("suite-status-cell") >= 2
    assert "2 cases, 2 candidates" in html
    assert "worst WARN, min norm 0.5" in html
    assert "max TOST p 0.03" in html
    assert "support 1/2 cases (gate fail)" in html
    assert "support 4/12 matched points (gate fail)" in html

suite_threshold_distribution_report = AuditReport(
    audit_id="suite_threshold_distribution_demo",
    title="Suite threshold distribution demo",
    items=[
        AuditItem(
            check_id="suite_threshold_distribution_demo.synthetic_suite.overview",
            status="FAIL",
            title="Synthetic threshold suite overview",
            criterion="Every compiled suite case should satisfy its declared criterion.",
            description="Synthetic suite summary item with a threshold split.",
            acceptable="The detailed suite cases pass.",
            acceptable_basis="Synthetic basis.",
            evidence={
                "suite_name": "suite_threshold_distribution_demo.synthetic_suite",
                "suite_kind": "Synthetic suite",
                "suite_case_count": 2,
                "suite_status_summary": {"PASS": 1, "WARN": 0, "FAIL": 1},
                "suite_norm_score_summary": {
                    "count": 2.0,
                    "mean": 0.5,
                    "median": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "total_weight": 12.0,
                    "weight_label": "matched points",
                    "weighted_mean": 0.25,
                },
                "suite_aggregate_score": {
                    "status_rollup": "worst_case",
                    "norm_rollup": "minimum",
                    "score_kind": "worst_case_minimum_norm",
                    "score_interpretation": "Aggregate suite score derived from the worst detailed-case status plus the minimum normalized case score.",
                    "status": "FAIL",
                    "score_text": "worst FAIL, min norm 0",
                    "score_value": 0.0,
                },
                "suite_statistical_summary": {
                    "score_family_category": "equivalence",
                    "statistical_test_family": "equivalence_tost",
                    "rollup_method": "max",
                    "rollup_source": "auto_default",
                    "rollup_pvalue": 0.08,
                    "available_case_count": 2,
                    "total_case_count": 2,
                    "available_case_fraction": 1.0,
                    "available_case_weight": 12.0,
                    "total_case_weight": 12.0,
                    "available_case_weight_fraction": 1.0,
                    "weight_label": "matched points",
                    "score_text": "max TOST p 0.08",
                    "score_interpretation": "Synthetic suite statistical summary.",
                    "threshold": 0.05,
                    "threshold_key": "equivalence_alpha",
                    "threshold_direction": "le",
                    "support_gate_passed": True,
                    "weight_support_gate_passed": True,
                    "threshold_gate_passed": False,
                    "gate_passed": False,
                    "threshold_passing_case_count": 1,
                    "threshold_failing_case_count": 1,
                    "threshold_passing_case_fraction": 0.5,
                    "threshold_failing_case_fraction": 0.5,
                    "threshold_passing_case_weight": 3.0,
                    "threshold_failing_case_weight": 9.0,
                    "threshold_passing_case_weight_fraction": 0.25,
                    "threshold_failing_case_weight_fraction": 0.75,
                    "case_pvalues": [0.02, 0.08],
                    "case_check_ids": [
                        "suite_threshold_distribution_demo.pass_detail",
                        "suite_threshold_distribution_demo.fail_detail",
                    ],
                    "threshold_passing_case_check_ids": ["suite_threshold_distribution_demo.pass_detail"],
                    "threshold_failing_case_check_ids": ["suite_threshold_distribution_demo.fail_detail"],
                },
                "warning_cases": [],
                "failed_cases": ["Threshold failing detail"],
                "suite_cases": [
                    {
                        "check_id": "suite_threshold_distribution_demo.pass_detail",
                        "title": "Threshold passing detail",
                        "status": "PASS",
                        "score_text": "TOST p 0.02",
                        "norm_score": 1.0,
                    },
                    {
                        "check_id": "suite_threshold_distribution_demo.fail_detail",
                        "title": "Threshold failing detail",
                        "status": "FAIL",
                        "score_text": "TOST p 0.08",
                        "norm_score": 0.0,
                    },
                ],
            },
            companion_visuals=[
                companion_visual_spec(kind="status_matrix", key="suite_cases", title="Suite case summary")
            ],
            group_id="suite_threshold_distribution_demo",
            group_title="Suite threshold distribution demo",
            detail_level="summary",
            summary_rollup_exempt=True,
        ),
    ],
)

with TemporaryDirectory() as tmp:
    output_dir = Path(tmp)
    export_audit_dashboard(suite_threshold_distribution_report, output_dir)
    html = (output_dir / "index.html").read_text()
    assert "threshold 1/2 pass (gate fail)" in html
    assert "threshold 3/12 matched points (gate fail)" in html
    assert "threshold fail" in html

print("audit_dashboard: OK")
