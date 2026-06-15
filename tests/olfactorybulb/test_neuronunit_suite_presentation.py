"""Coverage for the shared NeuronUnit-to-AuditItem presentation adapter."""

from __future__ import annotations

from types import SimpleNamespace

from olfactorybulb.audit import companion_visual_spec, series_visual_spec
from olfactorybulb.audit.core import AuditReport
from olfactorybulb.neuronunit.provenance import ValidationReview
from olfactorybulb.neuronunit.suite_presentation import (
    audit_item_adapter_spec_from_case,
    suite_case_result_from_spec,
    suite_items_from_case_results,
)
from olfactorybulb.neuronunit.suite_scores import SuiteCaseScorePayload, SuiteDescriptor


raw_series_visual = series_visual_spec(
    kind="fi_curve",
    keys=["currents_pA", "reference_values_Hz", "model_values_Hz"],
    style={"legend_loc": "lower center"},
)
raw_companion_visual = companion_visual_spec(
    kind="status_matrix",
    key="suite_cases",
    title="Suite case summary",
)

case = SimpleNamespace(
    check_id="synthetic_curve_match",
    title="Synthetic curve match",
    criterion="The synthetic curve should remain near the reference curve.",
    criterion_latex=r"\left|\bar{x} - \mu\right| \leq 2\sigma",
    criterion_formulae=[r"\mu = 5", r"\sigma = 1"],
    criterion_definitions=[
        {"symbol": r"\bar{x}", "definition": "observed synthetic mean"},
        {"symbol": r"\mu", "definition": "reference mean"},
    ],
    description="Synthetic adapter smoke test.",
    acceptable="The adapted case preserves all maintained presentation fields.",
    acceptable_basis="The shared adapter contract is the maintained source of truth.",
    note="Synthetic note.",
)
review = ValidationReview(
    status="approved",
    note="Reviewed by hand.",
    reviewer="qualified_domain_expert",
    required_expertise="single-cell electrophysiology",
    focus="adapter field preservation",
)
adapter_spec = audit_item_adapter_spec_from_case(
    case,
    validation_review=review,
    series_visuals=[raw_series_visual],
    companion_visuals=[raw_companion_visual],
    default_status_reason="Synthetic adapter status reason.",
)
result = suite_case_result_from_spec(
    adapter_spec,
    status="WARN",
    evidence={"observed": 3.2, "reference": 5.0},
    score_text="observed 3.2 Hz",
    norm_score=0.5,
    score_payload=SuiteCaseScorePayload(
        score_kind="synthetic_distance",
        score_value=1.8,
        score_units="Hz",
        score_interpretation="Synthetic adapter score payload.",
    ),
)
item = result.item

assert item.check_id == "synthetic_curve_match"
assert item.status == "WARN"
assert item.title == "Synthetic curve match"
assert item.criterion == "The synthetic curve should remain near the reference curve."
assert item.criterion_latex == r"\left|\bar{x} - \mu\right| \leq 2\sigma"
assert item.criterion_formulae == [r"\mu = 5", r"\sigma = 1"]
assert item.criterion_definitions[0]["symbol"] == r"\bar{x}"
assert item.description == "Synthetic adapter smoke test."
assert item.acceptable == "The adapted case preserves all maintained presentation fields."
assert item.acceptable_basis == "The shared adapter contract is the maintained source of truth."
assert item.note == "Synthetic note."
assert item.validation_design_review_status == "approved"
assert item.validation_design_review_note == "Reviewed by hand."
assert item.validation_design_review_reviewer == "qualified_domain_expert"
assert item.validation_design_review_required_expertise == "single-cell electrophysiology"
assert item.validation_design_review_focus == "adapter field preservation"
assert item.series_visuals[0]["kind"] == "fi_curve"
assert item.companion_visuals[0]["kind"] == "status_matrix"
assert item.status_reason == "Synthetic adapter status reason."
assert item.detail_level == "detail"
assert item.summary_rollup_exempt is False
assert item.evidence == {"observed": 3.2, "reference": 5.0}
assert result.score_text == "observed 3.2 Hz"
assert result.norm_score == 0.5
assert result.score_payload is not None
assert result.score_payload.score_kind == "synthetic_distance"

raw_series_visual["kind"] = "mutated"
raw_companion_visual["title"] = "mutated"
assert item.series_visuals[0]["kind"] == "fi_curve"
assert item.companion_visuals[0]["title"] == "Suite case summary"

items = suite_items_from_case_results(
    descriptor=SuiteDescriptor(
        suite_id="synthetic.adapter_suite",
        suite_kind_label="Series-comparison suite",
    ),
    case_results=[result],
)
assert len(items) == 2
assert items[0].check_id == "synthetic.adapter_suite.overview"
assert items[0].summary_rollup_exempt is True
assert items[0].evidence["suite_cases"][0]["check_id"] == "synthetic_curve_match"
assert items[0].evidence["suite_cases"][0]["score_text"] == "observed 3.2 Hz"
assert items[0].evidence["suite_cases"][0]["norm_score"] == 0.5
assert items[0].evidence["suite_cases"][0]["case_score"]["score_kind"] == "synthetic_distance"
assert items[0].evidence["suite_cases"][0]["case_score"]["score_value"] == 1.8

report = AuditReport(
    audit_id="synthetic_adapter_suite",
    title="Synthetic adapter suite",
    items=items,
)
assert report.summary == {"PASS": 0, "WARN": 1, "FAIL": 0}

print("neuronunit_suite_presentation: OK")
