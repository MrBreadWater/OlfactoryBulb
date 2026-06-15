"""Coverage for the SciUnit-backed comparison/exactness validation bridge."""

from __future__ import annotations

from argparse import Namespace

from olfactorybulb.audit.reference_validation_document import ValidationDesignReviewDefaultsSpec
from olfactorybulb.audit.core import AuditReport
from olfactorybulb.audit.reference_validation_rules import ValidationRuleContext, build_rule_items
from olfactorybulb.audit.reference_validation_rules import compile_rule_dispatches
from olfactorybulb.neuronunit.comparison_validation_suite import (
    ComparisonRuleCase,
    audit_items_from_comparison_rule_suite,
    compile_comparison_rule_suite,
)


def _rule_context(
    *,
    metrics: list[dict[str, object]],
    summary: dict[str, dict[str, float]],
    args: object,
    validation_id: str = "burton_urban_fi",
    protocol_result: object | None = None,
) -> ValidationRuleContext:
    return ValidationRuleContext(
        metrics=metrics,
        summary=summary,
        args=args,
        validation_id=validation_id,
        default_group="",
        notes_path="",
        design_review_defaults=ValidationDesignReviewDefaultsSpec(status="pending"),
        protocol_result=protocol_result,
    )


metrics = [
    {
        "cell_name": "MC1",
        "cell_type": "MC",
        "zero_step_rate_Hz": 0.0,
        "input_resistance_MOhm": 100.0,
        "AP_onset_mV": -42.0,
        "FWHM_ms": 1.1,
        "rheobase_pA": 100.0,
    },
    {
        "cell_name": "TC1",
        "cell_type": "TC",
        "zero_step_rate_Hz": 0.0,
        "input_resistance_MOhm": 110.0,
        "AP_onset_mV": -42.5,
        "FWHM_ms": 0.9,
        "rheobase_pA": 90.0,
    },
]

summary = {
    "MC": {
        "AP_onset_mV": -42.0,
        "FWHM_ms": 1.1,
        "rheobase_pA": 100.0,
    },
    "TC": {
        "AP_onset_mV": -42.5,
        "FWHM_ms": 0.9,
        "rheobase_pA": 90.0,
    },
}

cases = [
    ComparisonRuleCase(
        rule_kind="all_exact_metric",
        check_id="zero_current_quiescence_at_normalized_vm",
        title="Cells remain quiescent during the zero-picoampere step at the normalized membrane potential",
        criterion="Every audited cell should have zero firing rate at the zero-current step.",
        criterion_latex=r"\forall i,\ \left|x_i - c\right| \leq \epsilon",
        criterion_formulae=[],
        criterion_definitions=[
            {"symbol": r"x_i", "definition": "zero-current firing-rate value for each audited cell"},
            {"symbol": "c", "definition": "expected value"},
            {"symbol": r"\epsilon", "definition": "tolerance"},
        ],
        description="Synthetic exact-rule suite test.",
        acceptable="Every audited cell has exactly zero firing rate at the zero-current step.",
        acceptable_basis="Synthetic basis.",
        note="",
        metric_key="zero_step_rate_Hz",
        expected=0.0,
        tolerance=1e-9,
    ),
    ComparisonRuleCase(
        rule_kind="group_abs_diff_max",
        check_id="ap_threshold_similarity",
        title="Mitral-cell and tufted-cell action-potential thresholds remain similar",
        criterion="Thresholds should remain within five millivolts.",
        criterion_latex=r"\left|\bar{x}_{\mathrm{TC}} - \bar{x}_{\mathrm{MC}}\right| \leq 5",
        criterion_formulae=[],
        criterion_definitions=[
            {"symbol": r"\bar{x}_{\mathrm{MC}}", "definition": "MC mean AP_onset_mV"},
            {"symbol": r"\bar{x}_{\mathrm{TC}}", "definition": "TC mean AP_onset_mV"},
        ],
        description="Synthetic difference-rule suite test.",
        acceptable="The group means stay within five millivolts.",
        acceptable_basis="Synthetic basis.",
        note="",
        metric_key="AP_onset_mV",
        left_group="MC",
        right_group="TC",
        max_difference=5.0,
    ),
    ComparisonRuleCase(
        rule_kind="group_ordering",
        check_id="tc_action_potentials_narrower",
        title="Tufted-cell action potentials are narrower than mitral-cell action potentials",
        criterion="Tufted-cell full width at half maximum should be lower than the mitral-cell value.",
        criterion_latex=r"\bar{x}_{\mathrm{TC}} < \bar{x}_{\mathrm{MC}}",
        criterion_formulae=[],
        criterion_definitions=[
            {"symbol": r"\bar{x}_{\mathrm{MC}}", "definition": "MC mean FWHM_ms"},
            {"symbol": r"\bar{x}_{\mathrm{TC}}", "definition": "TC mean FWHM_ms"},
        ],
        description="Synthetic ordering-rule suite test.",
        acceptable="The tufted-cell mean is smaller than the mitral-cell mean.",
        acceptable_basis="Synthetic basis.",
        note="",
        metric_key="FWHM_ms",
        left_group="MC",
        right_group="TC",
        operator="<",
    ),
    ComparisonRuleCase(
        rule_kind="group_positive",
        check_id="rheobase_in_paper_regime",
        title="Mitral-cell and tufted-cell rheobases remain in a depolarizing-step regime",
        criterion="Both group means should stay strictly positive.",
        criterion_latex=r"\bar{x}_{\mathrm{MC}} > 0 \wedge \bar{x}_{\mathrm{TC}} > 0",
        criterion_formulae=[],
        criterion_definitions=[
            {"symbol": r"\bar{x}_{\mathrm{MC}}", "definition": "MC mean rheobase_pA"},
            {"symbol": r"\bar{x}_{\mathrm{TC}}", "definition": "TC mean rheobase_pA"},
        ],
        description="Synthetic positivity-rule suite test.",
        acceptable="Both group means are strictly positive.",
        acceptable_basis="Synthetic basis.",
        note="",
        metric_key="rheobase_pA",
        groups=("MC", "TC"),
    ),
    ComparisonRuleCase(
        rule_kind="all_finite_metric",
        check_id="input_resistance_recorded",
        title="Input resistance was measured for the firing-rate-versus-current gain comparison",
        criterion="Every audited cell should have a finite input resistance.",
        criterion_latex="",
        criterion_formulae=[],
        criterion_definitions=[],
        description="Synthetic finite-rule suite test.",
        acceptable="Every audited cell has a finite input resistance.",
        acceptable_basis="Synthetic basis.",
        note="",
        metric_key="input_resistance_MOhm",
    ),
]

compiled = compile_comparison_rule_suite(
    cases=cases,
    summary=summary,
    metrics=metrics,
    suite_name="synthetic comparison suite",
)
assert compiled.model.runtime_data.summary is compiled.model.summary
assert compiled.model.runtime_data.metrics is compiled.model.metrics
assert len(compiled.model.runtime_data.metrics.rows) == 2
assert cases[0].observation_payload() == {
    "rule_kind": "all_exact_metric",
    "metric_key": "zero_step_rate_Hz",
    "entity_key": "cell_name",
    "left_group": "",
    "right_group": "",
    "operator": ">",
    "groups": (),
    "expected": 0.0,
    "tolerance": 1e-09,
    "max_difference": 0.0,
}
assert cases[2].observation_payload() == {
    "rule_kind": "group_ordering",
    "metric_key": "FWHM_ms",
    "entity_key": "cell_name",
    "left_group": "MC",
    "right_group": "TC",
    "operator": "<",
    "groups": (),
    "expected": 0.0,
    "tolerance": 1e-09,
    "max_difference": 0.0,
}
judged = compiled.judge()
assert [score.status for _case, score in judged] == ["PASS", "PASS", "PASS", "PASS", "PASS"]

adapted_items = audit_items_from_comparison_rule_suite(compiled)
assert [item.status for item in adapted_items] == ["PASS", "PASS", "PASS", "PASS", "PASS", "PASS"]
assert adapted_items[0].detail_level == "summary"
assert adapted_items[0].summary_rollup_exempt is True
assert adapted_items[0].evidence["suite_status_summary"] == {"PASS": 5, "WARN": 0, "FAIL": 0}
assert adapted_items[0].evidence["suite_aggregate_score"]["status"] == "PASS"
assert adapted_items[0].evidence["suite_aggregate_score"]["score_text"] == "worst PASS, min norm 1"
assert adapted_items[0].evidence["suite_cases"][1]["score_text"] == "|Δ| 0.5 mV"
assert adapted_items[0].evidence["suite_cases"][2]["score_text"] == "Δ -0.2 ms"
assert adapted_items[0].evidence["suite_cases"][1]["case_score"]["score_kind"] == "group_abs_diff_max"
assert adapted_items[0].evidence["suite_cases"][1]["case_score"]["score_units"] == "mV"
assert adapted_items[1].evidence["expected"] == 0.0
assert adapted_items[1].evidence["metric_unit"] == "Hz"
assert adapted_items[2].evidence["absolute_difference"] == 0.5
assert adapted_items[2].evidence["metric_unit"] == "mV"
assert adapted_items[3].evidence["TC_minus_MC"] == -0.2
assert adapted_items[3].evidence["metric_unit"] == "ms"
assert adapted_items[4].evidence["MC_mean"] == 100.0
assert adapted_items[4].evidence["metric_unit"] == "pA"
assert adapted_items[5].evidence["cell_count"] == 2
assert adapted_items[5].evidence["metric_unit"] == "MOhm"
report = AuditReport(audit_id="synthetic_comparison_suite", title="Synthetic comparison suite", items=adapted_items)
assert report.summary == {"PASS": 5, "WARN": 0, "FAIL": 0}

rules = [
    {
        "kind": "all_exact_metric",
        "check_id": "zero_current_quiescence_at_normalized_vm",
        "metric_key": "zero_step_rate_Hz",
        "entity_key": "cell_name",
        "expected": 0.0,
        "tolerance": 1e-9,
        "title": "Cells remain quiescent during the zero-picoampere step at the normalized membrane potential",
        "criterion": "Every audited cell should have zero firing rate at the zero-current step.",
        "description": "Synthetic exact-rule suite test.",
        "acceptable": "Every audited cell has exactly zero firing rate at the zero-current step.",
        "acceptable_basis": "Synthetic basis.",
    },
    {
        "kind": "group_abs_diff_max",
        "check_id": "ap_threshold_similarity",
        "metric_key": "AP_onset_mV",
        "left_group": "MC",
        "right_group": "TC",
        "max_difference": 5.0,
        "title": "Mitral-cell and tufted-cell action-potential thresholds remain similar",
        "criterion": "Thresholds should remain within five millivolts.",
        "description": "Synthetic difference-rule suite test.",
        "acceptable": "The group means stay within five millivolts.",
        "acceptable_basis": "Synthetic basis.",
    },
    {
        "kind": "group_ordering",
        "check_id": "tc_action_potentials_narrower",
        "metric_key": "FWHM_ms",
        "left_group": "MC",
        "right_group": "TC",
        "operator": "<",
        "title": "Tufted-cell action potentials are narrower than mitral-cell action potentials",
        "criterion": "Tufted-cell full width at half maximum should be lower than the mitral-cell value.",
        "description": "Synthetic ordering-rule suite test.",
        "acceptable": "The tufted-cell mean is smaller than the mitral-cell mean.",
        "acceptable_basis": "Synthetic basis.",
    },
    {
        "kind": "group_positive",
        "check_id": "rheobase_in_paper_regime",
        "metric_key": "rheobase_pA",
        "groups": ["MC", "TC"],
        "title": "Mitral-cell and tufted-cell rheobases remain in a depolarizing-step regime",
        "criterion": "Both group means should stay strictly positive.",
        "description": "Synthetic positivity-rule suite test.",
        "acceptable": "Both group means are strictly positive.",
        "acceptable_basis": "Synthetic basis.",
    },
    {
        "kind": "all_finite_metric",
        "check_id": "input_resistance_recorded",
        "metric_key": "input_resistance_MOhm",
        "title": "Input resistance was measured for the firing-rate-versus-current gain comparison",
        "criterion": "Every audited cell should have a finite input resistance.",
        "description": "Synthetic finite-rule suite test.",
        "acceptable": "Every audited cell has a finite input resistance.",
        "acceptable_basis": "Synthetic basis.",
    },
]

context = _rule_context(
    metrics=metrics,
    summary=summary,
    args=Namespace(skip_neuron=False),
    protocol_result=None,
)
items = build_rule_items(compile_rule_dispatches(rules), context)
assert [item.check_id for item in items] == [
    "burton_urban_fi.comparison_rules.overview",
    "zero_current_quiescence_at_normalized_vm",
    "ap_threshold_similarity",
    "tc_action_potentials_narrower",
    "rheobase_in_paper_regime",
    "input_resistance_recorded",
]
assert items[0].detail_level == "summary"
assert [item.status for item in items] == ["PASS", "PASS", "PASS", "PASS", "PASS", "PASS"]
assert items[1].criterion_latex == r"\forall i,\ \left|x_i - c\right| \leq \epsilon"
assert items[2].evidence["absolute_difference"] == 0.5
assert items[3].evidence["TC_minus_MC"] == -0.2
assert items[4].criterion_latex == r"\bar{I}_{\mathrm{rh},\mathrm{MC}} > 0 \wedge \bar{I}_{\mathrm{rh},\mathrm{TC}} > 0"
assert items[5].evidence["cell_count"] == 2


print("neuronunit_comparison_validation_suite: OK")
