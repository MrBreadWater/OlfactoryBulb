"""Coverage for the SciUnit-backed summary-rule validation bridge."""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

from olfactorybulb.audit.reference_validation_document import ValidationDesignReviewDefaultsSpec
from olfactorybulb.audit.core import AuditReport
from olfactorybulb.audit.reference_validation_rules import (
    ValidationRuleContext,
    build_rule_items,
    compile_rule_dispatches,
)
from olfactorybulb.neuronunit.summary_validation_suite import (
    SummaryRuleCase,
    audit_items_from_summary_rule_suite,
    compile_summary_rule_suite,
)


def _rule_context(
    *,
    summary: dict[str, dict[str, float]],
    args: object,
    validation_id: str = "epli_correctness",
    protocol_result: object | None = None,
) -> ValidationRuleContext:
    return ValidationRuleContext(
        metrics=[],
        summary=summary,
        args=args,
        validation_id=validation_id,
        default_group="ungrouped",
        notes_path="",
        design_review_defaults=ValidationDesignReviewDefaultsSpec(status="pending"),
        protocol_result=protocol_result,
    )


cases = [
    SummaryRuleCase(
        rule_kind="summary_metric_min",
        check_id="baseline_slice_population_counts",
        title="Baseline dorsal slice contains nonzero principal and granule populations",
        criterion="Counts should stay nonzero.",
        criterion_latex=r"\bar{x}_{\mathrm{slice}} \geq 1",
        criterion_formulae=[],
        criterion_definitions=[{"symbol": r"\bar{x}_{\mathrm{slice}}", "definition": "slice summary count"}],
        description="Synthetic min-rule suite test.",
        acceptable="Counts are nonzero.",
        acceptable_basis="Synthetic basis.",
        note="",
        metric_key="baseline_population_min_count",
        group="ungrouped",
        minimum=1.0,
        evidence_metric_keys=["baseline_MCs_count", "baseline_TCs_count", "baseline_GCs_count"],
    ),
    SummaryRuleCase(
        rule_kind="summary_metric_status_map",
        check_id="epli_target_pattern_specificity",
        title="Default external plexiform layer interneuron target pattern encodes perisomatic principal territory",
        criterion="Selector should encode broader perisomatic territory.",
        criterion_latex="",
        criterion_formulae=[],
        criterion_definitions=[],
        description="Synthetic status-map suite test.",
        acceptable="Code 1 should warn.",
        acceptable_basis="Synthetic basis.",
        note="",
        metric_key="epli_target_pattern_specificity_code",
        group="ungrouped",
        pass_values=(2.0,),
        warn_values=(1.0,),
        fail_values=(0.0,),
    ),
    SummaryRuleCase(
        rule_kind="summary_metric_range",
        check_id="synthetic_soma_diameter",
        title="Synthetic external plexiform layer interneuron soma diameter matches target",
        criterion="Target soma diameter is 9.6 plus or minus 0.7 micrometers.",
        criterion_latex=r"\left|\bar{x}_{\mathrm{soma}} - 9.6\right| \leq 0.7",
        criterion_formulae=[],
        criterion_definitions=[{"symbol": r"\bar{x}_{\mathrm{soma}}", "definition": "observed soma diameter"}],
        description="Synthetic range-rule suite test.",
        acceptable="Observed diameter stays within range.",
        acceptable_basis="Synthetic basis.",
        note="",
        metric_key="soma_diameter_um",
        group="ungrouped",
        minimum=8.9,
        maximum=10.3,
    ),
]

summary = {
    "ungrouped": {
        "baseline_population_min_count": 4.0,
        "baseline_MCs_count": 12.0,
        "baseline_TCs_count": 9.0,
        "baseline_GCs_count": 41.0,
        "epli_target_pattern_specificity_code": 1.0,
        "soma_diameter_um": 9.7,
    }
}

compiled = compile_summary_rule_suite(cases=cases, summary=summary, suite_name="synthetic summary suite")
judged = compiled.judge()
assert [score.status for _case, score in judged] == ["PASS", "WARN", "PASS"]

adapted_items = audit_items_from_summary_rule_suite(compiled)
assert [item.status for item in adapted_items] == ["WARN", "PASS", "WARN", "PASS"]
assert adapted_items[0].detail_level == "summary"
assert adapted_items[0].summary_rollup_exempt is True
assert adapted_items[0].evidence["suite_status_summary"] == {"PASS": 2, "WARN": 1, "FAIL": 0}
assert adapted_items[0].evidence["suite_aggregate_score"]["status"] == "WARN"
assert adapted_items[0].evidence["suite_aggregate_score"]["score_text"] == "worst WARN, min norm 0.5"
assert adapted_items[0].evidence["suite_cases"][0]["score_text"] == "observed 4"
assert adapted_items[0].evidence["suite_cases"][1]["norm_score"] == 0.5
assert adapted_items[0].evidence["suite_cases"][1]["case_score"]["score_kind"] == "summary_metric_status_map"
assert adapted_items[0].evidence["suite_cases"][2]["score_text"] == "observed 9.7 um"
assert adapted_items[0].evidence["suite_cases"][2]["case_score"]["score_units"] == "um"
assert adapted_items[1].evidence["baseline_MCs_count"] == 12.0
assert adapted_items[2].evidence["warn_values"] == [1.0]
assert adapted_items[3].evidence["minimum"] == 8.9
assert adapted_items[3].evidence["maximum"] == 10.3
assert adapted_items[3].evidence["metric_unit"] == "um"
assert adapted_items[3].evidence["metric_quantity_name"] == "Soma Diameter"
report = AuditReport(audit_id="synthetic_summary_suite", title="Synthetic summary suite", items=adapted_items)
assert report.summary == {"PASS": 2, "WARN": 1, "FAIL": 0}


context = _rule_context(
    summary=summary,
    args=Namespace(skip_neuron=True),
    protocol_result=None,
)
rules = [
    {
        "kind": "summary_metric_min",
        "check_id": "baseline_slice_population_counts",
        "metric_key": "baseline_population_min_count",
        "minimum": 1.0,
        "title": "Baseline dorsal slice contains nonzero principal and granule populations",
        "criterion": "Counts should stay nonzero.",
        "description": "Synthetic min-rule suite test.",
        "acceptable": "Counts are nonzero.",
        "acceptable_basis": "Synthetic basis.",
        "evidence_metric_keys": ["baseline_MCs_count", "baseline_TCs_count", "baseline_GCs_count"],
        "suite_name": "synthetic_summary_policy_demo",
        "suite_aggregate_policy": {"norm_rollup": "mean"},
    },
    {
        "kind": "summary_metric_status_map",
        "check_id": "epli_target_pattern_specificity",
        "metric_key": "epli_target_pattern_specificity_code",
        "pass_values": [2],
        "warn_values": [1],
        "fail_values": [0],
        "title": "Default external plexiform layer interneuron target pattern encodes perisomatic principal territory",
        "criterion": "Selector should encode broader perisomatic territory.",
        "description": "Synthetic status-map suite test.",
        "acceptable": "Code 1 should warn.",
        "acceptable_basis": "Synthetic basis.",
        "suite_aggregate_policy": {"norm_rollup": "mean"},
    },
    {
        "kind": "summary_metric_range",
        "check_id": "synthetic_soma_diameter",
        "metric_key": "soma_diameter_um",
        "minimum": 8.9,
        "maximum": 10.3,
        "title": "Synthetic external plexiform layer interneuron soma diameter matches target",
        "criterion": "Target soma diameter is 9.6 plus or minus 0.7 micrometers.",
        "description": "Synthetic range-rule suite test.",
        "acceptable": "Observed diameter stays within range.",
        "acceptable_basis": "Synthetic basis.",
        "enabled_when_arg_falsey": "skip_neuron",
    },
]
items = build_rule_items(compile_rule_dispatches(rules), context)
assert [item.check_id for item in items] == [
    "synthetic_summary_policy_demo.overview",
    "baseline_slice_population_counts",
    "epli_target_pattern_specificity",
]
assert [item.status for item in items] == ["WARN", "PASS", "WARN"]
assert items[0].detail_level == "summary"
assert items[0].evidence["suite_aggregate_score"]["score_text"] == "worst WARN, mean norm 0.75"
assert items[1].criterion_latex == r"\bar{x}_{\mathrm{ungrouped}} \geq 1"
assert items[2].evidence["warn_values"] == [1.0]

non_skip_context = _rule_context(
    summary=summary,
    args=SimpleNamespace(skip_neuron=False),
    protocol_result=None,
)
items_non_skip = build_rule_items(compile_rule_dispatches(rules), non_skip_context)
assert [item.check_id for item in items_non_skip] == [
    "synthetic_summary_policy_demo.overview",
    "baseline_slice_population_counts",
    "epli_target_pattern_specificity",
    "synthetic_soma_diameter",
]
assert [item.status for item in items_non_skip] == ["WARN", "PASS", "WARN", "PASS"]


print("neuronunit_summary_validation_suite: OK")
