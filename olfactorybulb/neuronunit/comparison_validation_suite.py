"""SciUnit-backed suite bridge for maintained comparison and exactness rules."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import quantities as pq
import sciunit

from olfactorybulb.audit import AuditItem
from olfactorybulb.audit.core import rounded
from olfactorybulb.neuronunit.capabilities import ProvidesMetricRows, ProvidesMetricSummary
from olfactorybulb.neuronunit.reference_bands import numeric_value
from olfactorybulb.neuronunit.reference_validation_suite import ReferenceValidationModel
from olfactorybulb.neuronunit.suite_presentation import suite_case_result, suite_items_from_judged


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _rounded_dict(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            result[key] = _rounded_dict(value)
        elif isinstance(value, list):
            result[key] = [rounded(float(item)) if _is_finite_number(item) else item for item in value]
        elif _is_finite_number(value):
            result[key] = rounded(float(value))
        else:
            result[key] = value
    return result


@dataclass(frozen=True)
class ComparisonRuleCase:
    rule_kind: str
    check_id: str
    title: str
    criterion: str
    criterion_latex: str
    criterion_formulae: list[str]
    criterion_definitions: list[dict[str, Any]]
    description: str
    acceptable: str
    acceptable_basis: str
    note: str
    metric_key: str
    entity_key: str = "cell_name"
    left_group: str = ""
    right_group: str = ""
    operator: str = ">"
    groups: tuple[str, ...] = ()
    expected: float = 0.0
    tolerance: float = 1e-9
    max_difference: float = 0.0
    pass_status: str = "PASS"
    fail_status: str = "FAIL"


class ComparisonRuleScore(sciunit.Score):
    """Status-bearing score for comparison/exactness validation rules."""

    _allowed_types = (float, int, pq.Quantity)

    def __init__(
        self,
        score: float | int | pq.Quantity,
        *,
        status: str,
        evidence: dict[str, Any],
        case: ComparisonRuleCase,
    ) -> None:
        super().__init__(score)
        self.status = str(status)
        self.evidence = evidence
        self.case = case

    @property
    def norm_score(self) -> float:
        return 1.0 if self.status == "PASS" else 0.0

    def __str__(self) -> str:
        return self.status


class ComparisonRuleTest(sciunit.Test):
    required_capabilities = (ProvidesMetricSummary, ProvidesMetricRows)
    score_type = ComparisonRuleScore

    def __init__(self, case: ComparisonRuleCase) -> None:
        self.case = case
        observation = {
            "rule_kind": case.rule_kind,
            "metric_key": case.metric_key,
            "entity_key": case.entity_key,
            "left_group": case.left_group,
            "right_group": case.right_group,
            "operator": case.operator,
            "groups": case.groups,
            "expected": case.expected,
            "tolerance": case.tolerance,
            "max_difference": case.max_difference,
        }
        super().__init__(observation=observation, name=case.title)

    def validate_observation(self, observation: dict[str, Any]) -> None:
        required = {"rule_kind", "metric_key"}
        missing = sorted(required - set(observation))
        if missing:
            raise sciunit.ObservationError(
                f"ComparisonRuleTest observation is missing required keys: {', '.join(missing)}"
            )

    def generate_prediction(self, model: ReferenceValidationModel) -> dict[str, Any]:
        case = self.case
        if case.rule_kind in {"all_finite_metric", "all_exact_metric"}:
            return {
                "metric_value_map": model.get_metric_value_map(case.metric_key, entity_key=case.entity_key),
            }
        if case.rule_kind in {"group_ordering", "group_abs_diff_max"}:
            return {
                "left_value": model.get_metric_summary(case.left_group, case.metric_key),
                "right_value": model.get_metric_summary(case.right_group, case.metric_key),
            }
        if case.rule_kind == "group_positive":
            return {
                "group_values": {
                    group: model.get_metric_summary(group, case.metric_key)
                    for group in case.groups
                }
            }
        raise ValueError(f"Unsupported comparison rule kind {case.rule_kind!r}")

    def compute_score(self, observation: dict[str, Any], prediction: dict[str, Any]) -> ComparisonRuleScore:
        case = self.case
        if case.rule_kind == "all_finite_metric":
            metric_value_map = dict(prediction["metric_value_map"])
            failing = {
                entity: value
                for entity, value in metric_value_map.items()
                if not _is_finite_number(value)
            }
            evidence = _rounded_dict(
                {
                    "metric_key": case.metric_key,
                    "cell_count": len(metric_value_map),
                    "failing_values": failing,
                }
            )
            status = case.pass_status if not failing else case.fail_status
            return ComparisonRuleScore(len(failing), status=status, evidence=evidence, case=case)

        if case.rule_kind == "all_exact_metric":
            metric_value_map = dict(prediction["metric_value_map"])
            failing = {
                entity: value
                for entity, value in metric_value_map.items()
                if not (_is_finite_number(value) and abs(float(value) - case.expected) <= case.tolerance)
            }
            evidence = _rounded_dict(
                {
                    "metric_key": case.metric_key,
                    "expected": case.expected,
                    "tolerance": case.tolerance,
                    "failing_values": failing,
                }
            )
            status = case.pass_status if not failing else case.fail_status
            return ComparisonRuleScore(len(failing), status=status, evidence=evidence, case=case)

        if case.rule_kind == "group_ordering":
            left_value = numeric_value(prediction["left_value"])
            right_value = numeric_value(prediction["right_value"])
            if case.operator == ">":
                passed = right_value > left_value
                score_value = max(0.0, left_value - right_value)
            elif case.operator == "<":
                passed = right_value < left_value
                score_value = max(0.0, right_value - left_value)
            else:
                raise ValueError(f"Unsupported group_ordering operator {case.operator!r}")
            evidence = _rounded_dict(
                {
                    f"{case.left_group}_mean": left_value,
                    f"{case.right_group}_mean": right_value,
                    f"{case.right_group}_minus_{case.left_group}": right_value - left_value,
                }
            )
            status = case.pass_status if passed else case.fail_status
            return ComparisonRuleScore(score_value, status=status, evidence=evidence, case=case)

        if case.rule_kind == "group_abs_diff_max":
            left_value = numeric_value(prediction["left_value"])
            right_value = numeric_value(prediction["right_value"])
            difference = abs(right_value - left_value)
            passed = difference <= case.max_difference
            evidence = _rounded_dict(
                {
                    f"{case.left_group}_mean": left_value,
                    f"{case.right_group}_mean": right_value,
                    "absolute_difference": difference,
                    "max_difference": case.max_difference,
                }
            )
            score_value = max(0.0, difference - case.max_difference)
            status = case.pass_status if passed else case.fail_status
            return ComparisonRuleScore(score_value, status=status, evidence=evidence, case=case)

        if case.rule_kind == "group_positive":
            group_values = {
                group: numeric_value(value)
                for group, value in dict(prediction["group_values"]).items()
            }
            failing_groups = [
                group
                for group, value in group_values.items()
                if not (_is_finite_number(value) and float(value) > 0.0)
            ]
            evidence = _rounded_dict({f"{group}_mean": value for group, value in group_values.items()})
            if failing_groups:
                evidence["failing_groups"] = list(failing_groups)
            status = case.pass_status if not failing_groups else case.fail_status
            return ComparisonRuleScore(len(failing_groups), status=status, evidence=evidence, case=case)

        raise ValueError(f"Unsupported comparison rule kind {case.rule_kind!r}")


@dataclass(frozen=True)
class CompiledComparisonRuleSuite:
    suite: sciunit.TestSuite
    model: ReferenceValidationModel
    cases: list[ComparisonRuleCase]
    tests: list[ComparisonRuleTest]

    def judge(self) -> list[tuple[ComparisonRuleCase, ComparisonRuleScore]]:
        results: list[tuple[ComparisonRuleCase, ComparisonRuleScore]] = []
        for case, test in zip(self.cases, self.tests, strict=False):
            score = test.judge(self.model)
            results.append((case, score))
        return results


def compile_comparison_rule_suite(
    *,
    cases: list[ComparisonRuleCase],
    summary: dict[str, dict[str, float]],
    metrics: list[dict[str, Any]],
    suite_name: str,
) -> CompiledComparisonRuleSuite:
    tests = [ComparisonRuleTest(case) for case in cases]
    suite = sciunit.TestSuite(tests, name=suite_name)
    model = ReferenceValidationModel(summary=summary, metrics=metrics, name=f"{suite_name}-model")
    return CompiledComparisonRuleSuite(suite=suite, model=model, cases=cases, tests=tests)


def _comparison_score_text(case: ComparisonRuleCase, score: ComparisonRuleScore) -> str:
    if case.rule_kind == "all_finite_metric":
        return "all finite" if int(score.score) == 0 else f"missing {int(score.score)}"
    if case.rule_kind == "all_exact_metric":
        return "all exact" if int(score.score) == 0 else f"failing {int(score.score)}"
    if case.rule_kind == "group_ordering":
        delta = score.evidence.get(f"{case.right_group}_minus_{case.left_group}")
        if _is_finite_number(delta):
            return f"Δ {rounded(float(delta)):g}"
    if case.rule_kind == "group_abs_diff_max":
        difference = score.evidence.get("absolute_difference")
        if _is_finite_number(difference):
            return f"|Δ| {rounded(float(difference)):g}"
    if case.rule_kind == "group_positive":
        return "all positive" if int(score.score) == 0 else f"failing {int(score.score)}"
    return ""


def audit_items_from_comparison_rule_suite(compiled: CompiledComparisonRuleSuite) -> list[AuditItem]:
    judged = compiled.judge()
    def _result_builder(case: ComparisonRuleCase, score: ComparisonRuleScore):
        item = AuditItem(
            check_id=case.check_id,
            status=score.status,
            title=case.title,
            criterion=case.criterion,
            criterion_latex=case.criterion_latex,
            criterion_formulae=case.criterion_formulae,
            criterion_definitions=case.criterion_definitions,
            description=case.description,
            acceptable=case.acceptable,
            acceptable_basis=case.acceptable_basis,
            evidence=score.evidence,
            note=case.note,
        )
        return suite_case_result(
            item,
            score_text=_comparison_score_text(case, score),
            norm_score=score.norm_score,
        )

    return suite_items_from_judged(
        suite_name=str(compiled.suite.name or "comparison-rule-suite"),
        suite_kind_label="Comparison-rule suite",
        judged=judged,
        result_builder=_result_builder,
    )


__all__ = [
    "ComparisonRuleCase",
    "ComparisonRuleScore",
    "ComparisonRuleTest",
    "CompiledComparisonRuleSuite",
    "audit_items_from_comparison_rule_suite",
    "compile_comparison_rule_suite",
]
