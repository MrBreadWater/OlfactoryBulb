"""SciUnit-backed suite bridge for summary/comparison validation rules."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import quantities as pq
import sciunit

from olfactorybulb.audit.core import rounded
from olfactorybulb.neuronunit.capabilities import ProvidesMetricSummary
from olfactorybulb.neuronunit.reference_bands import numeric_value
from olfactorybulb.neuronunit.reference_validation_suite import ReferenceValidationModel
from olfactorybulb.neuronunit.suite_presentation import (
    audit_item_adapter_spec_from_case,
    suite_case_result_from_spec,
    suite_items_from_judged,
)
from olfactorybulb.neuronunit.suite_scores import SuiteCaseScorePayload, SuiteDescriptor


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
class SummaryRuleCase:
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
    group: str
    evidence_metric_keys: list[str] = field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None
    pass_values: tuple[float, ...] = ()
    warn_values: tuple[float, ...] = ()
    fail_values: tuple[float, ...] = ()
    pass_status: str = "PASS"
    fail_status: str = "FAIL"
    default_status: str = "FAIL"


class SummaryRuleScore(sciunit.Score):
    """Status-bearing score for summary metric rules."""

    _allowed_types = (float, int, pq.Quantity)

    def __init__(
        self,
        score: float | int | pq.Quantity,
        *,
        status: str,
        observed: float | int | pq.Quantity,
        case: SummaryRuleCase,
    ) -> None:
        super().__init__(score)
        self.status = str(status)
        self.observed = observed
        self.case = case

    @property
    def norm_score(self) -> float:
        if self.status == "PASS":
            return 1.0
        if self.status == "WARN":
            return 0.5
        return 0.0

    def __str__(self) -> str:
        return self.status


class SummaryRuleTest(sciunit.Test):
    required_capabilities = (ProvidesMetricSummary,)
    score_type = SummaryRuleScore

    def __init__(self, case: SummaryRuleCase) -> None:
        self.case = case
        observation = {
            "rule_kind": case.rule_kind,
            "metric_key": case.metric_key,
            "group": case.group,
        }
        if case.minimum is not None:
            observation["minimum"] = case.minimum
        if case.maximum is not None:
            observation["maximum"] = case.maximum
        if case.pass_values:
            observation["pass_values"] = case.pass_values
        if case.warn_values:
            observation["warn_values"] = case.warn_values
        if case.fail_values:
            observation["fail_values"] = case.fail_values
        super().__init__(observation=observation, name=case.title)

    def validate_observation(self, observation: dict[str, Any]) -> None:
        required = {"rule_kind", "metric_key", "group"}
        missing = sorted(required - set(observation))
        if missing:
            raise sciunit.ObservationError(
                f"SummaryRuleTest observation is missing required keys: {', '.join(missing)}"
            )

    def generate_prediction(self, model: ReferenceValidationModel) -> float | pq.Quantity:
        return model.get_metric_summary(self.case.group, self.case.metric_key)

    def compute_score(self, observation: dict[str, Any], prediction: float | pq.Quantity) -> SummaryRuleScore:
        observed_numeric = numeric_value(prediction)
        if self.case.rule_kind == "summary_metric_min":
            passed = _is_finite_number(observed_numeric) and observed_numeric >= float(self.case.minimum)
            distance = max(0.0, float(self.case.minimum) - observed_numeric) if _is_finite_number(observed_numeric) else float("inf")
            status = self.case.pass_status if passed else self.case.fail_status
            return SummaryRuleScore(distance, status=status, observed=prediction, case=self.case)
        if self.case.rule_kind == "summary_metric_max":
            passed = _is_finite_number(observed_numeric) and observed_numeric <= float(self.case.maximum)
            distance = max(0.0, observed_numeric - float(self.case.maximum)) if _is_finite_number(observed_numeric) else float("inf")
            status = self.case.pass_status if passed else self.case.fail_status
            return SummaryRuleScore(distance, status=status, observed=prediction, case=self.case)
        if self.case.rule_kind == "summary_metric_range":
            passed = _is_finite_number(observed_numeric) and float(self.case.minimum) <= observed_numeric <= float(self.case.maximum)
            if not _is_finite_number(observed_numeric):
                distance = float("inf")
            elif observed_numeric < float(self.case.minimum):
                distance = float(self.case.minimum) - observed_numeric
            elif observed_numeric > float(self.case.maximum):
                distance = observed_numeric - float(self.case.maximum)
            else:
                distance = 0.0
            status = self.case.pass_status if passed else self.case.fail_status
            return SummaryRuleScore(distance, status=status, observed=prediction, case=self.case)
        if self.case.rule_kind == "summary_metric_status_map":
            if not _is_finite_number(observed_numeric):
                status = "FAIL"
            elif observed_numeric in self.case.pass_values:
                status = "PASS"
            elif observed_numeric in self.case.warn_values:
                status = "WARN"
            elif observed_numeric in self.case.fail_values:
                status = "FAIL"
            else:
                status = self.case.default_status
            return SummaryRuleScore(observed_numeric, status=status, observed=prediction, case=self.case)
        raise ValueError(f"Unsupported summary rule kind {self.case.rule_kind!r}")


@dataclass(frozen=True)
class CompiledSummaryRuleSuite:
    suite: sciunit.TestSuite
    model: ReferenceValidationModel
    cases: list[SummaryRuleCase]
    tests: list[SummaryRuleTest]

    def judge(self) -> list[tuple[SummaryRuleCase, SummaryRuleScore]]:
        results: list[tuple[SummaryRuleCase, SummaryRuleScore]] = []
        for case, test in zip(self.cases, self.tests, strict=False):
            score = test.judge(self.model)
            results.append((case, score))
        return results


def compile_summary_rule_suite(
    *,
    cases: list[SummaryRuleCase],
    summary: dict[str, dict[str, float]],
    suite_name: str,
) -> CompiledSummaryRuleSuite:
    tests = [SummaryRuleTest(case) for case in cases]
    suite = sciunit.TestSuite(tests, name=suite_name)
    model = ReferenceValidationModel(summary=summary, name=f"{suite_name}-model")
    return CompiledSummaryRuleSuite(suite=suite, model=model, cases=cases, tests=tests)


def _summary_score_text(case: SummaryRuleCase, score: SummaryRuleScore) -> str:
    observed = numeric_value(score.observed)
    if not _is_finite_number(observed):
        return ""
    return f"observed {rounded(float(observed)):g}"


def _summary_score_payload(case: SummaryRuleCase, score: SummaryRuleScore) -> SuiteCaseScorePayload:
    observed = numeric_value(score.observed)
    prediction: dict[str, Any] = {}
    if case.rule_kind == "summary_metric_min":
        prediction["minimum"] = case.minimum
    elif case.rule_kind == "summary_metric_max":
        prediction["maximum"] = case.maximum
    elif case.rule_kind == "summary_metric_range":
        prediction["minimum"] = case.minimum
        prediction["maximum"] = case.maximum
    elif case.rule_kind == "summary_metric_status_map":
        prediction["pass_values"] = list(case.pass_values)
        prediction["warn_values"] = list(case.warn_values)
        prediction["fail_values"] = list(case.fail_values)
    return SuiteCaseScorePayload(
        score_kind=case.rule_kind,
        score_value=numeric_value(score.score),
        score_interpretation=(
            "Status-bearing summary-rule score derived from the observed group summary "
            "value against the configured summary-rule contract."
        ),
        observation={
            "group": case.group,
            "metric_key": case.metric_key,
            "observed": rounded(observed),
        },
        prediction=prediction or None,
        normalization={
            "norm_score": score.norm_score,
            "status": score.status,
        },
    )


def audit_items_from_summary_rule_suite(
    compiled: CompiledSummaryRuleSuite,
    *,
    descriptor: SuiteDescriptor | None = None,
) -> list[AuditItem]:
    judged = compiled.judge()
    if descriptor is None:
        descriptor = SuiteDescriptor(
            suite_id=str(compiled.suite.name or "summary-rule-suite"),
            suite_kind_label="Summary-rule suite",
        )
    def _result_builder(case: SummaryRuleCase, score: SummaryRuleScore):
        observed = numeric_value(score.observed)
        base: dict[str, Any] = {"group": case.group, "observed": observed}
        if case.rule_kind == "summary_metric_min":
            base["minimum"] = case.minimum
        elif case.rule_kind == "summary_metric_max":
            base["maximum"] = case.maximum
        elif case.rule_kind == "summary_metric_range":
            base["minimum"] = case.minimum
            base["maximum"] = case.maximum
        elif case.rule_kind == "summary_metric_status_map":
            base["pass_values"] = sorted(case.pass_values)
            base["warn_values"] = sorted(case.warn_values)
            base["fail_values"] = sorted(case.fail_values)
        for metric_key in case.evidence_metric_keys:
            base[metric_key] = compiled.model.summary.get(case.group, {}).get(metric_key, float("nan"))
        spec = audit_item_adapter_spec_from_case(case)
        return suite_case_result_from_spec(
            spec,
            status=score.status,
            evidence=_rounded_dict(base),
            score_text=_summary_score_text(case, score),
            norm_score=score.norm_score,
            score_payload=_summary_score_payload(case, score),
        )

    return suite_items_from_judged(
        descriptor=descriptor,
        judged=judged,
        result_builder=_result_builder,
    )


__all__ = [
    "CompiledSummaryRuleSuite",
    "SummaryRuleCase",
    "SummaryRuleScore",
    "SummaryRuleTest",
    "audit_items_from_summary_rule_suite",
    "compile_summary_rule_suite",
]
