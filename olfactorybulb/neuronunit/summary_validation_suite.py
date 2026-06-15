"""SciUnit-backed suite bridge for summary/comparison validation rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import quantities as pq
import sciunit

from olfactorybulb.audit.core import rounded
from olfactorybulb.neuronunit.capabilities import ProvidesMetricSummary
from olfactorybulb.neuronunit.metric_tables import MetricSummaryTable
from olfactorybulb.neuronunit.metric_quantities import (
    MetricQuantitySpec,
    resolve_metric_quantity,
)
from olfactorybulb.neuronunit.reference_bands import numeric_value
from olfactorybulb.neuronunit.reference_validation_suite import ReferenceValidationModel
from olfactorybulb.neuronunit.scalar_observations import (
    ScalarMetricValue,
    ScalarStatusMapPolicy,
    is_finite_scalar,
)
from olfactorybulb.neuronunit.suite_presentation import (
    audit_item_adapter_spec_from_case,
    suite_case_result_from_spec,
    suite_items_from_judged,
)
from olfactorybulb.neuronunit.suite_scores import SuiteCaseScorePayload, SuiteDescriptor


def _rounded_dict(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            result[key] = _rounded_dict(value)
        elif isinstance(value, list):
            result[key] = [rounded(numeric_value(item)) if is_finite_scalar(item) else item for item in value]
        elif is_finite_scalar(value):
            result[key] = rounded(numeric_value(value) if isinstance(value, pq.Quantity) else float(value))
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
    metric_quantity: MetricQuantitySpec | None = None
    evidence_metric_keys: list[str] = field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None
    pass_status: str = "PASS"
    fail_status: str = "FAIL"
    status_map_policy: ScalarStatusMapPolicy | None = None

    def __post_init__(self) -> None:
        if self.metric_quantity is None:
            object.__setattr__(self, "metric_quantity", resolve_metric_quantity(self.metric_key))
        object.__setattr__(self, "pass_status", str(self.pass_status or "PASS").strip().upper() or "PASS")
        object.__setattr__(self, "fail_status", str(self.fail_status or "FAIL").strip().upper() or "FAIL")
        if self.rule_kind == "summary_metric_status_map" and self.status_map_policy is None:
            raise ValueError("summary_metric_status_map cases require a status_map_policy")

    def observation_payload(self) -> dict[str, Any]:
        payload = {
            "rule_kind": self.rule_kind,
            "metric_key": self.metric_key,
            "group": self.group,
        }
        if self.minimum is not None:
            payload["minimum"] = self.minimum
        if self.maximum is not None:
            payload["maximum"] = self.maximum
        if self.status_map_policy is not None:
            payload.update(self.status_map_policy.observation_payload())
        return payload


class SummaryRuleScore(sciunit.Score):
    """Status-bearing score for summary metric rules."""

    _allowed_types = (float, int, pq.Quantity)

    def __init__(
        self,
        score: float | int | pq.Quantity,
        *,
        status: str,
        observed: ScalarMetricValue,
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
        super().__init__(observation=case.observation_payload(), name=case.title)

    def validate_observation(self, observation: dict[str, Any]) -> None:
        required = {"rule_kind", "metric_key", "group"}
        missing = sorted(required - set(observation))
        if missing:
            raise sciunit.ObservationError(
                f"SummaryRuleTest observation is missing required keys: {', '.join(missing)}"
            )

    def generate_prediction(self, model: ReferenceValidationModel) -> ScalarMetricValue:
        metric_quantity = self.case.metric_quantity or resolve_metric_quantity(self.case.metric_key)
        return ScalarMetricValue(
            metric_quantity=metric_quantity,
            group=self.case.group,
            value=model.get_metric_summary(
                self.case.group,
                self.case.metric_key,
                unit_text=metric_quantity.unit_text,
            ),
        )

    def compute_score(self, observation: dict[str, Any], prediction: ScalarMetricValue) -> SummaryRuleScore:
        observed_numeric = prediction.numeric
        if self.case.rule_kind == "summary_metric_min":
            passed = is_finite_scalar(observed_numeric) and observed_numeric >= float(self.case.minimum)
            distance = max(0.0, float(self.case.minimum) - observed_numeric) if is_finite_scalar(observed_numeric) else float("inf")
            status = self.case.pass_status if passed else self.case.fail_status
            return SummaryRuleScore(distance, status=status, observed=prediction, case=self.case)
        if self.case.rule_kind == "summary_metric_max":
            passed = is_finite_scalar(observed_numeric) and observed_numeric <= float(self.case.maximum)
            distance = max(0.0, observed_numeric - float(self.case.maximum)) if is_finite_scalar(observed_numeric) else float("inf")
            status = self.case.pass_status if passed else self.case.fail_status
            return SummaryRuleScore(distance, status=status, observed=prediction, case=self.case)
        if self.case.rule_kind == "summary_metric_range":
            passed = is_finite_scalar(observed_numeric) and float(self.case.minimum) <= observed_numeric <= float(self.case.maximum)
            if not is_finite_scalar(observed_numeric):
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
            status = self.case.status_map_policy.status_for(observed_numeric) if self.case.status_map_policy else "FAIL"
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
    summary: MetricSummaryTable | dict[str, dict[str, float]],
    suite_name: str,
) -> CompiledSummaryRuleSuite:
    tests = [SummaryRuleTest(case) for case in cases]
    suite = sciunit.TestSuite(tests, name=suite_name)
    model = ReferenceValidationModel(summary=summary, name=f"{suite_name}-model")
    return CompiledSummaryRuleSuite(suite=suite, model=model, cases=cases, tests=tests)


def _summary_score_text(case: SummaryRuleCase, score: SummaryRuleScore) -> str:
    observed = score.observed.numeric
    if not is_finite_scalar(observed):
        return ""
    unit_text = score.observed.unit_text
    unit_suffix = f" {unit_text}" if unit_text else ""
    return f"observed {rounded(float(observed)):g}{unit_suffix}"


def _summary_score_payload(case: SummaryRuleCase, score: SummaryRuleScore) -> SuiteCaseScorePayload:
    unit_text = score.observed.unit_text
    prediction: dict[str, Any] = {}
    if case.rule_kind == "summary_metric_min":
        prediction["minimum"] = case.minimum
    elif case.rule_kind == "summary_metric_max":
        prediction["maximum"] = case.maximum
    elif case.rule_kind == "summary_metric_range":
        prediction["minimum"] = case.minimum
        prediction["maximum"] = case.maximum
    elif case.rule_kind == "summary_metric_status_map":
        prediction.update(case.status_map_policy.observation_payload() if case.status_map_policy else {})
    return SuiteCaseScorePayload(
        score_kind=case.rule_kind,
        score_value=numeric_value(score.score),
        score_units=(
            unit_text
            if case.rule_kind in {"summary_metric_min", "summary_metric_max", "summary_metric_range"}
            else ""
        ),
        score_interpretation=(
            "Status-bearing summary-rule score derived from the observed group summary "
            "value against the configured summary-rule contract."
        ),
        observation=score.observed.observation_payload(),
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
        observed = score.observed.numeric
        base: dict[str, Any] = {"group": score.observed.group, "observed": observed}
        if score.observed.unit_text:
            base["metric_unit"] = score.observed.unit_text
        if score.observed.quantity_name:
            base["metric_quantity_name"] = score.observed.quantity_name
        if case.rule_kind == "summary_metric_min":
            base["minimum"] = case.minimum
        elif case.rule_kind == "summary_metric_max":
            base["maximum"] = case.maximum
        elif case.rule_kind == "summary_metric_range":
            base["minimum"] = case.minimum
            base["maximum"] = case.maximum
        elif case.rule_kind == "summary_metric_status_map":
            base.update(case.status_map_policy.observation_payload() if case.status_map_policy else {})
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
