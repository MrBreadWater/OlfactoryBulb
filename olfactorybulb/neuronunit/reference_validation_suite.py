"""SciUnit-backed suite bridge for maintained reference-band validations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import quantities as pq
import sciunit

from olfactorybulb.audit import AuditItem
from olfactorybulb.audit.core import rounded
from olfactorybulb.neuronunit.capabilities import (
    ProvidesMetricRows,
    ProvidesMetricSummary,
    ProvidesProtocolEvidenceRows,
)
from olfactorybulb.neuronunit.reference_bands import (
    ReferenceBandObservation,
    measurement_with_unit,
    numeric_value,
)


@dataclass(frozen=True)
class ReferenceBandCase:
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
    observation: ReferenceBandObservation
    reference_annotation: str = ""
    pass_status: str = "PASS"
    fail_status: str = "FAIL"


class ReferenceValidationModel(sciunit.Model, ProvidesMetricSummary, ProvidesMetricRows, ProvidesProtocolEvidenceRows):
    """SciUnit model wrapper around the maintained summary metric table."""

    def __init__(
        self,
        *,
        summary: dict[str, dict[str, float]],
        metrics: list[dict[str, Any]] | None = None,
        protocol_evidence: dict[str, Any] | None = None,
        name: str = "reference-validation-summary-model",
    ) -> None:
        super().__init__(name=name)
        self.summary = summary
        self.metrics = list(metrics or [])
        self.protocol_evidence = dict(protocol_evidence or {})

    def get_metric_summary(self, group: str, metric_key: str, *, unit_text: str = "") -> float | pq.Quantity:
        value = float(self.summary.get(group, {}).get(metric_key, float("nan")))
        return measurement_with_unit(value, unit_text)

    def get_metric_value_map(self, metric_key: str, entity_key: str = "cell_name") -> dict[str, Any]:
        values: dict[str, Any] = {}
        for index, row in enumerate(self.metrics):
            entity = str(row.get(entity_key, f"row_{index}"))
            values[entity] = row.get(metric_key)
        return values

    def get_protocol_evidence_rows(self, evidence_key: str) -> list[dict[str, Any]]:
        rows = self.protocol_evidence.get(evidence_key, [])
        if not isinstance(rows, list):
            return []
        return [dict(row) for row in rows]


class ReferenceBandScore(sciunit.Score):
    """Score object for a maintained reference-band check."""

    _allowed_types = (float, pq.Quantity)
    _best = 0.0
    _worst = float("inf")

    def __init__(
        self,
        score: float | pq.Quantity,
        *,
        passed: bool,
        observed: float | pq.Quantity,
        reference_mean: float | pq.Quantity,
        accepted_low: float | pq.Quantity,
        accepted_high: float | pq.Quantity,
        accepted_band,
    ) -> None:
        super().__init__(score)
        self.passed = bool(passed)
        self.observed = observed
        self.reference_mean = reference_mean
        self.accepted_low = accepted_low
        self.accepted_high = accepted_high
        self.accepted_band = accepted_band

    @property
    def norm_score(self) -> float:
        return 1.0 if self.passed else 0.0

    def __str__(self) -> str:
        return "Pass" if self.passed else "Fail"


class ReferenceBandTest(sciunit.Test):
    """SciUnit test for one reference-band comparison."""

    required_capabilities = (ProvidesMetricSummary,)
    score_type = ReferenceBandScore

    def __init__(self, case: ReferenceBandCase) -> None:
        self.case = case
        observation = {
            "property_name": case.observation.property_name,
            "group": case.observation.group,
            "metric_key": case.observation.metric_key,
            "reference_mean": case.observation.reference_mean,
            "reference_sd": case.observation.reference_sd,
            "unit_text": case.observation.unit_text,
            "band_mode": case.observation.policy.mode,
        }
        super().__init__(observation=observation, name=case.title)

    def validate_observation(self, observation: dict[str, Any]) -> None:
        required = {
            "property_name",
            "group",
            "metric_key",
            "reference_mean",
            "reference_sd",
            "unit_text",
            "band_mode",
        }
        missing = sorted(required - set(observation))
        if missing:
            raise sciunit.ObservationError(
                f"ReferenceBandTest observation is missing required keys: {', '.join(missing)}"
            )

    def generate_prediction(self, model: ReferenceValidationModel) -> float | pq.Quantity:
        obs = self.case.observation
        return model.get_metric_summary(
            obs.group,
            obs.metric_key,
            unit_text=obs.resolved_prediction_unit_text,
        )

    def compute_score(self, observation: dict[str, Any], prediction: float | pq.Quantity) -> ReferenceBandScore:
        obs = self.case.observation
        band = obs.accepted_band
        normalized_prediction = obs.normalize_prediction(prediction)
        accepted_low, accepted_high = obs.accepted_bounds_with_units()
        reference_mean = obs.reference_mean_measurement
        if isinstance(normalized_prediction, pq.Quantity):
            below = accepted_low - normalized_prediction if numeric_value(normalized_prediction) < numeric_value(accepted_low) else 0 * accepted_low.units
            above = normalized_prediction - accepted_high if numeric_value(normalized_prediction) > numeric_value(accepted_high) else 0 * accepted_high.units
            distance = below if numeric_value(below) > 0.0 else above
        else:
            below = float(accepted_low) - float(normalized_prediction) if float(normalized_prediction) < float(accepted_low) else 0.0
            above = float(normalized_prediction) - float(accepted_high) if float(normalized_prediction) > float(accepted_high) else 0.0
            distance = below if below > 0.0 else above
        passed = numeric_value(accepted_low) <= numeric_value(normalized_prediction) <= numeric_value(accepted_high)
        return ReferenceBandScore(
            distance,
            passed=passed,
            observed=normalized_prediction,
            reference_mean=reference_mean,
            accepted_low=accepted_low,
            accepted_high=accepted_high,
            accepted_band=band,
        )


@dataclass(frozen=True)
class CompiledReferenceBandSuite:
    suite: sciunit.TestSuite
    model: ReferenceValidationModel
    cases: list[ReferenceBandCase]
    tests: list[ReferenceBandTest]

    def judge(self) -> list[tuple[ReferenceBandCase, ReferenceBandScore]]:
        results: list[tuple[ReferenceBandCase, ReferenceBandScore]] = []
        for case, test in zip(self.cases, self.tests):
            score = test.judge(self.model)
            results.append((case, score))
        return results


def compile_reference_band_suite(
    *,
    cases: list[ReferenceBandCase],
    summary: dict[str, dict[str, float]],
    suite_name: str,
) -> CompiledReferenceBandSuite:
    tests = [ReferenceBandTest(case) for case in cases]
    suite = sciunit.TestSuite(tests, name=suite_name)
    model = ReferenceValidationModel(summary=summary)
    return CompiledReferenceBandSuite(suite=suite, model=model, cases=cases, tests=tests)


def _evidence_payload(case: ReferenceBandCase, score: ReferenceBandScore) -> dict[str, Any]:
    obs = case.observation
    band = score.accepted_band
    evidence_key = f"{obs.group}_mean"
    evidence: dict[str, Any] = {
        evidence_key: rounded(numeric_value(score.observed)),
        "reference_mean": rounded(numeric_value(score.reference_mean)),
        "reference_unit": obs.unit_text,
        "accepted_low": rounded(numeric_value(score.accepted_low)),
        "accepted_high": rounded(numeric_value(score.accepted_high)),
        "accepted_sigma_multiplier": band.sigma_multiplier,
        "accepted_interval_mode": band.mode,
        "accepted_interval_standard": band.standard_label,
        "__reference_annotations__": {evidence_key: case.reference_annotation},
    }
    if band.lower_bound is not None:
        evidence["accepted_lower_bound"] = rounded(float(band.lower_bound))
    if band.upper_bound is not None:
        evidence["accepted_upper_bound"] = rounded(float(band.upper_bound))
    if not np.isclose(band.raw_low, band.low):
        evidence["unbounded_low"] = rounded(float(band.raw_low))
    if not np.isclose(band.raw_high, band.high):
        evidence["unbounded_high"] = rounded(float(band.raw_high))
    return evidence


def audit_items_from_reference_band_suite(compiled: CompiledReferenceBandSuite) -> list[AuditItem]:
    items: list[AuditItem] = []
    for case, score in compiled.judge():
        obs = case.observation
        item = AuditItem(
            check_id=case.check_id,
            status=case.pass_status if score.passed else case.fail_status,
            title=case.title,
            criterion=case.criterion,
            criterion_latex=case.criterion_latex,
            criterion_formulae=case.criterion_formulae,
            criterion_definitions=case.criterion_definitions,
            description=case.description,
            acceptable=case.acceptable,
            acceptable_basis=case.acceptable_basis,
            evidence=_evidence_payload(case, score),
            note=case.note,
            validation_design_review_status=obs.review.status,
            validation_design_review_note=obs.review.note,
            validation_design_review_reviewer=obs.review.reviewer,
            validation_design_review_required_expertise=obs.review.required_expertise,
            validation_design_review_focus=obs.review.focus,
        )
        items.append(item)
    return items


__all__ = [
    "CompiledReferenceBandSuite",
    "ReferenceBandCase",
    "ReferenceBandScore",
    "ReferenceBandTest",
    "ReferenceValidationModel",
    "audit_items_from_reference_band_suite",
    "compile_reference_band_suite",
]
