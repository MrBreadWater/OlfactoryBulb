"""SciUnit-backed suite bridge for maintained reference-band validations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import quantities as pq
import sciunit

from olfactorybulb.audit.core import rounded
from olfactorybulb.audit.protocol_evidence import (
    ProtocolEvidenceBundle,
    ProtocolEvidenceRowTable,
    coerce_protocol_evidence_bundle,
)
from olfactorybulb.neuronunit.capabilities import (
    ProvidesMetricRows,
    ProvidesMetricSummary,
    ProvidesProtocolEvidenceBundle,
    ProvidesProtocolEvidenceMap,
    ProvidesProtocolEvidenceRows,
)
from olfactorybulb.neuronunit.metric_tables import (
    MetricValueMapPayload,
    MetricRowRecord,
    MetricSummaryRecord,
    MetricSummaryTable,
    MetricTable,
    coerce_metric_summary_table,
    coerce_metric_table,
)
from olfactorybulb.neuronunit.reference_bands import (
    ReferenceBandObservation,
    measurement_with_unit,
    numeric_value,
)
from olfactorybulb.neuronunit.suite_presentation import (
    audit_item_adapter_spec_from_case,
    suite_case_result_from_spec,
    suite_items_from_judged,
)
from olfactorybulb.neuronunit.suite_scores import SuiteCaseScorePayload, SuiteDescriptor
from olfactorybulb.neuronunit.validation_evidence import ReferenceBandEvidencePayload


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


@dataclass(frozen=True)
class ReferenceValidationRuntimeData:
    """Typed runtime bundle shared across migrated NeuronUnit validation suites."""

    summary: MetricSummaryTable | Mapping[str, MetricSummaryRecord | Mapping[str, float]]
    metrics: MetricTable | Sequence[MetricRowRecord | Mapping[str, Any]] = ()
    protocol_evidence: ProtocolEvidenceBundle | Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", coerce_metric_summary_table(self.summary))
        object.__setattr__(self, "metrics", coerce_metric_table(self.metrics or ()))
        object.__setattr__(self, "protocol_evidence", coerce_protocol_evidence_bundle(self.protocol_evidence))


class ReferenceValidationModel(
    sciunit.Model,
    ProvidesMetricSummary,
    ProvidesMetricRows,
    ProvidesProtocolEvidenceRows,
    ProvidesProtocolEvidenceMap,
    ProvidesProtocolEvidenceBundle,
):
    """SciUnit model wrapper around the maintained summary metric table."""

    def __init__(
        self,
        *,
        runtime_data: ReferenceValidationRuntimeData,
        name: str = "reference-validation-summary-model",
    ) -> None:
        super().__init__(name=name)
        self.runtime_data = runtime_data

    @property
    def summary(self) -> MetricSummaryTable:
        return self.runtime_data.summary

    @property
    def metrics(self) -> MetricTable:
        return self.runtime_data.metrics

    @property
    def protocol_evidence(self) -> ProtocolEvidenceBundle:
        return self.runtime_data.protocol_evidence

    def get_metric_summary(self, group: str, metric_key: str, *, unit_text: str = "") -> float | pq.Quantity:
        value = self.summary.metric_value(group, metric_key)
        return measurement_with_unit(value, unit_text)

    def get_metric_value_map(
        self,
        metric_key: str,
        entity_key: str = "cell_name",
        *,
        unit_text: str = "",
    ) -> MetricValueMapPayload:
        values: dict[str, Any] = {}
        for entity, value in self.metrics.metric_value_map(metric_key, entity_key=entity_key).items():
            if isinstance(value, bool) or value is None:
                values[entity] = value
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                values[entity] = value
                continue
            values[entity] = measurement_with_unit(numeric, unit_text)
        return MetricValueMapPayload.from_mapping(values)

    def get_protocol_evidence_rows(self, evidence_key: str) -> ProtocolEvidenceRowTable:
        return self.protocol_evidence.row_table(evidence_key)

    def get_protocol_evidence_map(self) -> dict[str, Any]:
        return self.protocol_evidence.to_dict()

    def get_protocol_evidence_bundle(self) -> ProtocolEvidenceBundle:
        return self.protocol_evidence


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
        super().__init__(observation=case.observation.observation_payload(), name=case.title)

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
    summary: MetricSummaryTable | Mapping[str, MetricSummaryRecord | Mapping[str, float]],
    protocol_evidence: ProtocolEvidenceBundle | Mapping[str, object] | None = None,
    suite_name: str,
) -> CompiledReferenceBandSuite:
    tests = [ReferenceBandTest(case) for case in cases]
    suite = sciunit.TestSuite(tests, name=suite_name)
    model = ReferenceValidationModel(
        runtime_data=ReferenceValidationRuntimeData(
            summary=summary,
            protocol_evidence=protocol_evidence,
        ),
    )
    return CompiledReferenceBandSuite(suite=suite, model=model, cases=cases, tests=tests)


def _evidence_payload(case: ReferenceBandCase, score: ReferenceBandScore) -> dict[str, Any]:
    obs = case.observation
    band = score.accepted_band
    evidence_key = f"{obs.group}_mean"
    return ReferenceBandEvidencePayload(
        observed_key=evidence_key,
        observed_value=numeric_value(score.observed),
        reference_mean=numeric_value(score.reference_mean),
        reference_unit=obs.unit_text,
        accepted_low=numeric_value(score.accepted_low),
        accepted_high=numeric_value(score.accepted_high),
        accepted_sigma_multiplier=band.sigma_multiplier,
        accepted_interval_mode=band.mode,
        accepted_interval_standard=band.standard_label,
        reference_annotation=case.reference_annotation,
        accepted_lower_bound=band.lower_bound,
        accepted_upper_bound=band.upper_bound,
        unbounded_low=float(band.raw_low) if not np.isclose(band.raw_low, band.low) else None,
        unbounded_high=float(band.raw_high) if not np.isclose(band.raw_high, band.high) else None,
    ).to_dict()


def _reference_band_score_text(case: ReferenceBandCase, score: ReferenceBandScore) -> str:
    observed_value = rounded(numeric_value(score.observed))
    unit_text = str(case.observation.unit_text or "").strip()
    if observed_value is None:
        return ""
    return f"observed {observed_value:g}{f' {unit_text}' if unit_text else ''}"


def _reference_band_score_payload(case: ReferenceBandCase, score: ReferenceBandScore) -> SuiteCaseScorePayload:
    unit_text = str(case.observation.unit_text or "").strip()
    return SuiteCaseScorePayload(
        score_kind="reference_band_distance",
        score_value=numeric_value(score.score),
        score_units=unit_text,
        score_interpretation=(
            "Distance from the accepted reference band in observed units. "
            "Zero means the observed value remained inside the accepted band."
        ),
        observation={
            "observed": rounded(numeric_value(score.observed)),
            "reference_mean": rounded(numeric_value(score.reference_mean)),
        },
        prediction={
            "accepted_low": rounded(numeric_value(score.accepted_low)),
            "accepted_high": rounded(numeric_value(score.accepted_high)),
            "interval_mode": str(case.observation.policy.mode),
        },
        normalization={
            "norm_score": score.norm_score,
            "passed": bool(score.passed),
        },
    )


def _reference_band_case_result(case: ReferenceBandCase, score: ReferenceBandScore):
    obs = case.observation
    spec = audit_item_adapter_spec_from_case(
        case,
        validation_review=obs.review,
    )
    return suite_case_result_from_spec(
        spec,
        status=case.pass_status if score.passed else case.fail_status,
        evidence=_evidence_payload(case, score),
        score_text=_reference_band_score_text(case, score),
        norm_score=score.norm_score,
        score_payload=_reference_band_score_payload(case, score),
    )


def audit_items_from_reference_band_suite(
    compiled: CompiledReferenceBandSuite,
    *,
    descriptor: SuiteDescriptor | None = None,
) -> list[AuditItem]:
    judged = compiled.judge()
    if descriptor is None:
        descriptor = SuiteDescriptor(
            suite_id=str(compiled.suite.name or "reference-band-suite"),
            suite_kind_label="Reference-band suite",
        )
    return suite_items_from_judged(
        descriptor=descriptor,
        judged=judged,
        result_builder=_reference_band_case_result,
    )


__all__ = [
    "CompiledReferenceBandSuite",
    "ReferenceBandCase",
    "ReferenceBandScore",
    "ReferenceBandTest",
    "ReferenceValidationModel",
    "ReferenceValidationRuntimeData",
    "audit_items_from_reference_band_suite",
    "compile_reference_band_suite",
]
