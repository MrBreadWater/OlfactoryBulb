"""Typed suite aggregation for SciUnit-backed maintained validation suites."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any

from olfactorybulb.audit.core import rounded


_STATUS_RANK = {"FAIL": 3, "WARN": 2, "PASS": 1}
_VALID_STATUSES = frozenset(_STATUS_RANK)
_EQUIVALENCE_CASE_SCORE_KINDS = frozenset({"equivalence_only", "hybrid_residual_equivalence"})
_WELCH_CASE_SCORE_KINDS = frozenset({"welch_only", "hybrid_residual_welch"})


def _normalized_status(value: str) -> str:
    status = str(value or "").upper().strip()
    if status not in _VALID_STATUSES:
        raise ValueError(f"Unsupported suite case status {value!r}")
    return status


def _normalized_norm_score(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(candidate):
        return None
    return rounded(candidate, digits=3)


def _normalized_mapping(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return copy.deepcopy(dict(value))


def _normalized_score_value(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(candidate):
        return None
    return rounded(candidate, digits=3)


def _normalized_case_weight(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(candidate) or candidate <= 0.0:
        return None
    return rounded(candidate, digits=3)


@dataclass(frozen=True)
class SuiteCaseScorePayload:
    score_kind: str
    score_value: float | None = None
    score_units: str = ""
    score_interpretation: str = ""
    observation: dict[str, Any] | None = None
    prediction: dict[str, Any] | None = None
    normalization: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "score_kind", str(self.score_kind).strip())
        object.__setattr__(self, "score_value", _normalized_score_value(self.score_value))
        object.__setattr__(self, "score_units", str(self.score_units).strip())
        object.__setattr__(self, "score_interpretation", str(self.score_interpretation).strip())
        object.__setattr__(self, "observation", _normalized_mapping(self.observation))
        object.__setattr__(self, "prediction", _normalized_mapping(self.prediction))
        object.__setattr__(self, "normalization", _normalized_mapping(self.normalization))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "score_kind": self.score_kind,
        }
        if self.score_value is not None:
            payload["score_value"] = float(self.score_value)
        if self.score_units:
            payload["score_units"] = self.score_units
        if self.score_interpretation:
            payload["score_interpretation"] = self.score_interpretation
        if self.observation is not None:
            payload["observation"] = copy.deepcopy(self.observation)
        if self.prediction is not None:
            payload["prediction"] = copy.deepcopy(self.prediction)
        if self.normalization is not None:
            payload["normalization"] = copy.deepcopy(self.normalization)
        return payload


@dataclass(frozen=True)
class SuiteCaseSummary:
    check_id: str
    title: str
    status: str
    score_text: str = ""
    norm_score: float | None = None
    case_score: SuiteCaseScorePayload | None = None
    case_weight: float | None = None
    case_weight_label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "check_id", str(self.check_id).strip())
        object.__setattr__(self, "title", str(self.title).strip())
        object.__setattr__(self, "status", _normalized_status(self.status))
        object.__setattr__(self, "score_text", str(self.score_text).strip())
        object.__setattr__(self, "norm_score", _normalized_norm_score(self.norm_score))
        object.__setattr__(self, "case_weight", _normalized_case_weight(self.case_weight))
        object.__setattr__(self, "case_weight_label", str(self.case_weight_label).strip())

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "check_id": self.check_id,
            "title": self.title,
            "status": self.status,
        }
        if self.score_text:
            payload["score_text"] = self.score_text
        if self.norm_score is not None:
            payload["norm_score"] = float(self.norm_score)
        if self.case_score is not None:
            payload["case_score"] = self.case_score.to_dict()
        if self.case_weight is not None:
            payload["case_weight"] = float(self.case_weight)
        if self.case_weight_label:
            payload["case_weight_label"] = self.case_weight_label
        return payload


@dataclass(frozen=True)
class SuiteStatisticalSummary:
    score_family_category: str
    statistical_test_family: str
    rollup_method: str
    rollup_source: str
    rollup_pvalue: float
    available_case_count: int
    total_case_count: int
    score_text: str
    score_interpretation: str
    threshold: float | None = None
    threshold_key: str = ""
    threshold_direction: str = ""
    gate_passed: bool | None = None
    case_pvalues: tuple[float, ...] = ()
    case_check_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "score_family_category", str(self.score_family_category).strip())
        object.__setattr__(self, "statistical_test_family", str(self.statistical_test_family).strip())
        object.__setattr__(self, "rollup_method", str(self.rollup_method).strip())
        object.__setattr__(self, "rollup_source", str(self.rollup_source).strip())
        object.__setattr__(self, "rollup_pvalue", float(self.rollup_pvalue))
        object.__setattr__(self, "available_case_count", int(self.available_case_count))
        object.__setattr__(self, "total_case_count", int(self.total_case_count))
        object.__setattr__(self, "score_text", str(self.score_text).strip())
        object.__setattr__(self, "score_interpretation", str(self.score_interpretation).strip())
        object.__setattr__(self, "threshold", _normalized_score_value(self.threshold))
        object.__setattr__(self, "threshold_key", str(self.threshold_key).strip())
        object.__setattr__(self, "threshold_direction", str(self.threshold_direction).strip())
        object.__setattr__(
            self,
            "case_pvalues",
            tuple(
                normalized
                for normalized in (_normalized_score_value(value) for value in self.case_pvalues)
                if normalized is not None
            ),
        )
        object.__setattr__(
            self,
            "case_check_ids",
            tuple(str(check_id).strip() for check_id in self.case_check_ids if str(check_id).strip()),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "score_family_category": self.score_family_category,
            "statistical_test_family": self.statistical_test_family,
            "rollup_method": self.rollup_method,
            "rollup_source": self.rollup_source,
            "rollup_pvalue": float(self.rollup_pvalue),
            "available_case_count": self.available_case_count,
            "total_case_count": self.total_case_count,
            "score_text": self.score_text,
            "score_interpretation": self.score_interpretation,
            "case_pvalues": [float(value) for value in self.case_pvalues],
            "case_check_ids": list(self.case_check_ids),
        }
        if self.threshold is not None:
            payload["threshold"] = float(self.threshold)
        if self.threshold_key:
            payload["threshold_key"] = self.threshold_key
        if self.threshold_direction:
            payload["threshold_direction"] = self.threshold_direction
        if self.gate_passed is not None:
            payload["gate_passed"] = bool(self.gate_passed)
        return payload


@dataclass(frozen=True)
class SuiteAggregatePolicy:
    status_rollup: str = "worst_case"
    norm_rollup: str = "minimum"

    def __post_init__(self) -> None:
        status_rollup = str(self.status_rollup or "").strip().lower()
        norm_rollup = str(self.norm_rollup or "").strip().lower()
        if status_rollup != "worst_case":
            raise ValueError(
                f"Unsupported suite status rollup {self.status_rollup!r}; "
                "the maintained bridge currently supports only worst_case"
            )
        if norm_rollup not in {"minimum", "mean", "median", "weighted_mean"}:
            raise ValueError(
                f"Unsupported suite norm rollup {self.norm_rollup!r}; "
                "the maintained bridge currently supports minimum, mean, median, or weighted_mean"
            )
        object.__setattr__(self, "status_rollup", status_rollup)
        object.__setattr__(self, "norm_rollup", norm_rollup)

    @property
    def score_kind(self) -> str:
        return f"{self.status_rollup}_{self.norm_rollup}_norm"

    @property
    def interpretation(self) -> str:
        norm_text = {
            "minimum": "minimum normalized case score",
            "mean": "mean normalized case score",
            "median": "median normalized case score",
            "weighted_mean": "weighted mean normalized case score using the per-case aggregate weights",
        }[self.norm_rollup]
        return (
            "Aggregate suite score derived from the worst detailed-case status "
            f"plus the {norm_text}."
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "status_rollup": self.status_rollup,
            "norm_rollup": self.norm_rollup,
            "score_kind": self.score_kind,
            "score_interpretation": self.interpretation,
        }


DEFAULT_SUITE_AGGREGATE_POLICY = SuiteAggregatePolicy()


@dataclass(frozen=True)
class SuiteDescriptor:
    suite_id: str
    suite_kind_label: str
    candidate_ids: tuple[str, ...] = ()
    aggregate_policy: SuiteAggregatePolicy = DEFAULT_SUITE_AGGREGATE_POLICY

    def __post_init__(self) -> None:
        suite_id = str(self.suite_id).strip()
        suite_kind_label = str(self.suite_kind_label).strip()
        if not suite_id:
            raise ValueError("suite_id must be non-empty")
        if not suite_kind_label:
            raise ValueError("suite_kind_label must be non-empty")
        object.__setattr__(self, "suite_id", suite_id)
        object.__setattr__(self, "suite_kind_label", suite_kind_label)
        object.__setattr__(
            self,
            "candidate_ids",
            tuple(str(candidate_id).strip() for candidate_id in self.candidate_ids if str(candidate_id).strip()),
        )


def _status_summary(case_summaries: tuple[SuiteCaseSummary, ...]) -> dict[str, int]:
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for case in case_summaries:
        counts[case.status] = counts.get(case.status, 0) + 1
    return counts


def _worst_case_status(case_summaries: tuple[SuiteCaseSummary, ...]) -> str:
    if not case_summaries:
        return "PASS"
    return max(case_summaries, key=lambda case: _STATUS_RANK.get(case.status, 0)).status


def _norm_score_summary(case_summaries: tuple[SuiteCaseSummary, ...]) -> dict[str, Any] | None:
    scores = [float(case.norm_score) for case in case_summaries if case.norm_score is not None]
    if not scores:
        return None
    weighted_cases = [case for case in case_summaries if case.norm_score is not None]
    weights = [float(case.case_weight) if case.case_weight is not None else 1.0 for case in weighted_cases]
    total_weight = sum(weights)
    weighted_mean = None
    if total_weight > 0.0:
        weighted_mean = rounded(
            sum(float(case.norm_score) * weight for case, weight in zip(weighted_cases, weights, strict=False)) / total_weight,
            digits=3,
        )
    scores.sort()
    midpoint = len(scores) // 2
    if len(scores) % 2 == 0:
        median = (scores[midpoint - 1] + scores[midpoint]) / 2.0
    else:
        median = scores[midpoint]
    summary: dict[str, Any] = {
        "mean": rounded(sum(scores) / len(scores), digits=3),
        "median": rounded(median, digits=3),
        "min": rounded(scores[0], digits=3),
        "max": rounded(scores[-1], digits=3),
        "count": float(len(scores)),
    }
    if weighted_mean is not None:
        summary["weighted_mean"] = weighted_mean
        summary["total_weight"] = rounded(total_weight, digits=3)
    explicit_labels = {
        case.case_weight_label
        for case in weighted_cases
        if case.case_weight is not None and case.case_weight_label
    }
    if explicit_labels:
        summary["weight_label"] = next(iter(explicit_labels)) if len(explicit_labels) == 1 else "mixed"
    return summary


def _aggregate_norm_score(
    norm_summary: dict[str, Any] | None,
    *,
    policy: SuiteAggregatePolicy,
) -> float | None:
    if norm_summary is None:
        return None
    key = {
        "minimum": "min",
        "mean": "mean",
        "median": "median",
        "weighted_mean": "weighted_mean",
    }[policy.norm_rollup]
    value = norm_summary.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return rounded(float(value), digits=3)


def _aggregate_score_text(status: str, aggregate_norm_score: float | None, *, policy: SuiteAggregatePolicy) -> str:
    if aggregate_norm_score is None:
        return f"worst {status}"
    norm_label = {
        "minimum": "min",
        "mean": "mean",
        "median": "median",
        "weighted_mean": "weighted mean",
    }[policy.norm_rollup]
    return f"worst {status}, {norm_label} norm {aggregate_norm_score:g}"


def _suite_statistical_case_entry(case_summary: SuiteCaseSummary) -> dict[str, Any] | None:
    payload = case_summary.case_score
    if payload is None or not isinstance(payload.observation, dict) or not isinstance(payload.prediction, dict):
        return None

    score_kind = str(payload.score_kind or "").strip()
    observation = payload.observation
    prediction = payload.prediction
    if score_kind in _EQUIVALENCE_CASE_SCORE_KINDS:
        pvalue = _normalized_score_value(observation.get("aggregate_statistical_pvalue"))
        threshold = _normalized_score_value(prediction.get("equivalence_alpha"))
        return {
            "category": "equivalence",
            "label": "TOST p",
            "rollup_method": "max",
            "rollup_source": "auto_default",
            "threshold": threshold,
            "threshold_key": "equivalence_alpha",
            "threshold_direction": "le",
            "pvalue": pvalue,
            "check_id": case_summary.check_id,
            "test_family": str(prediction.get("statistical_test_family") or "equivalence_tost").strip(),
        }
    if score_kind in _WELCH_CASE_SCORE_KINDS:
        pvalue = _normalized_score_value(observation.get("median_welch_pvalue"))
        threshold = _normalized_score_value(prediction.get("minimum_median_welch_pvalue"))
        return {
            "category": "welch_similarity",
            "label": "Welch p",
            "rollup_method": "min",
            "rollup_source": "auto_default",
            "threshold": threshold,
            "threshold_key": "minimum_median_welch_pvalue",
            "threshold_direction": "ge",
            "pvalue": pvalue,
            "check_id": case_summary.check_id,
            "test_family": str(prediction.get("statistical_test_family") or "legacy_welch_difference").strip(),
        }
    return None


def _suite_statistical_summary(case_summaries: tuple[SuiteCaseSummary, ...]) -> SuiteStatisticalSummary | None:
    entries = [
        entry
        for entry in (_suite_statistical_case_entry(case_summary) for case_summary in case_summaries)
        if entry is not None
    ]
    if not entries:
        return None

    categories = {str(entry["category"]) for entry in entries}
    if len(categories) != 1:
        return None
    category = next(iter(categories))
    if any(entry.get("pvalue") is None for entry in entries):
        return None

    rollup_method = str(entries[0]["rollup_method"])
    rollup_source = str(entries[0]["rollup_source"])
    label = str(entries[0]["label"])
    statistical_test_family = str(entries[0]["test_family"])
    threshold_key = str(entries[0]["threshold_key"])
    threshold_direction = str(entries[0]["threshold_direction"])
    thresholds = {
        entry["threshold"]
        for entry in entries
        if entry.get("threshold") is not None
    }
    threshold = next(iter(thresholds)) if len(thresholds) == 1 else None
    pvalues = [float(entry["pvalue"]) for entry in entries]
    if rollup_method == "max":
        rollup_pvalue = max(pvalues)
    else:
        rollup_pvalue = min(pvalues)
    rounded_rollup_pvalue = rounded(rollup_pvalue, digits=4)
    threshold_phrase = ""
    gate_passed: bool | None = None
    if threshold is not None:
        threshold_phrase = (
            f" (threshold {rounded(float(threshold), digits=4):g})"
        )
        if threshold_direction == "le":
            gate_passed = rollup_pvalue <= float(threshold)
        elif threshold_direction == "ge":
            gate_passed = rollup_pvalue >= float(threshold)
    return SuiteStatisticalSummary(
        score_family_category=category,
        statistical_test_family=statistical_test_family,
        rollup_method=rollup_method,
        rollup_source=rollup_source,
        rollup_pvalue=rollup_pvalue,
        available_case_count=len(entries),
        total_case_count=len(case_summaries),
        score_text=f"{rollup_method} {label} {rounded_rollup_pvalue:g}",
        score_interpretation=(
            "Diagnostic suite-level statistical summary derived from the case-level "
            f"{label} values. It does not replace the detailed per-case gates.{threshold_phrase}"
        ),
        threshold=threshold,
        threshold_key=threshold_key,
        threshold_direction=threshold_direction,
        gate_passed=gate_passed,
        case_pvalues=tuple(pvalues),
        case_check_ids=tuple(str(entry["check_id"]) for entry in entries),
    )


@dataclass(frozen=True)
class SuiteAggregateScore:
    suite_id: str
    case_summaries: tuple[SuiteCaseSummary, ...]
    policy: SuiteAggregatePolicy = DEFAULT_SUITE_AGGREGATE_POLICY
    candidate_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        suite_id = str(self.suite_id).strip()
        if not suite_id:
            raise ValueError("suite_id must be non-empty")
        object.__setattr__(self, "suite_id", suite_id)
        object.__setattr__(self, "case_summaries", tuple(self.case_summaries))
        object.__setattr__(
            self,
            "candidate_ids",
            tuple(str(candidate_id).strip() for candidate_id in self.candidate_ids if str(candidate_id).strip()),
        )

    @property
    def case_count(self) -> int:
        return len(self.case_summaries)

    @property
    def status_summary(self) -> dict[str, int]:
        return _status_summary(self.case_summaries)

    @property
    def status(self) -> str:
        return _worst_case_status(self.case_summaries)

    @property
    def norm_summary(self) -> dict[str, Any] | None:
        return _norm_score_summary(self.case_summaries)

    @property
    def aggregate_norm_score(self) -> float | None:
        return _aggregate_norm_score(self.norm_summary, policy=self.policy)

    @property
    def score_kind(self) -> str:
        return self.policy.score_kind

    @property
    def score_value(self) -> float | None:
        return self.aggregate_norm_score

    @property
    def score_text(self) -> str:
        return _aggregate_score_text(self.status, self.aggregate_norm_score, policy=self.policy)

    @property
    def score_interpretation(self) -> str:
        return self.policy.interpretation

    @property
    def statistical_summary(self) -> SuiteStatisticalSummary | None:
        return _suite_statistical_summary(self.case_summaries)

    @property
    def warning_case_titles(self) -> tuple[str, ...]:
        return tuple(case.title for case in self.case_summaries if case.status == "WARN")

    @property
    def failed_case_titles(self) -> tuple[str, ...]:
        return tuple(case.title for case in self.case_summaries if case.status == "FAIL")

    @property
    def case_check_ids(self) -> tuple[str, ...]:
        return tuple(case.check_id for case in self.case_summaries)

    def to_evidence(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "suite_id": self.suite_id,
            "suite_case_count": self.case_count,
            "suite_status_summary": self.status_summary,
            "suite_cases": [case.to_dict() for case in self.case_summaries],
            "warning_cases": list(self.warning_case_titles),
            "failed_cases": list(self.failed_case_titles),
            "suite_case_check_ids": list(self.case_check_ids),
            "suite_aggregate_score": {
                **self.policy.to_dict(),
                "status": self.status,
                "score_text": self.score_text,
            },
        }
        if self.candidate_ids:
            payload["suite_candidate_ids"] = list(self.candidate_ids)
        if self.norm_summary is not None:
            payload["suite_norm_score_summary"] = self.norm_summary
        if self.statistical_summary is not None:
            payload["suite_statistical_summary"] = self.statistical_summary.to_dict()
        if self.score_value is not None:
            payload["suite_aggregate_score"]["score_value"] = float(self.score_value)
        return payload


def build_suite_aggregate_score(
    *,
    suite_id: str,
    case_summaries: list[SuiteCaseSummary] | tuple[SuiteCaseSummary, ...],
    policy: SuiteAggregatePolicy = DEFAULT_SUITE_AGGREGATE_POLICY,
    candidate_ids: list[str] | tuple[str, ...] = (),
) -> SuiteAggregateScore:
    return SuiteAggregateScore(
        suite_id=suite_id,
        case_summaries=tuple(case_summaries),
        policy=policy,
        candidate_ids=tuple(candidate_ids),
    )


__all__ = [
    "DEFAULT_SUITE_AGGREGATE_POLICY",
    "SuiteAggregatePolicy",
    "SuiteAggregateScore",
    "SuiteCaseScorePayload",
    "SuiteCaseSummary",
    "SuiteDescriptor",
    "SuiteStatisticalSummary",
    "build_suite_aggregate_score",
]
