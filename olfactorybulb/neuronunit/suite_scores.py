"""Typed suite aggregation for SciUnit-backed maintained validation suites."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from olfactorybulb.audit.core import rounded


_STATUS_RANK = {"FAIL": 3, "WARN": 2, "PASS": 1}
_VALID_STATUSES = frozenset(_STATUS_RANK)


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


@dataclass(frozen=True)
class SuiteCaseSummary:
    check_id: str
    title: str
    status: str
    score_text: str = ""
    norm_score: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "check_id", str(self.check_id).strip())
        object.__setattr__(self, "title", str(self.title).strip())
        object.__setattr__(self, "status", _normalized_status(self.status))
        object.__setattr__(self, "score_text", str(self.score_text).strip())
        object.__setattr__(self, "norm_score", _normalized_norm_score(self.norm_score))

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
        if norm_rollup not in {"minimum", "mean", "median"}:
            raise ValueError(
                f"Unsupported suite norm rollup {self.norm_rollup!r}; "
                "the maintained bridge currently supports minimum, mean, or median"
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


def _norm_score_summary(case_summaries: tuple[SuiteCaseSummary, ...]) -> dict[str, float] | None:
    scores = [float(case.norm_score) for case in case_summaries if case.norm_score is not None]
    if not scores:
        return None
    scores.sort()
    midpoint = len(scores) // 2
    if len(scores) % 2 == 0:
        median = (scores[midpoint - 1] + scores[midpoint]) / 2.0
    else:
        median = scores[midpoint]
    return {
        "mean": rounded(sum(scores) / len(scores), digits=3),
        "median": rounded(median, digits=3),
        "min": rounded(scores[0], digits=3),
        "max": rounded(scores[-1], digits=3),
        "count": float(len(scores)),
    }


def _aggregate_norm_score(
    norm_summary: dict[str, float] | None,
    *,
    policy: SuiteAggregatePolicy,
) -> float | None:
    if norm_summary is None:
        return None
    key = {"minimum": "min", "mean": "mean", "median": "median"}[policy.norm_rollup]
    value = norm_summary.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return rounded(float(value), digits=3)


def _aggregate_score_text(status: str, aggregate_norm_score: float | None, *, policy: SuiteAggregatePolicy) -> str:
    if aggregate_norm_score is None:
        return f"worst {status}"
    norm_label = {"minimum": "min", "mean": "mean", "median": "median"}[policy.norm_rollup]
    return f"worst {status}, {norm_label} norm {aggregate_norm_score:g}"


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
    def norm_summary(self) -> dict[str, float] | None:
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
    "SuiteCaseSummary",
    "SuiteDescriptor",
    "build_suite_aggregate_score",
]
