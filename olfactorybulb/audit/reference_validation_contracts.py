"""Shared typed interface contracts for the maintained reference-validation stack."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ValidationReviewLike(Protocol):
    status: str
    note: str
    reviewer: str
    required_expertise: str
    focus: str


@runtime_checkable
class MetricSummaryLike(Protocol):
    def __len__(self) -> int: ...

    def __iter__(self): ...

    def get(self, key: str, default: Any = None) -> Any: ...


@runtime_checkable
class ValidationRuleContextLike(Protocol):
    args: Any
    default_group: str
    design_review_defaults: ValidationReviewLike
    summary: MetricSummaryLike


__all__ = [
    "MetricSummaryLike",
    "ValidationReviewLike",
    "ValidationRuleContextLike",
]
