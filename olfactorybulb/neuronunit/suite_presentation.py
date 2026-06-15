"""Shared presentation helpers for SciUnit-backed validation suites."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import re
from typing import Any, Callable, Iterable, Protocol, TypeVar, runtime_checkable

from olfactorybulb.audit import AuditItem, companion_visual_spec
from olfactorybulb.audit.reference_validation_contracts import ValidationReviewLike
from olfactorybulb.neuronunit.suite_scores import (
    SuiteCaseScorePayload,
    SuiteCaseSummary,
    SuiteDescriptor,
    build_suite_aggregate_score,
)


_CaseT = TypeVar("_CaseT")
_ScoreT = TypeVar("_ScoreT")


@runtime_checkable
class AuditItemCaseLike(Protocol):
    check_id: str
    title: str
    criterion: str
    criterion_latex: str
    criterion_formulae: tuple[str, ...] | list[str]
    criterion_definitions: tuple[dict[str, Any], ...] | list[dict[str, Any]]
    description: str
    acceptable: str
    acceptable_basis: str
    note: str


def _normalized_visual_payload(values: Iterable[dict[str, Any]] | None) -> tuple[dict[str, Any], ...]:
    if not values:
        return ()
    return tuple(copy.deepcopy(dict(value)) for value in values)


def _normalized_definition_payload(values: Iterable[dict[str, Any]] | None) -> tuple[dict[str, Any], ...]:
    if not values:
        return ()
    return tuple(copy.deepcopy(dict(value)) for value in values)


@dataclass(frozen=True)
class AuditItemAdapterSpec:
    check_id: str
    title: str
    criterion: str
    criterion_latex: str = ""
    criterion_formulae: tuple[str, ...] = ()
    criterion_definitions: tuple[dict[str, Any], ...] = ()
    description: str = ""
    acceptable: str = ""
    acceptable_basis: str = ""
    note: str = ""
    validation_design_review_status: str = ""
    validation_design_review_note: str = ""
    validation_design_review_reviewer: str = ""
    validation_design_review_required_expertise: str = ""
    validation_design_review_focus: str = ""
    series_visuals: tuple[dict[str, Any], ...] = ()
    companion_visuals: tuple[dict[str, Any], ...] = ()
    detail_level: str = "detail"
    summary_rollup_exempt: bool = False
    default_status_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "check_id", str(self.check_id))
        object.__setattr__(self, "title", str(self.title))
        object.__setattr__(self, "criterion", str(self.criterion))
        object.__setattr__(self, "criterion_latex", str(self.criterion_latex or ""))
        object.__setattr__(self, "criterion_formulae", tuple(str(value) for value in (self.criterion_formulae or ())))
        object.__setattr__(
            self,
            "criterion_definitions",
            _normalized_definition_payload(self.criterion_definitions),
        )
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "acceptable", str(self.acceptable))
        object.__setattr__(self, "acceptable_basis", str(self.acceptable_basis))
        object.__setattr__(self, "note", str(self.note))
        object.__setattr__(self, "validation_design_review_status", str(self.validation_design_review_status))
        object.__setattr__(self, "validation_design_review_note", str(self.validation_design_review_note))
        object.__setattr__(self, "validation_design_review_reviewer", str(self.validation_design_review_reviewer))
        object.__setattr__(
            self,
            "validation_design_review_required_expertise",
            str(self.validation_design_review_required_expertise),
        )
        object.__setattr__(self, "validation_design_review_focus", str(self.validation_design_review_focus))
        object.__setattr__(self, "series_visuals", _normalized_visual_payload(self.series_visuals))
        object.__setattr__(self, "companion_visuals", _normalized_visual_payload(self.companion_visuals))
        object.__setattr__(self, "detail_level", str(self.detail_level or "detail"))
        object.__setattr__(self, "summary_rollup_exempt", bool(self.summary_rollup_exempt))
        object.__setattr__(self, "default_status_reason", str(self.default_status_reason))

    def to_audit_item(
        self,
        *,
        status: str,
        evidence: dict[str, Any],
        status_reason: str = "",
    ) -> AuditItem:
        resolved_status_reason = str(status_reason or self.default_status_reason)
        return AuditItem(
            check_id=self.check_id,
            status=str(status),
            title=self.title,
            criterion=self.criterion,
            criterion_latex=self.criterion_latex,
            criterion_formulae=list(self.criterion_formulae),
            criterion_definitions=[copy.deepcopy(value) for value in self.criterion_definitions],
            description=self.description,
            acceptable=self.acceptable,
            acceptable_basis=self.acceptable_basis,
            evidence=copy.deepcopy(dict(evidence)),
            note=self.note,
            validation_design_review_status=self.validation_design_review_status,
            validation_design_review_note=self.validation_design_review_note,
            validation_design_review_reviewer=self.validation_design_review_reviewer,
            validation_design_review_required_expertise=self.validation_design_review_required_expertise,
            validation_design_review_focus=self.validation_design_review_focus,
            series_visuals=[copy.deepcopy(value) for value in self.series_visuals],
            companion_visuals=[copy.deepcopy(value) for value in self.companion_visuals],
            detail_level=self.detail_level,
            summary_rollup_exempt=self.summary_rollup_exempt,
            status_reason=resolved_status_reason,
        )


def audit_item_adapter_spec_from_case(
    case: AuditItemCaseLike,
    *,
    validation_review: ValidationReviewLike | None = None,
    series_visuals: Iterable[dict[str, Any]] | None = None,
    companion_visuals: Iterable[dict[str, Any]] | None = None,
    detail_level: str = "detail",
    summary_rollup_exempt: bool = False,
    default_status_reason: str = "",
) -> AuditItemAdapterSpec:
    review = validation_review
    return AuditItemAdapterSpec(
        check_id=getattr(case, "check_id"),
        title=getattr(case, "title"),
        criterion=getattr(case, "criterion"),
        criterion_latex=getattr(case, "criterion_latex", ""),
        criterion_formulae=tuple(getattr(case, "criterion_formulae", ()) or ()),
        criterion_definitions=tuple(getattr(case, "criterion_definitions", ()) or ()),
        description=getattr(case, "description"),
        acceptable=getattr(case, "acceptable"),
        acceptable_basis=getattr(case, "acceptable_basis"),
        note=getattr(case, "note", ""),
        validation_design_review_status=getattr(review, "status", "") if review is not None else "",
        validation_design_review_note=getattr(review, "note", "") if review is not None else "",
        validation_design_review_reviewer=getattr(review, "reviewer", "") if review is not None else "",
        validation_design_review_required_expertise=getattr(review, "required_expertise", "") if review is not None else "",
        validation_design_review_focus=getattr(review, "focus", "") if review is not None else "",
        series_visuals=tuple(series_visuals or ()),
        companion_visuals=tuple(companion_visuals or ()),
        detail_level=detail_level,
        summary_rollup_exempt=summary_rollup_exempt,
        default_status_reason=default_status_reason,
    )


@dataclass(frozen=True)
class SuiteCaseResult:
    item: AuditItem
    score_text: str = ""
    norm_score: float | None = None
    score_payload: SuiteCaseScorePayload | None = None
    case_weight: float | None = None
    case_weight_label: str = ""


def suite_case_result(
    item: AuditItem,
    *,
    score_text: str = "",
    norm_score: float | None = None,
    score_payload: SuiteCaseScorePayload | None = None,
    case_weight: float | None = None,
    case_weight_label: str = "",
) -> SuiteCaseResult:
    return SuiteCaseResult(
        item=item,
        score_text=str(score_text).strip(),
        norm_score=norm_score,
        score_payload=score_payload,
        case_weight=case_weight,
        case_weight_label=str(case_weight_label).strip(),
    )


def suite_case_result_from_spec(
    spec: AuditItemAdapterSpec,
    *,
    status: str,
    evidence: dict[str, Any],
    score_text: str = "",
    norm_score: float | None = None,
    score_payload: SuiteCaseScorePayload | None = None,
    case_weight: float | None = None,
    case_weight_label: str = "",
    status_reason: str = "",
) -> SuiteCaseResult:
    return suite_case_result(
        spec.to_audit_item(
            status=status,
            evidence=evidence,
            status_reason=status_reason,
        ),
        score_text=score_text,
        norm_score=norm_score,
        score_payload=score_payload,
        case_weight=case_weight,
        case_weight_label=case_weight_label,
    )


def suite_case_entry(result: SuiteCaseResult) -> dict[str, Any]:
    return suite_case_summary(result).to_dict()


def suite_case_summary(result: SuiteCaseResult) -> SuiteCaseSummary:
    return SuiteCaseSummary(
        check_id=str(result.item.check_id),
        title=str(result.item.title),
        status=str(result.item.status).upper(),
        score_text=result.score_text,
        norm_score=result.norm_score,
        case_score=result.score_payload,
        case_weight=result.case_weight,
        case_weight_label=result.case_weight_label,
    )


def suite_overview_item(
    *,
    descriptor: SuiteDescriptor,
    case_summaries: list[SuiteCaseSummary],
) -> AuditItem:
    suite_name_text = descriptor.suite_id
    check_id_prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", suite_name_text).strip("._") or "suite"
    suite_score = build_suite_aggregate_score(
        suite_id=suite_name_text,
        case_summaries=case_summaries,
        policy=descriptor.aggregate_policy,
        statistical_policy=descriptor.statistical_policy,
        candidate_ids=descriptor.candidate_ids,
    )
    summary = suite_score.status_summary
    worst_status = suite_score.status
    status_reason = ""
    if worst_status == "FAIL":
        status_reason = "One or more suite cases failed; inspect the detailed cases below."
    elif worst_status == "WARN":
        status_reason = "One or more suite cases remain at warning severity; inspect the detailed cases below."

    evidence = {
        "suite_name": suite_name_text,
        "suite_kind": descriptor.suite_kind_label,
        **suite_score.to_evidence(),
    }

    return AuditItem(
        check_id=f"{check_id_prefix}.overview",
        status=worst_status,
        title=f"{descriptor.suite_kind_label} overview",
        criterion="Every compiled case in this SciUnit-backed suite should satisfy its declared criterion.",
        description=(
            "This overview item summarizes the migrated NeuronUnit/SciUnit suite as one maintained validation surface "
            "while keeping the individual cases directly inspectable below."
        ),
        acceptable="The detailed suite cases pass. Warning and failure severities remain visible in the case matrix and the detailed cards below.",
        acceptable_basis=(
            f"The overview is derived mechanically from the compiled SciUnit suite {suite_name_text!r} so the "
            "suite-level presentation stays synchronized with the same maintained cases that generated the detailed items."
        ),
        evidence=evidence,
        companion_visuals=[
            companion_visual_spec(
                kind="status_matrix",
                key="suite_cases",
                title="Suite case summary",
            )
        ],
        status_reason=status_reason,
        detail_level="summary",
        summary_rollup_exempt=True,
    )


def suite_items_from_case_results(
    *,
    descriptor: SuiteDescriptor,
    case_results: Iterable[SuiteCaseResult],
) -> list[AuditItem]:
    normalized_results = list(case_results)
    overview_item = suite_overview_item(
        descriptor=descriptor,
        case_summaries=[suite_case_summary(result) for result in normalized_results],
    )
    return [overview_item, *(result.item for result in normalized_results)]


def suite_items_from_judged(
    *,
    descriptor: SuiteDescriptor,
    judged: Iterable[tuple[_CaseT, _ScoreT]],
    result_builder: Callable[[_CaseT, _ScoreT], SuiteCaseResult],
) -> list[AuditItem]:
    case_results = [result_builder(case, score) for case, score in judged]
    return suite_items_from_case_results(
        descriptor=descriptor,
        case_results=case_results,
    )


__all__ = [
    "AuditItemAdapterSpec",
    "SuiteCaseResult",
    "audit_item_adapter_spec_from_case",
    "suite_case_entry",
    "suite_case_summary",
    "suite_case_result",
    "suite_case_result_from_spec",
    "suite_items_from_case_results",
    "suite_items_from_judged",
    "suite_overview_item",
]
