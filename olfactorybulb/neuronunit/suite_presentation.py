"""Shared presentation helpers for SciUnit-backed validation suites."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Iterable, TypeVar

from olfactorybulb.audit import AuditItem, companion_visual_spec
from olfactorybulb.neuronunit.suite_scores import (
    DEFAULT_SUITE_AGGREGATE_POLICY,
    SuiteAggregatePolicy,
    SuiteCaseSummary,
    build_suite_aggregate_score,
)


_CaseT = TypeVar("_CaseT")
_ScoreT = TypeVar("_ScoreT")


@dataclass(frozen=True)
class SuiteCaseResult:
    item: AuditItem
    score_text: str = ""
    norm_score: float | None = None


def suite_case_result(
    item: AuditItem,
    *,
    score_text: str = "",
    norm_score: float | None = None,
) -> SuiteCaseResult:
    return SuiteCaseResult(
        item=item,
        score_text=str(score_text).strip(),
        norm_score=norm_score,
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
    )


def suite_overview_item(
    *,
    suite_name: str,
    suite_kind_label: str,
    case_summaries: list[SuiteCaseSummary],
    aggregate_policy: SuiteAggregatePolicy = DEFAULT_SUITE_AGGREGATE_POLICY,
    candidate_ids: list[str] | tuple[str, ...] = (),
) -> AuditItem:
    suite_name_text = str(suite_name).strip()
    check_id_prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", suite_name_text).strip("._") or "suite"
    suite_score = build_suite_aggregate_score(
        suite_id=suite_name_text,
        case_summaries=case_summaries,
        policy=aggregate_policy,
        candidate_ids=candidate_ids,
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
        "suite_kind": str(suite_kind_label).strip(),
        **suite_score.to_evidence(),
    }

    return AuditItem(
        check_id=f"{check_id_prefix}.overview",
        status=worst_status,
        title=f"{str(suite_kind_label).strip()} overview",
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
    suite_name: str,
    suite_kind_label: str,
    case_results: Iterable[SuiteCaseResult],
    aggregate_policy: SuiteAggregatePolicy = DEFAULT_SUITE_AGGREGATE_POLICY,
    candidate_ids: list[str] | tuple[str, ...] = (),
) -> list[AuditItem]:
    normalized_results = list(case_results)
    overview_item = suite_overview_item(
        suite_name=suite_name,
        suite_kind_label=suite_kind_label,
        case_summaries=[suite_case_summary(result) for result in normalized_results],
        aggregate_policy=aggregate_policy,
        candidate_ids=candidate_ids,
    )
    return [overview_item, *(result.item for result in normalized_results)]


def suite_items_from_judged(
    *,
    suite_name: str,
    suite_kind_label: str,
    judged: Iterable[tuple[_CaseT, _ScoreT]],
    result_builder: Callable[[_CaseT, _ScoreT], SuiteCaseResult],
    aggregate_policy: SuiteAggregatePolicy = DEFAULT_SUITE_AGGREGATE_POLICY,
    candidate_ids: list[str] | tuple[str, ...] = (),
) -> list[AuditItem]:
    case_results = [result_builder(case, score) for case, score in judged]
    return suite_items_from_case_results(
        suite_name=suite_name,
        suite_kind_label=suite_kind_label,
        case_results=case_results,
        aggregate_policy=aggregate_policy,
        candidate_ids=candidate_ids,
    )


__all__ = [
    "SuiteCaseResult",
    "suite_case_entry",
    "suite_case_summary",
    "suite_case_result",
    "suite_items_from_case_results",
    "suite_items_from_judged",
    "suite_overview_item",
]
