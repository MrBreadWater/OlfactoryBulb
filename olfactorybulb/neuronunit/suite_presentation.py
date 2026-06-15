"""Shared presentation helpers for SciUnit-backed validation suites."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Callable, Iterable, TypeVar

from olfactorybulb.audit import AuditItem, companion_visual_spec
from olfactorybulb.audit.core import rounded


_STATUS_RANK = {"FAIL": 3, "WARN": 2, "PASS": 1}
_CaseT = TypeVar("_CaseT")
_ScoreT = TypeVar("_ScoreT")


@dataclass(frozen=True)
class SuiteCaseResult:
    item: AuditItem
    score_text: str = ""
    norm_score: float | None = None


def _case_status_summary(entries: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for entry in entries:
        status = str(entry.get("status", "")).upper()
        if status not in counts:
            continue
        counts[status] = counts.get(status, 0) + 1
    return counts


def _worst_case_status(entries: Iterable[dict[str, Any]]) -> str:
    statuses = [str(entry.get("status", "")).upper() for entry in entries if str(entry.get("status", "")).strip()]
    return max(statuses, key=lambda status: _STATUS_RANK.get(status, 0), default="PASS")


def _norm_score_summary(entries: Iterable[dict[str, Any]]) -> dict[str, float] | None:
    norm_scores = [
        float(entry["norm_score"])
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("norm_score"), (int, float))
    ]
    finite_scores = [score for score in norm_scores if math.isfinite(score)]
    if not finite_scores:
        return None
    finite_scores.sort()
    midpoint = len(finite_scores) // 2
    if len(finite_scores) % 2 == 0:
        median_score = (finite_scores[midpoint - 1] + finite_scores[midpoint]) / 2.0
    else:
        median_score = finite_scores[midpoint]
    return {
        "mean": rounded(sum(finite_scores) / len(finite_scores), digits=3),
        "median": rounded(median_score, digits=3),
        "min": rounded(finite_scores[0], digits=3),
        "max": rounded(finite_scores[-1], digits=3),
        "count": float(len(finite_scores)),
    }


def suite_case_result(
    item: AuditItem,
    *,
    score_text: str = "",
    norm_score: float | None = None,
) -> SuiteCaseResult:
    normalized_norm_score: float | None = None
    if norm_score is not None:
        try:
            candidate = float(norm_score)
        except (TypeError, ValueError):
            candidate = float("nan")
        if math.isfinite(candidate):
            normalized_norm_score = rounded(candidate, digits=3)
    return SuiteCaseResult(
        item=item,
        score_text=str(score_text).strip(),
        norm_score=normalized_norm_score,
    )


def suite_case_entry(result: SuiteCaseResult) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "check_id": str(result.item.check_id),
        "title": str(result.item.title),
        "status": str(result.item.status).upper(),
    }
    if result.score_text:
        entry["score_text"] = result.score_text
    if result.norm_score is not None:
        entry["norm_score"] = float(result.norm_score)
    return entry


def suite_overview_item(
    *,
    suite_name: str,
    suite_kind_label: str,
    case_entries: list[dict[str, Any]],
) -> AuditItem:
    suite_name_text = str(suite_name).strip()
    check_id_prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", suite_name_text).strip("._") or "suite"
    summary = _case_status_summary(case_entries)
    worst_status = _worst_case_status(case_entries)
    warning_cases = [entry["title"] for entry in case_entries if entry.get("status") == "WARN"]
    failed_cases = [entry["title"] for entry in case_entries if entry.get("status") == "FAIL"]
    norm_summary = _norm_score_summary(case_entries)
    status_reason = ""
    if worst_status == "WARN":
        status_reason = "One or more suite cases remain at warning severity; inspect the detailed cases below."

    evidence = {
        "suite_name": suite_name_text,
        "suite_kind": str(suite_kind_label).strip(),
        "suite_case_count": len(case_entries),
        "suite_status_summary": summary,
        "warning_cases": warning_cases,
        "failed_cases": failed_cases,
        "suite_cases": [dict(entry) for entry in case_entries],
    }
    if norm_summary is not None:
        evidence["suite_norm_score_summary"] = norm_summary

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
) -> list[AuditItem]:
    normalized_results = list(case_results)
    overview_item = suite_overview_item(
        suite_name=suite_name,
        suite_kind_label=suite_kind_label,
        case_entries=[suite_case_entry(result) for result in normalized_results],
    )
    return [overview_item, *(result.item for result in normalized_results)]


def suite_items_from_judged(
    *,
    suite_name: str,
    suite_kind_label: str,
    judged: Iterable[tuple[_CaseT, _ScoreT]],
    result_builder: Callable[[_CaseT, _ScoreT], SuiteCaseResult],
) -> list[AuditItem]:
    case_results = [result_builder(case, score) for case, score in judged]
    return suite_items_from_case_results(
        suite_name=suite_name,
        suite_kind_label=suite_kind_label,
        case_results=case_results,
    )


__all__ = [
    "SuiteCaseResult",
    "suite_case_entry",
    "suite_case_result",
    "suite_items_from_case_results",
    "suite_items_from_judged",
    "suite_overview_item",
]
