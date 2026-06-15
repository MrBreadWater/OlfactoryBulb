"""Shared presentation helpers for SciUnit-backed validation suites."""

from __future__ import annotations

import re
from typing import Any, Iterable

from olfactorybulb.audit import AuditItem, companion_visual_spec


_STATUS_RANK = {"FAIL": 3, "WARN": 2, "PASS": 1}


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


def suite_case_entry(*, check_id: str, title: str, status: str) -> dict[str, str]:
    return {
        "check_id": str(check_id),
        "title": str(title),
        "status": str(status).upper(),
    }


def suite_overview_item(
    *,
    suite_name: str,
    suite_kind_label: str,
    case_entries: list[dict[str, str]],
) -> AuditItem:
    suite_name_text = str(suite_name).strip()
    check_id_prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", suite_name_text).strip("._") or "suite"
    summary = _case_status_summary(case_entries)
    worst_status = _worst_case_status(case_entries)
    warning_cases = [entry["title"] for entry in case_entries if entry.get("status") == "WARN"]
    failed_cases = [entry["title"] for entry in case_entries if entry.get("status") == "FAIL"]
    status_reason = ""
    if worst_status == "WARN":
        status_reason = "One or more suite cases remain at warning severity; inspect the detailed cases below."

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
        evidence={
            "suite_name": suite_name_text,
            "suite_kind": str(suite_kind_label).strip(),
            "suite_case_count": len(case_entries),
            "suite_status_summary": summary,
            "warning_cases": warning_cases,
            "failed_cases": failed_cases,
            "suite_cases": [dict(entry) for entry in case_entries],
        },
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


__all__ = [
    "suite_case_entry",
    "suite_overview_item",
]
