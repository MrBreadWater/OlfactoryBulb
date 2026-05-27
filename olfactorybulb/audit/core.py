"""Core datatypes for repo audits."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

STATUS_RANK = {"FAIL": 3, "WARN": 2, "PASS": 1}


@dataclass
class AuditItem:
    check_id: str
    status: str
    title: str
    criterion: str
    evidence: dict[str, Any] = field(default_factory=dict)
    note: str = ""


@dataclass
class AuditReport:
    audit_id: str
    title: str
    items: list[AuditItem]

    @property
    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {"PASS": 0, "WARN": 0, "FAIL": 0}
        for item in self.items:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts

    @property
    def worst_status(self) -> str:
        return max((item.status for item in self.items), key=lambda status: STATUS_RANK.get(status, 0), default="PASS")

    @property
    def exit_code(self) -> int:
        return 0 if self.worst_status != "FAIL" else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "title": self.title,
            "summary": self.summary,
            "worst_status": self.worst_status,
            "items": [asdict(item) for item in self.items],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def rounded(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def format_report(report: AuditReport) -> str:
    lines: list[str] = []
    lines.append(report.title)
    lines.append("=" * len(report.title))
    lines.append(
        "Summary: "
        + ", ".join(f"{key}={value}" for key, value in report.summary.items())
        + f" (worst={report.worst_status})"
    )
    lines.append("")

    for item in report.items:
        lines.append(f"[{item.status}] {item.check_id}: {item.title}")
        lines.append(f"  Criterion: {item.criterion}")
        if item.evidence:
            lines.append(f"  Evidence: {json.dumps(item.evidence, sort_keys=True)}")
        if item.note:
            lines.append(f"  Note: {item.note}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def collect_items(*groups: Iterable[AuditItem]) -> list[AuditItem]:
    items: list[AuditItem] = []
    for group in groups:
        items.extend(group)
    return items

