"""Core datatypes for repo audits."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

STATUS_RANK = {"FAIL": 3, "WARN": 2, "PASS": 1}
STATUS_ORDER = ("FAIL", "WARN", "PASS")
STATUS_COLOR = {"FAIL": "31", "WARN": "33", "PASS": "32"}
LABEL_COLOR = "36"
TITLE_COLOR = "96"
NOTE_COLOR = "35"
DIM = "2"
RESET = "\033[0m"


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


def _color_enabled(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    if os.environ.get("NO_COLOR"):
        return False
    return True


def _paint(text: str, *codes: str, enabled: bool) -> str:
    if not enabled or not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}{RESET}"


def _pretty_json_lines(payload: dict[str, Any]) -> list[str]:
    return json.dumps(payload, indent=2, sort_keys=True).splitlines()


def _summary_chunks(summary: dict[str, int], *, enabled: bool) -> list[str]:
    parts: list[str] = []
    for status in STATUS_ORDER:
        value = int(summary.get(status, 0))
        codes = [STATUS_COLOR[status]] if value > 0 else [DIM]
        parts.append(_paint(f"{status}={value}", *codes, enabled=enabled))
    return parts


def format_report(report: AuditReport, *, color: bool | None = None) -> str:
    enabled = _color_enabled(color)
    lines: list[str] = []
    title = _paint(report.title, "1", TITLE_COLOR, enabled=enabled)
    subtitle = _paint(f"audit_id={report.audit_id}", DIM, enabled=enabled)
    lines.append(title)
    lines.append(subtitle)
    lines.append(_paint("=" * max(len(report.title), len(f"audit_id={report.audit_id}")), DIM, enabled=enabled))
    lines.append(
        f"{_paint('Summary', '1', LABEL_COLOR, enabled=enabled)}  "
        + "  ".join(_summary_chunks(report.summary, enabled=enabled))
        + f"  {_paint('worst=', DIM, enabled=enabled)}{_paint(report.worst_status, STATUS_COLOR[report.worst_status], '1', enabled=enabled)}"
    )
    lines.append("")

    for item in report.items:
        status_tag = _paint(f"[{item.status}]", STATUS_COLOR.get(item.status, "37"), "1", enabled=enabled)
        check_id = _paint(item.check_id, "1", enabled=enabled)
        lines.append(f"{status_tag} {check_id}")
        lines.append(f"  {_paint(item.title, '1', enabled=enabled)}")
        lines.append(f"  {_paint('Criterion', LABEL_COLOR, enabled=enabled)}  {item.criterion}")
        if item.evidence:
            lines.append(f"  {_paint('Evidence', LABEL_COLOR, enabled=enabled)}")
            for evidence_line in _pretty_json_lines(item.evidence):
                lines.append(f"    {evidence_line}")
        if item.note:
            lines.append(f"  {_paint('Note', NOTE_COLOR, enabled=enabled)}  {_paint(item.note, DIM, enabled=enabled)}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def collect_items(*groups: Iterable[AuditItem]) -> list[AuditItem]:
    items: list[AuditItem] = []
    for group in groups:
        items.extend(group)
    return items
