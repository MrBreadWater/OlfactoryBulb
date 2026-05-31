"""Core datatypes for repo audits."""

from __future__ import annotations

import json
import os
import re
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
TEXT_TERM_REPLACEMENTS = [
    (r"\bCV_ISI\b", "coefficient of variation of interspike intervals"),
    (r"\bFWHM\b", "full width at half maximum"),
    (r"\bAHP\b", "afterhyperpolarization"),
    (r"\bT_AHP50%?\b", "afterhyperpolarization half-decay time"),
    (r"\bAP\b", "action potential"),
    (r"\bf-I\b", "firing-rate-versus-current"),
    (r"\bFI\b", "firing-rate-versus-current"),
    (r"\bVm\b", "membrane potential"),
    (r"\bPSD\b", "power spectral density"),
    (r"\bKDE\b", "kernel density estimate"),
    (r"\bHFO\b", "high-frequency oscillation"),
    (r"\bEPLI\b", "external plexiform layer interneuron"),
    (r"\bPVCRH\b", "parvalbumin- and corticotropin-releasing-hormone-positive interneuron"),
    (r"\bMCs\b", "mitral cells"),
    (r"\bTCs\b", "tufted cells"),
    (r"\bGCs\b", "granule cells"),
    (r"\bMC/TC\b", "mitral cell / tufted cell"),
    (r"\bM/T\b", "mitral cell / tufted cell"),
    (r"\bMC\b", "mitral cell"),
    (r"\bTC\b", "tufted cell"),
    (r"\bGC\b", "granule cell"),
    (r"\bKAR\b", "kainate receptor"),
    (r"\bOSN\b", "olfactory sensory neuron"),
    (r"\bDC\b", "direct-current"),
]
SPECIAL_EVIDENCE_LABELS = {
    "MC_mean": "mitral cell mean",
    "TC_mean": "tufted cell mean",
    "TC_minus_MC": "tufted cell mean minus mitral cell mean",
    "cv_isi": "coefficient of variation of interspike intervals",
    "cv_isi_step_pA": "current step used for coefficient-of-variation measurement in picoamperes",
    "cv_isi_mean_rate_Hz": "mean firing rate used for coefficient-of-variation measurement in hertz",
    "AP_onset_mV": "action-potential onset in millivolts",
    "Amplitude_mV": "action-potential amplitude in millivolts",
    "FWHM_ms": "action-potential full width at half maximum in milliseconds",
    "Rise_slope_mV_per_ms": "action-potential rise slope in millivolts per millisecond",
    "Fall_slope_mV_per_ms": "action-potential fall slope in millivolts per millisecond",
    "AHP_amplitude_mV": "afterhyperpolarization amplitude in millivolts",
    "T_AHP50_ms": "afterhyperpolarization half-decay time in milliseconds",
    "Peak_rate_Hz": "peak firing rate in hertz",
    "FI_gain_Hz_per_50pA": "firing-rate-versus-current gain in hertz per fifty picoamperes",
    "Rheobase_pA": "rheobase in picoamperes",
    "cell_types": "cell types",
}
INTERNAL_EVIDENCE_KEYS = {"__reference_annotations__"}


@dataclass
class AuditItem:
    check_id: str
    status: str
    title: str
    criterion: str
    description: str = ""
    acceptable: str = ""
    acceptable_basis: str = ""
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


def _expand_terms(text: str, *, sentence_case: bool = False) -> str:
    expanded = str(text)
    for pattern, replacement in TEXT_TERM_REPLACEMENTS:
        expanded = re.sub(pattern, replacement, expanded)
    if sentence_case and expanded:
        expanded = expanded[0].upper() + expanded[1:]
    return expanded


def _humanize_identifier(identifier: str) -> str:
    if identifier in SPECIAL_EVIDENCE_LABELS:
        return SPECIAL_EVIDENCE_LABELS[identifier]
    pieces = identifier.split("_")
    humanized = " ".join(piece for piece in pieces if piece)
    humanized = _expand_terms(humanized)
    humanized = humanized.replace(" mV", " millivolts")
    humanized = humanized.replace(" ms", " milliseconds")
    humanized = humanized.replace(" Hz", " hertz")
    humanized = humanized.replace(" pA", " picoamperes")
    humanized = humanized.replace(" um", " micrometers")
    return humanized


def _format_scalar(value: Any, *, key: str | None = None) -> str:
    if isinstance(value, str):
        if key in {"cell_types", "cell_type"}:
            parts = [part.strip() for part in value.split(",") if part.strip()]
            return ", ".join(_expand_terms(part, sentence_case=True) for part in parts)
        return value
    return json.dumps(value, sort_keys=True)


def _pretty_evidence_lines(payload: Any, *, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(payload, dict):
        lines: list[str] = []
        annotations = payload.get("__reference_annotations__", {}) if isinstance(payload.get("__reference_annotations__", {}), dict) else {}
        for key, value in payload.items():
            key_text = str(key)
            if key_text in INTERNAL_EVIDENCE_KEYS:
                continue
            label = _humanize_identifier(key_text)
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{label}:")
                lines.extend(_pretty_evidence_lines(value, indent=indent + 2))
            else:
                annotation_suffix = ""
                if key_text in annotations:
                    annotation_suffix = f" ({annotations[key_text]})"
                lines.append(f"{prefix}{label}: {_format_scalar(value, key=key_text)}{annotation_suffix}")
        return lines
    if isinstance(payload, list):
        lines = []
        for value in payload:
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_pretty_evidence_lines(value, indent=indent + 2))
            else:
                lines.append(f"{prefix}- {_format_scalar(value)}")
        return lines
    return [f"{prefix}{_format_scalar(payload)}"]


def _summary_chunks(summary: dict[str, int], *, enabled: bool) -> list[str]:
    parts: list[str] = []
    for status in STATUS_ORDER:
        value = int(summary.get(status, 0))
        codes = [STATUS_COLOR[status]] if value > 0 else [DIM]
        parts.append(_paint(f"{status}={value}", *codes, enabled=enabled))
    return parts


def _default_description(item: AuditItem) -> str:
    title = _expand_terms(item.title).rstrip(".")
    return (
        f"This check evaluates whether {title.lower()}. "
        "The criterion states the requirement, and the evidence shows how the current audit run was judged."
    )


def _default_acceptable(item: AuditItem) -> str:
    return (
        "This audit item does not define a separate numeric tolerance beyond the stated criterion. "
        "Interpret the criterion text as the acceptance rule for this check."
    )


def _default_acceptable_basis(item: AuditItem) -> str:
    return (
        "This audit item does not define a separate provenance note for its acceptance rule. "
        "Treat the criterion and description as the source of the decision logic."
    )


def format_report(report: AuditReport, *, color: bool | None = None) -> str:
    enabled = _color_enabled(color)
    lines: list[str] = []
    title = _paint(_expand_terms(report.title, sentence_case=True), "1", TITLE_COLOR, enabled=enabled)
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
        lines.append(f"  {_paint(_expand_terms(item.title, sentence_case=True), '1', enabled=enabled)}")
        lines.append(f"  {_paint('Criterion', LABEL_COLOR, enabled=enabled)}  {_expand_terms(item.criterion, sentence_case=True)}")
        lines.append(
            f"  {_paint('Description', LABEL_COLOR, enabled=enabled)}  "
            f"{_expand_terms(item.description or _default_description(item), sentence_case=True)}"
        )
        lines.append(
            f"  {_paint('Acceptable result', LABEL_COLOR, enabled=enabled)}  "
            f"{_expand_terms(item.acceptable or _default_acceptable(item), sentence_case=True)}"
        )
        lines.append(
            f"  {_paint('How Acceptable Result Was Determined', LABEL_COLOR, enabled=enabled)}  "
            f"{_expand_terms(item.acceptable_basis or _default_acceptable_basis(item), sentence_case=True)}"
        )
        if item.evidence:
            lines.append(f"  {_paint('Evidence', LABEL_COLOR, enabled=enabled)}")
            for evidence_line in _pretty_evidence_lines(item.evidence):
                lines.append(f"    {evidence_line}")
        if item.note:
            lines.append(
                f"  {_paint('Note', NOTE_COLOR, enabled=enabled)}  "
                f"{_paint(_expand_terms(item.note, sentence_case=True), DIM, enabled=enabled)}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def collect_items(*groups: Iterable[AuditItem]) -> list[AuditItem]:
    items: list[AuditItem] = []
    for group in groups:
        items.extend(group)
    return items
