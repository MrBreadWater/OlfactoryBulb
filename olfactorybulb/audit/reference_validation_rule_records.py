"""Typed declarative rule-record layer for reference-validation configs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from olfactorybulb.audit.reference_validation_contracts import ValidationRuleContextLike


@dataclass(frozen=True)
class ValidationRuleRecord:
    raw_rule: dict[str, Any]
    kind: str
    enabled_when_arg_truthy: str = ""
    enabled_when_arg_falsey: str = ""
    enabled_when_arg_in: str = ""
    enabled_values: tuple[str, ...] = ()
    review_status: str = ""
    review_note: str = ""
    review_reviewer: str = ""
    review_required_expertise: str = ""
    review_focus: str = ""

    @classmethod
    def from_rule(cls, rule: dict[str, Any]) -> "ValidationRuleRecord":
        raw_rule = dict(rule)
        kind = str(raw_rule.get("kind") or "").strip()
        if not kind:
            raise ValueError("Validation rule is missing required 'kind'")
        return cls(
            raw_rule=raw_rule,
            kind=kind,
            enabled_when_arg_truthy=str(raw_rule.get("enabled_when_arg_truthy", "") or "").strip(),
            enabled_when_arg_falsey=str(raw_rule.get("enabled_when_arg_falsey", "") or "").strip(),
            enabled_when_arg_in=str(raw_rule.get("enabled_when_arg_in", "") or "").strip(),
            enabled_values=tuple(
                str(value).strip()
                for value in raw_rule.get("enabled_values", [])
                if str(value).strip()
            ),
            review_status=str(raw_rule.get("validation_design_review_status", "") or "").strip(),
            review_note=str(raw_rule.get("validation_design_review_note", "") or "").strip(),
            review_reviewer=str(raw_rule.get("validation_design_review_reviewer", "") or "").strip(),
            review_required_expertise=str(
                raw_rule.get("validation_design_review_required_expertise", "") or ""
            ).strip(),
            review_focus=str(raw_rule.get("validation_design_review_focus", "") or "").strip(),
        )

    def is_enabled(self, args: Any) -> bool:
        truthy_arg = self.enabled_when_arg_truthy
        if truthy_arg and not bool(getattr(args, truthy_arg, None)):
            return False
        falsey_arg = self.enabled_when_arg_falsey
        if falsey_arg and bool(getattr(args, falsey_arg, None)):
            return False
        enabled_arg = self.enabled_when_arg_in
        if enabled_arg:
            current = set(_arg_values(getattr(args, enabled_arg, None)))
            if self.enabled_values and not current.intersection(self.enabled_values):
                return False
        return True

    def resolved_review_metadata(self, context: ValidationRuleContextLike) -> dict[str, str]:
        defaults = context.design_review_defaults
        return {
            "status": self.review_status or defaults.status,
            "note": self.review_note or defaults.note,
            "reviewer": self.review_reviewer or defaults.reviewer,
            "required_expertise": self.review_required_expertise or defaults.required_expertise,
            "focus": self.review_focus or defaults.focus,
        }


def coerce_validation_rule_records(
    rules: Iterable[ValidationRuleRecord | dict[str, Any]],
) -> tuple[ValidationRuleRecord, ...]:
    records: list[ValidationRuleRecord] = []
    for rule in rules:
        if isinstance(rule, ValidationRuleRecord):
            records.append(rule)
            continue
        records.append(ValidationRuleRecord.from_rule(rule))
    return tuple(records)


def _arg_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(entry).strip() for entry in value if str(entry).strip())
    text = str(value).strip()
    if not text:
        return ()
    return tuple(part.strip() for part in text.split(",") if part.strip())


__all__ = [
    "ValidationRuleRecord",
    "coerce_validation_rule_records",
]
