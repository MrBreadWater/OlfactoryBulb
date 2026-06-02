"""Audit declarative reference-validation configs for human-review metadata coverage."""

from __future__ import annotations

import argparse
from typing import Any

from olfactorybulb.audit.core import AuditItem, AuditReport, KNOWN_HUMAN_REVIEW_STATUSES
from olfactorybulb.audit.reference_validation_config import (
    list_reference_validation_ids,
    load_reference_validation_config,
    validation_human_review_defaults,
    validation_rule_specs,
    validation_skip_item,
)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = __doc__


def _resolved_status(
    explicit_status: Any,
    default_status: str,
) -> str:
    return str(explicit_status if explicit_status not in (None, "") else default_status).strip()


def _item_label(validation_id: str, *, check_id: str, property_name: str | None = None) -> str:
    suffix = f":{property_name}" if property_name else ""
    return f"{validation_id}:{check_id}{suffix}"


def _resolved_property_statuses(
    rule: dict[str, Any],
    *,
    validation_id: str,
    default_status: str,
    fallback_check_id: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    property_metric_map = dict(rule.get("property_metric_map", {}))
    property_statuses = dict(rule.get("property_human_review_statuses", {}))
    check_id = str(rule.get("check_id") or fallback_check_id)
    missing: list[str] = []
    unknown: list[str] = []
    provisional: list[str] = []
    pending: list[str] = []
    rule_level_status = _resolved_status(rule.get("human_review_status"), default_status)
    expected_properties = {str(property_name) for property_name in property_metric_map}
    for field_name in (
        "property_human_review_statuses",
        "property_human_review_notes",
        "property_human_review_reviewers",
    ):
        overrides = rule.get(field_name, {})
        if not isinstance(overrides, dict):
            continue
        extra_properties = sorted({str(property_name) for property_name in overrides} - expected_properties)
        for property_name in extra_properties:
            unknown.append(
                f"{validation_id}:{check_id}:{field_name}:{property_name}=unexpected_property_override"
            )
    for property_name in property_metric_map:
        resolved = _resolved_status(property_statuses.get(property_name), rule_level_status)
        label = _item_label(validation_id, check_id=check_id, property_name=str(property_name))
        if not resolved:
            missing.append(label)
            continue
        if resolved not in KNOWN_HUMAN_REVIEW_STATUSES:
            unknown.append(f"{label}={resolved}")
            continue
        if resolved == "pending_review":
            pending.append(label)
        elif resolved == "provisional":
            provisional.append(label)
    return missing, unknown, pending, provisional


def _resolved_check_statuses(
    rule: dict[str, Any],
    *,
    validation_id: str,
    default_status: str,
    fallback_check_id: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    check_id = str(rule.get("check_id") or fallback_check_id)
    resolved = _resolved_status(rule.get("human_review_status"), default_status)
    label = _item_label(validation_id, check_id=check_id)
    if str(rule.get("kind", "")).strip() == "reference_band_rows":
        return _resolved_property_statuses(
            rule,
            validation_id=validation_id,
            default_status=default_status,
            fallback_check_id=fallback_check_id,
        )
    if not resolved:
        return [label], [], [], []
    if resolved not in KNOWN_HUMAN_REVIEW_STATUSES:
        return [], [f"{label}={resolved}"], [], []
    if resolved == "pending_review":
        return [], [], [label], []
    if resolved == "provisional":
        return [], [], [], [label]
    return [], [], [], []


def run(args: argparse.Namespace) -> AuditReport:
    del args
    validation_ids = list_reference_validation_ids()
    missing_statuses: list[str] = []
    unknown_statuses: list[str] = []
    pending_statuses: list[str] = []
    provisional_statuses: list[str] = []
    default_statuses: dict[str, str] = {}

    for validation_id in validation_ids:
        config = load_reference_validation_config(validation_id=validation_id)
        defaults = validation_human_review_defaults(config)
        default_status = str(defaults.get("default_status", "")).strip()
        default_statuses[validation_id] = default_status
        if not default_status:
            missing_statuses.append(f"{validation_id}:[human_review].default_status")
        elif default_status not in KNOWN_HUMAN_REVIEW_STATUSES:
            unknown_statuses.append(f"{validation_id}:[human_review].default_status={default_status}")

        skip_item = validation_skip_item(config)
        if skip_item is not None:
            resolved = _resolved_status(skip_item.get("human_review_status"), default_status)
            label = _item_label(validation_id, check_id=str(skip_item.get("check_id", "skip_item")))
            if not resolved:
                missing_statuses.append(label)
            elif resolved not in KNOWN_HUMAN_REVIEW_STATUSES:
                unknown_statuses.append(f"{label}={resolved}")
            elif resolved == "pending_review":
                pending_statuses.append(label)
            elif resolved == "provisional":
                provisional_statuses.append(label)

        for rule_index, rule in enumerate(validation_rule_specs(config), start=1):
            missing, unknown, pending, provisional = _resolved_check_statuses(
                rule,
                validation_id=validation_id,
                default_status=default_status,
                fallback_check_id=f"rule_{rule_index}",
            )
            missing_statuses.extend(missing)
            unknown_statuses.extend(unknown)
            pending_statuses.extend(pending)
            provisional_statuses.extend(provisional)

    items = [
        AuditItem(
            check_id="reference_validation_review_status_coverage",
            status="PASS" if not missing_statuses else "FAIL",
            title="Every declarative reference-validation item resolves to a human-review status",
            criterion="Every declarative validation check, skip item, and generated reference-band property must resolve to a non-empty human-review status.",
            description="This audit prevents the validation framework from silently emitting literature-comparison items with no review-state metadata at all.",
            acceptable="Every validation config provides either an explicit human-review status for each item or a config-level default that resolves for every item.",
            acceptable_basis="The audit statically inspects the reference-validation TOML files and expands reference-band checks into per-property decisions. Missing resolved statuses are treated as failures because review-state coverage is now mandatory.",
            evidence={
                "validation_ids": validation_ids,
                "default_statuses": default_statuses,
                "missing_statuses": missing_statuses,
            },
            human_review_status="not_applicable",
        ),
        AuditItem(
            check_id="reference_validation_review_status_values",
            status="PASS" if not unknown_statuses else "FAIL",
            title="Every declarative reference-validation human-review status uses a known value",
            criterion="Human-review status values should stay inside the supported status vocabulary so downstream tools can interpret them consistently.",
            description="This audit enforces a small explicit vocabulary instead of letting each config invent ad hoc review states.",
            acceptable="All resolved statuses are one of accepted, provisional, pending_review, or not_applicable.",
            acceptable_basis="The accepted vocabulary is declared in the audit core module so all validation configs share the same status language.",
            evidence={
                "known_statuses": list(KNOWN_HUMAN_REVIEW_STATUSES),
                "unknown_statuses": unknown_statuses,
            },
            human_review_status="not_applicable",
        ),
        AuditItem(
            check_id="reference_validation_pending_review_items",
            status="WARN" if pending_statuses else "PASS",
            title="Pending-review validation items remain visible",
            criterion="Items that still rely on unreviewed LLM-authored choices should stay marked as pending review until a human accepts or revises them.",
            description="This warning is the tracking surface for unresolved review work. It is not a framework failure; it is a deliberate reminder that some validation decisions still lack human sign-off.",
            acceptable="The report lists every item still marked pending_review so it can be triaged explicitly.",
            acceptable_basis="Pending-review is a first-class status in the review vocabulary. The audit surfaces it as a warning rather than a failure so work can continue without hiding the unresolved review debt.",
            evidence={
                "pending_review_items": pending_statuses,
                "pending_review_count": len(pending_statuses),
            },
            human_review_status="not_applicable",
        ),
        AuditItem(
            check_id="reference_validation_provisional_items",
            status="WARN" if provisional_statuses else "PASS",
            title="Provisional validation items remain explicitly caveated",
            criterion="Any item that is still using a provisional acceptance rule should remain marked provisional until a better source-backed rule replaces it.",
            description="This warning separates intentionally provisional literature-comparison decisions from fully accepted ones.",
            acceptable="Every provisional item is listed explicitly so downstream readers can see where the current validation logic is still a stopgap.",
            acceptable_basis="Provisional is a first-class review status distinct from pending_review because a human may consciously accept a temporary rule while still marking it as caveated.",
            evidence={
                "provisional_items": provisional_statuses,
                "provisional_count": len(provisional_statuses),
            },
            human_review_status="not_applicable",
        ),
    ]
    return AuditReport(
        audit_id="human_review_status",
        title="Human review status audit",
        items=items,
    )
