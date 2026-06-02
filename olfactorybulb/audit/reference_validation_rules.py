"""Generic rule engine for literature-backed validation audits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import math

import numpy as np

from olfactorybulb.audit.core import AuditItem, rounded
from olfactorybulb.audit.reference_data import (
    REPO_ROOT,
    csv_rows,
    load_gc_protocol_rows,
    load_normalized_legacy_mc_tc_rows,
    load_pv_crh_epl_fsi_protocol_rows,
)
from olfactorybulb.audit.reference_notes import notes_for_rows


@dataclass(frozen=True)
class ValidationRuleContext:
    metrics: list[dict[str, Any]]
    summary: dict[str, dict[str, float]]
    args: Any
    config: dict[str, Any]
    protocol_result: Any | None = None


RuleHandler = Callable[[dict[str, Any], ValidationRuleContext], list[AuditItem]]


RULE_HANDLERS: dict[str, RuleHandler] = {}


REFERENCE_ROW_LOADERS: dict[str, Callable[[], list[dict[str, Any]]]] = {
    "legacy_mc_tc_ephys": load_normalized_legacy_mc_tc_rows,
    "pv_crh_epl_fsi_protocols": load_pv_crh_epl_fsi_protocol_rows,
    "gc_protocols": load_gc_protocol_rows,
}


def register_validation_rule(kind: str) -> Callable[[RuleHandler], RuleHandler]:
    def decorator(handler: RuleHandler) -> RuleHandler:
        RULE_HANDLERS[kind] = handler
        return handler

    return decorator


def summarize_numeric_metrics(
    metrics: list[dict[str, Any]],
    *,
    group_field: str = "cell_type",
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        group = str(metric.get(group_field, "")).strip() or "ungrouped"
        grouped.setdefault(group, []).append(metric)

    summary: dict[str, dict[str, float]] = {}
    for group, rows in grouped.items():
        numeric_keys: set[str] = set()
        for row in rows:
            for key, value in row.items():
                if isinstance(value, (bool, list, tuple, dict, str)) or value is None:
                    continue
                if isinstance(value, (int, float, np.integer, np.floating)):
                    numeric_keys.add(str(key))
        summary[group] = {
            key: _mean_metric(rows, key)
            for key in sorted(numeric_keys)
        }
    return summary


def build_rule_items(
    rules: list[dict[str, Any]],
    context: ValidationRuleContext,
) -> list[AuditItem]:
    items: list[AuditItem] = []
    for rule in rules:
        kind = str(rule.get("kind") or "").strip()
        if not kind:
            raise ValueError("Validation rule is missing required 'kind'")
        try:
            handler = RULE_HANDLERS[kind]
        except KeyError as exc:
            known = ", ".join(sorted(RULE_HANDLERS))
            raise KeyError(f"Unknown validation rule kind {kind!r}. Known rule kinds: {known}") from exc
        items.extend(handler(rule, context))
    return items


def _rule_item(
    rule: dict[str, Any],
    *,
    status: str,
    evidence: dict[str, Any] | None = None,
    note: str = "",
    title: str | None = None,
    criterion: str | None = None,
    description: str | None = None,
    acceptable: str | None = None,
    acceptable_basis: str | None = None,
    check_id: str | None = None,
) -> AuditItem:
    return AuditItem(
        check_id=str(check_id or rule["check_id"]),
        status=status,
        title=str(title or rule["title"]),
        criterion=str(criterion or rule["criterion"]),
        description=str(description or rule["description"]),
        acceptable=str(acceptable or rule["acceptable"]),
        acceptable_basis=str(acceptable_basis or rule["acceptable_basis"]),
        evidence=evidence or {},
        note=note,
    )


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [
        float(value)
        for value in (row.get(key) for row in rows)
        if _is_finite_number(value)
    ]
    if not values:
        return float("nan")
    return float(np.mean(values))


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _sigma_phrase(sigma_multiplier: float) -> str:
    if np.isclose(float(sigma_multiplier), 2.0):
        return "two standard deviations"
    if np.isclose(float(sigma_multiplier), 1.0):
        return "one standard deviation"
    return f"{rounded(float(sigma_multiplier), 3)} standard deviations"


def _rounded_dict(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            result[key] = _rounded_dict(value)
        elif _is_finite_number(value):
            result[key] = rounded(float(value))
        else:
            result[key] = value
    return result


def _load_rows(loader_spec: str) -> list[dict[str, Any]]:
    if loader_spec.startswith("csv:"):
        path_text = loader_spec.split(":", 1)[1]
        path = Path(path_text)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return csv_rows(path)
    try:
        loader = REFERENCE_ROW_LOADERS[loader_spec]
    except KeyError as exc:
        known = ", ".join(sorted(REFERENCE_ROW_LOADERS))
        raise KeyError(f"Unknown reference-row loader {loader_spec!r}. Known loaders: {known}") from exc
    return loader()


def _filter_rows(rows: list[dict[str, Any]], spec: dict[str, Any]) -> list[dict[str, Any]]:
    filtered = list(rows)
    filters = list(spec.get("filters", []))
    if spec.get("filter_field") and spec.get("filter_value") is not None:
        filters.append({"field": spec["filter_field"], "value": spec["filter_value"]})
    if spec.get("filter_field") and spec.get("filter_values") is not None:
        filters.append({"field": spec["filter_field"], "values": spec["filter_values"]})
    for rule in filters:
        field = str(rule.get("field") or rule.get("column") or "").strip()
        if not field:
            continue
        if "value" in rule:
            target = str(rule["value"]).strip()
            filtered = [row for row in filtered if str(row.get(field, "")).strip() == target]
        elif "values" in rule:
            allowed = {str(value).strip() for value in rule["values"]}
            filtered = [row for row in filtered if str(row.get(field, "")).strip() in allowed]
    return filtered


def _group_mean(summary: dict[str, dict[str, float]], group: str, metric_key: str) -> float:
    return float(summary.get(group, {}).get(metric_key, float("nan")))


def _reference_annotation(row: dict[str, Any]) -> str:
    mean = row.get("mean")
    sd = row.get("sd")
    units = str(row.get("unit", "")).strip()
    source = str(row.get("Source", "") or row.get("source", "")).strip()
    n_value = row.get("n")
    return (
        f"reference: {rounded(float(mean))} +/- {rounded(float(sd))} "
        f"{units} from {source} (n={n_value})"
    )


@register_validation_rule("protocol_executed")
def _protocol_executed(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    protocol_evidence = dict(getattr(context.protocol_result, "protocol_evidence", {}) or {})
    return [
        _rule_item(
            rule,
            status="PASS" if context.metrics else "FAIL",
            evidence=protocol_evidence,
        )
    ]


@register_validation_rule("all_finite_metric")
def _all_finite_metric(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    metric_key = str(rule["metric_key"])
    entity_key = str(rule.get("entity_key", "cell_name"))
    failing = {
        str(metric.get(entity_key, f"row_{index}")): metric.get(metric_key)
        for index, metric in enumerate(context.metrics)
        if not _is_finite_number(metric.get(metric_key))
    }
    evidence = {"metric_key": metric_key, "cell_count": len(context.metrics), "failing_values": _rounded_dict(failing)}
    return [_rule_item(rule, status="PASS" if not failing else "FAIL", evidence=evidence)]


@register_validation_rule("all_exact_metric")
def _all_exact_metric(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    metric_key = str(rule["metric_key"])
    entity_key = str(rule.get("entity_key", "cell_name"))
    expected = float(rule.get("expected", 0.0))
    tolerance = float(rule.get("tolerance", 1e-9))
    failing = {
        str(metric.get(entity_key, f"row_{index}")): metric.get(metric_key)
        for index, metric in enumerate(context.metrics)
        if not (_is_finite_number(metric.get(metric_key)) and abs(float(metric.get(metric_key)) - expected) <= tolerance)
    }
    evidence = {
        "metric_key": metric_key,
        "expected": expected,
        "tolerance": tolerance,
        "failing_values": _rounded_dict(failing),
    }
    return [_rule_item(rule, status="PASS" if not failing else "FAIL", evidence=evidence)]


@register_validation_rule("group_ordering")
def _group_ordering(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    metric_key = str(rule["metric_key"])
    left_group = str(rule["left_group"])
    right_group = str(rule["right_group"])
    operator = str(rule.get("operator", ">")).strip()
    left_value = _group_mean(context.summary, left_group, metric_key)
    right_value = _group_mean(context.summary, right_group, metric_key)
    if operator == ">":
        passed = right_value > left_value
        diff = right_value - left_value
    elif operator == "<":
        passed = right_value < left_value
        diff = right_value - left_value
    else:
        raise ValueError(f"Unsupported group_ordering operator {operator!r}")
    evidence = _rounded_dict(
        {
            f"{left_group}_mean": left_value,
            f"{right_group}_mean": right_value,
            f"{right_group}_minus_{left_group}": diff,
        }
    )
    return [_rule_item(rule, status="PASS" if passed else "FAIL", evidence=evidence)]


@register_validation_rule("group_abs_diff_max")
def _group_abs_diff_max(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    metric_key = str(rule["metric_key"])
    left_group = str(rule["left_group"])
    right_group = str(rule["right_group"])
    max_difference = float(rule["max_difference"])
    left_value = _group_mean(context.summary, left_group, metric_key)
    right_value = _group_mean(context.summary, right_group, metric_key)
    difference = abs(right_value - left_value)
    evidence = _rounded_dict(
        {
            f"{left_group}_mean": left_value,
            f"{right_group}_mean": right_value,
            "absolute_difference": difference,
            "max_difference": max_difference,
        }
    )
    return [_rule_item(rule, status="PASS" if difference <= max_difference else "FAIL", evidence=evidence)]


@register_validation_rule("group_positive")
def _group_positive(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    metric_key = str(rule["metric_key"])
    groups = [str(group) for group in rule.get("groups", [])]
    if not groups:
        raise ValueError("group_positive rule requires non-empty 'groups'")
    evidence = _rounded_dict({f"{group}_mean": _group_mean(context.summary, group, metric_key) for group in groups})
    passed = all(_is_finite_number(_group_mean(context.summary, group, metric_key)) and _group_mean(context.summary, group, metric_key) > 0.0 for group in groups)
    return [_rule_item(rule, status="PASS" if passed else "FAIL", evidence=evidence)]


@register_validation_rule("reference_band_rows")
def _reference_band_rows(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    loader = str(rule["loader"])
    reference_source = str(rule.get("reference_source", "")).strip()
    group_field = str(rule.get("group_field", "cell_type"))
    property_metric_map = {str(key): str(value) for key, value in dict(rule.get("property_metric_map", {})).items()}
    sigma_arg_name = str(rule.get("sigma_arg_name", "reference_sigma_multiplier"))
    sigma_multiplier = float(getattr(context.args, sigma_arg_name, rule.get("sigma_multiplier", 2.0)))
    sigma_phrase = _sigma_phrase(sigma_multiplier)
    rows = _filter_rows(_load_rows(loader), rule)
    items: list[AuditItem] = []
    for row in rows:
        if reference_source and str(row.get("Source", "")).strip() != reference_source:
            continue
        property_name = str(row.get("Property", "")).strip()
        metric_key = property_metric_map.get(property_name)
        if not metric_key:
            continue
        group = str(row.get(group_field, "")).strip()
        if not group:
            continue
        observed_value = _group_mean(context.summary, group, metric_key)
        if not (_is_finite_number(row.get("mean")) and _is_finite_number(row.get("sd"))):
            continue
        reference_mean = float(row["mean"])
        reference_sd = float(row["sd"])
        accepted_low = reference_mean - reference_sd * sigma_multiplier
        accepted_high = reference_mean + reference_sd * sigma_multiplier
        status = "PASS" if _is_finite_number(observed_value) and accepted_low <= observed_value <= accepted_high else "FAIL"
        item_id = f"{group.lower()}_{metric_key.lower()}_within_uploaded_reference_band".replace(".", "_")
        evidence_key = f"{group}_mean"
        evidence = _rounded_dict(
            {
                evidence_key: observed_value,
                "accepted_low": accepted_low,
                "accepted_high": accepted_high,
                "accepted_sigma_multiplier": sigma_multiplier,
                "__reference_annotations__": {evidence_key: _reference_annotation(row)},
            }
        )
        items.append(
            _rule_item(
                rule,
                check_id=item_id,
                status=status,
                title=f"{group} {property_name.lower()} stays within the uploaded reference band",
                criterion=(
                    f"The {group} mean {property_name.lower()} should remain within {sigma_phrase} "
                    f"of the uploaded reference value."
                ),
                description=(
                    f"This is the direct single-cell-type reference check derived from uploaded literature rows for "
                    f"{property_name} rather than from a cross-group ordering heuristic."
                ),
                acceptable=(
                    f"The observed {group} mean must lie between {rounded(accepted_low)} and {rounded(accepted_high)} "
                    f"{str(row.get('unit', '')).strip()}, which corresponds to mean plus or minus "
                    f"{sigma_phrase}."
                ),
                acceptable_basis=(
                    f"The accepted interval is computed from the uploaded literature row for {property_name} "
                    f"as mean plus or minus {sigma_phrase}. The sigma multiplier comes from the configurable "
                    f"'{sigma_arg_name}' setting."
                ),
                evidence=evidence,
            )
        )
    return items


@register_validation_rule("note_presence")
def _note_presence(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    scope = str(rule.get("scope", "")).strip() or None
    row_contexts = list(rule.get("row_contexts", []))
    rows: list[dict[str, Any]] = []
    for row_context in row_contexts:
        context_rows = _filter_rows(_load_rows(str(row_context["loader"])), row_context)
        if row_context.get("as_protocol_context"):
            property_name = str(row_context.get("property_name", "FI Protocol"))
            for row in context_rows:
                rows.append(
                    {
                        "protocol_id": row.get("protocol_id", ""),
                        "note_ids": "",
                        "Property": property_name,
                        "source": row.get("source", row.get("Source", "")),
                    }
                )
        else:
            rows.extend(context_rows)
    matched_notes = notes_for_rows(rows, scope=scope)
    evidence = {
        "protocol_ids_in_scope": sorted(
            {
                str(row.get("protocol_id", "")).strip()
                for row in rows
                if str(row.get("protocol_id", "")).strip()
            }
        ),
        "notes": [note.message for note in matched_notes],
        "note_ids": [note.note_id for note in matched_notes],
    }
    status = "WARN" if matched_notes else "PASS"
    return [_rule_item(rule, status=status, evidence=evidence)]


__all__ = [
    "REFERENCE_ROW_LOADERS",
    "RULE_HANDLERS",
    "ValidationRuleContext",
    "build_rule_items",
    "register_validation_rule",
    "summarize_numeric_metrics",
]
