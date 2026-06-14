"""Generic rule engine for literature-backed validation audits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
import math

import numpy as np

from olfactorybulb.audit import AuditItem, series_visual_spec
from olfactorybulb.audit.criterion_math import (
    criterion_math_for_absolute_difference,
    criterion_math_for_closed_range,
    criterion_math_for_exact_metric,
    criterion_math_for_lower_bound,
    criterion_math_for_ordering,
    criterion_math_for_reference_band,
    criterion_math_for_upper_bound,
    group_mean_symbol,
)
from olfactorybulb.audit.core import rounded
from olfactorybulb.audit.reference_data import (
    REPO_ROOT,
    csv_rows,
    load_dataset_output_rows,
    load_normalized_legacy_mc_tc_rows,
)
from olfactorybulb.audit.reference_notes import load_notes, notes_for_rows
from olfactorybulb.neuronunit.reference_bands import (
    ProvenanceRecord,
    ReferenceAcceptanceBand,
    ReferenceBandObservation,
    ReferenceBandPolicy,
    ValidationReview,
    compute_reference_acceptance_band,
    sigma_phrase as _sigma_phrase,
)
from olfactorybulb.neuronunit.reference_validation_suite import (
    ReferenceBandCase,
    audit_items_from_reference_band_suite,
    compile_reference_band_suite,
)
from olfactorybulb.neuronunit.comparison_validation_suite import (
    ComparisonRuleCase,
    audit_items_from_comparison_rule_suite,
    compile_comparison_rule_suite,
)
from olfactorybulb.neuronunit.summary_validation_suite import (
    SummaryRuleCase,
    audit_items_from_summary_rule_suite,
    compile_summary_rule_suite,
)


@dataclass(frozen=True)
class ValidationRuleContext:
    metrics: list[dict[str, Any]]
    summary: dict[str, dict[str, float]]
    args: Any
    config: dict[str, Any]
    protocol_result: Any | None = None


RuleHandler = Callable[[dict[str, Any], ValidationRuleContext], list[AuditItem]]


RULE_HANDLERS: dict[str, RuleHandler] = {}
SUMMARY_RULE_KINDS = {
    "summary_metric_min",
    "summary_metric_max",
    "summary_metric_range",
    "summary_metric_status_map",
}
COMPARISON_RULE_KINDS = {
    "all_finite_metric",
    "all_exact_metric",
    "group_ordering",
    "group_abs_diff_max",
    "group_positive",
}


REFERENCE_ROW_LOADERS: dict[str, Callable[[], list[dict[str, Any]]]] = {
    "legacy_mc_tc_ephys": load_normalized_legacy_mc_tc_rows,
    "epl_fsi_ephys": lambda: load_dataset_output_rows(dataset_id="pv_crh_epl_fsi", output_key="ephys"),
    "epl_fsi_fi_curve": lambda: load_dataset_output_rows(dataset_id="pv_crh_epl_fsi", output_key="fi_curve"),
    "epl_fsi_identity": lambda: load_dataset_output_rows(dataset_id="pv_crh_epl_fsi", output_key="identity"),
    "pv_crh_epl_fsi_protocols": lambda: load_dataset_output_rows(dataset_id="pv_crh_epl_fsi", output_key="protocols"),
    "gc_ephys": lambda: load_dataset_output_rows(dataset_id="granule_cells", output_key="ephys"),
    "gc_fi_curve": lambda: load_dataset_output_rows(dataset_id="granule_cells", output_key="fi_curve"),
    "gc_subtype_ephys": lambda: load_dataset_output_rows(dataset_id="granule_cells", output_key="subtype_ephys"),
    "gc_subtype_fi_curve": lambda: load_dataset_output_rows(dataset_id="granule_cells", output_key="subtype_fi_curve"),
    "gc_identity": lambda: load_dataset_output_rows(dataset_id="granule_cells", output_key="identity"),
    "gc_modulation": lambda: load_dataset_output_rows(dataset_id="granule_cells", output_key="modulation"),
    "gc_protocols": lambda: load_dataset_output_rows(dataset_id="granule_cells", output_key="protocols"),
    "gc_synaptic_latency": lambda: load_dataset_output_rows(dataset_id="granule_cells", output_key="synaptic_latency"),
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
    pending_summary_rules: list[dict[str, Any]] = []
    pending_comparison_rules: list[dict[str, Any]] = []

    def flush_pending_summary_rules() -> None:
        nonlocal pending_summary_rules
        if not pending_summary_rules:
            return
        generated_items = _build_summary_rule_items(pending_summary_rules, context)
        for rule, item in zip(pending_summary_rules, generated_items, strict=False):
            _apply_rule_level_validation_design_review([item], rule, context)
            items.append(item)
        pending_summary_rules = []

    def flush_pending_comparison_rules() -> None:
        nonlocal pending_comparison_rules
        if not pending_comparison_rules:
            return
        generated_items = _build_comparison_rule_items(pending_comparison_rules, context)
        for rule, item in zip(pending_comparison_rules, generated_items, strict=False):
            _apply_rule_level_validation_design_review([item], rule, context)
            items.append(item)
        pending_comparison_rules = []

    for rule in rules:
        if not _rule_enabled(rule, context.args):
            continue
        kind = str(rule.get("kind") or "").strip()
        if not kind:
            raise ValueError("Validation rule is missing required 'kind'")
        if kind in SUMMARY_RULE_KINDS:
            flush_pending_comparison_rules()
            pending_summary_rules.append(rule)
            continue
        if kind in COMPARISON_RULE_KINDS:
            flush_pending_summary_rules()
            pending_comparison_rules.append(rule)
            continue
        flush_pending_summary_rules()
        flush_pending_comparison_rules()
        try:
            handler = RULE_HANDLERS[kind]
        except KeyError as exc:
            known = ", ".join(sorted(RULE_HANDLERS))
            raise KeyError(f"Unknown validation rule kind {kind!r}. Known rule kinds: {known}") from exc
        rule_items = handler(rule, context)
        _apply_rule_level_validation_design_review(rule_items, rule, context)
        items.extend(rule_items)
    flush_pending_summary_rules()
    flush_pending_comparison_rules()
    return items


def _config_validation_design_review_defaults(context: ValidationRuleContext) -> dict[str, Any]:
    defaults = context.config.get("validation_design_review", {})
    return dict(defaults) if isinstance(defaults, dict) else {}


def _resolved_rule_validation_design_review(
    rule: dict[str, Any],
    context: ValidationRuleContext,
) -> dict[str, str]:
    defaults = _config_validation_design_review_defaults(context)
    return {
        "status": str(rule.get("validation_design_review_status", defaults.get("default_status", ""))).strip(),
        "note": str(rule.get("validation_design_review_note", defaults.get("default_note", ""))).strip(),
        "reviewer": str(rule.get("validation_design_review_reviewer", defaults.get("default_reviewer", ""))).strip(),
        "required_expertise": str(
            rule.get(
                "validation_design_review_required_expertise",
                defaults.get("default_required_expertise", ""),
            )
        ).strip(),
        "focus": str(
            rule.get(
                "validation_design_review_focus",
                defaults.get("default_focus", ""),
            )
        ).strip(),
    }


def _apply_rule_level_validation_design_review(
    items: list[AuditItem],
    rule: dict[str, Any],
    context: ValidationRuleContext,
) -> None:
    metadata = _resolved_rule_validation_design_review(rule, context)
    for item in items:
        if not item.validation_design_review_status and metadata["status"]:
            item.validation_design_review_status = metadata["status"]
        if not item.validation_design_review_note and metadata["note"]:
            item.validation_design_review_note = metadata["note"]
        if not item.validation_design_review_reviewer and metadata["reviewer"]:
            item.validation_design_review_reviewer = metadata["reviewer"]
        if not item.validation_design_review_required_expertise and metadata["required_expertise"]:
            item.validation_design_review_required_expertise = metadata["required_expertise"]
        if not item.validation_design_review_focus and metadata["focus"]:
            item.validation_design_review_focus = metadata["focus"]
        if item.status == "WARN" and not item.status_reason:
            if item.validation_design_review_status == "pending":
                item.status_reason = (
                    "This item is intentionally surfaced as a warning because the validation-design choice behind it is still pending expert review."
                )
            elif item.validation_design_review_status == "provisional":
                item.status_reason = (
                    "This item is intentionally surfaced as a warning because the validation-design choice behind it is still provisional."
                )


def _rule_item(
    rule: dict[str, Any],
    *,
    status: str,
    evidence: dict[str, Any] | None = None,
    note: str = "",
    status_reason: str = "",
    title: str | None = None,
    criterion: str | None = None,
    criterion_latex: str | None = None,
    criterion_formulae: list[str] | None = None,
    criterion_definitions: list[dict[str, Any]] | None = None,
    description: str | None = None,
    acceptable: str | None = None,
    acceptable_basis: str | None = None,
    check_id: str | None = None,
    series_visuals: list[dict[str, Any]] | None = None,
    companion_visuals: list[dict[str, Any]] | None = None,
) -> AuditItem:
    return AuditItem(
        check_id=str(check_id or rule["check_id"]),
        status=status,
        title=str(title or rule["title"]),
        criterion=str(criterion or rule["criterion"]),
        criterion_latex=str(criterion_latex if criterion_latex is not None else rule.get("criterion_latex", "")),
        criterion_formulae=(
            criterion_formulae if criterion_formulae is not None else rule.get("criterion_formulae", [])
        ),
        criterion_definitions=(
            criterion_definitions if criterion_definitions is not None else rule.get("criterion_definitions", [])
        ),
        description=str(description or rule["description"]),
        acceptable=str(acceptable or rule["acceptable"]),
        acceptable_basis=str(acceptable_basis or rule["acceptable_basis"]),
        evidence=evidence or {},
        series_visuals=list(series_visuals or []),
        companion_visuals=list(companion_visuals or []),
        note=note,
        status_reason=status_reason,
    )


def _rule_status(rule: dict[str, Any], passed: bool) -> str:
    return str(rule.get("pass_status", "PASS") if passed else rule.get("fail_status", "FAIL"))


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


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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


def _rule_enabled(rule: dict[str, Any], args: Any) -> bool:
    truthy_arg = str(rule.get("enabled_when_arg_truthy", "") or "").strip()
    if truthy_arg and not bool(getattr(args, truthy_arg, None)):
        return False
    falsey_arg = str(rule.get("enabled_when_arg_falsey", "") or "").strip()
    if falsey_arg and bool(getattr(args, falsey_arg, None)):
        return False
    enabled_arg = str(rule.get("enabled_when_arg_in", "") or "").strip()
    if enabled_arg:
        allowed = {str(value).strip() for value in rule.get("enabled_values", []) if str(value).strip()}
        current = set(_arg_values(getattr(args, enabled_arg, None)))
        if allowed and not current.intersection(allowed):
            return False
    return True


def _arg_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        tokens = value.replace(";", ",").split(",")
        return [token.strip() for token in tokens if token.strip()]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, dict)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _load_rows(loader_spec: str) -> list[dict[str, Any]]:
    if loader_spec.startswith("csv:"):
        path_text = loader_spec.split(":", 1)[1]
        path = Path(path_text)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return csv_rows(path)
    if loader_spec.startswith("dataset:"):
        try:
            _prefix, dataset_id, output_key = loader_spec.split(":", 2)
        except ValueError as exc:
            raise ValueError(
                "Dataset loader specs must look like 'dataset:<dataset_id>:<output_key>'"
            ) from exc
        return load_dataset_output_rows(dataset_id=dataset_id, output_key=output_key)
    try:
        loader = REFERENCE_ROW_LOADERS[loader_spec]
    except KeyError as exc:
        known = ", ".join(sorted(REFERENCE_ROW_LOADERS))
        raise KeyError(f"Unknown reference-row loader {loader_spec!r}. Known loaders: {known}") from exc
    return loader()


def _filter_rows(rows: list[dict[str, Any]], spec: dict[str, Any], *, args: Any | None = None) -> list[dict[str, Any]]:
    filtered = list(rows)
    filters = list(spec.get("filters", []))
    if spec.get("filter_field") and spec.get("filter_value") is not None:
        filters.append({"field": spec["filter_field"], "value": spec["filter_value"]})
    if spec.get("filter_field") and spec.get("filter_values") is not None:
        filters.append({"field": spec["filter_field"], "values": spec["filter_values"]})
    if args is not None and spec.get("filter_field") and spec.get("filter_value_arg"):
        filters.append({"field": spec["filter_field"], "value": getattr(args, str(spec["filter_value_arg"]), None)})
    if args is not None and spec.get("filter_field") and spec.get("filter_values_arg"):
        filters.append({"field": spec["filter_field"], "values": _arg_values(getattr(args, str(spec["filter_values_arg"]), None))})
    for rule in filters:
        field = str(rule.get("field") or rule.get("column") or "").strip()
        if not field:
            continue
        if args is not None and rule.get("value_arg"):
            rule = {**rule, "value": getattr(args, str(rule["value_arg"]), None)}
        if args is not None and rule.get("values_arg"):
            rule = {**rule, "values": _arg_values(getattr(args, str(rule["values_arg"]), None))}
        if "value" in rule:
            target = str(rule["value"]).strip()
            if not target:
                continue
            filtered = [row for row in filtered if str(row.get(field, "")).strip() == target]
        elif "values" in rule:
            allowed = {str(value).strip() for value in rule["values"]}
            if not allowed:
                continue
            filtered = [row for row in filtered if str(row.get(field, "")).strip() in allowed]
    return filtered


def _property_override(
    rule: dict[str, Any],
    field_name: str,
    property_name: str,
    default: Any = None,
) -> Any:
    overrides = rule.get(field_name, {})
    if not isinstance(overrides, dict):
        return default
    return overrides.get(property_name, default)


def _property_review_metadata(
    rule: dict[str, Any],
    context: ValidationRuleContext,
    property_name: str,
) -> dict[str, str]:
    defaults = _resolved_rule_validation_design_review(rule, context)
    return {
        "status": str(
            _property_override(rule, "property_validation_design_review_statuses", property_name, defaults["status"])
        ).strip(),
        "note": str(
            _property_override(rule, "property_validation_design_review_notes", property_name, defaults["note"])
        ).strip(),
        "reviewer": str(
            _property_override(rule, "property_validation_design_review_reviewers", property_name, defaults["reviewer"])
        ).strip(),
        "required_expertise": str(
            _property_override(
                rule,
                "property_validation_design_review_required_expertise",
                property_name,
                defaults["required_expertise"],
            )
        ).strip(),
        "focus": str(
            _property_override(
                rule,
                "property_validation_design_review_focuses",
                property_name,
                defaults["focus"],
            )
        ).strip(),
    }


def _property_band_modes(rule: dict[str, Any], property_metric_map: dict[str, str]) -> dict[str, str]:
    if "default_band_mode" in rule:
        raise ValueError(
            "reference_band_rows no longer supports 'default_band_mode'; "
            "choose 'property_band_modes' explicitly for every property"
        )
    raw_modes = rule.get("property_band_modes")
    if not isinstance(raw_modes, dict):
        raise ValueError(
            "reference_band_rows requires a 'property_band_modes' table that selects a band mode for every property"
        )
    normalized_modes = {
        str(property_name).strip(): str(mode).strip()
        for property_name, mode in raw_modes.items()
        if str(property_name).strip()
    }
    expected = set(property_metric_map)
    actual = set(normalized_modes)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append(f"missing explicit modes for: {', '.join(missing)}")
        if extra:
            parts.append(f"unexpected mode entries for: {', '.join(extra)}")
        raise ValueError(
            "reference_band_rows requires an explicit band mode for every property in property_metric_map; "
            + "; ".join(parts)
        )
    return normalized_modes


def _criterion_text_for_band(group: str, property_name: str, band: ReferenceAcceptanceBand, sigma_phrase: str) -> str:
    if band.mode == "quantile_interval":
        return (
            f"The {group} mean {property_name.lower()} should remain within the uploaded reported quantile interval."
        )
    if band.mode == "beta_sd":
        return (
            f"The {group} mean {property_name.lower()} should remain within the uploaded beta-reconstructed bounded probability interval."
        )
    if band.mode == "binary_indicator":
        return (
            f"The {group} mean {property_name.lower()} should match the uploaded binary reference indicator exactly."
        )
    if band.mode == "lognormal_sd":
        return (
            f"The {group} mean {property_name.lower()} should remain within {sigma_phrase} of the uploaded reference value "
            f"under a lognormal reconstruction."
        )
    return (
        f"The {group} mean {property_name.lower()} should remain within {sigma_phrase} of the uploaded reference value."
    )


def _title_text_for_band(group: str, property_name: str, band: ReferenceAcceptanceBand) -> str:
    if band.mode == "binary_indicator":
        return f"{group} {property_name.lower()} matches the uploaded binary reference indicator"
    return f"{group} {property_name.lower()} stays within the uploaded reference band"


def _row_field_name(
    rule: dict[str, Any],
    property_name: str,
    *,
    field_override_key: str,
    default_field_key: str,
    default: str,
) -> str:
    override = _property_override(rule, field_override_key, property_name, None)
    if override is not None:
        return str(override).strip()
    return str(rule.get(default_field_key, default) or default).strip()


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


def _summary_group(rule: dict[str, Any], context: ValidationRuleContext) -> str:
    explicit = str(rule.get("group", "") or "").strip()
    if explicit:
        return explicit
    if len(context.summary) == 1:
        return next(iter(context.summary))
    return str(context.config.get("default_group", "ungrouped"))


def _summary_evidence(
    rule: dict[str, Any],
    context: ValidationRuleContext,
    *,
    group: str,
    base: dict[str, Any],
) -> dict[str, Any]:
    evidence = dict(base)
    for metric_key in rule.get("evidence_metric_keys", []):
        metric_name = str(metric_key).strip()
        if not metric_name:
            continue
        evidence[metric_name] = _group_mean(context.summary, group, metric_name)
    return _rounded_dict(evidence)


def _summary_rule_case(rule: dict[str, Any], context: ValidationRuleContext) -> SummaryRuleCase:
    kind = str(rule["kind"])
    metric_key = str(rule["metric_key"])
    group = _summary_group(rule, context)
    observed_symbol = group_mean_symbol(group)
    base_kwargs = {
        "rule_kind": kind,
        "check_id": str(rule["check_id"]),
        "title": str(rule["title"]),
        "criterion": str(rule["criterion"]),
        "criterion_latex": str(rule.get("criterion_latex", "")),
        "criterion_formulae": list(rule.get("criterion_formulae", [])),
        "criterion_definitions": list(rule.get("criterion_definitions", [])),
        "description": str(rule["description"]),
        "acceptable": str(rule["acceptable"]),
        "acceptable_basis": str(rule["acceptable_basis"]),
        "note": str(rule.get("note", "")),
        "metric_key": metric_key,
        "group": group,
        "evidence_metric_keys": [str(metric).strip() for metric in rule.get("evidence_metric_keys", []) if str(metric).strip()],
        "pass_status": str(rule.get("pass_status", "PASS")),
        "fail_status": str(rule.get("fail_status", "FAIL")),
        "default_status": str(rule.get("default_status", "FAIL")),
    }
    if kind == "summary_metric_min":
        minimum = float(rule["minimum"])
        criterion_math = criterion_math_for_lower_bound(
            observed_symbol,
            minimum,
            definitions=[{"symbol": observed_symbol, "definition": f"{group} mean {metric_key}"}],
        )
        case_kwargs = dict(base_kwargs)
        case_kwargs["criterion_latex"] = criterion_math.latex
        case_kwargs["criterion_definitions"] = criterion_math.definitions
        case_kwargs["minimum"] = minimum
        return SummaryRuleCase(**case_kwargs)
    if kind == "summary_metric_max":
        maximum = float(rule["maximum"])
        criterion_math = criterion_math_for_upper_bound(
            observed_symbol,
            maximum,
            definitions=[{"symbol": observed_symbol, "definition": f"{group} mean {metric_key}"}],
        )
        case_kwargs = dict(base_kwargs)
        case_kwargs["criterion_latex"] = criterion_math.latex
        case_kwargs["criterion_definitions"] = criterion_math.definitions
        case_kwargs["maximum"] = maximum
        return SummaryRuleCase(**case_kwargs)
    if kind == "summary_metric_range":
        minimum = float(rule.get("minimum", float("-inf")))
        maximum = float(rule.get("maximum", float("inf")))
        criterion_math = criterion_math_for_closed_range(
            observed_symbol,
            minimum,
            maximum,
            definitions=[{"symbol": observed_symbol, "definition": f"{group} mean {metric_key}"}],
        )
        case_kwargs = dict(base_kwargs)
        case_kwargs["criterion_latex"] = criterion_math.latex
        case_kwargs["criterion_definitions"] = criterion_math.definitions
        case_kwargs["minimum"] = minimum
        case_kwargs["maximum"] = maximum
        return SummaryRuleCase(**case_kwargs)
    if kind == "summary_metric_status_map":
        return SummaryRuleCase(
            **base_kwargs,
            pass_values=tuple(float(value) for value in rule.get("pass_values", [])),
            warn_values=tuple(float(value) for value in rule.get("warn_values", [])),
            fail_values=tuple(float(value) for value in rule.get("fail_values", [])),
        )
    raise ValueError(f"Unsupported summary rule kind {kind!r}")


def _build_summary_rule_items(
    rules: list[dict[str, Any]],
    context: ValidationRuleContext,
) -> list[AuditItem]:
    if not rules:
        return []
    cases = [_summary_rule_case(rule, context) for rule in rules]
    suite_name = f"{str(context.config.get('validation_id', 'validation')).strip() or 'validation'}.summary_rules"
    compiled = compile_summary_rule_suite(cases=cases, summary=context.summary, suite_name=suite_name)
    return audit_items_from_summary_rule_suite(compiled)


def _comparison_rule_case(rule: dict[str, Any], context: ValidationRuleContext) -> ComparisonRuleCase:
    kind = str(rule["kind"])
    metric_key = str(rule["metric_key"])
    entity_key = str(rule.get("entity_key", "cell_name"))
    base_kwargs = {
        "rule_kind": kind,
        "check_id": str(rule["check_id"]),
        "title": str(rule["title"]),
        "criterion": str(rule["criterion"]),
        "criterion_latex": str(rule.get("criterion_latex", "")),
        "criterion_formulae": list(rule.get("criterion_formulae", [])),
        "criterion_definitions": list(rule.get("criterion_definitions", [])),
        "description": str(rule["description"]),
        "acceptable": str(rule["acceptable"]),
        "acceptable_basis": str(rule["acceptable_basis"]),
        "note": str(rule.get("note", "")),
        "metric_key": metric_key,
        "entity_key": entity_key,
        "pass_status": str(rule.get("pass_status", "PASS")),
        "fail_status": str(rule.get("fail_status", "FAIL")),
    }
    if kind == "all_finite_metric":
        return ComparisonRuleCase(**base_kwargs)
    if kind == "all_exact_metric":
        criterion_math = criterion_math_for_exact_metric(metric_key)
        case_kwargs = dict(base_kwargs)
        case_kwargs["criterion_latex"] = criterion_math.latex
        case_kwargs["criterion_definitions"] = criterion_math.definitions
        case_kwargs["expected"] = float(rule.get("expected", 0.0))
        case_kwargs["tolerance"] = float(rule.get("tolerance", 1e-9))
        return ComparisonRuleCase(**case_kwargs)
    if kind == "group_ordering":
        left_group = str(rule["left_group"])
        right_group = str(rule["right_group"])
        operator = str(rule.get("operator", ">")).strip()
        left_symbol = group_mean_symbol(left_group)
        right_symbol = group_mean_symbol(right_group)
        criterion_math = criterion_math_for_ordering(right_symbol, operator, left_symbol)
        case_kwargs = dict(base_kwargs)
        case_kwargs["criterion_latex"] = criterion_math.latex
        case_kwargs["criterion_definitions"] = [
            {"symbol": left_symbol, "definition": f"{left_group} mean {metric_key}"},
            {"symbol": right_symbol, "definition": f"{right_group} mean {metric_key}"},
        ]
        case_kwargs["left_group"] = left_group
        case_kwargs["right_group"] = right_group
        case_kwargs["operator"] = operator
        return ComparisonRuleCase(**case_kwargs)
    if kind == "group_abs_diff_max":
        left_group = str(rule["left_group"])
        right_group = str(rule["right_group"])
        max_difference = float(rule["max_difference"])
        left_symbol = group_mean_symbol(left_group)
        right_symbol = group_mean_symbol(right_group)
        criterion_math = criterion_math_for_absolute_difference(
            left_symbol,
            right_symbol,
            max_difference,
            definitions=[
                {"symbol": left_symbol, "definition": f"{left_group} mean {metric_key}"},
                {"symbol": right_symbol, "definition": f"{right_group} mean {metric_key}"},
            ],
        )
        case_kwargs = dict(base_kwargs)
        case_kwargs["criterion_latex"] = criterion_math.latex
        case_kwargs["criterion_definitions"] = criterion_math.definitions
        case_kwargs["left_group"] = left_group
        case_kwargs["right_group"] = right_group
        case_kwargs["max_difference"] = max_difference
        return ComparisonRuleCase(**case_kwargs)
    if kind == "group_positive":
        groups = tuple(str(group) for group in rule.get("groups", []))
        if not groups:
            raise ValueError("group_positive rule requires non-empty 'groups'")
        group_symbols = [group_mean_symbol(group) for group in groups]
        case_kwargs = dict(base_kwargs)
        case_kwargs["criterion_latex"] = " \\wedge ".join(rf"{symbol} > 0" for symbol in group_symbols)
        case_kwargs["criterion_definitions"] = [
            {"symbol": symbol, "definition": f"{group} mean {metric_key}"}
            for symbol, group in zip(group_symbols, groups, strict=False)
        ]
        case_kwargs["groups"] = groups
        return ComparisonRuleCase(**case_kwargs)
    raise ValueError(f"Unsupported comparison rule kind {kind!r}")


def _build_comparison_rule_items(
    rules: list[dict[str, Any]],
    context: ValidationRuleContext,
) -> list[AuditItem]:
    if not rules:
        return []
    cases = [_comparison_rule_case(rule, context) for rule in rules]
    suite_name = f"{str(context.config.get('validation_id', 'validation')).strip() or 'validation'}.comparison_rules"
    compiled = compile_comparison_rule_suite(
        cases=cases,
        summary=context.summary,
        metrics=context.metrics,
        suite_name=suite_name,
    )
    return audit_items_from_comparison_rule_suite(compiled)


def _notes_path(rule: dict[str, Any], context: ValidationRuleContext) -> Path | None:
    path_text = str(rule.get("notes_path") or context.config.get("notes_path") or "").strip()
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _curve_points_by_current(
    rows: list[dict[str, Any]],
    *,
    current_key: str,
    value_key: str,
    precision_digits: int = 6,
) -> dict[float, float]:
    buckets: dict[float, list[float]] = {}
    for row in rows:
        if not (_is_finite_number(row.get(current_key)) and _is_finite_number(row.get(value_key))):
            continue
        current_value = round(float(row[current_key]), int(precision_digits))
        value = float(row[value_key])
        buckets.setdefault(current_value, []).append(value)
    return {current: float(np.mean(values)) for current, values in sorted(buckets.items())}


@register_validation_rule("protocol_executed")
def _protocol_executed(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    protocol_evidence = dict(getattr(context.protocol_result, "protocol_evidence", {}) or {})
    series_visuals: list[dict[str, Any]] = []
    fi_curve_rows = protocol_evidence.get("fi_curve_rows")
    if isinstance(fi_curve_rows, list) and fi_curve_rows:
        series_visuals.append(
            series_visual_spec(
                keys=["fi_curve_rows"],
                style={
                    "line_width": 1.8,
                    "marker_size": 3.2,
                    "legend_loc": "lower center",
                },
            )
        )
    return [
        _rule_item(
            rule,
            status=_rule_status(rule, bool(context.metrics)),
            evidence=protocol_evidence,
            series_visuals=series_visuals,
        )
    ]


@register_validation_rule("all_finite_metric")
def _all_finite_metric(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    return _build_comparison_rule_items([rule], context)


@register_validation_rule("all_exact_metric")
def _all_exact_metric(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    return _build_comparison_rule_items([rule], context)


@register_validation_rule("group_ordering")
def _group_ordering(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    return _build_comparison_rule_items([rule], context)


@register_validation_rule("group_abs_diff_max")
def _group_abs_diff_max(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    return _build_comparison_rule_items([rule], context)


@register_validation_rule("group_positive")
def _group_positive(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    return _build_comparison_rule_items([rule], context)


@register_validation_rule("summary_metric_min")
def _summary_metric_min(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    return _build_summary_rule_items([rule], context)


@register_validation_rule("summary_metric_max")
def _summary_metric_max(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    return _build_summary_rule_items([rule], context)


@register_validation_rule("summary_metric_range")
def _summary_metric_range(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    return _build_summary_rule_items([rule], context)


@register_validation_rule("summary_metric_status_map")
def _summary_metric_status_map(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    return _build_summary_rule_items([rule], context)


@register_validation_rule("reference_band_rows")
def _reference_band_rows(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    loader = str(rule["loader"])
    reference_source = str(rule.get("reference_source", "")).strip()
    group_field = str(rule.get("group_field", "cell_type"))
    property_metric_map = {str(key): str(value) for key, value in dict(rule.get("property_metric_map", {})).items()}
    property_band_modes = _property_band_modes(rule, property_metric_map)
    sigma_arg_name = str(rule.get("sigma_arg_name", "reference_sigma_multiplier"))
    sigma_multiplier = float(getattr(context.args, sigma_arg_name, rule.get("sigma_multiplier", 2.0)))
    sigma_phrase = _sigma_phrase(sigma_multiplier)
    default_lower_bound = _optional_float(rule.get("default_lower_bound"))
    default_upper_bound = _optional_float(rule.get("default_upper_bound"))
    rows = _filter_rows(_load_rows(loader), rule, args=context.args)
    cases: list[ReferenceBandCase] = []
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
        band_mode = property_band_modes[property_name]
        lower_bound = _optional_float(
            _property_override(rule, "property_lower_bounds", property_name, default_lower_bound)
        )
        upper_bound = _optional_float(
            _property_override(rule, "property_upper_bounds", property_name, default_upper_bound)
        )
        quantile_low = None
        quantile_high = None
        quantile_low_label = None
        quantile_high_label = None
        if band_mode == "quantile_interval":
            low_field = _row_field_name(
                rule,
                property_name,
                field_override_key="property_quantile_low_fields",
                default_field_key="default_quantile_low_field",
                default="q_low",
            )
            high_field = _row_field_name(
                rule,
                property_name,
                field_override_key="property_quantile_high_fields",
                default_field_key="default_quantile_high_field",
                default="q_high",
            )
            low_label_field = _row_field_name(
                rule,
                property_name,
                field_override_key="property_quantile_low_label_fields",
                default_field_key="default_quantile_low_label_field",
                default="q_low_label",
            )
            high_label_field = _row_field_name(
                rule,
                property_name,
                field_override_key="property_quantile_high_label_fields",
                default_field_key="default_quantile_high_label_field",
                default="q_high_label",
            )
            quantile_low = _optional_float(row.get(low_field))
            quantile_high = _optional_float(row.get(high_field))
            quantile_low_label = str(row.get(low_label_field, "")).strip() or low_field
            quantile_high_label = str(row.get(high_label_field, "")).strip() or high_field
        band = compute_reference_acceptance_band(
            reference_mean=reference_mean,
            reference_sd=reference_sd,
            sigma_multiplier=sigma_multiplier,
            band_mode=band_mode,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            quantile_low_label=quantile_low_label,
            quantile_high_label=quantile_high_label,
        )
        item_id = f"{group.lower()}_{metric_key.lower()}_within_uploaded_reference_band".replace(".", "_")
        unit_text = str(row.get("unit", "")).strip()
        range_text = f"between {rounded(band.low)} and {rounded(band.high)}"
        if unit_text:
            range_text = f"{range_text} {unit_text}"
        review_metadata = _property_review_metadata(rule, context, property_name)
        criterion_math = criterion_math_for_reference_band(group, property_name, band)
        cases.append(
            ReferenceBandCase(
                check_id=item_id,
                title=_title_text_for_band(group, property_name, band),
                criterion=_criterion_text_for_band(group, property_name, band, sigma_phrase),
                criterion_latex=criterion_math.latex,
                criterion_formulae=criterion_math.formulae,
                criterion_definitions=criterion_math.definitions,
                description=(
                    f"This is the direct single-cell-type reference check derived from uploaded literature rows for "
                    f"{property_name} rather than from a cross-group ordering heuristic."
                ),
                acceptable=(
                    f"The observed {group} mean must lie {range_text}, using the configured "
                    f"{band.standard_label}."
                ),
                acceptable_basis=(
                    f"Derived from the uploaded literature row for {property_name} using the configured "
                    f"{band.standard_label}: {band.description}. "
                    f"The sigma multiplier comes from '{sigma_arg_name}' when that standard needs one."
                ),
                note=str(_property_override(rule, "property_notes", property_name, "")),
                observation=ReferenceBandObservation(
                    property_name=property_name,
                    group=group,
                    metric_key=metric_key,
                    reference_mean=reference_mean,
                    reference_sd=reference_sd,
                    unit_text=unit_text,
                    policy=ReferenceBandPolicy(
                        mode=band_mode,
                        sigma_multiplier=sigma_multiplier,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        quantile_low=quantile_low,
                        quantile_high=quantile_high,
                        quantile_low_label=quantile_low_label,
                        quantile_high_label=quantile_high_label,
                    ),
                    provenance=ProvenanceRecord.from_row(row),
                    review=ValidationReview(
                        status=review_metadata["status"],
                        note=review_metadata["note"],
                        reviewer=review_metadata["reviewer"],
                        required_expertise=review_metadata["required_expertise"],
                        focus=review_metadata["focus"],
                    ),
                ),
                reference_annotation=_reference_annotation(row),
                pass_status=str(rule.get("pass_status", "PASS")),
                fail_status=str(rule.get("fail_status", "FAIL")),
            )
        )
    compiled = compile_reference_band_suite(
        cases=cases,
        summary=context.summary,
        suite_name=str(rule.get("suite_name", rule.get("title", "reference-band-suite"))),
    )
    return audit_items_from_reference_band_suite(compiled)


@register_validation_rule("note_presence")
def _note_presence(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    scope = str(rule.get("scope", "")).strip() or None
    row_contexts = list(rule.get("row_contexts", []))
    rows: list[dict[str, Any]] = []
    for row_context in row_contexts:
        context_rows = _filter_rows(_load_rows(str(row_context["loader"])), row_context, args=context.args)
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
    for synthetic in list(rule.get("synthetic_contexts", [])):
        rows.append(dict(synthetic))
    notes_path = _notes_path(rule, context)
    matched_notes = notes_for_rows(rows, scope=scope, notes=load_notes(notes_path) if notes_path else None)
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
    status_reason = ""
    if status == "WARN":
        status_reason = (
            "This item is a warning because relevant validation caveats matched the current rows and should remain visible in the report."
        )
    return [_rule_item(rule, status=status, evidence=evidence, status_reason=status_reason)]


@register_validation_rule("reference_curve_match")
def _reference_curve_match(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    loader = str(rule["loader"])
    protocol_evidence_key = str(rule.get("protocol_evidence_key", "fi_curve_rows"))
    reference_current_key = str(rule.get("reference_current_key", "current_pA"))
    reference_value_key = str(rule.get("reference_value_key", "firing_rate_Hz"))
    model_current_key = str(rule.get("model_current_key", "current_pA"))
    model_value_key = str(rule.get("model_value_key", "firing_rate_Hz"))
    max_mae = float(rule.get("maximum_mae", float("inf")))
    max_rmse = float(rule.get("maximum_rmse", float("inf")))
    min_points = int(rule.get("minimum_point_count", 1))

    reference_rows = _filter_rows(_load_rows(loader), rule, args=context.args)
    model_rows = list((getattr(context.protocol_result, "protocol_evidence", {}) or {}).get(protocol_evidence_key, []))
    precision_digits = int(rule.get("current_precision_digits", 6))
    reference_curve = _curve_points_by_current(
        reference_rows,
        current_key=reference_current_key,
        value_key=reference_value_key,
        precision_digits=precision_digits,
    )
    model_curve = _curve_points_by_current(
        model_rows,
        current_key=model_current_key,
        value_key=model_value_key,
        precision_digits=precision_digits,
    )
    shared_currents = sorted(set(reference_curve).intersection(model_curve))
    diffs = [abs(model_curve[current] - reference_curve[current]) for current in shared_currents]
    mae = float(np.mean(diffs)) if diffs else float("nan")
    rmse = float(np.sqrt(np.mean(np.square(diffs)))) if diffs else float("nan")
    max_abs = float(np.max(diffs)) if diffs else float("nan")
    passed = (
        len(shared_currents) >= min_points
        and _is_finite_number(mae)
        and mae <= max_mae
        and _is_finite_number(rmse)
        and rmse <= max_rmse
    )
    evidence = _rounded_dict(
        {
            "matched_point_count": len(shared_currents),
            "currents_pA": shared_currents,
            "reference_values_Hz": [reference_curve[current] for current in shared_currents],
            "model_values_Hz": [model_curve[current] for current in shared_currents],
            "mean_absolute_error_Hz": mae,
            "root_mean_square_error_Hz": rmse,
            "max_absolute_error_Hz": max_abs,
            "maximum_mae_Hz": max_mae,
            "maximum_rmse_Hz": max_rmse,
        }
    )
    return [
        _rule_item(
            rule,
            status=_rule_status(rule, passed),
            evidence=evidence,
            series_visuals=[
                series_visual_spec(
                    keys=["currents_pA", "reference_values_Hz", "model_values_Hz"],
                    style={
                        "line_width": 1.8,
                        "marker_size": 3.2,
                        "legend_loc": "lower center",
                    },
                )
            ],
        )
    ]


__all__ = [
    "REFERENCE_ROW_LOADERS",
    "RULE_HANDLERS",
    "ReferenceAcceptanceBand",
    "ValidationRuleContext",
    "build_rule_items",
    "compute_reference_acceptance_band",
    "register_validation_rule",
    "summarize_numeric_metrics",
]
