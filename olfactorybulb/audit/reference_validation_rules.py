"""Generic rule engine for literature-backed validation audits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
import math

import numpy as np

from olfactorybulb.audit import AuditItem, series_visual_spec
from olfactorybulb.audit.protocol_evidence import protocol_series_spec_map
from olfactorybulb.audit.reference_data import (
    REPO_ROOT,
    csv_rows,
    load_dataset_output_rows,
    load_normalized_legacy_mc_tc_rows,
)
from olfactorybulb.audit.reference_notes import load_notes, notes_for_rows
from olfactorybulb.audit.reference_validation_specs import (
    ComparisonRuleSpec,
    NotePresenceRuleSpec,
    ProtocolExecutedRuleSpec,
    ReferenceBandRuleSpec,
    SeriesComparisonRuleSpec,
    SummaryRuleSpec,
    grouped_suite_descriptor as _grouped_suite_descriptor,
)
from olfactorybulb.neuronunit.reference_bands import (
    ReferenceAcceptanceBand,
    compute_reference_acceptance_band,
)
from olfactorybulb.neuronunit.reference_validation_suite import (
    audit_items_from_reference_band_suite,
    compile_reference_band_suite,
)
from olfactorybulb.neuronunit.comparison_validation_suite import (
    audit_items_from_comparison_rule_suite,
    compile_comparison_rule_suite,
)
from olfactorybulb.neuronunit.series_validation_suite import (
    audit_items_from_series_comparison_suite,
    compile_series_comparison_suite,
)
from olfactorybulb.neuronunit.summary_validation_suite import (
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
SERIES_RULE_KINDS = {
    "reference_curve_match",
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
    pending_series_rules: list[dict[str, Any]] = []

    def _append_grouped_suite_items(
        generated_items: list[AuditItem],
        source_rules: list[dict[str, Any]],
    ) -> None:
        if not generated_items:
            return
        detail_items = list(generated_items)
        if bool(detail_items[0].summary_rollup_exempt) and str(detail_items[0].detail_level or "detail") != "detail":
            items.append(detail_items[0])
            detail_items = detail_items[1:]
        for rule, item in zip(source_rules, detail_items, strict=False):
            _apply_rule_level_validation_design_review([item], rule, context)
            items.append(item)
        if len(detail_items) > len(source_rules):
            items.extend(detail_items[len(source_rules):])

    def flush_pending_summary_rules() -> None:
        nonlocal pending_summary_rules
        if not pending_summary_rules:
            return
        generated_items = _build_summary_rule_items(pending_summary_rules, context)
        _append_grouped_suite_items(generated_items, pending_summary_rules)
        pending_summary_rules = []

    def flush_pending_comparison_rules() -> None:
        nonlocal pending_comparison_rules
        if not pending_comparison_rules:
            return
        generated_items = _build_comparison_rule_items(pending_comparison_rules, context)
        _append_grouped_suite_items(generated_items, pending_comparison_rules)
        pending_comparison_rules = []

    def flush_pending_series_rules() -> None:
        nonlocal pending_series_rules
        if not pending_series_rules:
            return
        generated_items = _build_series_rule_items(pending_series_rules, context)
        _append_grouped_suite_items(generated_items, pending_series_rules)
        pending_series_rules = []

    for rule in rules:
        if not _rule_enabled(rule, context.args):
            continue
        kind = str(rule.get("kind") or "").strip()
        if not kind:
            raise ValueError("Validation rule is missing required 'kind'")
        if kind in SUMMARY_RULE_KINDS:
            flush_pending_series_rules()
            flush_pending_comparison_rules()
            pending_summary_rules.append(rule)
            continue
        if kind in COMPARISON_RULE_KINDS:
            flush_pending_series_rules()
            flush_pending_summary_rules()
            pending_comparison_rules.append(rule)
            continue
        if kind in SERIES_RULE_KINDS:
            flush_pending_summary_rules()
            flush_pending_comparison_rules()
            pending_series_rules.append(rule)
            continue
        flush_pending_summary_rules()
        flush_pending_comparison_rules()
        flush_pending_series_rules()
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
    flush_pending_series_rules()
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


def _group_mean(summary: dict[str, dict[str, float]], group: str, metric_key: str) -> float:
    return float(summary.get(group, {}).get(metric_key, float("nan")))


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


def _build_summary_rule_items(
    rules: list[dict[str, Any]],
    context: ValidationRuleContext,
) -> list[AuditItem]:
    if not rules:
        return []
    cases = [SummaryRuleSpec.from_rule(rule, context).to_case() for rule in rules]
    descriptor = _grouped_suite_descriptor(
        rules,
        default_suite_id=f"{str(context.config.get('validation_id', 'validation')).strip() or 'validation'}.summary_rules",
        suite_kind_label="Summary-rule suite",
    )
    compiled = compile_summary_rule_suite(cases=cases, summary=context.summary, suite_name=descriptor.suite_id)
    return audit_items_from_summary_rule_suite(compiled, descriptor=descriptor)


def _build_comparison_rule_items(
    rules: list[dict[str, Any]],
    context: ValidationRuleContext,
) -> list[AuditItem]:
    if not rules:
        return []
    cases = [ComparisonRuleSpec.from_rule(rule).to_case() for rule in rules]
    descriptor = _grouped_suite_descriptor(
        rules,
        default_suite_id=f"{str(context.config.get('validation_id', 'validation')).strip() or 'validation'}.comparison_rules",
        suite_kind_label="Comparison-rule suite",
    )
    compiled = compile_comparison_rule_suite(
        cases=cases,
        summary=context.summary,
        metrics=context.metrics,
        suite_name=descriptor.suite_id,
    )
    return audit_items_from_comparison_rule_suite(compiled, descriptor=descriptor)


def _build_series_rule_items(
    rules: list[dict[str, Any]],
    context: ValidationRuleContext,
) -> list[AuditItem]:
    if not rules:
        return []
    protocol_evidence = dict(getattr(context.protocol_result, "protocol_evidence", {}) or {})
    evidence_series_specs = protocol_series_spec_map(
        getattr(context.protocol_result, "evidence_series_specs", ()) or ()
    )
    cases = []
    for rule in rules:
        loader = str(rule["loader"])
        reference_rows = _filter_rows(_load_rows(loader), rule, args=context.args)
        protocol_series_spec = None
        if evidence_series_specs:
            protocol_evidence_key = str(rule.get("protocol_evidence_key", "fi_curve_rows")).strip()
            protocol_series_spec = evidence_series_specs.get(protocol_evidence_key)
        cases.append(
            SeriesComparisonRuleSpec.from_rule(
                rule,
                protocol_series_spec=protocol_series_spec,
            ).to_case(reference_rows=reference_rows)
        )
    raw_candidate_ids = protocol_evidence.get("cell_models", [])
    if not isinstance(raw_candidate_ids, list):
        raw_candidate_ids = []
    descriptor = _grouped_suite_descriptor(
        rules,
        default_suite_id=str(context.config.get("validation_id", "validation")).strip() or "validation",
        suite_kind_label="Series-comparison suite",
        candidate_ids=[str(candidate_id) for candidate_id in raw_candidate_ids if str(candidate_id).strip()],
    )
    compiled = compile_series_comparison_suite(
        cases=cases,
        summary=context.summary,
        metrics=context.metrics,
        protocol_evidence=protocol_evidence,
        suite_name=descriptor.suite_id,
    )
    return audit_items_from_series_comparison_suite(compiled, descriptor=descriptor)


def _notes_path(rule: dict[str, Any], context: ValidationRuleContext) -> Path | None:
    path_text = str(rule.get("notes_path") or context.config.get("notes_path") or "").strip()
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


@register_validation_rule("protocol_executed")
def _protocol_executed(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    parsed_spec = ProtocolExecutedRuleSpec.from_rule(rule)
    protocol_evidence = dict(getattr(context.protocol_result, "protocol_evidence", {}) or {})
    series_visuals: list[dict[str, Any]] = []
    evidence_series_specs = tuple(getattr(context.protocol_result, "evidence_series_specs", ()) or ())
    for evidence_spec in evidence_series_specs:
        series_visuals.append(evidence_spec.to_visual_spec())
    if not series_visuals:
        fallback_rows = protocol_evidence.get(parsed_spec.fallback_series_key)
        if isinstance(fallback_rows, list) and fallback_rows:
            series_visuals.append(
                series_visual_spec(
                    keys=[parsed_spec.fallback_series_key],
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
    spec = ReferenceBandRuleSpec.from_rule(rule, context)
    rows = _filter_rows(_load_rows(spec.loader), rule, args=context.args)
    cases = spec.build_cases(rows=rows)
    compiled = compile_reference_band_suite(
        cases=cases,
        summary=context.summary,
        suite_name=spec.suite_descriptor.suite_id,
    )
    return audit_items_from_reference_band_suite(compiled, descriptor=spec.suite_descriptor)


@register_validation_rule("note_presence")
def _note_presence(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    spec = NotePresenceRuleSpec.from_rule(rule)
    rows: list[dict[str, Any]] = []
    for row_context in spec.row_contexts:
        context_rows = _filter_rows(_load_rows(row_context.loader), row_context.to_filter_spec(), args=context.args)
        if row_context.as_protocol_context:
            property_name = row_context.property_name
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
    for synthetic in spec.synthetic_contexts:
        rows.append(dict(synthetic))
    notes_path = _notes_path(rule, context)
    matched_notes = notes_for_rows(rows, scope=spec.scope, notes=load_notes(notes_path) if notes_path else None)
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
    return _build_series_rule_items([rule], context)


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
