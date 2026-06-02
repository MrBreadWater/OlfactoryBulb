"""Generic rule engine for literature-backed validation audits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
import math

import numpy as np
from scipy.stats import beta as beta_distribution

from olfactorybulb.audit.core import AuditItem, rounded
from olfactorybulb.audit.reference_data import (
    GC_VALIDATION_NOTES_FILENAME,
    PV_CRH_EPL_FSI_EPHYS_FILENAME,
    PV_CRH_EPL_FSI_FI_CURVE_FILENAME,
    PV_CRH_EPL_FSI_IDENTITY_FILENAME,
    GC_EPHYS_FILENAME,
    GC_FI_CURVE_FILENAME,
    GC_SGC_DGC_EPHYS_FILENAME,
    GC_SGC_DGC_FI_CURVE_FILENAME,
    GC_IDENTITY_FILENAME,
    GC_MODULATION_FILENAME,
    GC_SYNAPTIC_LATENCY_FILENAME,
    REPO_ROOT,
    csv_rows,
    load_gc_ephys_rows,
    load_gc_fi_curve_rows,
    load_gc_identity_rows,
    load_gc_modulation_rows,
    load_gc_protocol_rows,
    load_gc_sgc_dgc_ephys_rows,
    load_gc_sgc_dgc_fi_curve_rows,
    load_gc_synaptic_latency_rows,
    load_normalized_legacy_mc_tc_rows,
    load_pv_crh_epl_fsi_ephys_rows,
    load_pv_crh_epl_fsi_fi_curve_rows,
    load_pv_crh_epl_fsi_identity_rows,
    load_pv_crh_epl_fsi_protocol_rows,
)
from olfactorybulb.audit.reference_notes import load_notes, notes_for_rows


@dataclass(frozen=True)
class ValidationRuleContext:
    metrics: list[dict[str, Any]]
    summary: dict[str, dict[str, float]]
    args: Any
    config: dict[str, Any]
    protocol_result: Any | None = None


@dataclass(frozen=True)
class ReferenceAcceptanceBand:
    low: float
    high: float
    mode: str
    standard_label: str
    raw_low: float
    raw_high: float
    lower_bound: float | None
    upper_bound: float | None
    description: str


RuleHandler = Callable[[dict[str, Any], ValidationRuleContext], list[AuditItem]]


RULE_HANDLERS: dict[str, RuleHandler] = {}


REFERENCE_ROW_LOADERS: dict[str, Callable[[], list[dict[str, Any]]]] = {
    "legacy_mc_tc_ephys": load_normalized_legacy_mc_tc_rows,
    "epl_fsi_ephys": load_pv_crh_epl_fsi_ephys_rows,
    "epl_fsi_fi_curve": load_pv_crh_epl_fsi_fi_curve_rows,
    "epl_fsi_identity": load_pv_crh_epl_fsi_identity_rows,
    "pv_crh_epl_fsi_protocols": load_pv_crh_epl_fsi_protocol_rows,
    "gc_ephys": load_gc_ephys_rows,
    "gc_fi_curve": load_gc_fi_curve_rows,
    "gc_subtype_ephys": load_gc_sgc_dgc_ephys_rows,
    "gc_subtype_fi_curve": load_gc_sgc_dgc_fi_curve_rows,
    "gc_identity": load_gc_identity_rows,
    "gc_modulation": load_gc_modulation_rows,
    "gc_protocols": load_gc_protocol_rows,
    "gc_synaptic_latency": load_gc_synaptic_latency_rows,
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
        if not _rule_enabled(rule, context.args):
            continue
        kind = str(rule.get("kind") or "").strip()
        if not kind:
            raise ValueError("Validation rule is missing required 'kind'")
        try:
            handler = RULE_HANDLERS[kind]
        except KeyError as exc:
            known = ", ".join(sorted(RULE_HANDLERS))
            raise KeyError(f"Unknown validation rule kind {kind!r}. Known rule kinds: {known}") from exc
        rule_items = handler(rule, context)
        _apply_rule_level_human_review(rule_items, rule, context)
        items.extend(rule_items)
    return items


def _config_human_review_defaults(context: ValidationRuleContext) -> dict[str, Any]:
    defaults = context.config.get("human_review", {})
    return dict(defaults) if isinstance(defaults, dict) else {}


def _resolved_rule_human_review(
    rule: dict[str, Any],
    context: ValidationRuleContext,
) -> dict[str, str]:
    defaults = _config_human_review_defaults(context)
    return {
        "status": str(rule.get("human_review_status", defaults.get("default_status", ""))).strip(),
        "note": str(rule.get("human_review_note", defaults.get("default_note", ""))).strip(),
        "reviewer": str(rule.get("human_review_reviewer", defaults.get("default_reviewer", ""))).strip(),
    }


def _apply_rule_level_human_review(
    items: list[AuditItem],
    rule: dict[str, Any],
    context: ValidationRuleContext,
) -> None:
    metadata = _resolved_rule_human_review(rule, context)
    for item in items:
        if not item.human_review_status and metadata["status"]:
            item.human_review_status = metadata["status"]
        if not item.human_review_note and metadata["note"]:
            item.human_review_note = metadata["note"]
        if not item.human_review_reviewer and metadata["reviewer"]:
            item.human_review_reviewer = metadata["reviewer"]


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


def _sigma_phrase(sigma_multiplier: float) -> str:
    if np.isclose(float(sigma_multiplier), 2.0):
        return "two standard deviations"
    if np.isclose(float(sigma_multiplier), 1.0):
        return "one standard deviation"
    return f"{rounded(float(sigma_multiplier), 3)} standard deviations"


def _central_mass_from_sigma(sigma_multiplier: float) -> float:
    sigma = abs(float(sigma_multiplier))
    return float(math.erf(sigma / math.sqrt(2.0)))


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def compute_reference_acceptance_band(
    *,
    reference_mean: float,
    reference_sd: float,
    sigma_multiplier: float,
    band_mode: str = "symmetric_sd",
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    quantile_low: float | None = None,
    quantile_high: float | None = None,
    quantile_low_label: str | None = None,
    quantile_high_label: str | None = None,
) -> ReferenceAcceptanceBand:
    mode = str(band_mode or "symmetric_sd").strip()
    sigma_phrase = _sigma_phrase(sigma_multiplier)
    if mode == "symmetric_sd":
        raw_low = float(reference_mean - reference_sd * sigma_multiplier)
        raw_high = float(reference_mean + reference_sd * sigma_multiplier)
        standard_label = "symmetric standard-deviation band"
        description = (
            f"the uploaded arithmetic mean plus or minus {sigma_phrase}"
        )
    elif mode == "lognormal_sd":
        if reference_mean <= 0.0:
            raise ValueError(
                "lognormal_sd acceptance bands require a strictly positive reference mean"
            )
        if reference_sd < 0.0:
            raise ValueError(
                "lognormal_sd acceptance bands require a non-negative reference standard deviation"
            )
        variance_ratio = (float(reference_sd) / float(reference_mean)) ** 2
        sigma_log = math.sqrt(math.log1p(variance_ratio))
        mu_log = math.log(float(reference_mean)) - 0.5 * sigma_log**2
        raw_low = float(math.exp(mu_log - float(sigma_multiplier) * sigma_log))
        raw_high = float(math.exp(mu_log + float(sigma_multiplier) * sigma_log))
        standard_label = "lognormal-reconstructed dispersion band"
        description = (
            f"a log-space band reconstructed from the uploaded arithmetic mean and standard deviation "
            f"assuming a lognormal distribution, then exponentiated back to the original units over {sigma_phrase}"
        )
    elif mode == "beta_sd":
        if reference_mean < 0.0 or reference_mean > 1.0:
            raise ValueError(
                "beta_sd acceptance bands require a reference mean between 0 and 1"
            )
        if reference_sd < 0.0:
            raise ValueError(
                "beta_sd acceptance bands require a non-negative reference standard deviation"
            )
        variance = float(reference_sd) ** 2
        if variance == 0.0:
            raw_low = float(reference_mean)
            raw_high = float(reference_mean)
            coverage_fraction = _central_mass_from_sigma(sigma_multiplier)
            standard_label = "beta-reconstructed bounded probability band"
            description = (
                f"an exact point interval at the uploaded mean because the reported standard deviation is zero; "
                f"the nominal central-mass target implied by {sigma_phrase} would have been {rounded(coverage_fraction * 100.0)} percent"
            )
        else:
            maximum_variance = float(reference_mean) * (1.0 - float(reference_mean))
            if variance >= maximum_variance:
                raise ValueError(
                    "beta_sd acceptance bands require variance smaller than mean*(1-mean) "
                    "to reconstruct valid beta-distribution parameters"
                )
            concentration = maximum_variance / variance - 1.0
            alpha = float(reference_mean) * concentration
            beta_param = (1.0 - float(reference_mean)) * concentration
            coverage_fraction = _central_mass_from_sigma(sigma_multiplier)
            tail_probability = (1.0 - coverage_fraction) / 2.0
            raw_low = float(beta_distribution.ppf(tail_probability, alpha, beta_param))
            raw_high = float(beta_distribution.ppf(1.0 - tail_probability, alpha, beta_param))
            standard_label = "beta-reconstructed bounded probability band"
            description = (
                f"a beta-distribution central interval reconstructed from the uploaded arithmetic mean and standard deviation, "
                f"with central mass matched to the normal-space coverage implied by {sigma_phrase} "
                f"({rounded(coverage_fraction * 100.0)} percent)"
            )
    elif mode == "quantile_interval":
        if quantile_low is None or quantile_high is None:
            raise ValueError(
                "quantile_interval acceptance bands require explicit quantile_low and quantile_high values"
            )
        low_label = str(quantile_low_label or "reported lower quantile").strip()
        high_label = str(quantile_high_label or "reported upper quantile").strip()
        raw_low = float(quantile_low)
        raw_high = float(quantile_high)
        standard_label = "reported quantile interval"
        description = (
            f"the explicitly reported quantile interval from {low_label} to {high_label}"
        )
    elif mode == "binary_indicator":
        if not np.isclose(reference_mean, round(reference_mean)) or int(round(reference_mean)) not in (0, 1):
            raise ValueError(
                "binary_indicator acceptance bands require the reference mean to encode an exact binary indicator of 0 or 1"
            )
        raw_low = float(int(round(reference_mean)))
        raw_high = float(int(round(reference_mean)))
        standard_label = "exact binary indicator"
        description = (
            "the exact binary indicator encoded by the uploaded reference row; the reported standard deviation is ignored "
            "because this metric is categorical rather than a continuous dispersion measure"
        )
    else:
        raise ValueError(
            f"Unknown reference-band mode {mode!r}. Supported modes: "
            "'symmetric_sd', 'lognormal_sd', 'beta_sd', 'quantile_interval', 'binary_indicator'."
        )

    low = raw_low
    high = raw_high
    bound_notes: list[str] = []
    if lower_bound is not None:
        if low < float(lower_bound):
            low = float(lower_bound)
            bound_notes.append(f"lower-bounded at {rounded(float(lower_bound))}")
    if upper_bound is not None:
        if high > float(upper_bound):
            high = float(upper_bound)
            bound_notes.append(f"upper-bounded at {rounded(float(upper_bound))}")
    if high < low:
        raise ValueError(
            f"Reference acceptance band bounds are inconsistent after clipping: low={low}, high={high}"
        )
    if bound_notes:
        description = f"{description}, then {' and '.join(bound_notes)}"
    return ReferenceAcceptanceBand(
        low=float(low),
        high=float(high),
        mode=mode,
        standard_label=standard_label,
        raw_low=float(raw_low),
        raw_high=float(raw_high),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        description=description,
    )


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
    defaults = _resolved_rule_human_review(rule, context)
    return {
        "status": str(
            _property_override(rule, "property_human_review_statuses", property_name, defaults["status"])
        ).strip(),
        "note": str(
            _property_override(rule, "property_human_review_notes", property_name, defaults["note"])
        ).strip(),
        "reviewer": str(
            _property_override(rule, "property_human_review_reviewers", property_name, defaults["reviewer"])
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
    return [
        _rule_item(
            rule,
            status=_rule_status(rule, bool(context.metrics)),
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
    return [_rule_item(rule, status=_rule_status(rule, not failing), evidence=evidence)]


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
    return [_rule_item(rule, status=_rule_status(rule, not failing), evidence=evidence)]


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
    return [_rule_item(rule, status=_rule_status(rule, passed), evidence=evidence)]


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
    return [_rule_item(rule, status=_rule_status(rule, difference <= max_difference), evidence=evidence)]


@register_validation_rule("group_positive")
def _group_positive(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    metric_key = str(rule["metric_key"])
    groups = [str(group) for group in rule.get("groups", [])]
    if not groups:
        raise ValueError("group_positive rule requires non-empty 'groups'")
    evidence = _rounded_dict({f"{group}_mean": _group_mean(context.summary, group, metric_key) for group in groups})
    passed = all(_is_finite_number(_group_mean(context.summary, group, metric_key)) and _group_mean(context.summary, group, metric_key) > 0.0 for group in groups)
    return [_rule_item(rule, status=_rule_status(rule, passed), evidence=evidence)]


@register_validation_rule("summary_metric_min")
def _summary_metric_min(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    metric_key = str(rule["metric_key"])
    minimum = float(rule["minimum"])
    group = _summary_group(rule, context)
    observed = _group_mean(context.summary, group, metric_key)
    passed = _is_finite_number(observed) and observed >= minimum
    evidence = _summary_evidence(rule, context, group=group, base={"group": group, "observed": observed, "minimum": minimum})
    return [_rule_item(rule, status=_rule_status(rule, passed), evidence=evidence)]


@register_validation_rule("summary_metric_max")
def _summary_metric_max(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    metric_key = str(rule["metric_key"])
    maximum = float(rule["maximum"])
    group = _summary_group(rule, context)
    observed = _group_mean(context.summary, group, metric_key)
    passed = _is_finite_number(observed) and observed <= maximum
    evidence = _summary_evidence(rule, context, group=group, base={"group": group, "observed": observed, "maximum": maximum})
    return [_rule_item(rule, status=_rule_status(rule, passed), evidence=evidence)]


@register_validation_rule("summary_metric_range")
def _summary_metric_range(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    metric_key = str(rule["metric_key"])
    minimum = float(rule.get("minimum", float("-inf")))
    maximum = float(rule.get("maximum", float("inf")))
    group = _summary_group(rule, context)
    observed = _group_mean(context.summary, group, metric_key)
    passed = _is_finite_number(observed) and minimum <= observed <= maximum
    evidence = _summary_evidence(
        rule,
        context,
        group=group,
        base={"group": group, "observed": observed, "minimum": minimum, "maximum": maximum},
    )
    return [_rule_item(rule, status=_rule_status(rule, passed), evidence=evidence)]


@register_validation_rule("summary_metric_status_map")
def _summary_metric_status_map(rule: dict[str, Any], context: ValidationRuleContext) -> list[AuditItem]:
    metric_key = str(rule["metric_key"])
    group = _summary_group(rule, context)
    observed = _group_mean(context.summary, group, metric_key)
    pass_values = {float(value) for value in rule.get("pass_values", [])}
    warn_values = {float(value) for value in rule.get("warn_values", [])}
    fail_values = {float(value) for value in rule.get("fail_values", [])}
    observed_value = float(observed) if _is_finite_number(observed) else float("nan")
    if observed_value in pass_values:
        status = "PASS"
    elif observed_value in warn_values:
        status = "WARN"
    elif observed_value in fail_values or not _is_finite_number(observed_value):
        status = "FAIL"
    else:
        status = str(rule.get("default_status", "FAIL"))
    evidence = _summary_evidence(
        rule,
        context,
        group=group,
        base={
            "group": group,
            "observed": observed,
            "pass_values": sorted(pass_values),
            "warn_values": sorted(warn_values),
            "fail_values": sorted(fail_values),
        },
    )
    return [_rule_item(rule, status=status, evidence=evidence)]


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
        accepted_low = band.low
        accepted_high = band.high
        passed = _is_finite_number(observed_value) and accepted_low <= observed_value <= accepted_high
        item_id = f"{group.lower()}_{metric_key.lower()}_within_uploaded_reference_band".replace(".", "_")
        evidence_key = f"{group}_mean"
        evidence_payload: dict[str, Any] = {
            evidence_key: observed_value,
            "accepted_low": accepted_low,
            "accepted_high": accepted_high,
            "accepted_sigma_multiplier": sigma_multiplier,
            "accepted_interval_mode": band.mode,
            "accepted_interval_standard": band.standard_label,
            "__reference_annotations__": {evidence_key: _reference_annotation(row)},
        }
        if lower_bound is not None:
            evidence_payload["accepted_lower_bound"] = lower_bound
        if upper_bound is not None:
            evidence_payload["accepted_upper_bound"] = upper_bound
        if not np.isclose(band.raw_low, band.low):
            evidence_payload["unbounded_low"] = band.raw_low
        if not np.isclose(band.raw_high, band.high):
            evidence_payload["unbounded_high"] = band.raw_high
        evidence = _rounded_dict(evidence_payload)
        unit_text = str(row.get("unit", "")).strip()
        range_text = f"between {rounded(accepted_low)} and {rounded(accepted_high)}"
        if unit_text:
            range_text = f"{range_text} {unit_text}"
        review_metadata = _property_review_metadata(rule, context, property_name)
        items.append(
            _rule_item(
                rule,
                check_id=item_id,
                status=_rule_status(rule, passed),
                title=_title_text_for_band(group, property_name, band),
                criterion=_criterion_text_for_band(group, property_name, band, sigma_phrase),
                description=(
                    f"This is the direct single-cell-type reference check derived from uploaded literature rows for "
                    f"{property_name} rather than from a cross-group ordering heuristic."
                ),
                acceptable=(
                    f"The observed {group} mean must lie {range_text}, using the configured "
                    f"{band.standard_label}."
                ),
                acceptable_basis=(
                    f"The accepted interval is computed from the uploaded literature row for {property_name} as "
                    f"{band.description}. The acceptance standard used here is {band.standard_label}. "
                    f"The sigma multiplier comes from the configurable '{sigma_arg_name}' setting when that standard needs one. "
                    f"This is a dispersion band, not a formal confidence interval."
                ),
                evidence=evidence,
                note=str(_property_override(rule, "property_notes", property_name, "")),
            )
        )
        items[-1].human_review_status = review_metadata["status"]
        items[-1].human_review_note = review_metadata["note"]
        items[-1].human_review_reviewer = review_metadata["reviewer"]
    return items


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
    return [_rule_item(rule, status=status, evidence=evidence)]


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
    return [_rule_item(rule, status=_rule_status(rule, passed), evidence=evidence)]


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
