"""Typed declarative config parsers for maintained reference-validation rules."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
import math
from typing import Any, Mapping

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
from olfactorybulb.audit.protocol_evidence import ProtocolEvidenceSeriesSpec
from olfactorybulb.audit.reference_rows import (
    ReferenceRowRecord,
    ReferenceRowTable,
    coerce_reference_row_table,
)
from olfactorybulb.audit.reference_validation_contracts import ValidationRuleContextLike
from olfactorybulb.neuronunit.reference_bands import sigma_phrase as _sigma_phrase
from olfactorybulb.neuronunit.reference_bands import (
    ProvenanceRecord,
    ReferenceAcceptanceBand,
    ReferenceBandObservation,
    ReferenceBandPolicy,
    ValidationReview,
    compute_reference_acceptance_band,
)
from olfactorybulb.neuronunit.comparison_validation_suite import ComparisonRuleCase
from olfactorybulb.neuronunit.metric_quantities import (
    MetricQuantitySpec,
    metric_definition_text,
    resolve_metric_quantity,
)
from olfactorybulb.neuronunit.reference_validation_suite import ReferenceBandCase
from olfactorybulb.neuronunit.scalar_observations import ScalarStatusMapPolicy
from olfactorybulb.neuronunit.series_validation_suite import (
    AxisTransform,
    SERIES_ALIGNMENT_POLICIES,
    SERIES_RESAMPLING_DOMAIN_POLICIES,
    SeriesComparisonCase,
    SeriesComparisonPolicy,
    SeriesDataSpec,
    SeriesDistributionObservation,
    SeriesVisualContract,
)
from olfactorybulb.neuronunit.summary_validation_suite import SummaryRuleCase
from olfactorybulb.neuronunit.suite_scores import (
    DEFAULT_SUITE_AGGREGATE_POLICY,
    DEFAULT_SUITE_STATISTICAL_POLICY,
    SuiteAggregatePolicy,
    SuiteDescriptor,
    SuiteStatisticalPolicy,
)


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def property_override(
    rule: Mapping[str, object],
    mapping_key: str,
    property_name: str,
    default: Any,
) -> Any:
    mapping = rule.get(mapping_key, {})
    if not isinstance(mapping, MappingABC):
        return default
    for key, value in mapping.items():
        if str(key).strip() == property_name:
            return value
    return default


def property_review_metadata(
    rule: Mapping[str, object],
    context: ValidationRuleContextLike,
    property_name: str,
) -> dict[str, str]:
    defaults = {
        "status": str(rule.get("validation_design_review_status", context.design_review_defaults.status)).strip(),
        "note": str(rule.get("validation_design_review_note", context.design_review_defaults.note)).strip(),
        "reviewer": str(rule.get("validation_design_review_reviewer", context.design_review_defaults.reviewer)).strip(),
        "required_expertise": str(
            rule.get(
                "validation_design_review_required_expertise",
                context.design_review_defaults.required_expertise,
            )
        ).strip(),
        "focus": str(
            rule.get(
                "validation_design_review_focus",
                context.design_review_defaults.focus,
            )
        ).strip(),
    }
    return {
        "status": str(
            property_override(rule, "property_validation_design_review_statuses", property_name, defaults["status"])
        ).strip(),
        "note": str(
            property_override(rule, "property_validation_design_review_notes", property_name, defaults["note"])
        ).strip(),
        "reviewer": str(
            property_override(rule, "property_validation_design_review_reviewers", property_name, defaults["reviewer"])
        ).strip(),
        "required_expertise": str(
            property_override(
                rule,
                "property_validation_design_review_required_expertise",
                property_name,
                defaults["required_expertise"],
            )
        ).strip(),
        "focus": str(
            property_override(
                rule,
                "property_validation_design_review_focuses",
                property_name,
                defaults["focus"],
            )
        ).strip(),
    }


def property_band_modes(rule: Mapping[str, object], property_metric_map: dict[str, str]) -> dict[str, str]:
    if "default_band_mode" in rule:
        raise ValueError(
            "reference_band_rows no longer supports 'default_band_mode'; "
            "choose 'property_band_modes' explicitly for every property"
        )
    raw_modes = rule.get("property_band_modes", {})
    if not isinstance(raw_modes, MappingABC):
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


def row_field_name(
    rule: Mapping[str, object],
    property_name: str,
    *,
    field_override_key: str,
    default_field_key: str,
    default: str,
) -> str:
    override = property_override(rule, field_override_key, property_name, None)
    if override is not None:
        return str(override).strip()
    return str(rule.get(default_field_key, default) or default).strip()


def summary_group(rule: Mapping[str, object], context: ValidationRuleContextLike) -> str:
    explicit = str(rule.get("group", "") or "").strip()
    if explicit:
        return explicit
    if len(context.summary) == 1:
        return next(iter(context.summary))
    return str(context.default_group or "ungrouped")


def suite_aggregate_policy_from_rule(rule: Mapping[str, object]) -> SuiteAggregatePolicy:
    raw_policy = rule.get("suite_aggregate_policy")
    if raw_policy in (None, ""):
        return DEFAULT_SUITE_AGGREGATE_POLICY
    if not isinstance(raw_policy, MappingABC):
        raise ValueError("suite_aggregate_policy must be a table when provided")
    status_rollup = str(raw_policy.get("status_rollup", DEFAULT_SUITE_AGGREGATE_POLICY.status_rollup)).strip()
    norm_rollup = str(raw_policy.get("norm_rollup", DEFAULT_SUITE_AGGREGATE_POLICY.norm_rollup)).strip()
    return SuiteAggregatePolicy(
        status_rollup=status_rollup or DEFAULT_SUITE_AGGREGATE_POLICY.status_rollup,
        norm_rollup=norm_rollup or DEFAULT_SUITE_AGGREGATE_POLICY.norm_rollup,
    )


def _explicit_suite_aggregate_policy_from_rule(rule: Mapping[str, object]) -> SuiteAggregatePolicy | None:
    raw_policy = rule.get("suite_aggregate_policy")
    if raw_policy in (None, ""):
        return None
    return suite_aggregate_policy_from_rule(rule)


def suite_statistical_policy_from_rule(rule: Mapping[str, object]) -> SuiteStatisticalPolicy:
    raw_policy = rule.get("suite_statistical_policy")
    if raw_policy in (None, ""):
        return DEFAULT_SUITE_STATISTICAL_POLICY
    if not isinstance(raw_policy, MappingABC):
        raise ValueError("suite_statistical_policy must be a table when provided")
    rollup_method = str(raw_policy.get("rollup_method", DEFAULT_SUITE_STATISTICAL_POLICY.rollup_method)).strip()
    minimum_available_case_count = raw_policy.get("minimum_available_case_count")
    minimum_available_case_fraction = raw_policy.get("minimum_available_case_fraction")
    return SuiteStatisticalPolicy(
        rollup_method=rollup_method or DEFAULT_SUITE_STATISTICAL_POLICY.rollup_method,
        minimum_available_case_count=(
            int(minimum_available_case_count)
            if minimum_available_case_count not in (None, "")
            else DEFAULT_SUITE_STATISTICAL_POLICY.minimum_available_case_count
        ),
        minimum_available_case_fraction=(
            float(minimum_available_case_fraction)
            if minimum_available_case_fraction not in (None, "")
            else DEFAULT_SUITE_STATISTICAL_POLICY.minimum_available_case_fraction
        ),
    )


def _explicit_suite_statistical_policy_from_rule(rule: Mapping[str, object]) -> SuiteStatisticalPolicy | None:
    raw_policy = rule.get("suite_statistical_policy")
    if raw_policy in (None, ""):
        return None
    return suite_statistical_policy_from_rule(rule)


def grouped_suite_descriptor(
    rules: list[Mapping[str, object]],
    *,
    default_suite_id: str,
    suite_kind_label: str,
    candidate_ids: list[str] | tuple[str, ...] = (),
) -> SuiteDescriptor:
    if not rules:
        raise ValueError("grouped_suite_descriptor requires at least one rule")
    explicit_suite_names = {
        str(rule.get("suite_name", "")).strip()
        for rule in rules
        if str(rule.get("suite_name", "")).strip()
    }
    if len(explicit_suite_names) > 1:
        raise ValueError(
            "Grouped SciUnit-backed rules require a consistent suite_name when compiled together; "
            f"got {sorted(explicit_suite_names)!r}"
        )
    explicit_policies = {
        policy
        for rule in rules
        for policy in [_explicit_suite_aggregate_policy_from_rule(rule)]
        if policy is not None
    }
    if len(explicit_policies) > 1:
        raise ValueError(
            "Grouped SciUnit-backed rules require a consistent suite_aggregate_policy when compiled together"
        )
    explicit_statistical_policies = {
        policy
        for rule in rules
        for policy in [_explicit_suite_statistical_policy_from_rule(rule)]
        if policy is not None
    }
    if len(explicit_statistical_policies) > 1:
        raise ValueError(
            "Grouped SciUnit-backed rules require a consistent suite_statistical_policy when compiled together"
        )
    suite_id = next(iter(explicit_suite_names), default_suite_id)
    aggregate_policy = next(iter(explicit_policies), DEFAULT_SUITE_AGGREGATE_POLICY)
    statistical_policy = next(iter(explicit_statistical_policies), DEFAULT_SUITE_STATISTICAL_POLICY)
    return SuiteDescriptor(
        suite_id=suite_id,
        suite_kind_label=suite_kind_label,
        candidate_ids=tuple(candidate_ids),
        aggregate_policy=aggregate_policy,
        statistical_policy=statistical_policy,
    )


def _group_mean(summary: dict[str, dict[str, float]], group: str, metric_key: str) -> float:
    return float(summary.get(group, {}).get(metric_key, float("nan")))


def _criterion_text_for_band(
    group: str,
    property_name: str,
    band: ReferenceAcceptanceBand,
    sigma_phrase: str,
) -> str:
    if band.mode == "quantile_interval":
        return f"The {group} mean {property_name.lower()} should remain within the uploaded reported quantile interval."
    if band.mode == "beta_sd":
        return (
            f"The {group} mean {property_name.lower()} should remain within the uploaded beta-reconstructed bounded "
            "probability interval."
        )
    if band.mode == "binary_indicator":
        return f"The {group} mean {property_name.lower()} should match the uploaded binary reference indicator exactly."
    if band.mode == "lognormal_sd":
        return (
            f"The {group} mean {property_name.lower()} should remain within {sigma_phrase} of the uploaded reference "
            "value under a lognormal reconstruction."
        )
    return f"The {group} mean {property_name.lower()} should remain within {sigma_phrase} of the uploaded reference value."


def _title_text_for_band(group: str, property_name: str, band: ReferenceAcceptanceBand) -> str:
    if band.mode == "binary_indicator":
        return f"{group} {property_name.lower()} matches the uploaded binary reference indicator"
    return f"{group} {property_name.lower()} stays within the uploaded reference band"


def _reference_annotation(row: Mapping[str, Any]) -> str:
    mean = row.get("mean")
    sd = row.get("sd")
    units = str(row.get("unit", "")).strip()
    source = str(row.get("Source", "") or row.get("source", "")).strip()
    n_value = row.get("n")
    return f"reference: {rounded(float(mean))} +/- {rounded(float(sd))} {units} from {source} (n={n_value})"


@dataclass(frozen=True)
class SummaryRuleSpec:
    rule_kind: str
    check_id: str
    title: str
    criterion: str
    criterion_latex: str
    criterion_formulae: list[str]
    criterion_definitions: list[dict[str, Any]]
    description: str
    acceptable: str
    acceptable_basis: str
    note: str
    metric_key: str
    metric_quantity: MetricQuantitySpec
    group: str
    evidence_metric_keys: list[str]
    pass_status: str
    fail_status: str
    status_map_policy: ScalarStatusMapPolicy | None = None
    minimum: float | None = None
    maximum: float | None = None

    @classmethod
    def from_rule(cls, rule: Mapping[str, object], context: ValidationRuleContextLike) -> "SummaryRuleSpec":
        return cls(
            rule_kind=str(rule["kind"]),
            check_id=str(rule["check_id"]),
            title=str(rule["title"]),
            criterion=str(rule["criterion"]),
            criterion_latex=str(rule.get("criterion_latex", "")),
            criterion_formulae=list(rule.get("criterion_formulae", [])),
            criterion_definitions=list(rule.get("criterion_definitions", [])),
            description=str(rule["description"]),
            acceptable=str(rule["acceptable"]),
            acceptable_basis=str(rule["acceptable_basis"]),
            note=str(rule.get("note", "")),
            metric_key=str(rule["metric_key"]),
            metric_quantity=resolve_metric_quantity(
                str(rule["metric_key"]),
                unit_text=str(rule.get("metric_unit_text", "")).strip(),
                quantity_name=str(rule.get("metric_quantity_name", "")).strip(),
                observed_symbol=str(rule.get("metric_observed_symbol", "")).strip(),
            ),
            group=summary_group(rule, context),
            evidence_metric_keys=[str(metric).strip() for metric in rule.get("evidence_metric_keys", []) if str(metric).strip()],
            pass_status=str(rule.get("pass_status", "PASS")),
            fail_status=str(rule.get("fail_status", "FAIL")),
            status_map_policy=(
                ScalarStatusMapPolicy.from_mapping(rule)
                if str(rule["kind"]) == "summary_metric_status_map"
                else None
            ),
            minimum=(
                float(rule["minimum"])
                if "minimum" in rule and rule.get("minimum") is not None
                else None
            ),
            maximum=(
                float(rule["maximum"])
                if "maximum" in rule and rule.get("maximum") is not None
                else None
            ),
        )

    def to_case(self) -> SummaryRuleCase:
        observed_symbol = self.metric_quantity.resolved_observed_symbol
        if observed_symbol == r"\bar{x}":
            observed_symbol = group_mean_symbol(self.group)
        base_kwargs = {
            "rule_kind": self.rule_kind,
            "check_id": self.check_id,
            "title": self.title,
            "criterion": self.criterion,
            "criterion_latex": self.criterion_latex,
            "criterion_formulae": self.criterion_formulae,
            "criterion_definitions": self.criterion_definitions,
            "description": self.description,
            "acceptable": self.acceptable,
            "acceptable_basis": self.acceptable_basis,
            "note": self.note,
            "metric_key": self.metric_key,
            "metric_quantity": self.metric_quantity,
            "group": self.group,
            "evidence_metric_keys": self.evidence_metric_keys,
            "pass_status": self.pass_status,
            "fail_status": self.fail_status,
        }
        if self.rule_kind == "summary_metric_min":
            criterion_math = criterion_math_for_lower_bound(
                observed_symbol,
                float(self.minimum),
                definitions=[{"symbol": observed_symbol, "definition": metric_definition_text(self.metric_quantity, group=self.group)}],
            )
            case_kwargs = dict(base_kwargs)
            case_kwargs["criterion_latex"] = criterion_math.latex
            case_kwargs["criterion_definitions"] = criterion_math.definitions
            case_kwargs["minimum"] = float(self.minimum)
            return SummaryRuleCase(**case_kwargs)
        if self.rule_kind == "summary_metric_max":
            criterion_math = criterion_math_for_upper_bound(
                observed_symbol,
                float(self.maximum),
                definitions=[{"symbol": observed_symbol, "definition": metric_definition_text(self.metric_quantity, group=self.group)}],
            )
            case_kwargs = dict(base_kwargs)
            case_kwargs["criterion_latex"] = criterion_math.latex
            case_kwargs["criterion_definitions"] = criterion_math.definitions
            case_kwargs["maximum"] = float(self.maximum)
            return SummaryRuleCase(**case_kwargs)
        if self.rule_kind == "summary_metric_range":
            minimum = float(self.minimum) if self.minimum is not None else float("-inf")
            maximum = float(self.maximum) if self.maximum is not None else float("inf")
            criterion_math = criterion_math_for_closed_range(
                observed_symbol,
                minimum,
                maximum,
                definitions=[{"symbol": observed_symbol, "definition": metric_definition_text(self.metric_quantity, group=self.group)}],
            )
            case_kwargs = dict(base_kwargs)
            case_kwargs["criterion_latex"] = criterion_math.latex
            case_kwargs["criterion_definitions"] = criterion_math.definitions
            case_kwargs["minimum"] = minimum
            case_kwargs["maximum"] = maximum
            return SummaryRuleCase(**case_kwargs)
        if self.rule_kind == "summary_metric_status_map":
            return SummaryRuleCase(
                **base_kwargs,
                status_map_policy=self.status_map_policy,
            )
        raise ValueError(f"Unsupported summary rule kind {self.rule_kind!r}")


@dataclass(frozen=True)
class ProtocolExecutedRuleSpec:
    fallback_series_key: str = "fi_curve_rows"

    @classmethod
    def from_rule(cls, rule: Mapping[str, object]) -> "ProtocolExecutedRuleSpec":
        return cls(
            fallback_series_key=str(rule.get("fallback_series_key", "fi_curve_rows")).strip() or "fi_curve_rows",
        )


@dataclass(frozen=True)
class NotePresenceRowContextSpec:
    loader: str
    as_protocol_context: bool = False
    property_name: str = "FI Protocol"
    filter_field: str = ""
    filter_value: Any = None
    filter_values: tuple[Any, ...] = ()
    filter_value_arg: str = ""
    filter_values_arg: str = ""
    filters: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_rule(cls, raw: Mapping[str, object]) -> "NotePresenceRowContextSpec":
        return cls(
            loader=str(raw["loader"]),
            as_protocol_context=bool(raw.get("as_protocol_context")),
            property_name=str(raw.get("property_name", "FI Protocol")).strip() or "FI Protocol",
            filter_field=str(raw.get("filter_field", "")).strip(),
            filter_value=raw.get("filter_value"),
            filter_values=tuple(raw.get("filter_values", []) or ()),
            filter_value_arg=str(raw.get("filter_value_arg", "")).strip(),
            filter_values_arg=str(raw.get("filter_values_arg", "")).strip(),
            filters=tuple(dict(item) for item in raw.get("filters", []) if isinstance(item, MappingABC)),
        )

    def to_filter_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {"loader": self.loader}
        if self.filter_field:
            spec["filter_field"] = self.filter_field
        if self.filter_value is not None:
            spec["filter_value"] = self.filter_value
        if self.filter_values:
            spec["filter_values"] = list(self.filter_values)
        if self.filter_value_arg:
            spec["filter_value_arg"] = self.filter_value_arg
        if self.filter_values_arg:
            spec["filter_values_arg"] = self.filter_values_arg
        if self.filters:
            spec["filters"] = [dict(item) for item in self.filters]
        return spec


@dataclass(frozen=True)
class NotePresenceRuleSpec:
    scope: str | None
    row_contexts: tuple[NotePresenceRowContextSpec, ...]
    synthetic_contexts: tuple[dict[str, Any], ...]

    @classmethod
    def from_rule(cls, rule: Mapping[str, object]) -> "NotePresenceRuleSpec":
        scope_text = str(rule.get("scope", "")).strip()
        return cls(
            scope=scope_text or None,
            row_contexts=tuple(
                NotePresenceRowContextSpec.from_rule(raw)
                for raw in rule.get("row_contexts", [])
                if isinstance(raw, MappingABC)
            ),
            synthetic_contexts=tuple(
                dict(raw)
                for raw in rule.get("synthetic_contexts", [])
                if isinstance(raw, MappingABC)
            ),
        )


@dataclass(frozen=True)
class ComparisonRuleSpec:
    rule_kind: str
    check_id: str
    title: str
    criterion: str
    criterion_latex: str
    criterion_formulae: list[str]
    criterion_definitions: list[dict[str, Any]]
    description: str
    acceptable: str
    acceptable_basis: str
    note: str
    metric_key: str
    metric_quantity: MetricQuantitySpec
    entity_key: str
    pass_status: str
    fail_status: str
    expected: float | None = None
    tolerance: float | None = None
    left_group: str = ""
    right_group: str = ""
    operator: str = ""
    max_difference: float | None = None
    groups: tuple[str, ...] = ()

    @classmethod
    def from_rule(cls, rule: Mapping[str, object]) -> "ComparisonRuleSpec":
        return cls(
            rule_kind=str(rule["kind"]),
            check_id=str(rule["check_id"]),
            title=str(rule["title"]),
            criterion=str(rule["criterion"]),
            criterion_latex=str(rule.get("criterion_latex", "")),
            criterion_formulae=list(rule.get("criterion_formulae", [])),
            criterion_definitions=list(rule.get("criterion_definitions", [])),
            description=str(rule["description"]),
            acceptable=str(rule["acceptable"]),
            acceptable_basis=str(rule["acceptable_basis"]),
            note=str(rule.get("note", "")),
            metric_key=str(rule["metric_key"]),
            metric_quantity=resolve_metric_quantity(
                str(rule["metric_key"]),
                unit_text=str(rule.get("metric_unit_text", "")).strip(),
                quantity_name=str(rule.get("metric_quantity_name", "")).strip(),
                observed_symbol=str(rule.get("metric_observed_symbol", "")).strip(),
            ),
            entity_key=str(rule.get("entity_key", "cell_name")),
            pass_status=str(rule.get("pass_status", "PASS")),
            fail_status=str(rule.get("fail_status", "FAIL")),
            expected=(
                float(rule.get("expected", 0.0))
                if str(rule["kind"]) == "all_exact_metric"
                else None
            ),
            tolerance=(
                float(rule.get("tolerance", 1e-9))
                if str(rule["kind"]) == "all_exact_metric"
                else None
            ),
            left_group=str(rule.get("left_group", "")),
            right_group=str(rule.get("right_group", "")),
            operator=str(rule.get("operator", ">")).strip(),
            max_difference=(
                float(rule["max_difference"])
                if "max_difference" in rule and rule.get("max_difference") is not None
                else None
            ),
            groups=tuple(str(group) for group in rule.get("groups", [])),
        )

    def to_case(self) -> ComparisonRuleCase:
        base_kwargs = {
            "rule_kind": self.rule_kind,
            "check_id": self.check_id,
            "title": self.title,
            "criterion": self.criterion,
            "criterion_latex": self.criterion_latex,
            "criterion_formulae": self.criterion_formulae,
            "criterion_definitions": self.criterion_definitions,
            "description": self.description,
            "acceptable": self.acceptable,
            "acceptable_basis": self.acceptable_basis,
            "note": self.note,
            "metric_key": self.metric_key,
            "metric_quantity": self.metric_quantity,
            "entity_key": self.entity_key,
            "pass_status": self.pass_status,
            "fail_status": self.fail_status,
        }
        if self.rule_kind == "all_finite_metric":
            return ComparisonRuleCase(**base_kwargs)
        if self.rule_kind == "all_exact_metric":
            criterion_math = criterion_math_for_exact_metric(self.metric_key)
            case_kwargs = dict(base_kwargs)
            case_kwargs["criterion_latex"] = criterion_math.latex
            case_kwargs["criterion_definitions"] = criterion_math.definitions
            case_kwargs["expected"] = float(self.expected)
            case_kwargs["tolerance"] = float(self.tolerance)
            return ComparisonRuleCase(**case_kwargs)
        if self.rule_kind == "group_ordering":
            left_symbol = group_mean_symbol(self.left_group, self.metric_quantity.resolved_observed_symbol)
            right_symbol = group_mean_symbol(self.right_group, self.metric_quantity.resolved_observed_symbol)
            criterion_math = criterion_math_for_ordering(right_symbol, self.operator, left_symbol)
            case_kwargs = dict(base_kwargs)
            case_kwargs["criterion_latex"] = criterion_math.latex
            case_kwargs["criterion_definitions"] = [
                {"symbol": left_symbol, "definition": metric_definition_text(self.metric_quantity, group=self.left_group)},
                {"symbol": right_symbol, "definition": metric_definition_text(self.metric_quantity, group=self.right_group)},
            ]
            case_kwargs["left_group"] = self.left_group
            case_kwargs["right_group"] = self.right_group
            case_kwargs["operator"] = self.operator
            return ComparisonRuleCase(**case_kwargs)
        if self.rule_kind == "group_abs_diff_max":
            left_symbol = group_mean_symbol(self.left_group, self.metric_quantity.resolved_observed_symbol)
            right_symbol = group_mean_symbol(self.right_group, self.metric_quantity.resolved_observed_symbol)
            criterion_math = criterion_math_for_absolute_difference(
                left_symbol,
                right_symbol,
                float(self.max_difference),
                definitions=[
                    {"symbol": left_symbol, "definition": metric_definition_text(self.metric_quantity, group=self.left_group)},
                    {"symbol": right_symbol, "definition": metric_definition_text(self.metric_quantity, group=self.right_group)},
                ],
            )
            case_kwargs = dict(base_kwargs)
            case_kwargs["criterion_latex"] = criterion_math.latex
            case_kwargs["criterion_definitions"] = criterion_math.definitions
            case_kwargs["left_group"] = self.left_group
            case_kwargs["right_group"] = self.right_group
            case_kwargs["max_difference"] = float(self.max_difference)
            return ComparisonRuleCase(**case_kwargs)
        if self.rule_kind == "group_positive":
            if not self.groups:
                raise ValueError("group_positive rule requires non-empty 'groups'")
            group_symbols = [
                group_mean_symbol(group, self.metric_quantity.resolved_observed_symbol)
                for group in self.groups
            ]
            case_kwargs = dict(base_kwargs)
            case_kwargs["criterion_latex"] = " \\wedge ".join(rf"{symbol} > 0" for symbol in group_symbols)
            case_kwargs["criterion_definitions"] = [
                {"symbol": symbol, "definition": metric_definition_text(self.metric_quantity, group=group)}
                for symbol, group in zip(group_symbols, self.groups, strict=False)
            ]
            case_kwargs["groups"] = self.groups
            return ComparisonRuleCase(**case_kwargs)
        raise ValueError(f"Unsupported comparison rule kind {self.rule_kind!r}")


@dataclass(frozen=True)
class ReferenceBandPropertyRuleSpec:
    property_name: str
    metric_key: str
    band_mode: str
    lower_bound: float | None
    upper_bound: float | None
    quantile_low_field: str
    quantile_high_field: str
    quantile_low_label_field: str
    quantile_high_label_field: str
    note: str
    review_status: str
    review_note: str
    review_reviewer: str
    review_required_expertise: str
    review_focus: str

    @classmethod
    def from_rule(
        cls,
        rule: Mapping[str, object],
        context: ValidationRuleContextLike,
        *,
        property_name: str,
        metric_key: str,
        band_mode: str,
        default_lower_bound: float | None,
        default_upper_bound: float | None,
    ) -> "ReferenceBandPropertyRuleSpec":
        review_metadata = property_review_metadata(rule, context, property_name)
        return cls(
            property_name=property_name,
            metric_key=metric_key,
            band_mode=band_mode,
            lower_bound=optional_float(
                property_override(rule, "property_lower_bounds", property_name, default_lower_bound)
            ),
            upper_bound=optional_float(
                property_override(rule, "property_upper_bounds", property_name, default_upper_bound)
            ),
            quantile_low_field=row_field_name(
                rule,
                property_name,
                field_override_key="property_quantile_low_fields",
                default_field_key="default_quantile_low_field",
                default="q_low",
            ),
            quantile_high_field=row_field_name(
                rule,
                property_name,
                field_override_key="property_quantile_high_fields",
                default_field_key="default_quantile_high_field",
                default="q_high",
            ),
            quantile_low_label_field=row_field_name(
                rule,
                property_name,
                field_override_key="property_quantile_low_label_fields",
                default_field_key="default_quantile_low_label_field",
                default="q_low_label",
            ),
            quantile_high_label_field=row_field_name(
                rule,
                property_name,
                field_override_key="property_quantile_high_label_fields",
                default_field_key="default_quantile_high_label_field",
                default="q_high_label",
            ),
            note=str(property_override(rule, "property_notes", property_name, "")),
            review_status=review_metadata["status"],
            review_note=review_metadata["note"],
            review_reviewer=review_metadata["reviewer"],
            review_required_expertise=review_metadata["required_expertise"],
            review_focus=review_metadata["focus"],
        )

    def quantile_interval_from_row(
        self,
        row: Mapping[str, Any],
    ) -> tuple[float | None, float | None, str | None, str | None]:
        quantile_low = optional_float(row.get(self.quantile_low_field))
        quantile_high = optional_float(row.get(self.quantile_high_field))
        quantile_low_label = str(row.get(self.quantile_low_label_field, "")).strip() or self.quantile_low_field
        quantile_high_label = str(row.get(self.quantile_high_label_field, "")).strip() or self.quantile_high_field
        return quantile_low, quantile_high, quantile_low_label, quantile_high_label


@dataclass(frozen=True)
class _ReferenceBandRowBinding:
    row: ReferenceRowRecord
    property_spec: ReferenceBandPropertyRuleSpec
    group: str
    sigma_arg_name: str
    sigma_multiplier: float
    sigma_phrase: str

    @property
    def property_name(self) -> str:
        return str(self.row.get("Property", "")).strip()

    @property
    def reference_mean(self) -> float:
        return float(self.row["mean"])

    @property
    def reference_sd(self) -> float:
        return float(self.row["sd"])

    @property
    def unit_text(self) -> str:
        return str(self.row.get("unit", "")).strip()

    @property
    def quantile_interval(self) -> tuple[float | None, float | None, str | None, str | None]:
        if self.property_spec.band_mode != "quantile_interval":
            return None, None, None, None
        return self.property_spec.quantile_interval_from_row(self.row)

    @property
    def accepted_band(self) -> ReferenceAcceptanceBand:
        quantile_low, quantile_high, quantile_low_label, quantile_high_label = self.quantile_interval
        return compute_reference_acceptance_band(
            reference_mean=self.reference_mean,
            reference_sd=self.reference_sd,
            sigma_multiplier=self.sigma_multiplier,
            band_mode=self.property_spec.band_mode,
            lower_bound=self.property_spec.lower_bound,
            upper_bound=self.property_spec.upper_bound,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            quantile_low_label=quantile_low_label,
            quantile_high_label=quantile_high_label,
        )

    @property
    def range_text(self) -> str:
        band = self.accepted_band
        range_text = f"between {rounded(band.low)} and {rounded(band.high)}"
        if self.unit_text:
            range_text = f"{range_text} {self.unit_text}"
        return range_text

    @property
    def criterion_math(self):
        return criterion_math_for_reference_band(self.group, self.property_name, self.accepted_band)

    @property
    def observation(self) -> ReferenceBandObservation:
        quantile_low, quantile_high, quantile_low_label, quantile_high_label = self.quantile_interval
        return ReferenceBandObservation(
            property_name=self.property_name,
            group=self.group,
            metric_key=self.property_spec.metric_key,
            reference_mean=self.reference_mean,
            reference_sd=self.reference_sd,
            unit_text=self.unit_text,
            policy=ReferenceBandPolicy(
                mode=self.property_spec.band_mode,
                sigma_multiplier=self.sigma_multiplier,
                lower_bound=self.property_spec.lower_bound,
                upper_bound=self.property_spec.upper_bound,
                quantile_low=quantile_low,
                quantile_high=quantile_high,
                quantile_low_label=quantile_low_label,
                quantile_high_label=quantile_high_label,
            ),
            provenance=ProvenanceRecord.from_row(self.row),
            review=ValidationReview(
                status=self.property_spec.review_status,
                note=self.property_spec.review_note,
                reviewer=self.property_spec.review_reviewer,
                required_expertise=self.property_spec.review_required_expertise,
                focus=self.property_spec.review_focus,
            ),
        )

    @property
    def acceptable(self) -> str:
        return (
            f"The observed {self.group} mean must lie {self.range_text}, using the configured "
            f"{self.accepted_band.standard_label}."
        )

    @property
    def acceptable_basis(self) -> str:
        band = self.accepted_band
        return (
            f"Derived from the uploaded literature row for {self.property_name} using the configured "
            f"{band.standard_label}: {band.description}. "
            f"The sigma multiplier comes from '{self.sigma_arg_name}' when that standard needs one."
        )

    @property
    def reference_annotation(self) -> str:
        return _reference_annotation(self.row)

    def to_case(self, *, pass_status: str, fail_status: str) -> ReferenceBandCase:
        band = self.accepted_band
        criterion_math = self.criterion_math
        return ReferenceBandCase(
            check_id=f"{self.group.lower()}_{self.property_spec.metric_key.lower()}_within_uploaded_reference_band".replace(".", "_"),
            title=_title_text_for_band(self.group, self.property_name, band),
            criterion=_criterion_text_for_band(self.group, self.property_name, band, self.sigma_phrase),
            criterion_latex=criterion_math.latex,
            criterion_formulae=criterion_math.formulae,
            criterion_definitions=criterion_math.definitions,
            description=(
                "This is the direct single-cell-type reference check derived from uploaded literature rows for "
                f"{self.property_name} rather than from a cross-group ordering heuristic."
            ),
            acceptable=self.acceptable,
            acceptable_basis=self.acceptable_basis,
            note=self.property_spec.note,
            observation=self.observation,
            reference_annotation=self.reference_annotation,
            pass_status=pass_status,
            fail_status=fail_status,
        )


@dataclass(frozen=True)
class ReferenceBandRuleSpec:
    loader: str
    reference_source: str
    group_field: str
    sigma_arg_name: str
    sigma_multiplier: float
    sigma_phrase: str
    suite_descriptor: SuiteDescriptor
    pass_status: str
    fail_status: str
    properties: dict[str, ReferenceBandPropertyRuleSpec]

    @classmethod
    def from_rule(
        cls,
        rule: Mapping[str, object],
        context: ValidationRuleContextLike,
    ) -> "ReferenceBandRuleSpec":
        property_metric_map = {
            str(key): str(value)
            for key, value in dict(rule.get("property_metric_map", {})).items()
        }
        resolved_property_band_modes = property_band_modes(rule, property_metric_map)
        sigma_arg_name = str(rule.get("sigma_arg_name", "reference_sigma_multiplier"))
        sigma_multiplier = float(getattr(context.args, sigma_arg_name, rule.get("sigma_multiplier", 2.0)))
        default_lower_bound = optional_float(rule.get("default_lower_bound"))
        default_upper_bound = optional_float(rule.get("default_upper_bound"))
        properties = {
            property_name: ReferenceBandPropertyRuleSpec.from_rule(
                rule,
                context,
                property_name=property_name,
                metric_key=metric_key,
                band_mode=resolved_property_band_modes[property_name],
                default_lower_bound=default_lower_bound,
                default_upper_bound=default_upper_bound,
            )
            for property_name, metric_key in property_metric_map.items()
        }
        return cls(
            loader=str(rule["loader"]),
            reference_source=str(rule.get("reference_source", "")).strip(),
            group_field=str(rule.get("group_field", "cell_type")),
            sigma_arg_name=sigma_arg_name,
            sigma_multiplier=sigma_multiplier,
            sigma_phrase=_sigma_phrase(sigma_multiplier),
            suite_descriptor=SuiteDescriptor(
                suite_id=str(rule.get("suite_name", rule.get("title", "reference-band-suite"))).strip() or "reference-band-suite",
                suite_kind_label="Reference-band suite",
                aggregate_policy=suite_aggregate_policy_from_rule(rule),
                statistical_policy=suite_statistical_policy_from_rule(rule),
            ),
            pass_status=str(rule.get("pass_status", "PASS")),
            fail_status=str(rule.get("fail_status", "FAIL")),
            properties=properties,
        )

    def build_cases(
        self,
        *,
        rows: ReferenceRowTable | list[ReferenceRowRecord | Mapping[str, Any]],
    ) -> list[ReferenceBandCase]:
        row_table = coerce_reference_row_table(rows)
        cases: list[ReferenceBandCase] = []
        for row in row_table:
            if self.reference_source and str(row.get("Source", "")).strip() != self.reference_source:
                continue
            property_name = str(row.get("Property", "")).strip()
            property_spec = self.properties.get(property_name)
            if property_spec is None:
                continue
            group = str(row.get(self.group_field, "")).strip()
            if not group:
                continue
            if not (optional_float(row.get("mean")) is not None and optional_float(row.get("sd")) is not None):
                continue
            binding = _ReferenceBandRowBinding(
                row=row,
                property_spec=property_spec,
                group=group,
                sigma_arg_name=self.sigma_arg_name,
                sigma_multiplier=self.sigma_multiplier,
                sigma_phrase=self.sigma_phrase,
            )
            cases.append(
                binding.to_case(
                    pass_status=self.pass_status,
                    fail_status=self.fail_status,
                )
            )
        return cases


@dataclass(frozen=True)
class SeriesComparisonRuleSpec:
    check_id: str
    title: str
    criterion: str
    criterion_latex: str
    criterion_formulae: list[str]
    criterion_definitions: list[dict[str, Any]]
    description: str
    acceptable: str
    acceptable_basis: str
    note: str
    pass_status: str
    fail_status: str
    protocol_evidence_key: str
    reference_spec: SeriesDataSpec
    model_spec: SeriesDataSpec
    comparison_x_unit_text: str
    comparison_y_unit_text: str
    x_quantity_name: str
    y_quantity_name: str
    visual_contract: SeriesVisualContract
    policy: SeriesComparisonPolicy

    @classmethod
    def from_rule(
        cls,
        rule: Mapping[str, object],
        *,
        protocol_series_spec: ProtocolEvidenceSeriesSpec | None = None,
    ) -> "SeriesComparisonRuleSpec":
        parser = _SeriesComparisonRuleParser(rule, protocol_series_spec=protocol_series_spec)
        return cls(
            check_id=str(rule["check_id"]),
            title=str(rule["title"]),
            criterion=str(rule["criterion"]),
            criterion_latex=str(rule.get("criterion_latex", "")),
            criterion_formulae=list(rule.get("criterion_formulae", [])),
            criterion_definitions=list(rule.get("criterion_definitions", [])),
            description=str(rule["description"]),
            acceptable=str(rule["acceptable"]),
            acceptable_basis=str(rule["acceptable_basis"]),
            note=str(rule.get("note", "")),
            pass_status=str(rule.get("pass_status", "PASS")),
            fail_status=str(rule.get("fail_status", "FAIL")),
            protocol_evidence_key=parser.string("protocol_evidence_key", "fi_curve_rows"),
            reference_spec=parser.data_spec("reference"),
            model_spec=parser.data_spec("model"),
            comparison_x_unit_text=parser.required_unit_text("comparison_x_unit_text"),
            comparison_y_unit_text=parser.required_unit_text("comparison_y_unit_text"),
            x_quantity_name=parser.string_with_fallback(
                "x_quantity_name",
                "Injected current",
                fallback=parser.protocol_series_string("x_quantity_name"),
            ),
            y_quantity_name=parser.string_with_fallback(
                "y_quantity_name",
                "Firing rate",
                fallback=parser.protocol_series_string("y_quantity_name"),
            ),
            visual_contract=parser.visual_contract(),
            policy=parser.policy(),
        )

    def to_case(
        self,
        *,
        reference_rows: ReferenceRowTable | list[ReferenceRowRecord | Mapping[str, Any]],
    ) -> SeriesComparisonCase:
        observation = SeriesDistributionObservation(
            protocol_evidence_key=self.protocol_evidence_key,
            reference_rows=coerce_reference_row_table(reference_rows),
            reference_spec=self.reference_spec,
            model_spec=self.model_spec,
            comparison_x_unit_text=self.comparison_x_unit_text,
            comparison_y_unit_text=self.comparison_y_unit_text,
            x_quantity_name=self.x_quantity_name,
            y_quantity_name=self.y_quantity_name,
            visual_contract=self.visual_contract,
            policy=self.policy,
        )
        return SeriesComparisonCase(
            check_id=self.check_id,
            title=self.title,
            criterion=self.criterion,
            criterion_latex=self.criterion_latex,
            criterion_formulae=self.criterion_formulae,
            criterion_definitions=self.criterion_definitions,
            description=self.description,
            acceptable=self.acceptable,
            acceptable_basis=self.acceptable_basis,
            note=self.note,
            observation=observation,
            pass_status=self.pass_status,
            fail_status=self.fail_status,
        )


@dataclass(frozen=True)
class _SeriesComparisonRuleParser:
    rule: Mapping[str, object]
    protocol_series_spec: ProtocolEvidenceSeriesSpec | None = None

    def string(self, key: str, default: str) -> str:
        return str(self.rule.get(key, default))

    def string_with_fallback(self, key: str, default: str, *, fallback: str = "") -> str:
        if key in self.rule:
            return str(self.rule.get(key, default))
        if fallback:
            return str(fallback)
        return str(default)

    def protocol_series_string(self, key: str) -> str:
        if self.protocol_series_spec is None:
            return ""
        value = getattr(self.protocol_series_spec, key, "")
        return str(value or "").strip()

    def required_unit_text(self, key: str) -> str:
        if key not in self.rule:
            raise ValueError(
                f"reference_curve_match requires explicit {key!r}; do not infer series-comparison units from field names"
            )
        return str(self.rule.get(key, "")).strip()

    def required_choice(self, key: str) -> str:
        if key not in self.rule:
            raise ValueError(
                f"reference_curve_match requires explicit {key!r}; do not let series-comparison policy fall back silently"
            )
        value = str(self.rule.get(key, "")).strip()
        if not value:
            raise ValueError(f"reference_curve_match requires non-empty {key!r}")
        return value

    def axis_transform(self, key: str) -> AxisTransform:
        raw = self.rule.get(key, {})
        if raw in (None, "", {}):
            return AxisTransform()
        if not isinstance(raw, MappingABC):
            raise ValueError(f"{key} must be a table/dict when provided")
        raw_points = raw.get("points", ())
        points: tuple[tuple[float, float], ...] = ()
        if raw_points not in (None, ""):
            if not isinstance(raw_points, (list, tuple)):
                raise ValueError(f"{key}.points must be a sequence when provided")
            if not raw_points:
                raw_points = ()
            normalized_points: list[tuple[float, float]] = []
            for index, point in enumerate(raw_points, start=1):
                if isinstance(point, MappingABC):
                    if "input" not in point or "output" not in point:
                        raise ValueError(f"{key}.points[{index}] must provide both 'input' and 'output'")
                    normalized_points.append((float(point["input"]), float(point["output"])))
                    continue
                if isinstance(point, (list, tuple)) and len(point) == 2:
                    normalized_points.append((float(point[0]), float(point[1])))
                    continue
                raise ValueError(
                    f"{key}.points[{index}] must be either a dict with input/output or a two-item list"
                )
            points = tuple(normalized_points)
        return AxisTransform(
            kind=str(raw.get("kind", "identity")),
            scale=float(raw.get("scale", 1.0)),
            offset=float(raw.get("offset", 0.0)),
            input_unit_text=str(raw.get("input_unit_text", "")).strip(),
            output_unit_text=str(raw.get("output_unit_text", "")).strip(),
            points=points,
            extrapolation_mode=str(raw.get("extrapolation_mode", "forbid")).strip(),
            scale_lookup_key=str(raw.get("scale_lookup_key", "")).strip(),
            offset_lookup_key=str(raw.get("offset_lookup_key", "")).strip(),
        )

    def _unit_text(self, key: str, *, fallback: str = "", explicit_required: bool = True) -> str:
        if key in self.rule:
            return str(self.rule.get(key, "")).strip()
        if fallback:
            return str(fallback).strip()
        if explicit_required:
            raise ValueError(
                f"reference_curve_match requires explicit {key!r}; do not infer series-comparison units from field names"
            )
        return ""

    def data_spec(self, side: str) -> SeriesDataSpec:
        side_prefix = str(side).strip().lower()
        if side_prefix not in {"reference", "model"}:
            raise ValueError(f"Unsupported series-data side {side!r}")
        defaults = {
            "reference": ("current_pA", "firing_rate_Hz", "cell_id"),
            "model": ("current_pA", "firing_rate_Hz", "cell_name"),
        }
        default_x_key, default_y_key, default_series_id_key = defaults[side_prefix]
        protocol_series_spec = self.protocol_series_spec if side_prefix == "model" else None
        fallback_x_key = protocol_series_spec.x_key if protocol_series_spec is not None else ""
        fallback_y_key = (
            protocol_series_spec.y_keys[0]
            if protocol_series_spec is not None and protocol_series_spec.y_keys
            else ""
        )
        fallback_series_id_key = protocol_series_spec.series_id_key if protocol_series_spec is not None else ""
        return SeriesDataSpec(
            x_key=self.string_with_fallback(
                f"{side_prefix}_current_key",
                default_x_key,
                fallback=fallback_x_key,
            ),
            y_key=self.string_with_fallback(
                f"{side_prefix}_value_key",
                default_y_key,
                fallback=fallback_y_key,
            ),
            x_unit_text=self._unit_text(
                f"{side_prefix}_x_unit_text",
                fallback=protocol_series_spec.x_unit_text if protocol_series_spec is not None else "",
                explicit_required=True,
            ),
            y_unit_text=self._unit_text(
                f"{side_prefix}_y_unit_text",
                fallback=protocol_series_spec.y_unit_text if protocol_series_spec is not None else "",
                explicit_required=True,
            ),
            x_transform=self.axis_transform(f"{side_prefix}_x_transform"),
            y_transform=self.axis_transform(f"{side_prefix}_y_transform"),
            series_id_key=self.string_with_fallback(
                f"{side_prefix}_series_id_key",
                default_series_id_key,
                fallback=fallback_series_id_key,
            ),
        )

    def visual_contract(self) -> SeriesVisualContract:
        return SeriesVisualContract(
            x_key=self.string("visual_x_key", "currents_pA"),
            reference_y_key=self.string("visual_reference_y_key", "reference_values_Hz"),
            model_y_key=self.string("visual_model_y_key", "model_values_Hz"),
            kind=self.string("visual_kind", "fi_curve"),
        )

    def policy(self) -> SeriesComparisonPolicy:
        minimum_point_count = int(self.rule.get("minimum_point_count", 1))
        if minimum_point_count < 1:
            raise ValueError("reference_curve_match requires 'minimum_point_count' >= 1")

        def _optional_fraction(key: str) -> float | None:
            if key not in self.rule or self.rule.get(key) is None:
                return None
            value = float(self.rule[key])
            if not math.isfinite(value) or not (0.0 <= value <= 1.0):
                raise ValueError(f"reference_curve_match {key!r} must be a finite fraction in [0, 1]")
            return value

        score_family = self.required_choice("score_family")
        alignment_policy = self.required_choice("alignment_policy")
        if alignment_policy not in SERIES_ALIGNMENT_POLICIES:
            raise ValueError(
                f"reference_curve_match alignment policy {alignment_policy!r} is unsupported; "
                f"expected one of {', '.join(sorted(SERIES_ALIGNMENT_POLICIES))}"
            )
        if alignment_policy in {"nearest_within_tolerance", "tolerance_clusters"}:
            if "x_match_tolerance" not in self.rule or self.rule.get("x_match_tolerance") is None:
                raise ValueError(
                    f"reference_curve_match alignment policy {alignment_policy!r} requires explicit 'x_match_tolerance'"
                )
            x_match_tolerance = float(self.rule["x_match_tolerance"])
        else:
            x_match_tolerance = None
        if alignment_policy == "resampled_grid":
            resampling_grid_source = str(self.rule.get("resampling_grid_source", "")).strip()
            resampling_domain_policy = str(self.rule.get("resampling_domain_policy", "")).strip()
            if resampling_domain_policy and resampling_domain_policy not in SERIES_RESAMPLING_DOMAIN_POLICIES:
                raise ValueError(
                    f"reference_curve_match resampling domain policy {resampling_domain_policy!r} is unsupported; "
                    f"expected one of {', '.join(sorted(SERIES_RESAMPLING_DOMAIN_POLICIES))}"
                )
            interpolation_method = str(self.rule.get("interpolation_method", "linear")).strip() or "linear"
            raw_resampling_grid_values = self.rule.get("resampling_grid_values", ())
            if raw_resampling_grid_values in (None, ""):
                resampling_grid_values = ()
            else:
                if not isinstance(raw_resampling_grid_values, (list, tuple)):
                    raise ValueError("reference_curve_match 'resampling_grid_values' must be a sequence when provided")
                resampling_grid_values = tuple(float(value) for value in raw_resampling_grid_values)
            if resampling_grid_source == "explicit_grid" and not resampling_grid_values:
                raise ValueError(
                    "reference_curve_match resampled-grid alignment with "
                    "resampling_grid_source='explicit_grid' requires explicit 'resampling_grid_values'"
                )
        else:
            if str(self.rule.get("resampling_domain_policy", "")).strip():
                raise ValueError(
                    "reference_curve_match 'resampling_domain_policy' only applies when "
                    "alignment_policy = 'resampled_grid'"
                )
            resampling_grid_source = ""
            resampling_domain_policy = ""
            resampling_grid_values = ()
            interpolation_method = "linear"
        pvalue_aggregation = str(self.rule.get("pvalue_aggregation", "auto")).strip()
        if score_family in {"welch_only", "hybrid_residual_welch"}:
            equivalence_margin = None
        elif score_family in {"equivalence_only", "hybrid_residual_equivalence"}:
            if "equivalence_margin" in self.rule and self.rule.get("equivalence_margin") is not None:
                equivalence_margin = float(self.rule["equivalence_margin"])
            else:
                equivalence_margin = None
                if not (
                    isinstance(self.rule.get("maximum_mae"), (int, float))
                    and math.isfinite(float(self.rule["maximum_mae"]))
                ):
                    raise ValueError(
                        f"reference_curve_match score family {score_family!r} requires explicit 'equivalence_margin' "
                        "or a finite 'maximum_mae' fallback"
                    )
        else:
            equivalence_margin = None
        return SeriesComparisonPolicy(
            minimum_point_count=minimum_point_count,
            minimum_reference_coverage_fraction=_optional_fraction("minimum_reference_coverage_fraction"),
            minimum_model_coverage_fraction=_optional_fraction("minimum_model_coverage_fraction"),
            maximum_mae=float(self.rule.get("maximum_mae", float("inf"))),
            maximum_rmse=float(self.rule.get("maximum_rmse", float("inf"))),
            minimum_median_welch_pvalue=(
                float(self.rule["minimum_median_welch_pvalue"])
                if "minimum_median_welch_pvalue" in self.rule and self.rule.get("minimum_median_welch_pvalue") is not None
                else None
            ),
            equivalence_margin=equivalence_margin,
            equivalence_alpha=float(self.rule.get("equivalence_alpha", 0.05)),
            x_precision_digits=int(self.rule.get("current_precision_digits", 6)),
            alignment_policy=alignment_policy,
            x_match_tolerance=x_match_tolerance,
            resampling_grid_source=resampling_grid_source,
            resampling_domain_policy=resampling_domain_policy,
            resampling_grid_values=resampling_grid_values,
            interpolation_method=interpolation_method,
            distribution_kind=self.required_choice("distribution_kind"),
            score_family=score_family,
            pvalue_aggregation=pvalue_aggregation,
        )
