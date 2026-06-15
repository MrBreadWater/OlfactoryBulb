"""Typed declarative config parsers for maintained reference-validation rules."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from olfactorybulb.audit.criterion_math import (
    criterion_math_for_absolute_difference,
    criterion_math_for_closed_range,
    criterion_math_for_exact_metric,
    criterion_math_for_lower_bound,
    criterion_math_for_ordering,
    criterion_math_for_upper_bound,
    group_mean_symbol,
)
from olfactorybulb.neuronunit.reference_bands import sigma_phrase as _sigma_phrase
from olfactorybulb.neuronunit.comparison_validation_suite import ComparisonRuleCase
from olfactorybulb.neuronunit.series_validation_suite import (
    AxisTransform,
    SERIES_ALIGNMENT_POLICIES,
    SeriesComparisonPolicy,
    SeriesDataSpec,
    SeriesVisualContract,
)
from olfactorybulb.neuronunit.summary_validation_suite import SummaryRuleCase


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def property_override(
    rule: dict[str, Any],
    mapping_key: str,
    property_name: str,
    default: Any,
) -> Any:
    mapping = rule.get(mapping_key, {})
    if not isinstance(mapping, dict):
        return default
    for key, value in mapping.items():
        if str(key).strip() == property_name:
            return value
    return default


def property_review_metadata(
    rule: dict[str, Any],
    context: Any,
    property_name: str,
) -> dict[str, str]:
    defaults = {
        "status": str(rule.get("validation_design_review_status", context.config.get("validation_design_review", {}).get("default_status", ""))).strip(),
        "note": str(rule.get("validation_design_review_note", context.config.get("validation_design_review", {}).get("default_note", ""))).strip(),
        "reviewer": str(rule.get("validation_design_review_reviewer", context.config.get("validation_design_review", {}).get("default_reviewer", ""))).strip(),
        "required_expertise": str(
            rule.get(
                "validation_design_review_required_expertise",
                context.config.get("validation_design_review", {}).get("default_required_expertise", ""),
            )
        ).strip(),
        "focus": str(
            rule.get(
                "validation_design_review_focus",
                context.config.get("validation_design_review", {}).get("default_focus", ""),
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


def property_band_modes(rule: dict[str, Any], property_metric_map: dict[str, str]) -> dict[str, str]:
    if "default_band_mode" in rule:
        raise ValueError(
            "reference_band_rows no longer supports 'default_band_mode'; "
            "choose 'property_band_modes' explicitly for every property"
        )
    raw_modes = rule.get("property_band_modes", {})
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


def row_field_name(
    rule: dict[str, Any],
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


def summary_group(rule: dict[str, Any], context: Any) -> str:
    explicit = str(rule.get("group", "") or "").strip()
    if explicit:
        return explicit
    if len(context.summary) == 1:
        return next(iter(context.summary))
    return str(context.config.get("default_group", "ungrouped"))


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
    group: str
    evidence_metric_keys: list[str]
    pass_status: str
    fail_status: str
    default_status: str
    minimum: float | None = None
    maximum: float | None = None
    pass_values: tuple[float, ...] = ()
    warn_values: tuple[float, ...] = ()
    fail_values: tuple[float, ...] = ()

    @classmethod
    def from_rule(cls, rule: dict[str, Any], context: Any) -> "SummaryRuleSpec":
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
            group=summary_group(rule, context),
            evidence_metric_keys=[str(metric).strip() for metric in rule.get("evidence_metric_keys", []) if str(metric).strip()],
            pass_status=str(rule.get("pass_status", "PASS")),
            fail_status=str(rule.get("fail_status", "FAIL")),
            default_status=str(rule.get("default_status", "FAIL")),
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
            pass_values=tuple(float(value) for value in rule.get("pass_values", [])),
            warn_values=tuple(float(value) for value in rule.get("warn_values", [])),
            fail_values=tuple(float(value) for value in rule.get("fail_values", [])),
        )

    def to_case(self) -> SummaryRuleCase:
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
            "group": self.group,
            "evidence_metric_keys": self.evidence_metric_keys,
            "pass_status": self.pass_status,
            "fail_status": self.fail_status,
            "default_status": self.default_status,
        }
        if self.rule_kind == "summary_metric_min":
            criterion_math = criterion_math_for_lower_bound(
                observed_symbol,
                float(self.minimum),
                definitions=[{"symbol": observed_symbol, "definition": f"{self.group} mean {self.metric_key}"}],
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
                definitions=[{"symbol": observed_symbol, "definition": f"{self.group} mean {self.metric_key}"}],
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
                definitions=[{"symbol": observed_symbol, "definition": f"{self.group} mean {self.metric_key}"}],
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
                pass_values=self.pass_values,
                warn_values=self.warn_values,
                fail_values=self.fail_values,
            )
        raise ValueError(f"Unsupported summary rule kind {self.rule_kind!r}")


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
    def from_rule(cls, rule: dict[str, Any]) -> "ComparisonRuleSpec":
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
            left_symbol = group_mean_symbol(self.left_group)
            right_symbol = group_mean_symbol(self.right_group)
            criterion_math = criterion_math_for_ordering(right_symbol, self.operator, left_symbol)
            case_kwargs = dict(base_kwargs)
            case_kwargs["criterion_latex"] = criterion_math.latex
            case_kwargs["criterion_definitions"] = [
                {"symbol": left_symbol, "definition": f"{self.left_group} mean {self.metric_key}"},
                {"symbol": right_symbol, "definition": f"{self.right_group} mean {self.metric_key}"},
            ]
            case_kwargs["left_group"] = self.left_group
            case_kwargs["right_group"] = self.right_group
            case_kwargs["operator"] = self.operator
            return ComparisonRuleCase(**case_kwargs)
        if self.rule_kind == "group_abs_diff_max":
            left_symbol = group_mean_symbol(self.left_group)
            right_symbol = group_mean_symbol(self.right_group)
            criterion_math = criterion_math_for_absolute_difference(
                left_symbol,
                right_symbol,
                float(self.max_difference),
                definitions=[
                    {"symbol": left_symbol, "definition": f"{self.left_group} mean {self.metric_key}"},
                    {"symbol": right_symbol, "definition": f"{self.right_group} mean {self.metric_key}"},
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
            group_symbols = [group_mean_symbol(group) for group in self.groups]
            case_kwargs = dict(base_kwargs)
            case_kwargs["criterion_latex"] = " \\wedge ".join(rf"{symbol} > 0" for symbol in group_symbols)
            case_kwargs["criterion_definitions"] = [
                {"symbol": symbol, "definition": f"{group} mean {self.metric_key}"}
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
        rule: dict[str, Any],
        context: Any,
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

    def quantile_interval_from_row(self, row: dict[str, Any]) -> tuple[float | None, float | None, str | None, str | None]:
        quantile_low = optional_float(row.get(self.quantile_low_field))
        quantile_high = optional_float(row.get(self.quantile_high_field))
        quantile_low_label = str(row.get(self.quantile_low_label_field, "")).strip() or self.quantile_low_field
        quantile_high_label = str(row.get(self.quantile_high_label_field, "")).strip() or self.quantile_high_field
        return quantile_low, quantile_high, quantile_low_label, quantile_high_label


@dataclass(frozen=True)
class ReferenceBandRuleSpec:
    loader: str
    reference_source: str
    group_field: str
    sigma_arg_name: str
    sigma_multiplier: float
    sigma_phrase: str
    suite_name: str
    pass_status: str
    fail_status: str
    properties: dict[str, ReferenceBandPropertyRuleSpec]

    @classmethod
    def from_rule(
        cls,
        rule: dict[str, Any],
        context: Any,
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
            suite_name=str(rule.get("suite_name", rule.get("title", "reference-band-suite"))),
            pass_status=str(rule.get("pass_status", "PASS")),
            fail_status=str(rule.get("fail_status", "FAIL")),
            properties=properties,
        )


@dataclass(frozen=True)
class SeriesComparisonRuleSpec:
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
    def from_rule(cls, rule: dict[str, Any]) -> "SeriesComparisonRuleSpec":
        parser = _SeriesComparisonRuleParser(rule)
        return cls(
            protocol_evidence_key=parser.string("protocol_evidence_key", "fi_curve_rows"),
            reference_spec=parser.data_spec("reference"),
            model_spec=parser.data_spec("model"),
            comparison_x_unit_text=parser.required_unit_text("comparison_x_unit_text"),
            comparison_y_unit_text=parser.required_unit_text("comparison_y_unit_text"),
            x_quantity_name=parser.string("x_quantity_name", "Injected current"),
            y_quantity_name=parser.string("y_quantity_name", "Firing rate"),
            visual_contract=parser.visual_contract(),
            policy=parser.policy(),
        )


@dataclass(frozen=True)
class _SeriesComparisonRuleParser:
    rule: dict[str, Any]

    def string(self, key: str, default: str) -> str:
        return str(self.rule.get(key, default))

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
        if not isinstance(raw, dict):
            raise ValueError(f"{key} must be a table/dict when provided")
        raw_points = raw.get("points", [])
        points: tuple[tuple[float, float], ...] = ()
        if raw_points not in (None, "", []):
            if not isinstance(raw_points, list):
                raise ValueError(f"{key}.points must be a list when provided")
            normalized_points: list[tuple[float, float]] = []
            for index, point in enumerate(raw_points, start=1):
                if isinstance(point, dict):
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
        )

    def data_spec(self, side: str) -> SeriesDataSpec:
        side_prefix = str(side).strip().lower()
        if side_prefix not in {"reference", "model"}:
            raise ValueError(f"Unsupported series-data side {side!r}")
        defaults = {
            "reference": ("current_pA", "firing_rate_Hz", "cell_id"),
            "model": ("current_pA", "firing_rate_Hz", "cell_name"),
        }
        default_x_key, default_y_key, default_series_id_key = defaults[side_prefix]
        return SeriesDataSpec(
            x_key=self.string(f"{side_prefix}_current_key", default_x_key),
            y_key=self.string(f"{side_prefix}_value_key", default_y_key),
            x_unit_text=self.required_unit_text(f"{side_prefix}_x_unit_text"),
            y_unit_text=self.required_unit_text(f"{side_prefix}_y_unit_text"),
            x_transform=self.axis_transform(f"{side_prefix}_x_transform"),
            y_transform=self.axis_transform(f"{side_prefix}_y_transform"),
            series_id_key=self.string(f"{side_prefix}_series_id_key", default_series_id_key),
        )

    def visual_contract(self) -> SeriesVisualContract:
        return SeriesVisualContract(
            x_key=self.string("visual_x_key", "currents_pA"),
            reference_y_key=self.string("visual_reference_y_key", "reference_values_Hz"),
            model_y_key=self.string("visual_model_y_key", "model_values_Hz"),
            kind=self.string("visual_kind", "fi_curve"),
        )

    def policy(self) -> SeriesComparisonPolicy:
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
            interpolation_method = str(self.rule.get("interpolation_method", "linear")).strip() or "linear"
            raw_resampling_grid_values = self.rule.get("resampling_grid_values", [])
            if raw_resampling_grid_values in (None, ""):
                resampling_grid_values = ()
            else:
                if not isinstance(raw_resampling_grid_values, list):
                    raise ValueError("reference_curve_match 'resampling_grid_values' must be a list when provided")
                resampling_grid_values = tuple(float(value) for value in raw_resampling_grid_values)
            if resampling_grid_source == "explicit_grid" and not resampling_grid_values:
                raise ValueError(
                    "reference_curve_match resampled-grid alignment with "
                    "resampling_grid_source='explicit_grid' requires explicit 'resampling_grid_values'"
                )
        else:
            resampling_grid_source = ""
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
            minimum_point_count=int(self.rule.get("minimum_point_count", 1)),
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
            resampling_grid_values=resampling_grid_values,
            interpolation_method=interpolation_method,
            distribution_kind=self.required_choice("distribution_kind"),
            score_family=score_family,
            pvalue_aggregation=pvalue_aggregation,
        )
