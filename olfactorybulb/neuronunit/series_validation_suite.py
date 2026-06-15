"""SciUnit-backed suite bridge for unit-aware series/distribution comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import warnings
from typing import Any

import numpy as np
import quantities as pq
from scipy.stats import t as student_t
from scipy.stats import ttest_ind
import sciunit

from olfactorybulb.audit import series_visual_spec
from olfactorybulb.audit.core import rounded
from olfactorybulb.audit.protocol_evidence import ProtocolEvidenceBundle
from olfactorybulb.neuronunit.capabilities import (
    ProvidesProtocolEvidenceBundle,
)
from olfactorybulb.neuronunit.metric_tables import MetricSummaryTable, MetricTable
from olfactorybulb.neuronunit.provenance import SeriesObservationProvenance, SeriesProvenanceSummary
from olfactorybulb.neuronunit.reference_bands import measurement_with_unit, numeric_value, quantity_unit_for_text
from olfactorybulb.neuronunit.reference_validation_suite import ReferenceValidationModel
from olfactorybulb.neuronunit.suite_presentation import (
    audit_item_adapter_spec_from_case,
    suite_case_result_from_spec,
    suite_items_from_judged,
)
from olfactorybulb.neuronunit.suite_scores import (
    SuiteCaseScorePayload,
    SuiteCaseStatisticalPayload,
    SuiteDescriptor,
)


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _rounded_list(values: list[float]) -> list[float]:
    return [rounded(float(value)) for value in values]


def _sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def _welch_pvalue(reference_values: list[float], model_values: list[float]) -> float:
    if len(reference_values) < 2 or len(model_values) < 2:
        return float("nan")
    reference_array = np.asarray(reference_values, dtype=float)
    model_array = np.asarray(model_values, dtype=float)
    if not np.all(np.isfinite(reference_array)) or not np.all(np.isfinite(model_array)):
        return float("nan")
    if np.allclose(reference_array, reference_array[0]) and np.allclose(model_array, model_array[0]):
        return 1.0 if np.isclose(reference_array[0], model_array[0]) else 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        statistic, pvalue = ttest_ind(reference_array, model_array, equal_var=False, nan_policy="omit")
    del statistic
    if not np.isfinite(pvalue):
        return float("nan")
    return float(pvalue)


def _aggregate_pvalues(pvalues: list[float], *, method: str) -> float:
    if not pvalues:
        return float("nan")
    if method == "max":
        return float(np.max(np.asarray(pvalues, dtype=float)))
    if method == "median":
        return float(np.median(np.asarray(pvalues, dtype=float)))
    raise ValueError(f"Unsupported p-value aggregation method {method!r}")


SERIES_SCORE_FAMILIES = {
    "residual_only",
    "equivalence_only",
    "hybrid_residual_equivalence",
    "welch_only",
    "hybrid_residual_welch",
}

SERIES_ALIGNMENT_POLICIES = {
    "exact_transformed_x",
    "nearest_within_tolerance",
    "tolerance_clusters",
    "resampled_grid",
}

SERIES_RESAMPLING_GRID_SOURCES = {
    "reference_observed_x",
    "model_observed_x",
    "union_observed_x",
    "explicit_grid",
}

SERIES_INTERPOLATION_METHODS = {
    "linear",
    "nearest",
    "step_hold",
}

EQUIVALENCE_SERIES_SCORE_FAMILIES = {
    "equivalence_only",
    "hybrid_residual_equivalence",
}

LEGACY_WELCH_SERIES_SCORE_FAMILIES = {
    "welch_only",
    "hybrid_residual_welch",
}


def _resolved_pvalue_aggregation(policy: "SeriesComparisonPolicy", *, equivalence_family: bool) -> tuple[str, str]:
    declared = str(policy.pvalue_aggregation or "auto").strip().lower()
    if declared in {"", "auto"}:
        return ("max" if equivalence_family else "median"), "auto_default"
    if declared not in {"max", "median"}:
        raise ValueError(
            f"Unsupported series p-value aggregation {policy.pvalue_aggregation!r}; "
            "the current bridge only supports max or median aggregation"
        )
    return declared, "explicit"


def _resolved_equivalence_margin(
    policy: "SeriesComparisonPolicy",
    *,
    equivalence_family: bool,
) -> tuple[float | None, str | None]:
    if not equivalence_family:
        return None, None
    if _is_finite_number(policy.equivalence_margin):
        return float(policy.equivalence_margin), "explicit"
    if _is_finite_number(policy.maximum_mae):
        return float(policy.maximum_mae), "maximum_mae_default"
    return None, None


def _resolved_resampling_grid_source(policy: "SeriesComparisonPolicy") -> tuple[str, str]:
    declared = str(policy.resampling_grid_source or "").strip().lower()
    if declared:
        return declared, "explicit"
    return "union_observed_x", "default_union_observed_x"


def _deterministic_equivalence_pvalue(mean_difference: float, *, equivalence_margin: float) -> float:
    if abs(mean_difference) < float(equivalence_margin):
        return 0.0
    return 1.0


def _welch_satterthwaite_df(variance_a_over_n: float, variance_b_over_n: float, *, n_a: int, n_b: int) -> float:
    numerator = float(variance_a_over_n + variance_b_over_n) ** 2
    denominator = 0.0
    if n_a > 1 and variance_a_over_n > 0.0:
        denominator += float(variance_a_over_n**2) / float(n_a - 1)
    if n_b > 1 and variance_b_over_n > 0.0:
        denominator += float(variance_b_over_n**2) / float(n_b - 1)
    if denominator <= 0.0:
        return float("nan")
    return numerator / denominator


def _one_sample_tost_pvalue(sample_values: list[float], point_target: float, *, equivalence_margin: float) -> float:
    if len(sample_values) < 2:
        return float("nan")
    sample_array = np.asarray(sample_values, dtype=float)
    if not np.all(np.isfinite(sample_array)):
        return float("nan")
    sample_mean = float(np.mean(sample_array))
    sample_sd = _sample_sd([float(value) for value in sample_array])
    mean_difference = sample_mean - float(point_target)
    if sample_sd <= 0.0:
        return _deterministic_equivalence_pvalue(mean_difference, equivalence_margin=equivalence_margin)
    standard_error = sample_sd / math.sqrt(float(len(sample_array)))
    if standard_error <= 0.0:
        return _deterministic_equivalence_pvalue(mean_difference, equivalence_margin=equivalence_margin)
    df = float(len(sample_array) - 1)
    lower_statistic = (mean_difference + float(equivalence_margin)) / standard_error
    upper_statistic = (mean_difference - float(equivalence_margin)) / standard_error
    lower_pvalue = 1.0 - float(student_t.cdf(lower_statistic, df))
    upper_pvalue = float(student_t.cdf(upper_statistic, df))
    return max(lower_pvalue, upper_pvalue)


def _two_sample_welch_tost_pvalue(
    sample_a_values: list[float],
    sample_b_values: list[float],
    *,
    equivalence_margin: float,
) -> float:
    if len(sample_a_values) < 2 or len(sample_b_values) < 2:
        return float("nan")
    sample_a = np.asarray(sample_a_values, dtype=float)
    sample_b = np.asarray(sample_b_values, dtype=float)
    if not np.all(np.isfinite(sample_a)) or not np.all(np.isfinite(sample_b)):
        return float("nan")
    mean_difference = float(np.mean(sample_a) - np.mean(sample_b))
    variance_a = float(np.var(sample_a, ddof=1))
    variance_b = float(np.var(sample_b, ddof=1))
    variance_a_over_n = variance_a / float(len(sample_a))
    variance_b_over_n = variance_b / float(len(sample_b))
    standard_error = math.sqrt(max(variance_a_over_n + variance_b_over_n, 0.0))
    if standard_error <= 0.0:
        return _deterministic_equivalence_pvalue(mean_difference, equivalence_margin=equivalence_margin)
    df = _welch_satterthwaite_df(
        variance_a_over_n,
        variance_b_over_n,
        n_a=len(sample_a),
        n_b=len(sample_b),
    )
    if not _is_finite_number(df) or float(df) <= 0.0:
        return float("nan")
    lower_statistic = (mean_difference + float(equivalence_margin)) / standard_error
    upper_statistic = (mean_difference - float(equivalence_margin)) / standard_error
    lower_pvalue = 1.0 - float(student_t.cdf(lower_statistic, float(df)))
    upper_pvalue = float(student_t.cdf(upper_statistic, float(df)))
    return max(lower_pvalue, upper_pvalue)


def _equivalence_test_result(
    reference_values: list[float],
    model_values: list[float],
    *,
    equivalence_margin: float,
) -> dict[str, Any]:
    if len(reference_values) >= 2 and len(model_values) >= 2:
        return {
            "supported": True,
            "test_kind": "welch_tost",
            "pvalue": _two_sample_welch_tost_pvalue(
                reference_values,
                model_values,
                equivalence_margin=equivalence_margin,
            ),
        }
    if len(reference_values) >= 2 and len(model_values) >= 1:
        return {
            "supported": True,
            "test_kind": "one_sample_reference_tost",
            "pvalue": _one_sample_tost_pvalue(
                reference_values,
                float(np.mean(np.asarray(model_values, dtype=float))),
                equivalence_margin=equivalence_margin,
            ),
        }
    if len(model_values) >= 2 and len(reference_values) >= 1:
        return {
            "supported": True,
            "test_kind": "one_sample_model_tost",
            "pvalue": _one_sample_tost_pvalue(
                model_values,
                float(np.mean(np.asarray(reference_values, dtype=float))),
                equivalence_margin=equivalence_margin,
            ),
        }
    return {"supported": False, "test_kind": "unsupported", "pvalue": float("nan")}


def _norm_ratio_at_most(observed: Any, threshold: Any) -> float | None:
    if not (_is_finite_number(observed) and _is_finite_number(threshold)):
        return None
    observed_value = float(observed)
    threshold_value = float(threshold)
    if threshold_value <= 0.0:
        return None
    if observed_value <= 0.0:
        return 1.0
    return max(0.0, min(1.0, threshold_value / observed_value))


def _norm_ratio_at_least(observed: Any, threshold: Any) -> float | None:
    if not (_is_finite_number(observed) and _is_finite_number(threshold)):
        return None
    observed_value = float(observed)
    threshold_value = float(threshold)
    if threshold_value <= 0.0:
        return None
    return max(0.0, min(1.0, observed_value / threshold_value))


def _residual_norm_score(*, mae: Any, maximum_mae: Any, rmse: Any, maximum_rmse: Any) -> float | None:
    components = [
        component
        for component in (
            _norm_ratio_at_most(mae, maximum_mae),
            _norm_ratio_at_most(rmse, maximum_rmse),
        )
        if component is not None
    ]
    if not components:
        return None
    return min(components)


def _series_statistical_norm_score(
    *,
    score_family: str,
    aggregate_statistical_pvalue: Any,
    equivalence_alpha: Any,
    median_welch_pvalue: Any,
    minimum_median_welch_pvalue: Any,
) -> float | None:
    if score_family in EQUIVALENCE_SERIES_SCORE_FAMILIES:
        return _norm_ratio_at_most(aggregate_statistical_pvalue, equivalence_alpha)
    if score_family in LEGACY_WELCH_SERIES_SCORE_FAMILIES:
        return _norm_ratio_at_least(median_welch_pvalue, minimum_median_welch_pvalue)
    return None


def _series_overall_norm_score(
    *,
    score_family: str,
    residual_norm_score: float | None,
    statistical_norm_score: float | None,
    fallback_status: str,
) -> float:
    if score_family == "residual_only":
        candidate = residual_norm_score
    elif score_family == "equivalence_only":
        candidate = statistical_norm_score
    else:
        components = [component for component in (residual_norm_score, statistical_norm_score) if component is not None]
        candidate = min(components) if components else None
    if candidate is None or not math.isfinite(float(candidate)):
        return 1.0 if str(fallback_status).upper() == "PASS" else 0.0
    return max(0.0, min(1.0, float(candidate)))


def _sorted_piecewise_points(points: tuple[tuple[float, float], ...]) -> list[tuple[float, float]]:
    if len(points) < 2:
        raise ValueError("piecewise_linear transform requires at least two control points")
    normalized = [(float(input_value), float(output_value)) for input_value, output_value in points]
    normalized.sort(key=lambda item: item[0])
    input_values = [input_value for input_value, _output_value in normalized]
    if len(set(input_values)) != len(input_values):
        raise ValueError("piecewise_linear transform control points must have distinct input values")
    return normalized


def _piecewise_linear_value(
    raw_input: float,
    *,
    points: tuple[tuple[float, float], ...],
    extrapolation_mode: str,
) -> float:
    sorted_points = _sorted_piecewise_points(points)
    input_values = [input_value for input_value, _output_value in sorted_points]
    output_values = [output_value for _input_value, output_value in sorted_points]
    if input_values[0] <= float(raw_input) <= input_values[-1]:
        return float(np.interp(float(raw_input), np.asarray(input_values, dtype=float), np.asarray(output_values, dtype=float)))

    mode = str(extrapolation_mode or "forbid").strip().lower()
    if mode == "forbid":
        raise ValueError(
            f"piecewise_linear transform cannot extrapolate input value {float(raw_input):g}; "
            f"supported domain is [{input_values[0]:g}, {input_values[-1]:g}]"
        )
    if mode == "constant":
        return float(output_values[0] if float(raw_input) < input_values[0] else output_values[-1])
    if mode == "linear":
        if float(raw_input) < input_values[0]:
            left_a, left_b = sorted_points[0], sorted_points[1]
        else:
            left_a, left_b = sorted_points[-2], sorted_points[-1]
        input_span = float(left_b[0] - left_a[0])
        if input_span == 0.0:
            raise ValueError("piecewise_linear transform extrapolation requires distinct neighboring input values")
        slope = float(left_b[1] - left_a[1]) / input_span
        return float(left_a[1]) + slope * (float(raw_input) - float(left_a[0]))
    raise ValueError(
        f"Unsupported piecewise_linear extrapolation mode {extrapolation_mode!r}; "
        "expected one of forbid, constant, linear"
    )


def _nested_mapping_value(mapping: dict[str, Any] | None, dotted_key: str) -> Any:
    if not mapping:
        return None
    direct_key = str(dotted_key or "").strip()
    if not direct_key:
        return None
    if direct_key in mapping:
        return mapping.get(direct_key)
    current: Any = mapping
    for token in direct_key.split("."):
        if not isinstance(current, dict) or token not in current:
            return None
        current = current[token]
    return current


def _lookup_transform_numeric_value(
    lookup_key: str,
    *,
    row: dict[str, Any] | None,
    context: dict[str, Any] | None,
) -> float:
    normalized_key = str(lookup_key or "").strip()
    if not normalized_key:
        raise ValueError("Transform lookup keys must be non-empty")
    for mapping_name, mapping in (("row", row), ("context", context)):
        candidate = _nested_mapping_value(mapping, normalized_key)
        if candidate is None or isinstance(candidate, bool):
            continue
        try:
            numeric = numeric_value(candidate) if isinstance(candidate, pq.Quantity) else float(candidate)
        except (TypeError, ValueError):
            raise ValueError(
                f"Transform lookup key {normalized_key!r} resolved from {mapping_name} metadata "
                f"but did not contain a finite numeric value"
            ) from None
        if not math.isfinite(numeric):
            raise ValueError(
                f"Transform lookup key {normalized_key!r} resolved from {mapping_name} metadata "
                "but did not contain a finite numeric value"
            )
        return float(numeric)
    raise ValueError(
        f"Transform lookup key {normalized_key!r} was not found in the available row/context metadata"
    )


@dataclass(frozen=True)
class AxisTransform:
    kind: str = "identity"
    scale: float = 1.0
    offset: float = 0.0
    input_unit_text: str = ""
    output_unit_text: str = ""
    points: tuple[tuple[float, float], ...] = ()
    extrapolation_mode: str = "forbid"
    scale_lookup_key: str = ""
    offset_lookup_key: str = ""

    def apply(
        self,
        raw_value: float,
        *,
        source_unit_text: str,
        comparison_unit_text: str,
        row: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> float | pq.Quantity:
        kind = str(self.kind or "identity").strip().lower()
        input_unit_text = str(self.input_unit_text or source_unit_text or "").strip()
        output_unit_text = str(self.output_unit_text or input_unit_text or "").strip()

        if kind == "identity":
            measurement = measurement_with_unit(float(raw_value), source_unit_text)
            return _coerce_to_comparison_unit(
                measurement,
                fallback_unit_text=input_unit_text or source_unit_text,
                comparison_unit_text=comparison_unit_text,
            )

        if kind == "affine":
            if input_unit_text and source_unit_text:
                measurement = measurement_with_unit(float(raw_value), source_unit_text)
                input_unit = quantity_unit_for_text(input_unit_text)
                if isinstance(measurement, pq.Quantity) and input_unit is not None:
                    try:
                        base_numeric = float(measurement.rescale(input_unit).magnitude)
                    except Exception as exc:  # pragma: no cover - defensive path
                        raise ValueError(
                            f"Cannot rescale source unit {source_unit_text!r} into affine-transform input unit {input_unit_text!r}"
                        ) from exc
                else:
                    base_numeric = float(raw_value)
            else:
                base_numeric = float(raw_value)
            transformed = float(self.scale) * base_numeric + float(self.offset)
            measurement = measurement_with_unit(transformed, output_unit_text)
            return _coerce_to_comparison_unit(
                measurement,
                fallback_unit_text=output_unit_text,
                comparison_unit_text=comparison_unit_text,
            )

        if kind == "affine_lookup":
            if input_unit_text and source_unit_text:
                measurement = measurement_with_unit(float(raw_value), source_unit_text)
                input_unit = quantity_unit_for_text(input_unit_text)
                if isinstance(measurement, pq.Quantity) and input_unit is not None:
                    try:
                        base_numeric = float(measurement.rescale(input_unit).magnitude)
                    except Exception as exc:  # pragma: no cover - defensive path
                        raise ValueError(
                            f"Cannot rescale source unit {source_unit_text!r} into affine-lookup input unit {input_unit_text!r}"
                        ) from exc
                else:
                    base_numeric = float(raw_value)
            else:
                base_numeric = float(raw_value)
            lookup_scale = 1.0
            lookup_offset = 0.0
            if str(self.scale_lookup_key or "").strip():
                lookup_scale = _lookup_transform_numeric_value(
                    self.scale_lookup_key,
                    row=row,
                    context=context,
                )
            if str(self.offset_lookup_key or "").strip():
                lookup_offset = _lookup_transform_numeric_value(
                    self.offset_lookup_key,
                    row=row,
                    context=context,
                )
            transformed = float(self.scale) * base_numeric * float(lookup_scale) + float(self.offset) + float(lookup_offset)
            measurement = measurement_with_unit(transformed, output_unit_text)
            return _coerce_to_comparison_unit(
                measurement,
                fallback_unit_text=output_unit_text,
                comparison_unit_text=comparison_unit_text,
            )

        if kind == "piecewise_linear":
            if input_unit_text and source_unit_text:
                measurement = measurement_with_unit(float(raw_value), source_unit_text)
                input_unit = quantity_unit_for_text(input_unit_text)
                if isinstance(measurement, pq.Quantity) and input_unit is not None:
                    try:
                        base_numeric = float(measurement.rescale(input_unit).magnitude)
                    except Exception as exc:  # pragma: no cover - defensive path
                        raise ValueError(
                            f"Cannot rescale source unit {source_unit_text!r} into piecewise-linear input unit {input_unit_text!r}"
                        ) from exc
                else:
                    base_numeric = float(raw_value)
            else:
                base_numeric = float(raw_value)
            transformed = _piecewise_linear_value(
                base_numeric,
                points=self.points,
                extrapolation_mode=self.extrapolation_mode,
            )
            measurement = measurement_with_unit(transformed, output_unit_text)
            return _coerce_to_comparison_unit(
                measurement,
                fallback_unit_text=output_unit_text,
                comparison_unit_text=comparison_unit_text,
            )

        raise ValueError(f"Unsupported axis transform kind {self.kind!r}")

    def description(self) -> str:
        kind = str(self.kind or "identity").strip().lower()
        if kind == "identity":
            return "identity"
        if kind == "affine":
            return (
                f"affine(scale={float(self.scale):g}, offset={float(self.offset):g}, "
                f"input_unit={self.input_unit_text or '-'}, output_unit={self.output_unit_text or '-'})"
            )
        if kind == "affine_lookup":
            return (
                f"affine_lookup(scale={float(self.scale):g}, offset={float(self.offset):g}, "
                f"scale_lookup={self.scale_lookup_key or '-'}, offset_lookup={self.offset_lookup_key or '-'}, "
                f"input_unit={self.input_unit_text or '-'}, output_unit={self.output_unit_text or '-'})"
            )
        if kind == "piecewise_linear":
            return (
                f"piecewise_linear(points={len(self.points)}, extrapolation={self.extrapolation_mode or 'forbid'}, "
                f"input_unit={self.input_unit_text or '-'}, output_unit={self.output_unit_text or '-'})"
            )
        return kind


def _coerce_to_comparison_unit(
    measurement: float | pq.Quantity,
    *,
    fallback_unit_text: str,
    comparison_unit_text: str,
) -> float | pq.Quantity:
    if isinstance(measurement, pq.Quantity):
        comparison_unit = quantity_unit_for_text(comparison_unit_text)
        if comparison_unit is None:
            return float(measurement.magnitude)
        try:
            return measurement.rescale(comparison_unit)
        except Exception as exc:
            raise ValueError(
                f"Cannot rescale quantity into comparison unit {comparison_unit_text!r}"
            ) from exc
    if not comparison_unit_text:
        return float(measurement)
    comparison_unit = quantity_unit_for_text(comparison_unit_text)
    if comparison_unit is None:
        return float(measurement)
    if fallback_unit_text:
        source_unit = quantity_unit_for_text(fallback_unit_text)
        if source_unit is not None:
            try:
                return (float(measurement) * source_unit).rescale(comparison_unit)
            except Exception as exc:
                raise ValueError(
                    f"Cannot rescale fallback unit {fallback_unit_text!r} into comparison unit {comparison_unit_text!r}"
                ) from exc
    return float(measurement) * comparison_unit


@dataclass(frozen=True)
class SeriesComparisonPolicy:
    minimum_point_count: int = 1
    maximum_mae: float = float("inf")
    maximum_rmse: float = float("inf")
    minimum_median_welch_pvalue: float | None = None
    equivalence_margin: float | None = None
    equivalence_alpha: float = 0.05
    x_precision_digits: int = 6
    alignment_policy: str = "exact_transformed_x"
    x_match_tolerance: float | None = None
    resampling_grid_source: str = ""
    resampling_grid_values: tuple[float, ...] = ()
    interpolation_method: str = "linear"
    distribution_kind: str = "empirical_by_x"
    score_family: str = "residual_only"
    pvalue_aggregation: str = "auto"


@dataclass(frozen=True)
class SeriesDataSpec:
    x_key: str
    y_key: str
    x_unit_text: str
    y_unit_text: str
    x_transform: AxisTransform = field(default_factory=AxisTransform)
    y_transform: AxisTransform = field(default_factory=AxisTransform)
    series_id_key: str = ""

    def bins(
        self,
        rows: list[dict[str, Any]],
        *,
        comparison_x_unit_text: str,
        comparison_y_unit_text: str,
        precision_digits: int,
        context: dict[str, Any] | None = None,
    ) -> dict[float, list[float]]:
        return _series_bins(
            rows,
            x_key=self.x_key,
            y_key=self.y_key,
            x_unit_text=self.x_unit_text,
            y_unit_text=self.y_unit_text,
            comparison_x_unit_text=comparison_x_unit_text,
            comparison_y_unit_text=comparison_y_unit_text,
            x_transform=self.x_transform,
            y_transform=self.y_transform,
            precision_digits=precision_digits,
            context=context,
        )

    def paths(
        self,
        rows: list[dict[str, Any]],
        *,
        comparison_x_unit_text: str,
        comparison_y_unit_text: str,
        precision_digits: int,
        context: dict[str, Any] | None = None,
    ) -> dict[str, list[tuple[float, float]]]:
        return _series_paths(
            rows,
            series_id_key=self.series_id_key,
            x_key=self.x_key,
            y_key=self.y_key,
            x_unit_text=self.x_unit_text,
            y_unit_text=self.y_unit_text,
            comparison_x_unit_text=comparison_x_unit_text,
            comparison_y_unit_text=comparison_y_unit_text,
            x_transform=self.x_transform,
            y_transform=self.y_transform,
            precision_digits=precision_digits,
            context=context,
        )

    def provenance_summary(
        self,
        rows: list[dict[str, Any]],
        *,
        context: dict[str, Any] | None = None,
        exclude_context_keys: set[str] | None = None,
    ) -> SeriesProvenanceSummary:
        return SeriesProvenanceSummary.from_rows(
            rows,
            series_id_key=self.series_id_key,
            x_key=self.x_key,
            y_key=self.y_key,
            x_unit_text=self.x_unit_text,
            y_unit_text=self.y_unit_text,
            context=context,
            exclude_context_keys=exclude_context_keys,
        )


@dataclass(frozen=True)
class SeriesVisualContract:
    x_key: str = "currents_pA"
    reference_y_key: str = "reference_values_Hz"
    model_y_key: str = "model_values_Hz"
    kind: str = "fi_curve"


@dataclass(frozen=True)
class SeriesObservedDataset:
    rows: list[dict[str, Any]]
    spec: SeriesDataSpec
    comparison_x_unit_text: str
    comparison_y_unit_text: str
    context: dict[str, Any] = field(default_factory=dict)
    exclude_provenance_context_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", list(self.rows))
        object.__setattr__(self, "context", dict(self.context))
        object.__setattr__(
            self,
            "exclude_provenance_context_keys",
            tuple(
                str(key).strip()
                for key in self.exclude_provenance_context_keys
                if str(key).strip()
            ),
        )

    @property
    def x_key(self) -> str:
        return self.spec.x_key

    @property
    def y_key(self) -> str:
        return self.spec.y_key

    @property
    def x_unit_text(self) -> str:
        return self.spec.x_unit_text

    @property
    def y_unit_text(self) -> str:
        return self.spec.y_unit_text

    @property
    def series_id_key(self) -> str:
        return self.spec.series_id_key

    @property
    def x_transform(self) -> AxisTransform:
        return self.spec.x_transform

    @property
    def y_transform(self) -> AxisTransform:
        return self.spec.y_transform

    def bins(self, *, precision_digits: int) -> dict[float, list[float]]:
        return self.spec.bins(
            self.rows,
            comparison_x_unit_text=self.comparison_x_unit_text,
            comparison_y_unit_text=self.comparison_y_unit_text,
            precision_digits=precision_digits,
            context=self.context or None,
        )

    def paths(self, *, precision_digits: int) -> dict[str, list[tuple[float, float]]]:
        return self.spec.paths(
            self.rows,
            comparison_x_unit_text=self.comparison_x_unit_text,
            comparison_y_unit_text=self.comparison_y_unit_text,
            precision_digits=precision_digits,
            context=self.context or None,
        )

    def provenance_summary(self) -> SeriesProvenanceSummary:
        return self.spec.provenance_summary(
            self.rows,
            context=self.context or None,
            exclude_context_keys=set(self.exclude_provenance_context_keys) or None,
        )


@dataclass(frozen=True)
class SeriesDistributionObservation:
    protocol_evidence_key: str
    reference_rows: list[dict[str, Any]]
    reference_spec: SeriesDataSpec
    model_spec: SeriesDataSpec
    comparison_x_unit_text: str
    comparison_y_unit_text: str
    x_quantity_name: str = "series x-value"
    y_quantity_name: str = "series y-value"
    visual_contract: SeriesVisualContract = field(default_factory=SeriesVisualContract)
    policy: SeriesComparisonPolicy = field(default_factory=SeriesComparisonPolicy)

    @property
    def reference_x_key(self) -> str:
        return self.reference_spec.x_key

    @property
    def reference_y_key(self) -> str:
        return self.reference_spec.y_key

    @property
    def model_x_key(self) -> str:
        return self.model_spec.x_key

    @property
    def model_y_key(self) -> str:
        return self.model_spec.y_key

    @property
    def reference_x_unit_text(self) -> str:
        return self.reference_spec.x_unit_text

    @property
    def reference_y_unit_text(self) -> str:
        return self.reference_spec.y_unit_text

    @property
    def model_x_unit_text(self) -> str:
        return self.model_spec.x_unit_text

    @property
    def model_y_unit_text(self) -> str:
        return self.model_spec.y_unit_text

    @property
    def reference_x_transform(self) -> AxisTransform:
        return self.reference_spec.x_transform

    @property
    def reference_y_transform(self) -> AxisTransform:
        return self.reference_spec.y_transform

    @property
    def model_x_transform(self) -> AxisTransform:
        return self.model_spec.x_transform

    @property
    def model_y_transform(self) -> AxisTransform:
        return self.model_spec.y_transform

    @property
    def reference_series_id_key(self) -> str:
        return self.reference_spec.series_id_key

    @property
    def model_series_id_key(self) -> str:
        return self.model_spec.series_id_key

    @property
    def visual_x_key(self) -> str:
        return self.visual_contract.x_key

    @property
    def visual_reference_y_key(self) -> str:
        return self.visual_contract.reference_y_key

    @property
    def visual_model_y_key(self) -> str:
        return self.visual_contract.model_y_key

    @property
    def visual_kind(self) -> str:
        return self.visual_contract.kind

    def reference_dataset(self) -> SeriesObservedDataset:
        return SeriesObservedDataset(
            rows=self.reference_rows,
            spec=self.reference_spec,
            comparison_x_unit_text=self.comparison_x_unit_text,
            comparison_y_unit_text=self.comparison_y_unit_text,
        )

    def model_dataset(self, prediction: "SeriesPredictionBundle") -> SeriesObservedDataset:
        return SeriesObservedDataset(
            rows=prediction.rows,
            spec=self.model_spec,
            comparison_x_unit_text=self.comparison_x_unit_text,
            comparison_y_unit_text=self.comparison_y_unit_text,
            context=prediction.context,
            exclude_provenance_context_keys=(self.protocol_evidence_key,),
        )

    def observation_payload(self) -> dict[str, Any]:
        return {
            "protocol_evidence_key": self.protocol_evidence_key,
            "reference_x_key": self.reference_x_key,
            "reference_y_key": self.reference_y_key,
            "model_x_key": self.model_x_key,
            "model_y_key": self.model_y_key,
            "comparison_x_unit_text": self.comparison_x_unit_text,
            "comparison_y_unit_text": self.comparison_y_unit_text,
            "equivalence_margin": self.policy.equivalence_margin,
            "equivalence_alpha": self.policy.equivalence_alpha,
            "alignment_policy": self.policy.alignment_policy,
            "x_match_tolerance": self.policy.x_match_tolerance,
            "resampling_grid_source": self.policy.resampling_grid_source,
            "resampling_grid_values": self.policy.resampling_grid_values,
            "interpolation_method": self.policy.interpolation_method,
            "distribution_kind": self.policy.distribution_kind,
            "score_family": self.policy.score_family,
            "pvalue_aggregation": self.policy.pvalue_aggregation,
        }


@dataclass(frozen=True)
class SeriesComparisonCase:
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
    observation: SeriesDistributionObservation
    pass_status: str = "PASS"
    fail_status: str = "FAIL"


def _coerced_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerced_float_sequence(values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    result: list[float] = []
    for value in values:
        candidate = _coerced_float(value)
        if candidate is None:
            continue
        result.append(float(candidate))
    return tuple(result)


def _coerced_optional_float_sequence(
    values: list[float | None] | tuple[float | None, ...],
) -> tuple[float | None, ...]:
    result: list[float | None] = []
    for value in values:
        candidate = _coerced_float(value)
        result.append(candidate)
    return tuple(result)


def _coerced_nested_float_sequences(
    values: list[list[float]] | tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(_coerced_float_sequence(list(group)) for group in values)


def _coerced_int_sequence(values: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return tuple(result)


def _rounded_float_or_raw(value: float | None) -> float | None:
    if value is None:
        return None
    if _is_finite_number(value):
        return rounded(float(value))
    return float(value)


def _rounded_optional_float_list(values: tuple[float | None, ...]) -> list[float | None]:
    return [_rounded_float_or_raw(value) for value in values]


@dataclass(frozen=True)
class SeriesComparisonEvidencePayload:
    visual_x_key: str
    visual_reference_y_key: str
    visual_model_y_key: str
    visual_x_values: tuple[float, ...]
    reference_matched_x_values: tuple[float, ...]
    model_matched_x_values: tuple[float, ...]
    matched_x_differences: tuple[float, ...]
    reference_cluster_x_groups: tuple[tuple[float, ...], ...] = ()
    model_cluster_x_groups: tuple[tuple[float, ...], ...] = ()
    reference_values: tuple[float, ...] = ()
    model_values: tuple[float, ...] = ()
    reference_sd_values: tuple[float, ...] = ()
    model_sd_values: tuple[float, ...] = ()
    reference_count_values: tuple[int, ...] = ()
    model_count_values: tuple[int, ...] = ()
    matched_point_count: int = 0
    mean_absolute_error: float | None = None
    root_mean_square_error: float | None = None
    max_absolute_error: float | None = None
    maximum_mae: float | None = None
    maximum_rmse: float | None = None
    error_unit_text: str = ""
    score_family: str = "residual_only"
    declared_pvalue_aggregation: str = "auto"
    welch_pvalues: tuple[float | None, ...] = ()
    median_welch_pvalue: float | None = None
    minimum_median_welch_pvalue: float | None = None
    finite_welch_pvalue_count: int = 0
    equivalence_margin: float | None = None
    declared_equivalence_margin: float | None = None
    equivalence_margin_source: str | None = None
    equivalence_alpha: float | None = None
    statistical_test_family: str = "none"
    statistical_test_kinds: tuple[str, ...] = ()
    statistical_pvalues: tuple[float | None, ...] = ()
    aggregate_statistical_pvalue: float | None = None
    supported_statistical_bin_count: int = 0
    unsupported_statistical_x_values: tuple[float, ...] = ()
    pvalue_aggregation: str = ""
    pvalue_aggregation_source: str = ""
    residual_gate_passed: bool = False
    pvalue_gate_passed: bool = False
    statistical_gate_passed: bool = False
    residual_norm_score: float | None = None
    statistical_norm_score: float | None = None
    overall_norm_score: float | None = None
    alignment_policy: str = ""
    x_match_tolerance: float | None = None
    declared_resampling_grid_source: str = ""
    resampling_grid_source: str = ""
    resampling_grid_source_origin: str = ""
    declared_resampling_grid_values: tuple[float, ...] = ()
    resampling_grid_values: tuple[float, ...] = ()
    interpolation_method: str = "linear"
    reference_resampled_support_counts: tuple[int, ...] = ()
    model_resampled_support_counts: tuple[int, ...] = ()
    distribution_kind: str = "empirical_by_x"
    x_quantity_name: str = "series x-value"
    y_quantity_name: str = "series y-value"
    comparison_x_unit_text: str = ""
    comparison_y_unit_text: str = ""
    reference_x_key: str = ""
    reference_y_key: str = ""
    reference_x_unit_text: str = ""
    reference_y_unit_text: str = ""
    reference_series_id_key: str = ""
    model_x_key: str = ""
    model_y_key: str = ""
    model_x_unit_text: str = ""
    model_y_unit_text: str = ""
    model_series_id_key: str = ""
    reference_x_transform: str = ""
    model_x_transform: str = ""
    series_provenance: SeriesObservationProvenance = field(default_factory=SeriesObservationProvenance)

    def __post_init__(self) -> None:
        object.__setattr__(self, "visual_x_key", str(self.visual_x_key).strip())
        object.__setattr__(self, "visual_reference_y_key", str(self.visual_reference_y_key).strip())
        object.__setattr__(self, "visual_model_y_key", str(self.visual_model_y_key).strip())
        object.__setattr__(self, "visual_x_values", _coerced_float_sequence(self.visual_x_values))
        object.__setattr__(self, "reference_matched_x_values", _coerced_float_sequence(self.reference_matched_x_values))
        object.__setattr__(self, "model_matched_x_values", _coerced_float_sequence(self.model_matched_x_values))
        object.__setattr__(self, "matched_x_differences", _coerced_float_sequence(self.matched_x_differences))
        object.__setattr__(self, "reference_cluster_x_groups", _coerced_nested_float_sequences(self.reference_cluster_x_groups))
        object.__setattr__(self, "model_cluster_x_groups", _coerced_nested_float_sequences(self.model_cluster_x_groups))
        object.__setattr__(self, "reference_values", _coerced_float_sequence(self.reference_values))
        object.__setattr__(self, "model_values", _coerced_float_sequence(self.model_values))
        object.__setattr__(self, "reference_sd_values", _coerced_float_sequence(self.reference_sd_values))
        object.__setattr__(self, "model_sd_values", _coerced_float_sequence(self.model_sd_values))
        object.__setattr__(self, "reference_count_values", _coerced_int_sequence(self.reference_count_values))
        object.__setattr__(self, "model_count_values", _coerced_int_sequence(self.model_count_values))
        object.__setattr__(self, "matched_point_count", int(self.matched_point_count))
        object.__setattr__(self, "mean_absolute_error", _coerced_float(self.mean_absolute_error))
        object.__setattr__(self, "root_mean_square_error", _coerced_float(self.root_mean_square_error))
        object.__setattr__(self, "max_absolute_error", _coerced_float(self.max_absolute_error))
        object.__setattr__(self, "maximum_mae", _coerced_float(self.maximum_mae))
        object.__setattr__(self, "maximum_rmse", _coerced_float(self.maximum_rmse))
        object.__setattr__(self, "error_unit_text", str(self.error_unit_text).strip())
        object.__setattr__(self, "score_family", str(self.score_family).strip())
        object.__setattr__(self, "declared_pvalue_aggregation", str(self.declared_pvalue_aggregation).strip())
        object.__setattr__(self, "welch_pvalues", _coerced_optional_float_sequence(self.welch_pvalues))
        object.__setattr__(self, "median_welch_pvalue", _coerced_float(self.median_welch_pvalue))
        object.__setattr__(self, "minimum_median_welch_pvalue", _coerced_float(self.minimum_median_welch_pvalue))
        object.__setattr__(self, "finite_welch_pvalue_count", int(self.finite_welch_pvalue_count))
        object.__setattr__(self, "equivalence_margin", _coerced_float(self.equivalence_margin))
        object.__setattr__(self, "declared_equivalence_margin", _coerced_float(self.declared_equivalence_margin))
        object.__setattr__(self, "equivalence_margin_source", str(self.equivalence_margin_source or "").strip() or None)
        object.__setattr__(self, "equivalence_alpha", _coerced_float(self.equivalence_alpha))
        object.__setattr__(self, "statistical_test_family", str(self.statistical_test_family).strip())
        object.__setattr__(
            self,
            "statistical_test_kinds",
            tuple(str(value).strip() for value in self.statistical_test_kinds if str(value).strip()),
        )
        object.__setattr__(self, "statistical_pvalues", _coerced_optional_float_sequence(self.statistical_pvalues))
        object.__setattr__(self, "aggregate_statistical_pvalue", _coerced_float(self.aggregate_statistical_pvalue))
        object.__setattr__(self, "supported_statistical_bin_count", int(self.supported_statistical_bin_count))
        object.__setattr__(self, "unsupported_statistical_x_values", _coerced_float_sequence(self.unsupported_statistical_x_values))
        object.__setattr__(self, "pvalue_aggregation", str(self.pvalue_aggregation).strip())
        object.__setattr__(self, "pvalue_aggregation_source", str(self.pvalue_aggregation_source).strip())
        object.__setattr__(self, "residual_gate_passed", bool(self.residual_gate_passed))
        object.__setattr__(self, "pvalue_gate_passed", bool(self.pvalue_gate_passed))
        object.__setattr__(self, "statistical_gate_passed", bool(self.statistical_gate_passed))
        object.__setattr__(self, "residual_norm_score", _coerced_float(self.residual_norm_score))
        object.__setattr__(self, "statistical_norm_score", _coerced_float(self.statistical_norm_score))
        object.__setattr__(self, "overall_norm_score", _coerced_float(self.overall_norm_score))
        object.__setattr__(self, "alignment_policy", str(self.alignment_policy).strip())
        object.__setattr__(self, "x_match_tolerance", _coerced_float(self.x_match_tolerance))
        object.__setattr__(self, "declared_resampling_grid_source", str(self.declared_resampling_grid_source).strip())
        object.__setattr__(self, "resampling_grid_source", str(self.resampling_grid_source).strip())
        object.__setattr__(self, "resampling_grid_source_origin", str(self.resampling_grid_source_origin).strip())
        object.__setattr__(self, "declared_resampling_grid_values", _coerced_float_sequence(self.declared_resampling_grid_values))
        object.__setattr__(self, "resampling_grid_values", _coerced_float_sequence(self.resampling_grid_values))
        object.__setattr__(self, "interpolation_method", str(self.interpolation_method).strip())
        object.__setattr__(self, "reference_resampled_support_counts", _coerced_int_sequence(self.reference_resampled_support_counts))
        object.__setattr__(self, "model_resampled_support_counts", _coerced_int_sequence(self.model_resampled_support_counts))
        object.__setattr__(self, "distribution_kind", str(self.distribution_kind).strip())
        object.__setattr__(self, "x_quantity_name", str(self.x_quantity_name).strip())
        object.__setattr__(self, "y_quantity_name", str(self.y_quantity_name).strip())
        object.__setattr__(self, "comparison_x_unit_text", str(self.comparison_x_unit_text).strip())
        object.__setattr__(self, "comparison_y_unit_text", str(self.comparison_y_unit_text).strip())
        object.__setattr__(self, "reference_x_key", str(self.reference_x_key).strip())
        object.__setattr__(self, "reference_y_key", str(self.reference_y_key).strip())
        object.__setattr__(self, "reference_x_unit_text", str(self.reference_x_unit_text).strip())
        object.__setattr__(self, "reference_y_unit_text", str(self.reference_y_unit_text).strip())
        object.__setattr__(self, "reference_series_id_key", str(self.reference_series_id_key).strip())
        object.__setattr__(self, "model_x_key", str(self.model_x_key).strip())
        object.__setattr__(self, "model_y_key", str(self.model_y_key).strip())
        object.__setattr__(self, "model_x_unit_text", str(self.model_x_unit_text).strip())
        object.__setattr__(self, "model_y_unit_text", str(self.model_y_unit_text).strip())
        object.__setattr__(self, "model_series_id_key", str(self.model_series_id_key).strip())
        object.__setattr__(self, "reference_x_transform", str(self.reference_x_transform).strip())
        object.__setattr__(self, "model_x_transform", str(self.model_x_transform).strip())

    @property
    def aggregate_pvalue_for_display(self) -> float | None:
        if self.score_family in EQUIVALENCE_SERIES_SCORE_FAMILIES:
            return self.aggregate_statistical_pvalue
        if self.score_family in LEGACY_WELCH_SERIES_SCORE_FAMILIES:
            return self.median_welch_pvalue
        return self.aggregate_statistical_pvalue

    @property
    def norm_score(self) -> float | None:
        if _is_finite_number(self.overall_norm_score):
            return max(0.0, min(1.0, float(self.overall_norm_score)))
        return None

    def score_text(self) -> str:
        residual_text = ""
        if _is_finite_number(self.mean_absolute_error):
            residual_text = f"MAE {rounded(float(self.mean_absolute_error)):g}" + (
                f" {self.error_unit_text}" if self.error_unit_text else ""
            )
        aggregate_pvalue = self.aggregate_pvalue_for_display
        statistical_text = ""
        if _is_finite_number(aggregate_pvalue):
            pvalue_text = f"{rounded(float(aggregate_pvalue), digits=4):g}"
            if self.score_family in EQUIVALENCE_SERIES_SCORE_FAMILIES:
                statistical_text = f"TOST p {pvalue_text}"
            elif self.score_family in LEGACY_WELCH_SERIES_SCORE_FAMILIES:
                statistical_text = f"Welch p {pvalue_text}"
            else:
                statistical_text = f"p {pvalue_text}"
        if self.score_family == "residual_only":
            return residual_text
        if self.score_family == "equivalence_only":
            return statistical_text
        if residual_text and statistical_text:
            return f"{residual_text} | {statistical_text}"
        return residual_text or statistical_text

    def suite_case_statistical_payload(self) -> SuiteCaseStatisticalPayload | None:
        if self.score_family in EQUIVALENCE_SERIES_SCORE_FAMILIES and _is_finite_number(self.aggregate_statistical_pvalue):
            return SuiteCaseStatisticalPayload(
                score_family_category="equivalence",
                statistical_test_family=self.statistical_test_family or "equivalence_tost",
                pvalue=float(self.aggregate_statistical_pvalue),
                label="TOST p",
                default_rollup_method="max",
                threshold=self.equivalence_alpha,
                threshold_key="equivalence_alpha",
                threshold_direction="le",
            )
        if self.score_family in LEGACY_WELCH_SERIES_SCORE_FAMILIES and _is_finite_number(self.median_welch_pvalue):
            return SuiteCaseStatisticalPayload(
                score_family_category="welch_similarity",
                statistical_test_family=self.statistical_test_family or "legacy_welch_difference",
                pvalue=float(self.median_welch_pvalue),
                label="Welch p",
                default_rollup_method="min",
                threshold=self.minimum_median_welch_pvalue,
                threshold_key="minimum_median_welch_pvalue",
                threshold_direction="ge",
            )
        return None

    def case_weight(self) -> float | None:
        if self.matched_point_count <= 0:
            return None
        return float(self.matched_point_count)

    def to_suite_case_score_payload(self, *, score_value: float | int | None) -> SuiteCaseScorePayload:
        score_units = ""
        if self.score_family == "residual_only":
            score_units = self.error_unit_text
        if self.score_family in {"equivalence_only", "hybrid_residual_equivalence", "welch_only", "hybrid_residual_welch"}:
            score_units = ""
        return SuiteCaseScorePayload(
            score_kind=self.score_family,
            score_value=numeric_value(score_value) if score_value is not None else None,
            score_units=score_units,
            score_interpretation=(
                "Unit-aware series-comparison score derived from the configured alignment, "
                "distribution, and statistical policy."
            ),
            observation={
                "matched_point_count": self.matched_point_count,
                "mean_absolute_error": _rounded_float_or_raw(self.mean_absolute_error),
                "root_mean_square_error": _rounded_float_or_raw(self.root_mean_square_error),
                "aggregate_statistical_pvalue": _rounded_float_or_raw(self.aggregate_statistical_pvalue),
                "median_welch_pvalue": _rounded_float_or_raw(self.median_welch_pvalue),
            },
            prediction={
                "score_family": self.score_family,
                "statistical_test_family": self.statistical_test_family,
                "maximum_mae": _rounded_float_or_raw(self.maximum_mae),
                "maximum_rmse": _rounded_float_or_raw(self.maximum_rmse),
                "equivalence_margin": _rounded_float_or_raw(self.equivalence_margin),
                "equivalence_alpha": _rounded_float_or_raw(self.equivalence_alpha),
                "minimum_median_welch_pvalue": _rounded_float_or_raw(self.minimum_median_welch_pvalue),
                "pvalue_aggregation": self.pvalue_aggregation,
            },
            normalization={
                "norm_score": self.norm_score,
                "residual_norm_score": _rounded_float_or_raw(self.residual_norm_score),
                "statistical_norm_score": _rounded_float_or_raw(self.statistical_norm_score),
                "overall_norm_score": _rounded_float_or_raw(self.overall_norm_score),
            },
            statistical_summary=self.suite_case_statistical_payload(),
        )

    def to_dict(self) -> dict[str, Any]:
        evidence = {
            self.visual_x_key: _rounded_list(list(self.visual_x_values)),
            "reference_matched_x_values": _rounded_list(list(self.reference_matched_x_values)),
            "model_matched_x_values": _rounded_list(list(self.model_matched_x_values)),
            "matched_x_differences": _rounded_list(list(self.matched_x_differences)),
            "reference_cluster_x_groups": [_rounded_list(list(values)) for values in self.reference_cluster_x_groups],
            "model_cluster_x_groups": [_rounded_list(list(values)) for values in self.model_cluster_x_groups],
            self.visual_reference_y_key: _rounded_list(list(self.reference_values)),
            self.visual_model_y_key: _rounded_list(list(self.model_values)),
            "reference_sd_values": _rounded_list(list(self.reference_sd_values)),
            "model_sd_values": _rounded_list(list(self.model_sd_values)),
            "reference_count_values": list(self.reference_count_values),
            "model_count_values": list(self.model_count_values),
            "matched_point_count": self.matched_point_count,
            "mean_absolute_error": _rounded_float_or_raw(self.mean_absolute_error),
            "root_mean_square_error": _rounded_float_or_raw(self.root_mean_square_error),
            "max_absolute_error": _rounded_float_or_raw(self.max_absolute_error),
            "maximum_mae": _rounded_float_or_raw(self.maximum_mae),
            "maximum_rmse": _rounded_float_or_raw(self.maximum_rmse),
            "error_unit_text": self.error_unit_text,
            "score_family": self.score_family,
            "declared_pvalue_aggregation": self.declared_pvalue_aggregation,
            "welch_pvalues": _rounded_optional_float_list(self.welch_pvalues),
            "median_welch_pvalue": _rounded_float_or_raw(self.median_welch_pvalue),
            "minimum_median_welch_pvalue": _rounded_float_or_raw(self.minimum_median_welch_pvalue),
            "finite_welch_pvalue_count": self.finite_welch_pvalue_count,
            "equivalence_margin": _rounded_float_or_raw(self.equivalence_margin),
            "declared_equivalence_margin": _rounded_float_or_raw(self.declared_equivalence_margin),
            "equivalence_margin_source": self.equivalence_margin_source,
            "equivalence_alpha": _rounded_float_or_raw(self.equivalence_alpha),
            "statistical_test_family": self.statistical_test_family,
            "statistical_test_kinds": list(self.statistical_test_kinds),
            "statistical_pvalues": _rounded_optional_float_list(self.statistical_pvalues),
            "aggregate_statistical_pvalue": _rounded_float_or_raw(self.aggregate_statistical_pvalue),
            "supported_statistical_bin_count": self.supported_statistical_bin_count,
            "unsupported_statistical_bin_count": len(self.unsupported_statistical_x_values),
            "unsupported_statistical_x_values": _rounded_list(list(self.unsupported_statistical_x_values)),
            "pvalue_aggregation": self.pvalue_aggregation,
            "pvalue_aggregation_source": self.pvalue_aggregation_source,
            "residual_gate_passed": self.residual_gate_passed,
            "pvalue_gate_passed": self.pvalue_gate_passed,
            "statistical_gate_passed": self.statistical_gate_passed,
            "residual_norm_score": _rounded_float_or_raw(self.residual_norm_score),
            "statistical_norm_score": _rounded_float_or_raw(self.statistical_norm_score),
            "overall_norm_score": _rounded_float_or_raw(self.overall_norm_score),
            "alignment_policy": self.alignment_policy,
            "x_match_tolerance": _rounded_float_or_raw(self.x_match_tolerance),
            "declared_resampling_grid_source": self.declared_resampling_grid_source,
            "resampling_grid_source": self.resampling_grid_source,
            "resampling_grid_source_origin": self.resampling_grid_source_origin,
            "declared_resampling_grid_values": _rounded_list(list(self.declared_resampling_grid_values)),
            "resampling_grid_values": _rounded_list(list(self.resampling_grid_values)),
            "interpolation_method": self.interpolation_method,
            "reference_resampled_support_counts": list(self.reference_resampled_support_counts),
            "model_resampled_support_counts": list(self.model_resampled_support_counts),
            "distribution_kind": self.distribution_kind,
            "x_quantity_name": self.x_quantity_name,
            "y_quantity_name": self.y_quantity_name,
            "comparison_x_unit_text": self.comparison_x_unit_text,
            "comparison_y_unit_text": self.comparison_y_unit_text,
            "reference_x_key": self.reference_x_key,
            "reference_y_key": self.reference_y_key,
            "reference_x_unit_text": self.reference_x_unit_text,
            "reference_y_unit_text": self.reference_y_unit_text,
            "reference_series_id_key": self.reference_series_id_key,
            "model_x_key": self.model_x_key,
            "model_y_key": self.model_y_key,
            "model_x_unit_text": self.model_x_unit_text,
            "model_y_unit_text": self.model_y_unit_text,
            "model_series_id_key": self.model_series_id_key,
            "reference_x_transform": self.reference_x_transform,
            "model_x_transform": self.model_x_transform,
            "series_provenance": self.series_provenance.to_dict(),
        }
        return _with_legacy_hz_aliases(
            evidence,
            comparison_y_unit_text=self.comparison_y_unit_text,
        )


class SeriesComparisonScore(sciunit.Score):
    _allowed_types = (float, int, pq.Quantity)

    def __init__(
        self,
        score: float | int | pq.Quantity,
        *,
        status: str,
        evidence_payload: SeriesComparisonEvidencePayload,
        case: SeriesComparisonCase,
    ) -> None:
        super().__init__(score)
        self.status = str(status)
        self.evidence_payload = evidence_payload
        self.evidence = evidence_payload.to_dict()
        self.case = case

    @property
    def norm_score(self) -> float:
        candidate = self.evidence_payload.norm_score
        if candidate is not None:
            return float(candidate)
        return 1.0 if self.status == "PASS" else 0.0

    def __str__(self) -> str:
        return self.status


@dataclass(frozen=True)
class SeriesPredictionBundle:
    protocol_evidence_key: str
    protocol_evidence: ProtocolEvidenceBundle = field(default_factory=ProtocolEvidenceBundle)

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self.protocol_evidence.rows(self.protocol_evidence_key)

    @property
    def context(self) -> dict[str, Any]:
        return self.protocol_evidence.to_dict()


def _aligned_x_pairs(
    reference_bins: dict[float, list[float]],
    model_bins: dict[float, list[float]],
    *,
    alignment_policy: str,
    x_match_tolerance: float | None,
) -> list[tuple[float, float]]:
    if alignment_policy == "exact_transformed_x":
        return [(x_value, x_value) for x_value in sorted(set(reference_bins).intersection(model_bins))]
    if alignment_policy == "tolerance_clusters":
        return [(x_value, x_value) for x_value in sorted(set(reference_bins).intersection(model_bins))]
    if alignment_policy == "resampled_grid":
        return [(x_value, x_value) for x_value in sorted(set(reference_bins).intersection(model_bins))]

    if alignment_policy != "nearest_within_tolerance":
        raise ValueError(
            f"Unsupported series alignment policy {alignment_policy!r}; "
            f"expected one of {', '.join(sorted(SERIES_ALIGNMENT_POLICIES))}"
        )

    if not _is_finite_number(x_match_tolerance) or float(x_match_tolerance) < 0.0:
        raise ValueError(
            "Alignment policy 'nearest_within_tolerance' requires a finite non-negative "
            "'x_match_tolerance' value"
        )

    tolerance = float(x_match_tolerance)
    reference_x_values = sorted(reference_bins)
    model_x_values = sorted(model_bins)
    pairs: list[tuple[float, float]] = []
    reference_index = 0
    model_index = 0
    while reference_index < len(reference_x_values) and model_index < len(model_x_values):
        reference_x = reference_x_values[reference_index]
        model_x = model_x_values[model_index]
        difference = model_x - reference_x
        if abs(difference) <= tolerance:
            pairs.append((reference_x, model_x))
            reference_index += 1
            model_index += 1
        elif difference < -tolerance:
            model_index += 1
        else:
            reference_index += 1
    return pairs


def _tolerance_cluster_bins(
    reference_bins: dict[float, list[float]],
    model_bins: dict[float, list[float]],
    *,
    x_match_tolerance: float | None,
    precision_digits: int,
) -> tuple[dict[float, list[float]], dict[float, list[float]], dict[float, dict[str, list[float]]]]:
    if not _is_finite_number(x_match_tolerance) or float(x_match_tolerance) < 0.0:
        raise ValueError(
            "Alignment policy 'tolerance_clusters' requires a finite non-negative "
            "'x_match_tolerance' value"
        )
    tolerance = float(x_match_tolerance)
    union_x_values = sorted(set(reference_bins).union(model_bins))
    if not union_x_values:
        return {}, {}, {}

    clusters: list[list[float]] = []
    current_cluster = [union_x_values[0]]
    cluster_start = union_x_values[0]
    for x_value in union_x_values[1:]:
        if float(x_value) - float(cluster_start) <= tolerance:
            current_cluster.append(float(x_value))
            continue
        clusters.append(current_cluster)
        current_cluster = [float(x_value)]
        cluster_start = float(x_value)
    clusters.append(current_cluster)

    clustered_reference_bins: dict[float, list[float]] = {}
    clustered_model_bins: dict[float, list[float]] = {}
    cluster_metadata: dict[float, dict[str, list[float]]] = {}
    for cluster_x_values in clusters:
        reference_x_values = [float(x_value) for x_value in cluster_x_values if x_value in reference_bins]
        model_x_values = [float(x_value) for x_value in cluster_x_values if x_value in model_bins]
        if not reference_x_values or not model_x_values:
            continue
        cluster_center = round(float(np.mean(np.asarray(cluster_x_values, dtype=float))), int(precision_digits))
        reference_values: list[float] = []
        for reference_x in reference_x_values:
            reference_values.extend(float(value) for value in reference_bins[reference_x])
        model_values: list[float] = []
        for model_x in model_x_values:
            model_values.extend(float(value) for value in model_bins[model_x])
        clustered_reference_bins[cluster_center] = reference_values
        clustered_model_bins[cluster_center] = model_values
        cluster_metadata[cluster_center] = {
            "reference_x_values": reference_x_values,
            "model_x_values": model_x_values,
        }
    return clustered_reference_bins, clustered_model_bins, cluster_metadata


def _series_bins(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    x_unit_text: str,
    y_unit_text: str,
    comparison_x_unit_text: str,
    comparison_y_unit_text: str,
    x_transform: AxisTransform,
    y_transform: AxisTransform,
    precision_digits: int,
    context: dict[str, Any] | None = None,
) -> dict[float, list[float]]:
    bins: dict[float, list[float]] = {}
    for row in rows:
        x_raw = row.get(x_key)
        y_raw = row.get(y_key)
        if not (_is_finite_number(x_raw) and _is_finite_number(y_raw)):
            continue
        x_value = x_transform.apply(
            float(x_raw),
            source_unit_text=x_unit_text,
            comparison_unit_text=comparison_x_unit_text,
            row=row,
            context=context,
        )
        y_value = y_transform.apply(
            float(y_raw),
            source_unit_text=y_unit_text,
            comparison_unit_text=comparison_y_unit_text,
            row=row,
            context=context,
        )
        bucket_key = round(numeric_value(x_value), int(precision_digits))
        bins.setdefault(bucket_key, []).append(float(numeric_value(y_value)))
    return {x_value: list(values) for x_value, values in sorted(bins.items())}


def _series_paths(
    rows: list[dict[str, Any]],
    *,
    series_id_key: str,
    x_key: str,
    y_key: str,
    x_unit_text: str,
    y_unit_text: str,
    comparison_x_unit_text: str,
    comparison_y_unit_text: str,
    x_transform: AxisTransform,
    y_transform: AxisTransform,
    precision_digits: int,
    context: dict[str, Any] | None = None,
) -> dict[str, list[tuple[float, float]]]:
    grouped_points: dict[str, dict[float, list[float]]] = {}
    for row in rows:
        series_id = str(row.get(series_id_key, "")).strip()
        if not series_id:
            continue
        x_raw = row.get(x_key)
        y_raw = row.get(y_key)
        if not (_is_finite_number(x_raw) and _is_finite_number(y_raw)):
            continue
        x_value = x_transform.apply(
            float(x_raw),
            source_unit_text=x_unit_text,
            comparison_unit_text=comparison_x_unit_text,
            row=row,
            context=context,
        )
        y_value = y_transform.apply(
            float(y_raw),
            source_unit_text=y_unit_text,
            comparison_unit_text=comparison_y_unit_text,
            row=row,
            context=context,
        )
        bucket_key = round(numeric_value(x_value), int(precision_digits))
        grouped_points.setdefault(series_id, {}).setdefault(bucket_key, []).append(float(numeric_value(y_value)))

    paths: dict[str, list[tuple[float, float]]] = {}
    for series_id, bucket_map in grouped_points.items():
        paths[series_id] = [
            (x_value, float(np.mean(np.asarray(values, dtype=float))))
            for x_value, values in sorted(bucket_map.items())
        ]
    return paths


def _resolved_resampling_grid(
    reference_bins: dict[float, list[float]],
    model_bins: dict[float, list[float]],
    *,
    grid_source: str,
    grid_values: tuple[float, ...],
    precision_digits: int,
) -> list[float]:
    normalized_source = str(grid_source or "").strip().lower()
    if normalized_source == "reference_observed_x":
        return sorted(reference_bins)
    if normalized_source == "model_observed_x":
        return sorted(model_bins)
    if normalized_source == "union_observed_x":
        return sorted(set(reference_bins).union(model_bins))
    if normalized_source == "explicit_grid":
        values = [
            round(float(value), int(precision_digits))
            for value in grid_values
            if _is_finite_number(value)
        ]
        if not values:
            raise ValueError(
                "Alignment policy 'resampled_grid' with resampling_grid_source='explicit_grid' "
                "requires non-empty finite 'resampling_grid_values'"
            )
        return sorted(set(values))
    raise ValueError(
        f"Unsupported resampling grid source {grid_source!r}; expected one of "
        f"{', '.join(sorted(SERIES_RESAMPLING_GRID_SOURCES))}"
    )


def _interpolated_series_value(
    path: list[tuple[float, float]],
    target_x: float,
    *,
    interpolation_method: str,
) -> float | None:
    if len(path) < 2:
        return None
    normalized_method = str(interpolation_method or "linear").strip().lower()
    if normalized_method not in SERIES_INTERPOLATION_METHODS:
        raise ValueError(
            f"Unsupported interpolation method {interpolation_method!r}; the current bridge only supports "
            f"{', '.join(sorted(SERIES_INTERPOLATION_METHODS))}"
        )
    x_values = [point[0] for point in path]
    y_values = [point[1] for point in path]
    if target_x < x_values[0] or target_x > x_values[-1]:
        return None
    for x_value, y_value in path:
        if np.isclose(x_value, target_x):
            return float(y_value)
    if normalized_method == "nearest":
        insert_at = int(np.searchsorted(np.asarray(x_values, dtype=float), float(target_x), side="left"))
        if insert_at <= 0:
            return float(y_values[0])
        if insert_at >= len(path):
            return float(y_values[-1])
        left_x, left_y = path[insert_at - 1]
        right_x, right_y = path[insert_at]
        if abs(float(target_x) - float(left_x)) <= abs(float(right_x) - float(target_x)):
            return float(left_y)
        return float(right_y)
    if normalized_method == "step_hold":
        insert_at = int(np.searchsorted(np.asarray(x_values, dtype=float), float(target_x), side="right")) - 1
        if insert_at < 0 or insert_at >= len(path):
            return None
        return float(path[insert_at][1])
    insert_at = int(np.searchsorted(np.asarray(x_values, dtype=float), float(target_x), side="left"))
    if insert_at <= 0 or insert_at >= len(path):
        return None
    x0, y0 = path[insert_at - 1]
    x1, y1 = path[insert_at]
    if np.isclose(x1, x0):
        return float(y0)
    fraction = (float(target_x) - float(x0)) / float(x1 - x0)
    return float(y0) + fraction * (float(y1) - float(y0))


def _resampled_bins_from_paths(
    series_paths: dict[str, list[tuple[float, float]]],
    *,
    target_grid: list[float],
    interpolation_method: str,
) -> tuple[dict[float, list[float]], dict[float, list[str]]]:
    bins: dict[float, list[float]] = {}
    support_ids: dict[float, list[str]] = {}
    for target_x in target_grid:
        values: list[float] = []
        ids: list[str] = []
        for series_id, path in series_paths.items():
            value = _interpolated_series_value(
                path,
                float(target_x),
                interpolation_method=interpolation_method,
            )
            if not _is_finite_number(value):
                continue
            values.append(float(value))
            ids.append(series_id)
        if values:
            bins[float(target_x)] = values
            support_ids[float(target_x)] = ids
    return bins, support_ids

def _with_legacy_hz_aliases(
    evidence: dict[str, Any],
    *,
    comparison_y_unit_text: str,
) -> dict[str, Any]:
    if str(comparison_y_unit_text or "").strip() != "Hz":
        return evidence
    alias_pairs = {
        "reference_sd_values": "reference_sd_values_Hz",
        "model_sd_values": "model_sd_values_Hz",
        "mean_absolute_error": "mean_absolute_error_Hz",
        "root_mean_square_error": "root_mean_square_error_Hz",
        "max_absolute_error": "max_absolute_error_Hz",
        "maximum_mae": "maximum_mae_Hz",
        "maximum_rmse": "maximum_rmse_Hz",
        "equivalence_margin": "equivalence_margin_Hz",
        "declared_equivalence_margin": "declared_equivalence_margin_Hz",
    }
    for canonical_key, alias_key in alias_pairs.items():
        evidence[alias_key] = evidence.get(canonical_key)
    return evidence


class SeriesComparisonTest(sciunit.Test):
    required_capabilities = (ProvidesProtocolEvidenceBundle,)
    score_type = SeriesComparisonScore

    def __init__(self, case: SeriesComparisonCase) -> None:
        self.case = case
        super().__init__(observation=case.observation.observation_payload(), name=case.title)

    def validate_observation(self, observation: dict[str, Any]) -> None:
        required = {
            "protocol_evidence_key",
            "reference_x_key",
            "reference_y_key",
            "model_x_key",
            "model_y_key",
            "comparison_x_unit_text",
            "comparison_y_unit_text",
            "equivalence_margin",
            "equivalence_alpha",
            "alignment_policy",
            "x_match_tolerance",
            "resampling_grid_source",
            "resampling_grid_values",
            "interpolation_method",
            "distribution_kind",
            "score_family",
            "pvalue_aggregation",
        }
        missing = sorted(required - set(observation))
        if missing:
            raise sciunit.ObservationError(
                f"SeriesComparisonTest observation is missing required keys: {', '.join(missing)}"
            )

    def generate_prediction(self, model: ReferenceValidationModel) -> SeriesPredictionBundle:
        return SeriesPredictionBundle(
            protocol_evidence_key=self.case.observation.protocol_evidence_key,
            protocol_evidence=model.get_protocol_evidence_bundle(),
        )

    def compute_score(self, observation: dict[str, Any], prediction: SeriesPredictionBundle) -> SeriesComparisonScore:
        del observation
        obs = self.case.observation
        visual_contract = obs.visual_contract
        reference_dataset = obs.reference_dataset()
        model_dataset = obs.model_dataset(prediction)
        if obs.policy.alignment_policy not in SERIES_ALIGNMENT_POLICIES:
            raise ValueError(
                f"Unsupported series alignment policy {obs.policy.alignment_policy!r}; "
                "the current bridge only supports: "
                + ", ".join(sorted(SERIES_ALIGNMENT_POLICIES))
            )
        if obs.policy.alignment_policy == "exact_transformed_x" and obs.policy.x_match_tolerance is not None:
            raise ValueError(
                "Series alignment policy 'exact_transformed_x' should not also declare "
                "'x_match_tolerance'; use 'nearest_within_tolerance' or 'tolerance_clusters' instead"
            )
        if obs.policy.alignment_policy == "resampled_grid" and obs.policy.x_match_tolerance is not None:
            raise ValueError(
                "Series alignment policy 'resampled_grid' should not also declare "
                "'x_match_tolerance'; declare a resampling grid source instead"
            )
        if obs.policy.distribution_kind != "empirical_by_x":
            raise ValueError(
                f"Unsupported series distribution kind {obs.policy.distribution_kind!r}; "
                "the current bridge only supports empirical per-x distributions"
            )
        if obs.policy.score_family not in SERIES_SCORE_FAMILIES:
            raise ValueError(
                f"Unsupported series score family {obs.policy.score_family!r}; "
                "expected one of residual_only, equivalence_only, hybrid_residual_equivalence, "
                "welch_only, hybrid_residual_welch"
            )
        equivalence_family = obs.policy.score_family in EQUIVALENCE_SERIES_SCORE_FAMILIES
        legacy_welch_family = obs.policy.score_family in LEGACY_WELCH_SERIES_SCORE_FAMILIES
        resolved_pvalue_aggregation, pvalue_aggregation_source = _resolved_pvalue_aggregation(
            obs.policy,
            equivalence_family=equivalence_family,
        )
        resolved_equivalence_margin, equivalence_margin_source = _resolved_equivalence_margin(
            obs.policy,
            equivalence_family=equivalence_family,
        )
        if obs.policy.alignment_policy == "resampled_grid":
            resolved_resampling_grid_source, resampling_grid_source_origin = _resolved_resampling_grid_source(obs.policy)
        else:
            resolved_resampling_grid_source, resampling_grid_source_origin = "", ""
        if legacy_welch_family and (
            obs.policy.minimum_median_welch_pvalue is None
        ):
            raise ValueError(
                f"Series score family {obs.policy.score_family!r} requires an explicit "
                "'minimum_median_welch_pvalue' threshold"
            )
        if obs.policy.score_family == "residual_only" and obs.policy.minimum_median_welch_pvalue is not None:
            raise ValueError(
                "Series score family 'residual_only' should not also declare "
                "'minimum_median_welch_pvalue'; use a Welch-based score family instead"
            )
        if equivalence_family and obs.policy.minimum_median_welch_pvalue is not None:
            raise ValueError(
                f"Series score family {obs.policy.score_family!r} should not declare "
                "'minimum_median_welch_pvalue'; use equivalence settings instead"
            )
        if equivalence_family and (not _is_finite_number(resolved_equivalence_margin) or float(resolved_equivalence_margin) <= 0.0):
            raise ValueError(
                f"Series score family {obs.policy.score_family!r} requires a positive finite equivalence margin"
            )
        if equivalence_family and (not _is_finite_number(obs.policy.equivalence_alpha) or not (0.0 < float(obs.policy.equivalence_alpha) < 1.0)):
            raise ValueError(
                f"Series score family {obs.policy.score_family!r} requires equivalence alpha in (0, 1)"
            )
        if not equivalence_family and _is_finite_number(obs.policy.equivalence_margin):
            raise ValueError(
                f"Series score family {obs.policy.score_family!r} should not declare an equivalence margin"
            )
        reference_bins = reference_dataset.bins(precision_digits=obs.policy.x_precision_digits)
        model_bins = model_dataset.bins(precision_digits=obs.policy.x_precision_digits)
        cluster_metadata: dict[float, dict[str, list[float]]] = {}
        resampling_metadata: dict[str, Any] = {}
        if obs.policy.alignment_policy == "resampled_grid":
            reference_paths = reference_dataset.paths(precision_digits=obs.policy.x_precision_digits)
            model_paths = model_dataset.paths(precision_digits=obs.policy.x_precision_digits)
            if not reference_paths:
                raise ValueError(
                    "Alignment policy 'resampled_grid' requires non-empty reference series ids; "
                    "the current bridge cannot interpolate a distribution from rows without per-series identity"
                )
            if not model_paths:
                raise ValueError(
                    "Alignment policy 'resampled_grid' requires non-empty model series ids; "
                    "the current bridge cannot interpolate a distribution from rows without per-series identity"
                )
            target_grid = _resolved_resampling_grid(
                reference_bins,
                model_bins,
                grid_source=resolved_resampling_grid_source,
                grid_values=obs.policy.resampling_grid_values,
                precision_digits=obs.policy.x_precision_digits,
            )
            reference_bins, reference_support_ids = _resampled_bins_from_paths(
                reference_paths,
                target_grid=target_grid,
                interpolation_method=obs.policy.interpolation_method,
            )
            model_bins, model_support_ids = _resampled_bins_from_paths(
                model_paths,
                target_grid=target_grid,
                interpolation_method=obs.policy.interpolation_method,
            )
            resampling_metadata = {
                "target_grid": list(target_grid),
                "reference_support_ids": {
                    float(target_x): list(series_ids)
                    for target_x, series_ids in reference_support_ids.items()
                },
                "model_support_ids": {
                    float(target_x): list(series_ids)
                    for target_x, series_ids in model_support_ids.items()
                },
            }
        if obs.policy.alignment_policy == "tolerance_clusters":
            reference_bins, model_bins, cluster_metadata = _tolerance_cluster_bins(
                reference_bins,
                model_bins,
                x_match_tolerance=obs.policy.x_match_tolerance,
                precision_digits=obs.policy.x_precision_digits,
            )
        aligned_pairs = _aligned_x_pairs(
            reference_bins,
            model_bins,
            alignment_policy=obs.policy.alignment_policy,
            x_match_tolerance=obs.policy.x_match_tolerance,
        )
        aligned_reference_keys = [reference_x for reference_x, _model_x in aligned_pairs]
        aligned_model_keys = [model_x for _reference_x, model_x in aligned_pairs]
        if cluster_metadata:
            reference_x_values = [
                float(np.mean(np.asarray(cluster_metadata[cluster_center]["reference_x_values"], dtype=float)))
                for cluster_center in aligned_reference_keys
            ]
            model_x_values = [
                float(np.mean(np.asarray(cluster_metadata[cluster_center]["model_x_values"], dtype=float)))
                for cluster_center in aligned_reference_keys
            ]
            visual_x_values = [cluster_center for cluster_center in aligned_reference_keys]
            matched_x_differences = [
                abs(model_x - reference_x)
                for reference_x, model_x in zip(reference_x_values, model_x_values, strict=False)
            ]
        else:
            reference_x_values = list(aligned_reference_keys)
            model_x_values = list(aligned_model_keys)
            visual_x_values = list(reference_x_values)
            matched_x_differences = [
                abs(model_x - reference_x)
                for reference_x, model_x in aligned_pairs
            ]
        reference_mean_values = [float(np.mean(reference_bins[reference_x])) for reference_x in aligned_reference_keys]
        model_mean_values = [float(np.mean(model_bins[model_x])) for model_x in aligned_model_keys]
        reference_sd_values = [_sample_sd(reference_bins[reference_x]) for reference_x in aligned_reference_keys]
        model_sd_values = [_sample_sd(model_bins[model_x]) for model_x in aligned_model_keys]
        reference_count_values = [len(reference_bins[reference_x]) for reference_x in aligned_reference_keys]
        model_count_values = [len(model_bins[model_x]) for model_x in aligned_model_keys]
        absolute_differences = [
            abs(model_mean - reference_mean)
            for reference_mean, model_mean in zip(reference_mean_values, model_mean_values, strict=False)
        ]
        mae = float(np.mean(absolute_differences)) if absolute_differences else float("nan")
        rmse = (
            float(np.sqrt(np.mean(np.square(np.asarray(absolute_differences, dtype=float)))))
            if absolute_differences
            else float("nan")
        )
        max_abs = float(np.max(absolute_differences)) if absolute_differences else float("nan")
        residual_gate_passed = (
            len(aligned_pairs) >= int(obs.policy.minimum_point_count)
            and _is_finite_number(mae)
            and mae <= float(obs.policy.maximum_mae)
            and _is_finite_number(rmse)
            and rmse <= float(obs.policy.maximum_rmse)
        )
        welch_pvalues: list[float] = []
        finite_welch_pvalues: list[float] = []
        median_welch_pvalue = float("nan")
        legacy_difference_gate_passed = False
        if legacy_welch_family:
            welch_pvalues = [
                _welch_pvalue(reference_bins[reference_x], model_bins[model_x])
                for reference_x, model_x in aligned_pairs
            ]
            finite_welch_pvalues = [value for value in welch_pvalues if _is_finite_number(value)]
            median_welch_pvalue = _aggregate_pvalues(
                finite_welch_pvalues,
                method=resolved_pvalue_aggregation,
            ) if finite_welch_pvalues else float("nan")
            legacy_difference_gate_passed = (
                len(aligned_pairs) >= int(obs.policy.minimum_point_count)
                and _is_finite_number(median_welch_pvalue)
                and obs.policy.minimum_median_welch_pvalue is not None
                and median_welch_pvalue >= float(obs.policy.minimum_median_welch_pvalue)
            )
        statistical_pvalues: list[float | None] = []
        statistical_test_kinds: list[str] = []
        aggregate_statistical_pvalue = float("nan")
        supported_statistical_bin_count = 0
        unsupported_statistical_x_values: list[float] = []
        statistical_gate_passed = False
        if equivalence_family:
            equivalence_results = [
                _equivalence_test_result(
                    reference_bins[reference_x],
                    model_bins[model_x],
                    equivalence_margin=float(resolved_equivalence_margin),
                )
                for reference_x, model_x in aligned_pairs
            ]
            statistical_test_kinds = [str(result["test_kind"]) for result in equivalence_results]
            statistical_pvalues = [
                rounded(float(result["pvalue"])) if _is_finite_number(result["pvalue"]) else None
                for result in equivalence_results
            ]
            finite_statistical_pvalues = [
                float(result["pvalue"])
                for result in equivalence_results
                if bool(result["supported"]) and _is_finite_number(result["pvalue"])
            ]
            supported_statistical_bin_count = len(finite_statistical_pvalues)
            unsupported_statistical_x_values = [
                float(reference_x)
                for (reference_x, _model_x), result in zip(aligned_pairs, equivalence_results, strict=False)
                if not bool(result["supported"]) or not _is_finite_number(result["pvalue"])
            ]
            if supported_statistical_bin_count == len(aligned_pairs) and finite_statistical_pvalues:
                aggregate_statistical_pvalue = _aggregate_pvalues(
                    finite_statistical_pvalues,
                    method=resolved_pvalue_aggregation,
                )
            statistical_gate_passed = (
                len(aligned_pairs) >= int(obs.policy.minimum_point_count)
                and supported_statistical_bin_count == len(aligned_pairs)
                and _is_finite_number(aggregate_statistical_pvalue)
                and aggregate_statistical_pvalue <= float(obs.policy.equivalence_alpha)
            )
        pvalue_gate_passed = legacy_difference_gate_passed if legacy_welch_family else statistical_gate_passed
        if obs.policy.score_family == "residual_only":
            passed = residual_gate_passed
        elif obs.policy.score_family == "equivalence_only":
            passed = statistical_gate_passed
        elif obs.policy.score_family == "hybrid_residual_equivalence":
            passed = residual_gate_passed and statistical_gate_passed
        elif obs.policy.score_family == "welch_only":
            passed = len(aligned_pairs) >= int(obs.policy.minimum_point_count) and legacy_difference_gate_passed
        else:
            passed = residual_gate_passed and legacy_difference_gate_passed
        status = self.case.pass_status if passed else self.case.fail_status
        residual_norm_score = _residual_norm_score(
            mae=mae,
            maximum_mae=obs.policy.maximum_mae,
            rmse=rmse,
            maximum_rmse=obs.policy.maximum_rmse,
        )
        statistical_norm_score = _series_statistical_norm_score(
            score_family=obs.policy.score_family,
            aggregate_statistical_pvalue=aggregate_statistical_pvalue,
            equivalence_alpha=obs.policy.equivalence_alpha,
            median_welch_pvalue=median_welch_pvalue,
            minimum_median_welch_pvalue=obs.policy.minimum_median_welch_pvalue,
        )
        overall_norm_score = _series_overall_norm_score(
            score_family=obs.policy.score_family,
            residual_norm_score=residual_norm_score,
            statistical_norm_score=statistical_norm_score,
            fallback_status=status,
        )
        reference_provenance_summary = reference_dataset.provenance_summary()
        model_provenance_summary = model_dataset.provenance_summary()
        series_provenance = SeriesObservationProvenance(
            reference=reference_provenance_summary,
            model=model_provenance_summary,
        )
        evidence_payload = SeriesComparisonEvidencePayload(
            visual_x_key=visual_contract.x_key,
            visual_reference_y_key=visual_contract.reference_y_key,
            visual_model_y_key=visual_contract.model_y_key,
            visual_x_values=tuple(visual_x_values),
            reference_matched_x_values=tuple(reference_x_values),
            model_matched_x_values=tuple(model_x_values),
            matched_x_differences=tuple(matched_x_differences),
            reference_cluster_x_groups=tuple(
                tuple(cluster_metadata[cluster_center]["reference_x_values"])
                for cluster_center in aligned_reference_keys
            )
            if cluster_metadata
            else (),
            model_cluster_x_groups=tuple(
                tuple(cluster_metadata[cluster_center]["model_x_values"])
                for cluster_center in aligned_reference_keys
            )
            if cluster_metadata
            else (),
            reference_values=tuple(reference_mean_values),
            model_values=tuple(model_mean_values),
            reference_sd_values=tuple(reference_sd_values),
            model_sd_values=tuple(model_sd_values),
            reference_count_values=tuple(reference_count_values),
            model_count_values=tuple(model_count_values),
            matched_point_count=len(aligned_pairs),
            mean_absolute_error=mae,
            root_mean_square_error=rmse,
            max_absolute_error=max_abs,
            maximum_mae=obs.policy.maximum_mae,
            maximum_rmse=obs.policy.maximum_rmse,
            error_unit_text=obs.comparison_y_unit_text,
            score_family=obs.policy.score_family,
            declared_pvalue_aggregation=str(obs.policy.pvalue_aggregation or "auto"),
            welch_pvalues=tuple(
                finite_welch_pvalues
                if len(finite_welch_pvalues) == len(welch_pvalues)
                else [float(value) if _is_finite_number(value) else None for value in welch_pvalues]
            ),
            median_welch_pvalue=median_welch_pvalue,
            minimum_median_welch_pvalue=obs.policy.minimum_median_welch_pvalue,
            finite_welch_pvalue_count=len(finite_welch_pvalues),
            equivalence_margin=resolved_equivalence_margin,
            declared_equivalence_margin=obs.policy.equivalence_margin,
            equivalence_margin_source=equivalence_margin_source,
            equivalence_alpha=obs.policy.equivalence_alpha,
            statistical_test_family=(
                "equivalence_tost"
                if equivalence_family
                else ("legacy_welch_difference" if legacy_welch_family else "none")
            ),
            statistical_test_kinds=tuple(statistical_test_kinds),
            statistical_pvalues=tuple(statistical_pvalues),
            aggregate_statistical_pvalue=aggregate_statistical_pvalue,
            supported_statistical_bin_count=supported_statistical_bin_count,
            unsupported_statistical_x_values=tuple(unsupported_statistical_x_values),
            pvalue_aggregation=resolved_pvalue_aggregation,
            pvalue_aggregation_source=pvalue_aggregation_source,
            residual_gate_passed=residual_gate_passed,
            pvalue_gate_passed=pvalue_gate_passed,
            statistical_gate_passed=(statistical_gate_passed if equivalence_family else legacy_difference_gate_passed),
            residual_norm_score=residual_norm_score,
            statistical_norm_score=statistical_norm_score,
            overall_norm_score=overall_norm_score,
            alignment_policy=obs.policy.alignment_policy,
            x_match_tolerance=obs.policy.x_match_tolerance,
            declared_resampling_grid_source=obs.policy.resampling_grid_source,
            resampling_grid_source=resolved_resampling_grid_source,
            resampling_grid_source_origin=resampling_grid_source_origin,
            declared_resampling_grid_values=tuple(obs.policy.resampling_grid_values),
            resampling_grid_values=tuple(resampling_metadata.get("target_grid", [])),
            interpolation_method=obs.policy.interpolation_method,
            reference_resampled_support_counts=tuple(
                len(resampling_metadata["reference_support_ids"].get(float(target_x), []))
                for target_x in aligned_reference_keys
            )
            if resampling_metadata
            else (),
            model_resampled_support_counts=tuple(
                len(resampling_metadata["model_support_ids"].get(float(target_x), []))
                for target_x in aligned_reference_keys
            )
            if resampling_metadata
            else (),
            distribution_kind=obs.policy.distribution_kind,
            x_quantity_name=obs.x_quantity_name,
            y_quantity_name=obs.y_quantity_name,
            comparison_x_unit_text=obs.comparison_x_unit_text,
            comparison_y_unit_text=obs.comparison_y_unit_text,
            reference_x_key=reference_dataset.x_key,
            reference_y_key=reference_dataset.y_key,
            reference_x_unit_text=reference_dataset.x_unit_text,
            reference_y_unit_text=reference_dataset.y_unit_text,
            reference_series_id_key=reference_dataset.series_id_key,
            model_x_key=model_dataset.x_key,
            model_y_key=model_dataset.y_key,
            model_x_unit_text=model_dataset.x_unit_text,
            model_y_unit_text=model_dataset.y_unit_text,
            model_series_id_key=model_dataset.series_id_key,
            reference_x_transform=reference_dataset.x_transform.description(),
            model_x_transform=model_dataset.x_transform.description(),
            series_provenance=series_provenance,
        )
        if obs.policy.score_family == "equivalence_only":
            if _is_finite_number(aggregate_statistical_pvalue):
                score_value = float(aggregate_statistical_pvalue) - float(obs.policy.equivalence_alpha)
            else:
                score_value = float("inf")
        elif obs.policy.score_family == "welch_only":
            if _is_finite_number(median_welch_pvalue) and obs.policy.minimum_median_welch_pvalue is not None:
                score_value = float(obs.policy.minimum_median_welch_pvalue) - float(median_welch_pvalue)
            else:
                score_value = float("inf")
        else:
            score_value = mae if _is_finite_number(mae) else float("inf")
        return SeriesComparisonScore(
            score_value,
            status=status,
            evidence_payload=evidence_payload,
            case=self.case,
        )


@dataclass(frozen=True)
class CompiledSeriesComparisonSuite:
    suite: sciunit.TestSuite
    model: ReferenceValidationModel
    cases: list[SeriesComparisonCase]
    tests: list[SeriesComparisonTest]

    def judge(self) -> list[tuple[SeriesComparisonCase, SeriesComparisonScore]]:
        results: list[tuple[SeriesComparisonCase, SeriesComparisonScore]] = []
        for case, test in zip(self.cases, self.tests, strict=False):
            score = test.judge(self.model)
            results.append((case, score))
        return results


def compile_series_comparison_suite(
    *,
    cases: list[SeriesComparisonCase],
    summary: MetricSummaryTable | dict[str, dict[str, float]],
    metrics: MetricTable | list[dict[str, Any]],
    protocol_evidence: ProtocolEvidenceBundle,
    suite_name: str,
) -> CompiledSeriesComparisonSuite:
    tests = [SeriesComparisonTest(case) for case in cases]
    suite = sciunit.TestSuite(tests, name=suite_name)
    model = ReferenceValidationModel(
        summary=summary,
        metrics=metrics,
        protocol_evidence=protocol_evidence,
        name=f"{suite_name}-model",
    )
    return CompiledSeriesComparisonSuite(suite=suite, model=model, cases=cases, tests=tests)


def _series_score_text(case: SeriesComparisonCase, score: SeriesComparisonScore) -> str:
    del case
    return score.evidence_payload.score_text()


def _series_score_payload(case: SeriesComparisonCase, score: SeriesComparisonScore) -> SuiteCaseScorePayload:
    del case
    return score.evidence_payload.to_suite_case_score_payload(score_value=score.score)


def _series_case_weight(score: SeriesComparisonScore) -> float | None:
    return score.evidence_payload.case_weight()


def audit_items_from_series_comparison_suite(
    compiled: CompiledSeriesComparisonSuite,
    *,
    descriptor: SuiteDescriptor | None = None,
) -> list[AuditItem]:
    judged = compiled.judge()
    protocol_context = compiled.model.get_protocol_evidence_bundle().to_dict()
    candidate_ids = protocol_context.get("cell_models", [])
    if not isinstance(candidate_ids, list):
        candidate_ids = []

    if descriptor is None:
        descriptor = SuiteDescriptor(
            suite_id=str(compiled.suite.name or "series-comparison-suite"),
            suite_kind_label="Series-comparison suite",
            candidate_ids=tuple(str(candidate_id) for candidate_id in candidate_ids if str(candidate_id).strip()),
        )

    def _result_builder(case: SeriesComparisonCase, score: SeriesComparisonScore):
        obs = case.observation
        visual_contract = obs.visual_contract
        spec = audit_item_adapter_spec_from_case(
            case,
            series_visuals=[
                series_visual_spec(
                    kind=visual_contract.kind,
                    keys=[visual_contract.x_key, visual_contract.reference_y_key, visual_contract.model_y_key],
                    style={
                        "line_width": 1.8,
                        "marker_size": 3.2,
                        "legend_loc": "lower center",
                    },
                )
            ],
        )
        return suite_case_result_from_spec(
            spec,
            status=score.status,
            evidence=score.evidence,
            score_text=_series_score_text(case, score),
            norm_score=score.norm_score,
            score_payload=_series_score_payload(case, score),
            case_weight=_series_case_weight(score),
            case_weight_label="matched points",
        )

    return suite_items_from_judged(
        descriptor=descriptor,
        judged=judged,
        result_builder=_result_builder,
    )


__all__ = [
    "AxisTransform",
    "CompiledSeriesComparisonSuite",
    "SERIES_ALIGNMENT_POLICIES",
    "SERIES_INTERPOLATION_METHODS",
    "SERIES_RESAMPLING_GRID_SOURCES",
    "SeriesComparisonCase",
    "SeriesComparisonEvidencePayload",
    "SeriesDataSpec",
    "SeriesComparisonPolicy",
    "SeriesComparisonScore",
    "SeriesComparisonTest",
    "SeriesObservedDataset",
    "SeriesPredictionBundle",
    "SeriesDistributionObservation",
    "SeriesVisualContract",
    "audit_items_from_series_comparison_suite",
    "compile_series_comparison_suite",
]
