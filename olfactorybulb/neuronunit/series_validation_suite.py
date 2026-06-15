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

from olfactorybulb.audit import AuditItem, series_visual_spec
from olfactorybulb.audit.core import rounded
from olfactorybulb.neuronunit.capabilities import (
    ProvidesProtocolEvidenceMap,
    ProvidesProtocolEvidenceRows,
)
from olfactorybulb.neuronunit.provenance import SeriesProvenanceSummary
from olfactorybulb.neuronunit.reference_bands import measurement_with_unit, numeric_value, quantity_unit_for_text
from olfactorybulb.neuronunit.reference_validation_suite import ReferenceValidationModel
from olfactorybulb.neuronunit.suite_presentation import suite_case_result, suite_items_from_judged


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


@dataclass(frozen=True)
class AxisTransform:
    kind: str = "identity"
    scale: float = 1.0
    offset: float = 0.0
    input_unit_text: str = ""
    output_unit_text: str = ""
    points: tuple[tuple[float, float], ...] = ()
    extrapolation_mode: str = "forbid"

    def apply(self, raw_value: float, *, source_unit_text: str, comparison_unit_text: str) -> float | pq.Quantity:
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
        )

    def paths(
        self,
        rows: list[dict[str, Any]],
        *,
        comparison_x_unit_text: str,
        comparison_y_unit_text: str,
        precision_digits: int,
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


class SeriesComparisonScore(sciunit.Score):
    _allowed_types = (float, int, pq.Quantity)

    def __init__(
        self,
        score: float | int | pq.Quantity,
        *,
        status: str,
        evidence: dict[str, Any],
        case: SeriesComparisonCase,
    ) -> None:
        super().__init__(score)
        self.status = str(status)
        self.evidence = evidence
        self.case = case

    @property
    def norm_score(self) -> float:
        candidate = self.evidence.get("overall_norm_score")
        if _is_finite_number(candidate):
            return max(0.0, min(1.0, float(candidate)))
        return 1.0 if self.status == "PASS" else 0.0

    def __str__(self) -> str:
        return self.status


@dataclass(frozen=True)
class SeriesPredictionBundle:
    rows: list[dict[str, Any]]
    context: dict[str, Any]


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
        )
        y_value = y_transform.apply(
            float(y_raw),
            source_unit_text=y_unit_text,
            comparison_unit_text=comparison_y_unit_text,
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
        )
        y_value = y_transform.apply(
            float(y_raw),
            source_unit_text=y_unit_text,
            comparison_unit_text=comparison_y_unit_text,
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
    required_capabilities = (ProvidesProtocolEvidenceRows, ProvidesProtocolEvidenceMap)
    score_type = SeriesComparisonScore

    def __init__(self, case: SeriesComparisonCase) -> None:
        self.case = case
        observation = {
            "protocol_evidence_key": case.observation.protocol_evidence_key,
            "reference_x_key": case.observation.reference_x_key,
            "reference_y_key": case.observation.reference_y_key,
            "model_x_key": case.observation.model_x_key,
            "model_y_key": case.observation.model_y_key,
            "comparison_x_unit_text": case.observation.comparison_x_unit_text,
            "comparison_y_unit_text": case.observation.comparison_y_unit_text,
            "equivalence_margin": case.observation.policy.equivalence_margin,
            "equivalence_alpha": case.observation.policy.equivalence_alpha,
            "alignment_policy": case.observation.policy.alignment_policy,
            "x_match_tolerance": case.observation.policy.x_match_tolerance,
            "resampling_grid_source": case.observation.policy.resampling_grid_source,
            "resampling_grid_values": case.observation.policy.resampling_grid_values,
            "interpolation_method": case.observation.policy.interpolation_method,
            "distribution_kind": case.observation.policy.distribution_kind,
            "score_family": case.observation.policy.score_family,
            "pvalue_aggregation": case.observation.policy.pvalue_aggregation,
        }
        super().__init__(observation=observation, name=case.title)

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
            rows=model.get_protocol_evidence_rows(self.case.observation.protocol_evidence_key),
            context=model.get_protocol_evidence_map(),
        )

    def compute_score(self, observation: dict[str, Any], prediction: SeriesPredictionBundle) -> SeriesComparisonScore:
        del observation
        obs = self.case.observation
        reference_spec = obs.reference_spec
        model_spec = obs.model_spec
        visual_contract = obs.visual_contract
        prediction_rows = list(prediction.rows)
        prediction_context = dict(prediction.context)
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
        reference_bins = reference_spec.bins(
            obs.reference_rows,
            comparison_x_unit_text=obs.comparison_x_unit_text,
            comparison_y_unit_text=obs.comparison_y_unit_text,
            precision_digits=obs.policy.x_precision_digits,
        )
        model_bins = model_spec.bins(
            prediction_rows,
            comparison_x_unit_text=obs.comparison_x_unit_text,
            comparison_y_unit_text=obs.comparison_y_unit_text,
            precision_digits=obs.policy.x_precision_digits,
        )
        cluster_metadata: dict[float, dict[str, list[float]]] = {}
        resampling_metadata: dict[str, Any] = {}
        if obs.policy.alignment_policy == "resampled_grid":
            reference_paths = reference_spec.paths(
                obs.reference_rows,
                comparison_x_unit_text=obs.comparison_x_unit_text,
                comparison_y_unit_text=obs.comparison_y_unit_text,
                precision_digits=obs.policy.x_precision_digits,
            )
            model_paths = model_spec.paths(
                prediction_rows,
                comparison_x_unit_text=obs.comparison_x_unit_text,
                comparison_y_unit_text=obs.comparison_y_unit_text,
                precision_digits=obs.policy.x_precision_digits,
            )
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
        reference_provenance_summary = reference_spec.provenance_summary(
            obs.reference_rows,
        )
        model_provenance_summary = model_spec.provenance_summary(
            prediction_rows,
            context=prediction_context,
            exclude_context_keys={obs.protocol_evidence_key},
        )
        evidence = {
            visual_contract.x_key: _rounded_list(visual_x_values),
            "reference_matched_x_values": _rounded_list(reference_x_values),
            "model_matched_x_values": _rounded_list(model_x_values),
            "matched_x_differences": _rounded_list(matched_x_differences),
            "reference_cluster_x_groups": [
                _rounded_list(cluster_metadata[cluster_center]["reference_x_values"])
                for cluster_center in aligned_reference_keys
            ]
            if cluster_metadata
            else [],
            "model_cluster_x_groups": [
                _rounded_list(cluster_metadata[cluster_center]["model_x_values"])
                for cluster_center in aligned_reference_keys
            ]
            if cluster_metadata
            else [],
            visual_contract.reference_y_key: _rounded_list(reference_mean_values),
            visual_contract.model_y_key: _rounded_list(model_mean_values),
            "reference_sd_values": _rounded_list(reference_sd_values),
            "model_sd_values": _rounded_list(model_sd_values),
            "reference_count_values": list(reference_count_values),
            "model_count_values": list(model_count_values),
            "matched_point_count": len(aligned_pairs),
            "mean_absolute_error": rounded(mae) if _is_finite_number(mae) else mae,
            "root_mean_square_error": rounded(rmse) if _is_finite_number(rmse) else rmse,
            "max_absolute_error": rounded(max_abs) if _is_finite_number(max_abs) else max_abs,
            "maximum_mae": rounded(float(obs.policy.maximum_mae))
            if _is_finite_number(obs.policy.maximum_mae)
            else obs.policy.maximum_mae,
            "maximum_rmse": rounded(float(obs.policy.maximum_rmse))
            if _is_finite_number(obs.policy.maximum_rmse)
            else obs.policy.maximum_rmse,
            "error_unit_text": obs.comparison_y_unit_text,
            "score_family": obs.policy.score_family,
            "declared_pvalue_aggregation": str(obs.policy.pvalue_aggregation or "auto"),
            "welch_pvalues": _rounded_list(finite_welch_pvalues)
            if len(finite_welch_pvalues) == len(welch_pvalues)
            else [rounded(float(value)) if _is_finite_number(value) else None for value in welch_pvalues],
            "median_welch_pvalue": rounded(median_welch_pvalue)
            if _is_finite_number(median_welch_pvalue)
            else median_welch_pvalue,
            "minimum_median_welch_pvalue": rounded(float(obs.policy.minimum_median_welch_pvalue))
            if _is_finite_number(obs.policy.minimum_median_welch_pvalue)
            else obs.policy.minimum_median_welch_pvalue,
            "finite_welch_pvalue_count": len(finite_welch_pvalues),
            "equivalence_margin": rounded(float(resolved_equivalence_margin))
            if _is_finite_number(resolved_equivalence_margin)
            else resolved_equivalence_margin,
            "declared_equivalence_margin": rounded(float(obs.policy.equivalence_margin))
            if _is_finite_number(obs.policy.equivalence_margin)
            else obs.policy.equivalence_margin,
            "equivalence_margin_source": equivalence_margin_source,
            "equivalence_alpha": rounded(float(obs.policy.equivalence_alpha))
            if _is_finite_number(obs.policy.equivalence_alpha)
            else obs.policy.equivalence_alpha,
            "statistical_test_family": (
                "equivalence_tost"
                if equivalence_family
                else ("legacy_welch_difference" if legacy_welch_family else "none")
            ),
            "statistical_test_kinds": statistical_test_kinds,
            "statistical_pvalues": statistical_pvalues,
            "aggregate_statistical_pvalue": rounded(float(aggregate_statistical_pvalue))
            if _is_finite_number(aggregate_statistical_pvalue)
            else aggregate_statistical_pvalue,
            "supported_statistical_bin_count": supported_statistical_bin_count,
            "unsupported_statistical_bin_count": len(unsupported_statistical_x_values),
            "unsupported_statistical_x_values": _rounded_list(unsupported_statistical_x_values),
            "pvalue_aggregation": resolved_pvalue_aggregation,
            "pvalue_aggregation_source": pvalue_aggregation_source,
            "residual_gate_passed": residual_gate_passed,
            "pvalue_gate_passed": pvalue_gate_passed,
            "statistical_gate_passed": statistical_gate_passed if equivalence_family else legacy_difference_gate_passed,
            "residual_norm_score": rounded(float(residual_norm_score))
            if _is_finite_number(residual_norm_score)
            else residual_norm_score,
            "statistical_norm_score": rounded(float(statistical_norm_score))
            if _is_finite_number(statistical_norm_score)
            else statistical_norm_score,
            "overall_norm_score": rounded(float(overall_norm_score))
            if _is_finite_number(overall_norm_score)
            else overall_norm_score,
            "alignment_policy": obs.policy.alignment_policy,
            "x_match_tolerance": rounded(float(obs.policy.x_match_tolerance))
            if _is_finite_number(obs.policy.x_match_tolerance)
            else obs.policy.x_match_tolerance,
            "declared_resampling_grid_source": obs.policy.resampling_grid_source,
            "resampling_grid_source": resolved_resampling_grid_source,
            "resampling_grid_source_origin": resampling_grid_source_origin,
            "declared_resampling_grid_values": _rounded_list(list(obs.policy.resampling_grid_values))
            if obs.policy.resampling_grid_values
            else [],
            "resampling_grid_values": _rounded_list(list(resampling_metadata.get("target_grid", []))),
            "interpolation_method": obs.policy.interpolation_method,
            "reference_resampled_support_counts": [
                len(resampling_metadata["reference_support_ids"].get(float(target_x), []))
                for target_x in aligned_reference_keys
            ]
            if resampling_metadata
            else [],
            "model_resampled_support_counts": [
                len(resampling_metadata["model_support_ids"].get(float(target_x), []))
                for target_x in aligned_reference_keys
            ]
            if resampling_metadata
            else [],
            "distribution_kind": obs.policy.distribution_kind,
            "x_quantity_name": obs.x_quantity_name,
            "y_quantity_name": obs.y_quantity_name,
            "comparison_x_unit_text": obs.comparison_x_unit_text,
            "comparison_y_unit_text": obs.comparison_y_unit_text,
            "reference_series_count": reference_provenance_summary.series_count,
            "model_series_count": model_provenance_summary.series_count,
            "reference_x_transform": reference_spec.x_transform.description(),
            "model_x_transform": model_spec.x_transform.description(),
            "reference_provenance": reference_provenance_summary.to_dict(),
            "model_provenance": model_provenance_summary.to_dict(),
        }
        evidence = _with_legacy_hz_aliases(
            evidence,
            comparison_y_unit_text=obs.comparison_y_unit_text,
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
        return SeriesComparisonScore(score_value, status=status, evidence=evidence, case=self.case)


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
    summary: dict[str, dict[str, float]],
    metrics: list[dict[str, Any]],
    protocol_evidence: dict[str, Any],
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
    score_family = str(score.evidence.get("score_family") or "").strip()
    mae = score.evidence.get("mean_absolute_error")
    error_unit_text = str(score.evidence.get("error_unit_text", "")).strip()
    residual_text = ""
    if _is_finite_number(mae):
        residual_text = f"MAE {rounded(float(mae)):g}" + (f" {error_unit_text}" if error_unit_text else "")
    if score_family in EQUIVALENCE_SERIES_SCORE_FAMILIES:
        aggregate_pvalue = score.evidence.get("aggregate_statistical_pvalue")
    elif score_family in LEGACY_WELCH_SERIES_SCORE_FAMILIES:
        aggregate_pvalue = score.evidence.get("median_welch_pvalue")
    else:
        aggregate_pvalue = score.evidence.get("aggregate_statistical_pvalue")
    statistical_text = ""
    if _is_finite_number(aggregate_pvalue):
        pvalue_text = f"{rounded(float(aggregate_pvalue), digits=4):g}"
        if score_family in EQUIVALENCE_SERIES_SCORE_FAMILIES:
            statistical_text = f"TOST p {pvalue_text}"
        elif score_family in LEGACY_WELCH_SERIES_SCORE_FAMILIES:
            statistical_text = f"Welch p {pvalue_text}"
        else:
            statistical_text = f"p {pvalue_text}"
    if score_family == "residual_only":
        return residual_text
    if score_family == "equivalence_only":
        return statistical_text
    if residual_text and statistical_text:
        return f"{residual_text} | {statistical_text}"
    return residual_text or statistical_text


def audit_items_from_series_comparison_suite(compiled: CompiledSeriesComparisonSuite) -> list[AuditItem]:
    judged = compiled.judge()
    def _result_builder(case: SeriesComparisonCase, score: SeriesComparisonScore):
        obs = case.observation
        visual_contract = obs.visual_contract
        item = AuditItem(
            check_id=case.check_id,
            status=score.status,
            title=case.title,
            criterion=case.criterion,
            criterion_latex=case.criterion_latex,
            criterion_formulae=case.criterion_formulae,
            criterion_definitions=case.criterion_definitions,
            description=case.description,
            acceptable=case.acceptable,
            acceptable_basis=case.acceptable_basis,
            evidence=score.evidence,
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
            note=case.note,
        )
        return suite_case_result(
            item,
            score_text=_series_score_text(case, score),
            norm_score=score.norm_score,
        )

    return suite_items_from_judged(
        suite_name=str(compiled.suite.name or "series-comparison-suite"),
        suite_kind_label="Series-comparison suite",
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
    "SeriesDataSpec",
    "SeriesComparisonPolicy",
    "SeriesComparisonScore",
    "SeriesComparisonTest",
    "SeriesPredictionBundle",
    "SeriesDistributionObservation",
    "SeriesVisualContract",
    "audit_items_from_series_comparison_suite",
    "compile_series_comparison_suite",
]
