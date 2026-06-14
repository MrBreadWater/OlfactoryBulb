"""SciUnit-backed suite bridge for unit-aware series/distribution comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import warnings
from typing import Any

import numpy as np
import quantities as pq
from scipy.stats import ttest_ind
import sciunit

from olfactorybulb.audit import AuditItem, series_visual_spec
from olfactorybulb.audit.core import rounded
from olfactorybulb.neuronunit.capabilities import ProvidesProtocolEvidenceRows
from olfactorybulb.neuronunit.reference_bands import measurement_with_unit, numeric_value, quantity_unit_for_text
from olfactorybulb.neuronunit.reference_validation_suite import ReferenceValidationModel


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


@dataclass(frozen=True)
class AxisTransform:
    kind: str = "identity"
    scale: float = 1.0
    offset: float = 0.0
    input_unit_text: str = ""
    output_unit_text: str = ""

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
    x_precision_digits: int = 6
    alignment_policy: str = "exact_transformed_x"
    distribution_kind: str = "empirical_by_x"


@dataclass(frozen=True)
class SeriesDistributionObservation:
    protocol_evidence_key: str
    reference_rows: list[dict[str, Any]]
    reference_x_key: str
    reference_y_key: str
    model_x_key: str
    model_y_key: str
    reference_x_unit_text: str
    reference_y_unit_text: str
    model_x_unit_text: str
    model_y_unit_text: str
    comparison_x_unit_text: str
    comparison_y_unit_text: str
    reference_x_transform: AxisTransform = field(default_factory=AxisTransform)
    reference_y_transform: AxisTransform = field(default_factory=AxisTransform)
    model_x_transform: AxisTransform = field(default_factory=AxisTransform)
    model_y_transform: AxisTransform = field(default_factory=AxisTransform)
    reference_series_id_key: str = "cell_id"
    model_series_id_key: str = "cell_name"
    x_quantity_name: str = "series x-value"
    y_quantity_name: str = "series y-value"
    visual_x_key: str = "currents_pA"
    visual_reference_y_key: str = "reference_values_Hz"
    visual_model_y_key: str = "model_values_Hz"
    visual_kind: str = "fi_curve"
    policy: SeriesComparisonPolicy = field(default_factory=SeriesComparisonPolicy)


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
        return 1.0 if self.status == "PASS" else 0.0

    def __str__(self) -> str:
        return self.status


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


def _series_id_count(rows: list[dict[str, Any]], *, series_id_key: str) -> int:
    if not series_id_key:
        return 0
    return len(
        {
            str(row.get(series_id_key, "")).strip()
            for row in rows
            if str(row.get(series_id_key, "")).strip()
        }
    )


class SeriesComparisonTest(sciunit.Test):
    required_capabilities = (ProvidesProtocolEvidenceRows,)
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
            "alignment_policy": case.observation.policy.alignment_policy,
            "distribution_kind": case.observation.policy.distribution_kind,
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
            "alignment_policy",
            "distribution_kind",
        }
        missing = sorted(required - set(observation))
        if missing:
            raise sciunit.ObservationError(
                f"SeriesComparisonTest observation is missing required keys: {', '.join(missing)}"
            )

    def generate_prediction(self, model: ReferenceValidationModel) -> list[dict[str, Any]]:
        return model.get_protocol_evidence_rows(self.case.observation.protocol_evidence_key)

    def compute_score(self, observation: dict[str, Any], prediction: list[dict[str, Any]]) -> SeriesComparisonScore:
        del observation
        obs = self.case.observation
        reference_bins = _series_bins(
            obs.reference_rows,
            x_key=obs.reference_x_key,
            y_key=obs.reference_y_key,
            x_unit_text=obs.reference_x_unit_text,
            y_unit_text=obs.reference_y_unit_text,
            comparison_x_unit_text=obs.comparison_x_unit_text,
            comparison_y_unit_text=obs.comparison_y_unit_text,
            x_transform=obs.reference_x_transform,
            y_transform=obs.reference_y_transform,
            precision_digits=obs.policy.x_precision_digits,
        )
        model_bins = _series_bins(
            list(prediction),
            x_key=obs.model_x_key,
            y_key=obs.model_y_key,
            x_unit_text=obs.model_x_unit_text,
            y_unit_text=obs.model_y_unit_text,
            comparison_x_unit_text=obs.comparison_x_unit_text,
            comparison_y_unit_text=obs.comparison_y_unit_text,
            x_transform=obs.model_x_transform,
            y_transform=obs.model_y_transform,
            precision_digits=obs.policy.x_precision_digits,
        )
        shared_x_values = sorted(set(reference_bins).intersection(model_bins))
        reference_mean_values = [float(np.mean(reference_bins[x_value])) for x_value in shared_x_values]
        model_mean_values = [float(np.mean(model_bins[x_value])) for x_value in shared_x_values]
        reference_sd_values = [_sample_sd(reference_bins[x_value]) for x_value in shared_x_values]
        model_sd_values = [_sample_sd(model_bins[x_value]) for x_value in shared_x_values]
        reference_count_values = [len(reference_bins[x_value]) for x_value in shared_x_values]
        model_count_values = [len(model_bins[x_value]) for x_value in shared_x_values]
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
        welch_pvalues = [
            _welch_pvalue(reference_bins[x_value], model_bins[x_value])
            for x_value in shared_x_values
        ]
        finite_welch_pvalues = [value for value in welch_pvalues if _is_finite_number(value)]
        median_welch_pvalue = float(np.median(finite_welch_pvalues)) if finite_welch_pvalues else float("nan")
        passed = (
            len(shared_x_values) >= int(obs.policy.minimum_point_count)
            and _is_finite_number(mae)
            and mae <= float(obs.policy.maximum_mae)
            and _is_finite_number(rmse)
            and rmse <= float(obs.policy.maximum_rmse)
        )
        if obs.policy.minimum_median_welch_pvalue is not None:
            passed = (
                passed
                and _is_finite_number(median_welch_pvalue)
                and median_welch_pvalue >= float(obs.policy.minimum_median_welch_pvalue)
            )
        evidence = {
            obs.visual_x_key: _rounded_list(shared_x_values),
            obs.visual_reference_y_key: _rounded_list(reference_mean_values),
            obs.visual_model_y_key: _rounded_list(model_mean_values),
            "reference_sd_values_Hz": _rounded_list(reference_sd_values),
            "model_sd_values_Hz": _rounded_list(model_sd_values),
            "reference_count_values": list(reference_count_values),
            "model_count_values": list(model_count_values),
            "matched_point_count": len(shared_x_values),
            "mean_absolute_error_Hz": rounded(mae) if _is_finite_number(mae) else mae,
            "root_mean_square_error_Hz": rounded(rmse) if _is_finite_number(rmse) else rmse,
            "max_absolute_error_Hz": rounded(max_abs) if _is_finite_number(max_abs) else max_abs,
            "maximum_mae_Hz": rounded(float(obs.policy.maximum_mae))
            if _is_finite_number(obs.policy.maximum_mae)
            else obs.policy.maximum_mae,
            "maximum_rmse_Hz": rounded(float(obs.policy.maximum_rmse))
            if _is_finite_number(obs.policy.maximum_rmse)
            else obs.policy.maximum_rmse,
            "welch_pvalues": _rounded_list(finite_welch_pvalues)
            if len(finite_welch_pvalues) == len(welch_pvalues)
            else [rounded(float(value)) if _is_finite_number(value) else None for value in welch_pvalues],
            "median_welch_pvalue": rounded(median_welch_pvalue)
            if _is_finite_number(median_welch_pvalue)
            else median_welch_pvalue,
            "finite_welch_pvalue_count": len(finite_welch_pvalues),
            "alignment_policy": obs.policy.alignment_policy,
            "distribution_kind": obs.policy.distribution_kind,
            "x_quantity_name": obs.x_quantity_name,
            "y_quantity_name": obs.y_quantity_name,
            "comparison_x_unit_text": obs.comparison_x_unit_text,
            "comparison_y_unit_text": obs.comparison_y_unit_text,
            "reference_series_count": _series_id_count(obs.reference_rows, series_id_key=obs.reference_series_id_key),
            "model_series_count": _series_id_count(list(prediction), series_id_key=obs.model_series_id_key),
            "reference_x_transform": obs.reference_x_transform.description(),
            "model_x_transform": obs.model_x_transform.description(),
        }
        status = self.case.pass_status if passed else self.case.fail_status
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


def audit_items_from_series_comparison_suite(compiled: CompiledSeriesComparisonSuite) -> list[AuditItem]:
    items: list[AuditItem] = []
    for case, score in compiled.judge():
        obs = case.observation
        items.append(
            AuditItem(
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
                        kind=obs.visual_kind,
                        keys=[obs.visual_x_key, obs.visual_reference_y_key, obs.visual_model_y_key],
                        style={
                            "line_width": 1.8,
                            "marker_size": 3.2,
                            "legend_loc": "lower center",
                        },
                    )
                ],
                note=case.note,
            )
        )
    return items


__all__ = [
    "AxisTransform",
    "CompiledSeriesComparisonSuite",
    "SeriesComparisonCase",
    "SeriesComparisonPolicy",
    "SeriesComparisonScore",
    "SeriesComparisonTest",
    "SeriesDistributionObservation",
    "audit_items_from_series_comparison_suite",
    "compile_series_comparison_suite",
]
