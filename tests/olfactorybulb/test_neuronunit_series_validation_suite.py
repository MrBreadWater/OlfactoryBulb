"""Coverage for the SciUnit-backed unit-aware series/distribution bridge."""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

from olfactorybulb.audit.reference_validation_rules import ValidationRuleContext, build_rule_items
from olfactorybulb.neuronunit.series_validation_suite import (
    AxisTransform,
    SeriesComparisonCase,
    SeriesComparisonPolicy,
    SeriesDistributionObservation,
    audit_items_from_series_comparison_suite,
    compile_series_comparison_suite,
)


reference_rows = [
    {"cell_id": "RefA", "current_pA": 100.0, "firing_rate_Hz": 5.0},
    {"cell_id": "RefB", "current_pA": 100.0, "firing_rate_Hz": 7.0},
    {"cell_id": "RefA", "current_pA": 200.0, "firing_rate_Hz": 10.0},
    {"cell_id": "RefB", "current_pA": 200.0, "firing_rate_Hz": 12.0},
]

model_rows = [
    {"cell_name": "Model1", "current_flux": 0.10, "firing_rate_Hz": 6.0},
    {"cell_name": "Model2", "current_flux": 0.10, "firing_rate_Hz": 8.0},
    {"cell_name": "Model1", "current_flux": 0.20, "firing_rate_Hz": 11.0},
    {"cell_name": "Model2", "current_flux": 0.20, "firing_rate_Hz": 13.0},
]

observation = SeriesDistributionObservation(
    protocol_evidence_key="fi_curve_rows",
    reference_rows=reference_rows,
    reference_x_key="current_pA",
    reference_y_key="firing_rate_Hz",
    model_x_key="current_flux",
    model_y_key="firing_rate_Hz",
    reference_x_unit_text="pA",
    reference_y_unit_text="Hz",
    model_x_unit_text="",
    model_y_unit_text="Hz",
    comparison_x_unit_text="pA",
    comparison_y_unit_text="Hz",
    model_x_transform=AxisTransform(kind="affine", scale=1000.0, output_unit_text="pA"),
    policy=SeriesComparisonPolicy(
        minimum_point_count=2,
        maximum_mae=2.0,
        maximum_rmse=2.0,
        minimum_median_welch_pvalue=0.01,
        x_precision_digits=6,
    ),
)

case = SeriesComparisonCase(
    check_id="synthetic_series_match",
    title="Synthetic series comparison stays within distribution-aware tolerances",
    criterion="The transformed model series should stay close to the reference distribution at shared x values.",
    criterion_latex="",
    criterion_formulae=[],
    criterion_definitions=[],
    description="Synthetic series-comparison suite test.",
    acceptable="The matched bins satisfy the configured MAE/RMSE tolerances.",
    acceptable_basis="Synthetic basis.",
    note="",
    observation=observation,
)

compiled = compile_series_comparison_suite(
    cases=[case],
    summary={},
    metrics=[],
    protocol_evidence={"fi_curve_rows": model_rows},
    suite_name="synthetic series suite",
)
judged = compiled.judge()
assert [score.status for _case, score in judged] == ["PASS"]

adapted_items = audit_items_from_series_comparison_suite(compiled)
assert [item.status for item in adapted_items] == ["PASS"]
assert adapted_items[0].evidence["currents_pA"] == [100.0, 200.0]
assert adapted_items[0].evidence["reference_values_Hz"] == [6.0, 11.0]
assert adapted_items[0].evidence["model_values_Hz"] == [7.0, 12.0]
assert adapted_items[0].evidence["reference_count_values"] == [2, 2]
assert adapted_items[0].evidence["model_count_values"] == [2, 2]
assert adapted_items[0].evidence["mean_absolute_error_Hz"] == 1.0
assert adapted_items[0].evidence["reference_series_count"] == 2
assert adapted_items[0].evidence["model_series_count"] == 2
assert adapted_items[0].evidence["model_x_transform"].startswith("affine(")

rule = {
    "kind": "reference_curve_match",
    "check_id": "synthetic_series_match",
    "loader": "csv:/tmp/unused.csv",
    "protocol_evidence_key": "fi_curve_rows",
    "reference_current_key": "current_pA",
    "reference_value_key": "firing_rate_Hz",
    "model_current_key": "current_flux",
    "model_value_key": "firing_rate_Hz",
    "reference_x_unit_text": "pA",
    "reference_y_unit_text": "Hz",
    "model_x_unit_text": "",
    "model_y_unit_text": "Hz",
    "comparison_x_unit_text": "pA",
    "comparison_y_unit_text": "Hz",
    "model_x_transform": {"kind": "affine", "scale": 1000.0, "output_unit_text": "pA"},
    "maximum_mae": 2.0,
    "maximum_rmse": 2.0,
    "minimum_median_welch_pvalue": 0.01,
    "minimum_point_count": 2,
    "title": "Synthetic series comparison stays within distribution-aware tolerances",
    "criterion": "The transformed model series should stay close to the reference distribution at shared x values.",
    "description": "Synthetic series-comparison suite test.",
    "acceptable": "The matched bins satisfy the configured MAE/RMSE tolerances.",
    "acceptable_basis": "Synthetic basis.",
}

context = ValidationRuleContext(
    metrics=[],
    summary={},
    args=Namespace(),
    config={"validation_id": "synthetic_series_validation"},
    protocol_result=SimpleNamespace(protocol_evidence={"fi_curve_rows": model_rows}),
)

from olfactorybulb.audit import reference_validation_rules as rules_module

original_load_rows = rules_module._load_rows
rules_module._load_rows = lambda loader_spec: reference_rows if loader_spec == "csv:/tmp/unused.csv" else original_load_rows(loader_spec)
try:
    items = build_rule_items([rule], context)
finally:
    rules_module._load_rows = original_load_rows

assert len(items) == 1
assert items[0].status == "PASS"
assert items[0].series_visuals[0]["keys"] == ["currents_pA", "reference_values_Hz", "model_values_Hz"]
assert items[0].evidence["median_welch_pvalue"] is not None

missing_units_rule = dict(rule)
del missing_units_rule["reference_x_unit_text"]
rules_module._load_rows = lambda loader_spec: reference_rows if loader_spec == "csv:/tmp/unused.csv" else original_load_rows(loader_spec)
try:
    try:
        build_rule_items([missing_units_rule], context)
        raise AssertionError("Expected reference_curve_match to require explicit unit metadata")
    except ValueError as exc:
        assert "requires explicit 'reference_x_unit_text'" in str(exc)
finally:
    rules_module._load_rows = original_load_rows


print("neuronunit_series_validation_suite: OK")
