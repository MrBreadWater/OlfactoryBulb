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
    {
        "cell_id": "RefA",
        "current_pA": 100.0,
        "firing_rate_Hz": 5.0,
        "protocol_id": "SYNTHETIC_PROTOCOL",
        "source": "Synthetic Reference Source",
        "source_file": "synthetic_curve.csv",
        "source_location": "sheet1",
        "source_url": "https://example.org/reference",
        "extraction_method": "source_spreadsheet",
        "sample_scope": "example_cell",
        "note_ids": "NOTE_A;NOTE_B",
        "rate_definition": "median_inverse_isi",
    },
    {
        "cell_id": "RefB",
        "current_pA": 100.0,
        "firing_rate_Hz": 7.0,
        "protocol_id": "SYNTHETIC_PROTOCOL",
        "source": "Synthetic Reference Source",
        "source_file": "synthetic_curve.csv",
        "source_location": "sheet1",
        "source_url": "https://example.org/reference",
        "extraction_method": "source_spreadsheet",
        "sample_scope": "example_cell",
        "note_ids": "NOTE_B",
        "rate_definition": "median_inverse_isi",
    },
    {
        "cell_id": "RefA",
        "current_pA": 200.0,
        "firing_rate_Hz": 10.0,
        "protocol_id": "SYNTHETIC_PROTOCOL",
        "source": "Synthetic Reference Source",
        "source_file": "synthetic_curve.csv",
        "source_location": "sheet1",
        "source_url": "https://example.org/reference",
        "extraction_method": "source_spreadsheet",
        "sample_scope": "example_cell",
        "note_ids": "NOTE_A",
        "rate_definition": "median_inverse_isi",
    },
    {
        "cell_id": "RefB",
        "current_pA": 200.0,
        "firing_rate_Hz": 12.0,
        "protocol_id": "SYNTHETIC_PROTOCOL",
        "source": "Synthetic Reference Source",
        "source_file": "synthetic_curve.csv",
        "source_location": "sheet1",
        "source_url": "https://example.org/reference",
        "extraction_method": "source_spreadsheet",
        "sample_scope": "example_cell",
        "note_ids": "NOTE_B",
        "rate_definition": "median_inverse_isi",
    },
]

model_rows = [
    {
        "cell_name": "Model1",
        "cell_type": "SyntheticFSI",
        "current_flux": 0.10,
        "firing_rate_Hz": 6.0,
        "rate_definition": "median_inverse_isi",
        "sample_scope": "model_population",
    },
    {
        "cell_name": "Model2",
        "cell_type": "SyntheticFSI",
        "current_flux": 0.10,
        "firing_rate_Hz": 8.0,
        "rate_definition": "median_inverse_isi",
        "sample_scope": "model_population",
    },
    {
        "cell_name": "Model1",
        "cell_type": "SyntheticFSI",
        "current_flux": 0.20,
        "firing_rate_Hz": 11.0,
        "rate_definition": "median_inverse_isi",
        "sample_scope": "model_population",
    },
    {
        "cell_name": "Model2",
        "cell_type": "SyntheticFSI",
        "current_flux": 0.20,
        "firing_rate_Hz": 13.0,
        "rate_definition": "median_inverse_isi",
        "sample_scope": "model_population",
    },
]

offset_model_rows = [
    {
        "cell_name": "Model1",
        "cell_type": "SyntheticFSI",
        "current_flux": 0.1004,
        "firing_rate_Hz": 6.0,
        "rate_definition": "median_inverse_isi",
        "sample_scope": "model_population",
    },
    {
        "cell_name": "Model2",
        "cell_type": "SyntheticFSI",
        "current_flux": 0.1004,
        "firing_rate_Hz": 8.0,
        "rate_definition": "median_inverse_isi",
        "sample_scope": "model_population",
    },
    {
        "cell_name": "Model1",
        "cell_type": "SyntheticFSI",
        "current_flux": 0.2004,
        "firing_rate_Hz": 11.0,
        "rate_definition": "median_inverse_isi",
        "sample_scope": "model_population",
    },
    {
        "cell_name": "Model2",
        "cell_type": "SyntheticFSI",
        "current_flux": 0.2004,
        "firing_rate_Hz": 13.0,
        "rate_definition": "median_inverse_isi",
        "sample_scope": "model_population",
    },
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
        alignment_policy="exact_transformed_x",
        distribution_kind="empirical_by_x",
        score_family="hybrid_residual_welch",
        pvalue_aggregation="median",
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
    protocol_evidence={
        "fi_curve_rows": model_rows,
        "cell_models": ["SyntheticModel1", "SyntheticModel2"],
        "step_duration_ms": 500.0,
        "target_vm_mV": -60.0,
    },
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
assert adapted_items[0].evidence["score_family"] == "hybrid_residual_welch"
assert adapted_items[0].evidence["pvalue_aggregation"] == "median"
assert adapted_items[0].evidence["residual_gate_passed"] is True
assert adapted_items[0].evidence["pvalue_gate_passed"] is True
assert adapted_items[0].evidence["reference_provenance"]["source_files"] == ["synthetic_curve.csv"]
assert adapted_items[0].evidence["reference_provenance"]["note_ids"] == ["NOTE_A", "NOTE_B"]
assert adapted_items[0].evidence["model_provenance"]["sample_scopes"] == ["model_population"]
assert adapted_items[0].evidence["model_provenance"]["protocol_context"]["cell_models"] == [
    "SyntheticModel1",
    "SyntheticModel2",
]
assert adapted_items[0].evidence["model_provenance"]["protocol_context"]["step_duration_ms"] == 500.0

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
    "alignment_policy": "exact_transformed_x",
    "distribution_kind": "empirical_by_x",
    "score_family": "hybrid_residual_welch",
    "pvalue_aggregation": "median",
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
    protocol_result=SimpleNamespace(
        protocol_evidence={
            "fi_curve_rows": model_rows,
            "cell_models": ["SyntheticModel1", "SyntheticModel2"],
            "step_duration_ms": 500.0,
            "target_vm_mV": -60.0,
        }
    ),
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
assert items[0].evidence["score_family"] == "hybrid_residual_welch"
assert items[0].evidence["reference_provenance"]["protocol_ids"] == ["SYNTHETIC_PROTOCOL"]
assert items[0].evidence["model_provenance"]["protocol_context"]["target_vm_mV"] == -60.0

residual_only_rule = dict(rule)
residual_only_rule["score_family"] = "residual_only"
del residual_only_rule["minimum_median_welch_pvalue"]
rules_module._load_rows = lambda loader_spec: reference_rows if loader_spec == "csv:/tmp/unused.csv" else original_load_rows(loader_spec)
try:
    residual_items = build_rule_items([residual_only_rule], context)
finally:
    rules_module._load_rows = original_load_rows

assert len(residual_items) == 1
assert residual_items[0].status == "PASS"
assert residual_items[0].evidence["score_family"] == "residual_only"
assert residual_items[0].evidence["minimum_median_welch_pvalue"] is None

nearest_rule = dict(residual_only_rule)
nearest_rule["alignment_policy"] = "nearest_within_tolerance"
nearest_rule["x_match_tolerance"] = 0.5
nearest_context = ValidationRuleContext(
    metrics=[],
    summary={},
    args=Namespace(),
    config={"validation_id": "synthetic_series_validation"},
    protocol_result=SimpleNamespace(
        protocol_evidence={
            "fi_curve_rows": offset_model_rows,
            "cell_models": ["SyntheticModel1", "SyntheticModel2"],
            "step_duration_ms": 500.0,
        }
    ),
)
rules_module._load_rows = lambda loader_spec: reference_rows if loader_spec == "csv:/tmp/unused.csv" else original_load_rows(loader_spec)
try:
    nearest_items = build_rule_items([nearest_rule], nearest_context)
finally:
    rules_module._load_rows = original_load_rows

assert len(nearest_items) == 1
assert nearest_items[0].status == "PASS"
assert nearest_items[0].evidence["alignment_policy"] == "nearest_within_tolerance"
assert nearest_items[0].evidence["x_match_tolerance"] == 0.5
assert nearest_items[0].evidence["reference_matched_x_values"] == [100.0, 200.0]
assert nearest_items[0].evidence["model_matched_x_values"] == [100.4, 200.4]
assert nearest_items[0].evidence["matched_x_differences"] == [0.4, 0.4]

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

missing_score_family_rule = dict(rule)
del missing_score_family_rule["score_family"]
rules_module._load_rows = lambda loader_spec: reference_rows if loader_spec == "csv:/tmp/unused.csv" else original_load_rows(loader_spec)
try:
    try:
        build_rule_items([missing_score_family_rule], context)
        raise AssertionError("Expected reference_curve_match to require explicit score-family metadata")
    except ValueError as exc:
        assert "requires explicit 'score_family'" in str(exc)
finally:
    rules_module._load_rows = original_load_rows

missing_pvalue_aggregation_rule = dict(rule)
del missing_pvalue_aggregation_rule["pvalue_aggregation"]
rules_module._load_rows = lambda loader_spec: reference_rows if loader_spec == "csv:/tmp/unused.csv" else original_load_rows(loader_spec)
try:
    try:
        build_rule_items([missing_pvalue_aggregation_rule], context)
        raise AssertionError("Expected Welch-based series comparison to require explicit p-value aggregation")
    except ValueError as exc:
        assert "requires explicit 'pvalue_aggregation'" in str(exc)
finally:
    rules_module._load_rows = original_load_rows

missing_tolerance_rule = dict(nearest_rule)
del missing_tolerance_rule["x_match_tolerance"]
rules_module._load_rows = lambda loader_spec: reference_rows if loader_spec == "csv:/tmp/unused.csv" else original_load_rows(loader_spec)
try:
    try:
        build_rule_items([missing_tolerance_rule], nearest_context)
        raise AssertionError("Expected tolerance-based alignment to require an explicit x-match tolerance")
    except ValueError as exc:
        assert "requires explicit 'x_match_tolerance'" in str(exc)
finally:
    rules_module._load_rows = original_load_rows


print("neuronunit_series_validation_suite: OK")
