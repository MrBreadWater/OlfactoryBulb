"""Coverage for the SciUnit-backed unit-aware series/distribution bridge."""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

from olfactorybulb.audit.core import AuditReport
from olfactorybulb.audit.reference_validation_rules import ValidationRuleContext, build_rule_items
from olfactorybulb.neuronunit.provenance import SeriesProvenanceSummary
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

equivalence_reference_rows = [
    {"cell_id": "EqRef1", "current_pA": 100.0, "firing_rate_Hz": 6.00},
    {"cell_id": "EqRef2", "current_pA": 100.0, "firing_rate_Hz": 6.05},
    {"cell_id": "EqRef3", "current_pA": 100.0, "firing_rate_Hz": 5.95},
    {"cell_id": "EqRef4", "current_pA": 100.0, "firing_rate_Hz": 6.02},
    {"cell_id": "EqRef5", "current_pA": 100.0, "firing_rate_Hz": 5.98},
    {"cell_id": "EqRef1", "current_pA": 200.0, "firing_rate_Hz": 10.00},
    {"cell_id": "EqRef2", "current_pA": 200.0, "firing_rate_Hz": 10.05},
    {"cell_id": "EqRef3", "current_pA": 200.0, "firing_rate_Hz": 9.95},
    {"cell_id": "EqRef4", "current_pA": 200.0, "firing_rate_Hz": 10.02},
    {"cell_id": "EqRef5", "current_pA": 200.0, "firing_rate_Hz": 9.98},
]

equivalence_model_rows = [
    {"cell_name": "EqModel1", "current_pA": 100.0, "firing_rate_Hz": 6.04},
    {"cell_name": "EqModel2", "current_pA": 100.0, "firing_rate_Hz": 6.08},
    {"cell_name": "EqModel3", "current_pA": 100.0, "firing_rate_Hz": 6.01},
    {"cell_name": "EqModel4", "current_pA": 100.0, "firing_rate_Hz": 6.06},
    {"cell_name": "EqModel5", "current_pA": 100.0, "firing_rate_Hz": 6.03},
    {"cell_name": "EqModel1", "current_pA": 200.0, "firing_rate_Hz": 10.04},
    {"cell_name": "EqModel2", "current_pA": 200.0, "firing_rate_Hz": 10.06},
    {"cell_name": "EqModel3", "current_pA": 200.0, "firing_rate_Hz": 10.02},
    {"cell_name": "EqModel4", "current_pA": 200.0, "firing_rate_Hz": 10.05},
    {"cell_name": "EqModel5", "current_pA": 200.0, "firing_rate_Hz": 10.03},
]

equivalence_singleton_model_rows = [
    {"cell_name": "EqSingle", "current_pA": 100.0, "firing_rate_Hz": 6.03},
    {"cell_name": "EqSingle", "current_pA": 200.0, "firing_rate_Hz": 10.03},
]

piecewise_reference_rows = [
    {"cell_id": "PwRef1", "current_pA": 100.0, "firing_rate_Hz": 8.0},
    {"cell_id": "PwRef2", "current_pA": 100.0, "firing_rate_Hz": 9.0},
    {"cell_id": "PwRef1", "current_pA": 210.0, "firing_rate_Hz": 12.0},
    {"cell_id": "PwRef2", "current_pA": 210.0, "firing_rate_Hz": 13.0},
    {"cell_id": "PwRef1", "current_pA": 330.0, "firing_rate_Hz": 18.0},
    {"cell_id": "PwRef2", "current_pA": 330.0, "firing_rate_Hz": 19.0},
]

piecewise_model_rows = [
    {"cell_name": "PwModel1", "drive_flux": 0.10, "firing_rate_Hz": 8.4},
    {"cell_name": "PwModel2", "drive_flux": 0.10, "firing_rate_Hz": 8.6},
    {"cell_name": "PwModel1", "drive_flux": 0.20, "firing_rate_Hz": 12.4},
    {"cell_name": "PwModel2", "drive_flux": 0.20, "firing_rate_Hz": 12.6},
    {"cell_name": "PwModel1", "drive_flux": 0.30, "firing_rate_Hz": 18.4},
    {"cell_name": "PwModel2", "drive_flux": 0.30, "firing_rate_Hz": 18.6},
]

cluster_reference_rows = [
    {"cell_id": "ClRef1", "current_pA": 100.0, "firing_rate_Hz": 5.0},
    {"cell_id": "ClRef2", "current_pA": 100.3, "firing_rate_Hz": 5.2},
    {"cell_id": "ClRef1", "current_pA": 200.0, "firing_rate_Hz": 10.0},
    {"cell_id": "ClRef2", "current_pA": 200.3, "firing_rate_Hz": 10.2},
]

cluster_model_rows = [
    {"cell_name": "ClModel1", "current_pA": 100.1, "firing_rate_Hz": 5.1},
    {"cell_name": "ClModel2", "current_pA": 100.4, "firing_rate_Hz": 5.3},
    {"cell_name": "ClModel1", "current_pA": 200.1, "firing_rate_Hz": 10.1},
    {"cell_name": "ClModel2", "current_pA": 200.4, "firing_rate_Hz": 10.3},
]

resampled_reference_rows = [
    {"cell_id": "RsRef1", "current_pA": 100.0, "firing_rate_Hz": 5.0},
    {"cell_id": "RsRef2", "current_pA": 100.0, "firing_rate_Hz": 6.0},
    {"cell_id": "RsRef1", "current_pA": 200.0, "firing_rate_Hz": 10.0},
    {"cell_id": "RsRef2", "current_pA": 200.0, "firing_rate_Hz": 11.0},
    {"cell_id": "RsRef1", "current_pA": 300.0, "firing_rate_Hz": 15.0},
    {"cell_id": "RsRef2", "current_pA": 300.0, "firing_rate_Hz": 16.0},
]

resampled_model_rows = [
    {"cell_name": "RsModel1", "current_pA": 125.0, "firing_rate_Hz": 6.25},
    {"cell_name": "RsModel2", "current_pA": 125.0, "firing_rate_Hz": 7.25},
    {"cell_name": "RsModel1", "current_pA": 225.0, "firing_rate_Hz": 11.25},
    {"cell_name": "RsModel2", "current_pA": 225.0, "firing_rate_Hz": 12.25},
    {"cell_name": "RsModel1", "current_pA": 325.0, "firing_rate_Hz": 16.25},
    {"cell_name": "RsModel2", "current_pA": 325.0, "firing_rate_Hz": 17.25},
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
assert observation.reference_spec.x_key == "current_pA"
assert observation.reference_spec.series_id_key == "cell_id"
assert observation.model_spec.x_key == "current_flux"
assert observation.visual_contract.reference_y_key == "reference_values_Hz"

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
assert [item.status for item in adapted_items] == ["PASS", "PASS"]
assert adapted_items[0].detail_level == "summary"
assert adapted_items[0].summary_rollup_exempt is True
assert adapted_items[0].evidence["suite_status_summary"] == {"PASS": 1, "WARN": 0, "FAIL": 0}
assert adapted_items[0].evidence["suite_cases"][0]["score_text"].startswith("MAE 1 Hz | Welch p ")
assert adapted_items[0].evidence["suite_cases"][0]["norm_score"] == 1.0
assert adapted_items[0].evidence["suite_norm_score_summary"] == {
    "count": 1.0,
    "max": 1.0,
    "mean": 1.0,
    "median": 1.0,
    "min": 1.0,
}
assert adapted_items[1].evidence["currents_pA"] == [100.0, 200.0]
assert adapted_items[1].evidence["reference_values_Hz"] == [6.0, 11.0]
assert adapted_items[1].evidence["model_values_Hz"] == [7.0, 12.0]
assert adapted_items[1].evidence["reference_count_values"] == [2, 2]
assert adapted_items[1].evidence["model_count_values"] == [2, 2]
assert adapted_items[1].evidence["reference_sd_values"] == [1.414, 1.414]
assert adapted_items[1].evidence["model_sd_values"] == [1.414, 1.414]
assert adapted_items[1].evidence["mean_absolute_error"] == 1.0
assert adapted_items[1].evidence["root_mean_square_error"] == 1.0
assert adapted_items[1].evidence["max_absolute_error"] == 1.0
assert adapted_items[1].evidence["maximum_mae"] == 2.0
assert adapted_items[1].evidence["maximum_rmse"] == 2.0
assert adapted_items[1].evidence["error_unit_text"] == "Hz"
assert adapted_items[1].evidence["mean_absolute_error_Hz"] == 1.0
assert adapted_items[1].evidence["reference_series_count"] == 2
assert adapted_items[1].evidence["model_series_count"] == 2
assert adapted_items[1].evidence["model_x_transform"].startswith("affine(")
assert adapted_items[1].evidence["score_family"] == "hybrid_residual_welch"
assert adapted_items[1].evidence["pvalue_aggregation"] == "median"
assert adapted_items[1].evidence["residual_gate_passed"] is True
assert adapted_items[1].evidence["pvalue_gate_passed"] is True
assert adapted_items[1].evidence["residual_norm_score"] == 1.0
assert adapted_items[1].evidence["statistical_norm_score"] == 1.0
assert adapted_items[1].evidence["overall_norm_score"] == 1.0
assert adapted_items[1].evidence["reference_provenance"]["source_files"] == ["synthetic_curve.csv"]
assert adapted_items[1].evidence["reference_provenance"]["note_ids"] == ["NOTE_A", "NOTE_B"]
assert adapted_items[1].evidence["model_provenance"]["sample_scopes"] == ["model_population"]
assert adapted_items[1].evidence["model_provenance"]["protocol_context"]["cell_models"] == [
    "SyntheticModel1",
    "SyntheticModel2",
]
assert adapted_items[1].evidence["model_provenance"]["protocol_context"]["step_duration_ms"] == 500.0
assert adapted_items[1].evidence["reference_provenance"] == SeriesProvenanceSummary.from_rows(
    reference_rows,
    series_id_key=observation.reference_series_id_key,
    x_key=observation.reference_x_key,
    y_key=observation.reference_y_key,
    x_unit_text=observation.reference_x_unit_text,
    y_unit_text=observation.reference_y_unit_text,
).to_dict()
assert adapted_items[1].evidence["model_provenance"] == SeriesProvenanceSummary.from_rows(
    model_rows,
    series_id_key=observation.model_series_id_key,
    x_key=observation.model_x_key,
    y_key=observation.model_y_key,
    x_unit_text=observation.model_x_unit_text,
    y_unit_text=observation.model_y_unit_text,
    context={
        "fi_curve_rows": model_rows,
        "cell_models": ["SyntheticModel1", "SyntheticModel2"],
        "step_duration_ms": 500.0,
        "target_vm_mV": -60.0,
    },
    exclude_context_keys={"fi_curve_rows"},
).to_dict()
report = AuditReport(audit_id="synthetic_series_suite", title="Synthetic series suite", items=adapted_items)
assert report.summary == {"PASS": 1, "WARN": 0, "FAIL": 0}

voltage_reference_rows = [
    {"cell_id": "VmRef1", "current_pA": 100.0, "response_mV": -60.0},
    {"cell_id": "VmRef2", "current_pA": 100.0, "response_mV": -58.0},
    {"cell_id": "VmRef1", "current_pA": 200.0, "response_mV": -55.0},
    {"cell_id": "VmRef2", "current_pA": 200.0, "response_mV": -53.0},
]
voltage_model_rows = [
    {"cell_name": "VmModel1", "current_pA": 100.0, "response_mV": -59.0},
    {"cell_name": "VmModel2", "current_pA": 100.0, "response_mV": -57.0},
    {"cell_name": "VmModel1", "current_pA": 200.0, "response_mV": -54.0},
    {"cell_name": "VmModel2", "current_pA": 200.0, "response_mV": -52.0},
]
voltage_observation = SeriesDistributionObservation(
    protocol_evidence_key="response_rows",
    reference_rows=voltage_reference_rows,
    reference_x_key="current_pA",
    reference_y_key="response_mV",
    model_x_key="current_pA",
    model_y_key="response_mV",
    reference_x_unit_text="pA",
    reference_y_unit_text="mV",
    model_x_unit_text="pA",
    model_y_unit_text="mV",
    comparison_x_unit_text="pA",
    comparison_y_unit_text="mV",
    visual_reference_y_key="reference_values_mV",
    visual_model_y_key="model_values_mV",
    policy=SeriesComparisonPolicy(
        minimum_point_count=2,
        maximum_mae=1.0,
        maximum_rmse=1.0,
        score_family="residual_only",
    ),
)
voltage_case = SeriesComparisonCase(
    check_id="synthetic_voltage_series_match",
    title="Synthetic voltage-valued series comparison stays unit-aware",
    criterion="The aligned voltage response series should stay within the configured residual tolerances.",
    criterion_latex="",
    criterion_formulae=[],
    criterion_definitions=[],
    description="Synthetic non-Hz series-comparison suite test.",
    acceptable="The matched bins satisfy the configured MAE/RMSE tolerances.",
    acceptable_basis="Synthetic basis.",
    note="",
    observation=voltage_observation,
)
voltage_compiled = compile_series_comparison_suite(
    cases=[voltage_case],
    summary={},
    metrics=[],
    protocol_evidence={"response_rows": voltage_model_rows},
    suite_name="synthetic voltage series suite",
)
voltage_items = audit_items_from_series_comparison_suite(voltage_compiled)
assert voltage_items[0].evidence["suite_cases"][0]["score_text"] == "MAE 1 mV"
assert voltage_items[1].evidence["error_unit_text"] == "mV"
assert voltage_items[1].evidence["mean_absolute_error"] == 1.0
assert voltage_items[1].evidence["reference_sd_values"] == [1.414, 1.414]
assert voltage_items[1].evidence["model_sd_values"] == [1.414, 1.414]
assert "mean_absolute_error_Hz" not in voltage_items[1].evidence

equivalence_observation = SeriesDistributionObservation(
    protocol_evidence_key="fi_curve_rows",
    reference_rows=equivalence_reference_rows,
    reference_x_key="current_pA",
    reference_y_key="firing_rate_Hz",
    model_x_key="current_pA",
    model_y_key="firing_rate_Hz",
    reference_x_unit_text="pA",
    reference_y_unit_text="Hz",
    model_x_unit_text="pA",
    model_y_unit_text="Hz",
    comparison_x_unit_text="pA",
    comparison_y_unit_text="Hz",
    policy=SeriesComparisonPolicy(
        minimum_point_count=2,
        maximum_mae=0.2,
        maximum_rmse=0.2,
        score_family="hybrid_residual_equivalence",
    ),
)

equivalence_case = SeriesComparisonCase(
    check_id="synthetic_series_equivalence",
    title="Synthetic replicated series pass the equivalence gate",
    criterion="The aligned replicated series should satisfy both residual and equivalence gates.",
    criterion_latex="",
    criterion_formulae=[],
    criterion_definitions=[],
    description="Synthetic equivalence suite test.",
    acceptable="The matched bins satisfy the configured MAE/RMSE tolerances and per-bin equivalence tests.",
    acceptable_basis="Synthetic basis.",
    note="",
    observation=equivalence_observation,
)

equivalence_compiled = compile_series_comparison_suite(
    cases=[equivalence_case],
    summary={},
    metrics=[],
    protocol_evidence={"fi_curve_rows": equivalence_model_rows},
    suite_name="synthetic equivalence suite",
)
equivalence_judged = equivalence_compiled.judge()
assert [score.status for _case, score in equivalence_judged] == ["PASS"]
equivalence_items = audit_items_from_series_comparison_suite(equivalence_compiled)
assert "MAE " in equivalence_items[0].evidence["suite_cases"][0]["score_text"]
assert "TOST p " in equivalence_items[0].evidence["suite_cases"][0]["score_text"]
assert equivalence_items[1].evidence["score_family"] == "hybrid_residual_equivalence"
assert equivalence_items[1].evidence["statistical_test_family"] == "equivalence_tost"
assert equivalence_items[1].evidence["statistical_test_kinds"] == ["welch_tost", "welch_tost"]
assert equivalence_items[1].evidence["pvalue_aggregation"] == "max"
assert equivalence_items[1].evidence["pvalue_aggregation_source"] == "auto_default"
assert equivalence_items[1].evidence["equivalence_margin"] == 0.2
assert equivalence_items[1].evidence["declared_equivalence_margin"] is None
assert equivalence_items[1].evidence["equivalence_margin_Hz"] == 0.2
assert equivalence_items[1].evidence["declared_equivalence_margin_Hz"] is None
assert equivalence_items[1].evidence["equivalence_margin_source"] == "maximum_mae_default"
assert equivalence_items[1].evidence["supported_statistical_bin_count"] == 2
assert equivalence_items[1].evidence["unsupported_statistical_bin_count"] == 0
assert equivalence_items[1].evidence["statistical_gate_passed"] is True
assert equivalence_items[1].evidence["aggregate_statistical_pvalue"] <= 0.05
assert equivalence_items[1].evidence["residual_norm_score"] == 1.0
assert equivalence_items[1].evidence["statistical_norm_score"] == 1.0
assert equivalence_items[1].evidence["overall_norm_score"] == 1.0

singleton_equivalence_observation = SeriesDistributionObservation(
    protocol_evidence_key="fi_curve_rows",
    reference_rows=equivalence_reference_rows,
    reference_x_key="current_pA",
    reference_y_key="firing_rate_Hz",
    model_x_key="current_pA",
    model_y_key="firing_rate_Hz",
    reference_x_unit_text="pA",
    reference_y_unit_text="Hz",
    model_x_unit_text="pA",
    model_y_unit_text="Hz",
    comparison_x_unit_text="pA",
    comparison_y_unit_text="Hz",
    policy=SeriesComparisonPolicy(
        minimum_point_count=2,
        maximum_mae=0.2,
        maximum_rmse=0.2,
        equivalence_margin=0.2,
        score_family="equivalence_only",
    ),
)

singleton_equivalence_case = SeriesComparisonCase(
    check_id="synthetic_singleton_series_equivalence",
    title="Synthetic singleton model bins still use one-sample equivalence when supported",
    criterion="The aligned singleton model bins should still support one-sample equivalence tests.",
    criterion_latex="",
    criterion_formulae=[],
    criterion_definitions=[],
    description="Synthetic one-sample equivalence suite test.",
    acceptable="The matched bins satisfy the per-bin one-sample equivalence tests.",
    acceptable_basis="Synthetic basis.",
    note="",
    observation=singleton_equivalence_observation,
)

singleton_equivalence_compiled = compile_series_comparison_suite(
    cases=[singleton_equivalence_case],
    summary={},
    metrics=[],
    protocol_evidence={"fi_curve_rows": equivalence_singleton_model_rows},
    suite_name="synthetic singleton equivalence suite",
)
singleton_equivalence_items = audit_items_from_series_comparison_suite(singleton_equivalence_compiled)
assert singleton_equivalence_items[1].status == "PASS"
assert singleton_equivalence_items[0].evidence["suite_cases"][0]["score_text"].startswith("TOST p ")
assert singleton_equivalence_items[1].evidence["statistical_test_kinds"] == [
    "one_sample_reference_tost",
    "one_sample_reference_tost",
]
assert singleton_equivalence_items[1].evidence["equivalence_margin_source"] == "explicit"
assert singleton_equivalence_items[1].evidence["statistical_gate_passed"] is True

piecewise_observation = SeriesDistributionObservation(
    protocol_evidence_key="fi_curve_rows",
    reference_rows=piecewise_reference_rows,
    reference_x_key="current_pA",
    reference_y_key="firing_rate_Hz",
    model_x_key="drive_flux",
    model_y_key="firing_rate_Hz",
    reference_x_unit_text="pA",
    reference_y_unit_text="Hz",
    model_x_unit_text="",
    model_y_unit_text="Hz",
    comparison_x_unit_text="pA",
    comparison_y_unit_text="Hz",
    model_x_transform=AxisTransform(
        kind="piecewise_linear",
        output_unit_text="pA",
        points=((0.10, 100.0), (0.20, 210.0), (0.30, 330.0)),
    ),
    policy=SeriesComparisonPolicy(
        minimum_point_count=3,
        maximum_mae=0.2,
        maximum_rmse=0.2,
        score_family="residual_only",
    ),
)

piecewise_case = SeriesComparisonCase(
    check_id="synthetic_piecewise_series_match",
    title="Synthetic piecewise transform matches a non-affine x-axis mapping",
    criterion="The declared piecewise-linear transform should align the model and reference x axes before residual comparison.",
    criterion_latex="",
    criterion_formulae=[],
    criterion_definitions=[],
    description="Synthetic piecewise-transform suite test.",
    acceptable="The transformed model bins satisfy the configured residual tolerances.",
    acceptable_basis="Synthetic basis.",
    note="",
    observation=piecewise_observation,
)

piecewise_compiled = compile_series_comparison_suite(
    cases=[piecewise_case],
    summary={},
    metrics=[],
    protocol_evidence={"fi_curve_rows": piecewise_model_rows},
    suite_name="synthetic piecewise series suite",
)
piecewise_items = audit_items_from_series_comparison_suite(piecewise_compiled)
assert piecewise_items[1].status == "PASS"
assert piecewise_items[1].evidence["currents_pA"] == [100.0, 210.0, 330.0]
assert piecewise_items[1].evidence["model_x_transform"].startswith("piecewise_linear(")

cluster_observation = SeriesDistributionObservation(
    protocol_evidence_key="fi_curve_rows",
    reference_rows=cluster_reference_rows,
    reference_x_key="current_pA",
    reference_y_key="firing_rate_Hz",
    model_x_key="current_pA",
    model_y_key="firing_rate_Hz",
    reference_x_unit_text="pA",
    reference_y_unit_text="Hz",
    model_x_unit_text="pA",
    model_y_unit_text="Hz",
    comparison_x_unit_text="pA",
    comparison_y_unit_text="Hz",
    policy=SeriesComparisonPolicy(
        minimum_point_count=2,
        maximum_mae=0.2,
        maximum_rmse=0.2,
        alignment_policy="tolerance_clusters",
        x_match_tolerance=0.5,
        score_family="residual_only",
    ),
)

cluster_case = SeriesComparisonCase(
    check_id="synthetic_clustered_series_match",
    title="Synthetic clustered alignment pools nearby x bins into shared comparison groups",
    criterion="Nearby transformed x values should be pooled into shared tolerance clusters before comparison.",
    criterion_latex="",
    criterion_formulae=[],
    criterion_definitions=[],
    description="Synthetic tolerance-cluster suite test.",
    acceptable="The pooled cluster means satisfy the configured residual tolerances.",
    acceptable_basis="Synthetic basis.",
    note="",
    observation=cluster_observation,
)

cluster_compiled = compile_series_comparison_suite(
    cases=[cluster_case],
    summary={},
    metrics=[],
    protocol_evidence={"fi_curve_rows": cluster_model_rows},
    suite_name="synthetic cluster series suite",
)
cluster_items = audit_items_from_series_comparison_suite(cluster_compiled)
assert cluster_items[1].status == "PASS"
assert cluster_items[1].evidence["alignment_policy"] == "tolerance_clusters"
assert cluster_items[1].evidence["currents_pA"] == [100.2, 200.2]
assert cluster_items[1].evidence["reference_cluster_x_groups"] == [[100.0, 100.3], [200.0, 200.3]]
assert cluster_items[1].evidence["model_cluster_x_groups"] == [[100.1, 100.4], [200.1, 200.4]]
assert cluster_items[1].evidence["reference_count_values"] == [2, 2]
assert cluster_items[1].evidence["model_count_values"] == [2, 2]

resampled_observation = SeriesDistributionObservation(
    protocol_evidence_key="fi_curve_rows",
    reference_rows=resampled_reference_rows,
    reference_x_key="current_pA",
    reference_y_key="firing_rate_Hz",
    model_x_key="current_pA",
    model_y_key="firing_rate_Hz",
    reference_x_unit_text="pA",
    reference_y_unit_text="Hz",
    model_x_unit_text="pA",
    model_y_unit_text="Hz",
    comparison_x_unit_text="pA",
    comparison_y_unit_text="Hz",
    policy=SeriesComparisonPolicy(
        minimum_point_count=4,
        maximum_mae=0.01,
        maximum_rmse=0.01,
        alignment_policy="resampled_grid",
        score_family="residual_only",
    ),
)

resampled_case = SeriesComparisonCase(
    check_id="synthetic_resampled_series_match",
    title="Synthetic resampled alignment interpolates series onto a shared comparison grid",
    criterion="A shared resampling grid should align offset series before residual comparison.",
    criterion_latex="",
    criterion_formulae=[],
    criterion_definitions=[],
    description="Synthetic resampled-grid suite test.",
    acceptable="The interpolated comparison grid satisfies the configured residual tolerances.",
    acceptable_basis="Synthetic basis.",
    note="",
    observation=resampled_observation,
)

resampled_compiled = compile_series_comparison_suite(
    cases=[resampled_case],
    summary={},
    metrics=[],
    protocol_evidence={"fi_curve_rows": resampled_model_rows},
    suite_name="synthetic resampled series suite",
)
resampled_items = audit_items_from_series_comparison_suite(resampled_compiled)
assert resampled_items[1].status == "PASS"
assert resampled_items[1].evidence["alignment_policy"] == "resampled_grid"
assert resampled_items[1].evidence["declared_resampling_grid_source"] == ""
assert resampled_items[1].evidence["resampling_grid_source"] == "union_observed_x"
assert resampled_items[1].evidence["resampling_grid_source_origin"] == "default_union_observed_x"
assert resampled_items[1].evidence["currents_pA"] == [125.0, 200.0, 225.0, 300.0]
assert resampled_items[1].evidence["matched_point_count"] == 4
assert resampled_items[1].evidence["reference_resampled_support_counts"] == [2, 2, 2, 2]
assert resampled_items[1].evidence["model_resampled_support_counts"] == [2, 2, 2, 2]
assert resampled_items[1].evidence["mean_absolute_error"] == 0.0
assert resampled_items[1].evidence["mean_absolute_error_Hz"] == 0.0

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

assert len(items) == 2
assert items[0].detail_level == "summary"
assert items[1].status == "PASS"
assert items[1].series_visuals[0]["keys"] == ["currents_pA", "reference_values_Hz", "model_values_Hz"]
assert items[1].evidence["median_welch_pvalue"] is not None
assert items[1].evidence["score_family"] == "hybrid_residual_welch"
assert items[1].evidence["pvalue_aggregation"] == "median"
assert items[1].evidence["pvalue_aggregation_source"] == "auto_default"
assert items[1].evidence["reference_provenance"]["protocol_ids"] == ["SYNTHETIC_PROTOCOL"]
assert items[1].evidence["model_provenance"]["protocol_context"]["target_vm_mV"] == -60.0

residual_only_rule = dict(rule)
residual_only_rule["score_family"] = "residual_only"
del residual_only_rule["minimum_median_welch_pvalue"]
rules_module._load_rows = lambda loader_spec: reference_rows if loader_spec == "csv:/tmp/unused.csv" else original_load_rows(loader_spec)
try:
    residual_items = build_rule_items([residual_only_rule], context)
finally:
    rules_module._load_rows = original_load_rows

assert len(residual_items) == 2
assert residual_items[1].status == "PASS"
assert residual_items[1].evidence["score_family"] == "residual_only"
assert residual_items[1].evidence["minimum_median_welch_pvalue"] is None

second_residual_rule = dict(residual_only_rule)
second_residual_rule["check_id"] = "synthetic_series_match_second"
second_residual_rule["title"] = "Synthetic second series comparison"
second_residual_rule["criterion"] = "A second synthetic series check should share the same compiled suite."
rules_module._load_rows = lambda loader_spec: reference_rows if loader_spec == "csv:/tmp/unused.csv" else original_load_rows(loader_spec)
try:
    grouped_series_items = build_rule_items([residual_only_rule, second_residual_rule], context)
finally:
    rules_module._load_rows = original_load_rows

assert len(grouped_series_items) == 3
assert grouped_series_items[0].detail_level == "summary"
assert grouped_series_items[0].summary_rollup_exempt is True
assert grouped_series_items[0].evidence["suite_case_count"] == 2
assert grouped_series_items[0].evidence["suite_norm_score_summary"]["mean"] == 1.0
assert [item.check_id for item in grouped_series_items[1:]] == [
    "synthetic_series_match",
    "synthetic_series_match_second",
]

equivalence_rule = dict(rule)
equivalence_rule["loader"] = "csv:/tmp/equivalence.csv"
equivalence_rule["model_current_key"] = "current_pA"
equivalence_rule["model_x_unit_text"] = "pA"
equivalence_rule["score_family"] = "hybrid_residual_equivalence"
equivalence_rule["maximum_mae"] = 0.2
equivalence_rule["maximum_rmse"] = 0.2
del equivalence_rule["minimum_median_welch_pvalue"]
del equivalence_rule["model_x_transform"]
equivalence_context = ValidationRuleContext(
    metrics=[],
    summary={},
    args=Namespace(),
    config={"validation_id": "synthetic_series_validation"},
    protocol_result=SimpleNamespace(
        protocol_evidence={"fi_curve_rows": equivalence_model_rows}
    ),
)
rules_module._load_rows = (
    lambda loader_spec: equivalence_reference_rows
    if loader_spec == "csv:/tmp/equivalence.csv"
    else original_load_rows(loader_spec)
)
try:
    equivalence_rule_items = build_rule_items([equivalence_rule], equivalence_context)
finally:
    rules_module._load_rows = original_load_rows

assert len(equivalence_rule_items) == 2
assert equivalence_rule_items[1].status == "PASS"
assert equivalence_rule_items[1].evidence["score_family"] == "hybrid_residual_equivalence"
assert equivalence_rule_items[1].evidence["pvalue_aggregation"] == "max"
assert equivalence_rule_items[1].evidence["pvalue_aggregation_source"] == "auto_default"
assert equivalence_rule_items[1].evidence["equivalence_margin"] == 0.2
assert equivalence_rule_items[1].evidence["equivalence_margin_Hz"] == 0.2
assert equivalence_rule_items[1].evidence["equivalence_margin_source"] == "maximum_mae_default"
assert equivalence_rule_items[1].evidence["statistical_test_family"] == "equivalence_tost"
assert equivalence_rule_items[1].evidence["statistical_gate_passed"] is True

piecewise_rule = dict(residual_only_rule)
piecewise_rule["loader"] = "csv:/tmp/piecewise.csv"
piecewise_rule["reference_current_key"] = "current_pA"
piecewise_rule["model_current_key"] = "drive_flux"
piecewise_rule["model_x_unit_text"] = ""
piecewise_rule["maximum_mae"] = 0.2
piecewise_rule["maximum_rmse"] = 0.2
piecewise_rule["minimum_point_count"] = 3
piecewise_rule["model_x_transform"] = {
    "kind": "piecewise_linear",
    "output_unit_text": "pA",
    "points": [
        {"input": 0.10, "output": 100.0},
        {"input": 0.20, "output": 210.0},
        {"input": 0.30, "output": 330.0},
    ],
}
piecewise_context = ValidationRuleContext(
    metrics=[],
    summary={},
    args=Namespace(),
    config={"validation_id": "synthetic_series_validation"},
    protocol_result=SimpleNamespace(protocol_evidence={"fi_curve_rows": piecewise_model_rows}),
)
rules_module._load_rows = (
    lambda loader_spec: piecewise_reference_rows
    if loader_spec == "csv:/tmp/piecewise.csv"
    else original_load_rows(loader_spec)
)
try:
    piecewise_rule_items = build_rule_items([piecewise_rule], piecewise_context)
finally:
    rules_module._load_rows = original_load_rows

assert piecewise_rule_items[1].status == "PASS"
assert piecewise_rule_items[1].evidence["model_x_transform"].startswith("piecewise_linear(")

cluster_rule = dict(residual_only_rule)
cluster_rule["loader"] = "csv:/tmp/cluster.csv"
cluster_rule["model_current_key"] = "current_pA"
cluster_rule["model_x_unit_text"] = "pA"
cluster_rule["alignment_policy"] = "tolerance_clusters"
cluster_rule["x_match_tolerance"] = 0.5
cluster_rule["maximum_mae"] = 0.2
cluster_rule["maximum_rmse"] = 0.2
del cluster_rule["model_x_transform"]
cluster_context = ValidationRuleContext(
    metrics=[],
    summary={},
    args=Namespace(),
    config={"validation_id": "synthetic_series_validation"},
    protocol_result=SimpleNamespace(protocol_evidence={"fi_curve_rows": cluster_model_rows}),
)
rules_module._load_rows = (
    lambda loader_spec: cluster_reference_rows
    if loader_spec == "csv:/tmp/cluster.csv"
    else original_load_rows(loader_spec)
)
try:
    cluster_rule_items = build_rule_items([cluster_rule], cluster_context)
finally:
    rules_module._load_rows = original_load_rows

assert cluster_rule_items[1].status == "PASS"
assert cluster_rule_items[1].evidence["alignment_policy"] == "tolerance_clusters"
assert cluster_rule_items[1].evidence["reference_cluster_x_groups"] == [[100.0, 100.3], [200.0, 200.3]]
assert cluster_rule_items[1].evidence["model_cluster_x_groups"] == [[100.1, 100.4], [200.1, 200.4]]

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

assert len(nearest_items) == 2
assert nearest_items[1].status == "PASS"
assert nearest_items[1].evidence["alignment_policy"] == "nearest_within_tolerance"
assert nearest_items[1].evidence["x_match_tolerance"] == 0.5
assert nearest_items[1].evidence["reference_matched_x_values"] == [100.0, 200.0]
assert nearest_items[1].evidence["model_matched_x_values"] == [100.4, 200.4]
assert nearest_items[1].evidence["matched_x_differences"] == [0.4, 0.4]

resampled_rule = dict(residual_only_rule)
resampled_rule["loader"] = "csv:/tmp/resampled.csv"
resampled_rule["model_current_key"] = "current_pA"
resampled_rule["model_x_unit_text"] = "pA"
resampled_rule["alignment_policy"] = "resampled_grid"
resampled_rule["maximum_mae"] = 0.01
resampled_rule["maximum_rmse"] = 0.01
resampled_rule["minimum_point_count"] = 2
del resampled_rule["model_x_transform"]
resampled_rule["resampling_grid_source"] = "explicit_grid"
resampled_rule["resampling_grid_values"] = [150.0, 250.0]
resampled_context = ValidationRuleContext(
    metrics=[],
    summary={},
    args=Namespace(),
    config={"validation_id": "synthetic_series_validation"},
    protocol_result=SimpleNamespace(protocol_evidence={"fi_curve_rows": resampled_model_rows}),
)
rules_module._load_rows = (
    lambda loader_spec: resampled_reference_rows
    if loader_spec == "csv:/tmp/resampled.csv"
    else original_load_rows(loader_spec)
)
try:
    resampled_rule_items = build_rule_items([resampled_rule], resampled_context)
finally:
    rules_module._load_rows = original_load_rows

assert resampled_rule_items[1].status == "PASS"
assert resampled_rule_items[1].evidence["alignment_policy"] == "resampled_grid"
assert resampled_rule_items[1].evidence["resampling_grid_source"] == "explicit_grid"
assert resampled_rule_items[1].evidence["declared_resampling_grid_values"] == [150.0, 250.0]
assert resampled_rule_items[1].evidence["currents_pA"] == [150.0, 250.0]
assert resampled_rule_items[1].evidence["matched_point_count"] == 2

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

missing_cluster_tolerance_rule = dict(cluster_rule)
del missing_cluster_tolerance_rule["x_match_tolerance"]
rules_module._load_rows = lambda loader_spec: cluster_reference_rows if loader_spec == "csv:/tmp/cluster.csv" else original_load_rows(loader_spec)
try:
    try:
        build_rule_items([missing_cluster_tolerance_rule], cluster_context)
        raise AssertionError("Expected tolerance-cluster alignment to require an explicit x-match tolerance")
    except ValueError as exc:
        assert "requires explicit 'x_match_tolerance'" in str(exc)
finally:
    rules_module._load_rows = original_load_rows

missing_explicit_resampling_grid_rule = dict(resampled_rule)
del missing_explicit_resampling_grid_rule["resampling_grid_values"]
rules_module._load_rows = (
    lambda loader_spec: resampled_reference_rows
    if loader_spec == "csv:/tmp/resampled.csv"
    else original_load_rows(loader_spec)
)
try:
    try:
        build_rule_items([missing_explicit_resampling_grid_rule], resampled_context)
        raise AssertionError("Expected explicit-grid resampling to require explicit grid values")
    except ValueError as exc:
        assert "requires explicit 'resampling_grid_values'" in str(exc)
finally:
    rules_module._load_rows = original_load_rows


print("neuronunit_series_validation_suite: OK")
