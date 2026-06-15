"""Coverage for the SciUnit-backed reference-band validation bridge."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import quantities as pq

from olfactorybulb.audit.core import AuditReport
from olfactorybulb.audit.reference_validation_rules import ValidationRuleContext, build_rule_items
from olfactorybulb.neuronunit.reference_bands import (
    ReferenceBandObservation,
    ReferenceBandPolicy,
    ValidationReview,
    numeric_value,
)
from olfactorybulb.neuronunit.reference_validation_suite import (
    ReferenceBandCase,
    ReferenceBandTest,
    audit_items_from_reference_band_suite,
    compile_reference_band_suite,
)


case = ReferenceBandCase(
    check_id="input_resistance_within_band",
    title="Input resistance stays within the uploaded reference band",
    criterion="The mitral cell mean input resistance should remain within two standard deviations of the uploaded reference value.",
    criterion_latex="|\\bar{R}_{\\mathrm{in}} - \\mu_{\\mathrm{ref}}| \\le 2\\sigma_{\\mathrm{ref}}",
    criterion_formulae=[],
    criterion_definitions=[
        {"symbol": "\\bar{R}_{\\mathrm{in}}", "definition": "mitral-cell mean input resistance"},
        {"symbol": "\\mu_{\\mathrm{ref}}", "definition": "uploaded reference mean input resistance"},
        {"symbol": "\\sigma_{\\mathrm{ref}}", "definition": "uploaded reference standard deviation"},
    ],
    description="Synthetic unit-conversion smoke test.",
    acceptable="The observed mean stays inside the accepted interval.",
    acceptable_basis="The thresholds are declared directly in the synthetic suite fixture.",
    note="",
    observation=ReferenceBandObservation(
        property_name="Input Resistance",
        group="MC",
        metric_key="input_resistance_MOhm",
        reference_mean=100.0,
        reference_sd=10.0,
        unit_text="MOhm",
        policy=ReferenceBandPolicy(mode="symmetric_sd", sigma_multiplier=2.0),
        review=ValidationReview(status="approved", reviewer="qualified_domain_expert"),
    ),
    reference_annotation="reference: 100 +/- 10 MOhm from Synthetic Study (n=12)",
)

compiled = compile_reference_band_suite(
    cases=[case],
    summary={"MC": {"input_resistance_MOhm": 102.0}},
    suite_name="Synthetic reference-band suite",
)
judged = compiled.judge()
assert len(judged) == 1
score = judged[0][1]
assert score.passed is True
assert numeric_value(score.observed) == 102.0

unit_conversion_score = ReferenceBandTest(case).compute_score(
    {},
    100000.0 * pq.kOhm,
)
assert unit_conversion_score.passed is True
assert round(numeric_value(unit_conversion_score.observed), 6) == 100.0

adapted_items = audit_items_from_reference_band_suite(compiled)
assert len(adapted_items) == 2
assert adapted_items[0].check_id == "Synthetic_reference-band_suite.overview"
assert adapted_items[0].detail_level == "summary"
assert adapted_items[0].summary_rollup_exempt is True
assert adapted_items[0].companion_visuals[0]["kind"] == "status_matrix"
assert adapted_items[0].evidence["suite_status_summary"] == {"PASS": 1, "WARN": 0, "FAIL": 0}
report = AuditReport(audit_id="synthetic_reference_band", title="Synthetic reference band", items=adapted_items)
assert report.summary == {"PASS": 1, "WARN": 0, "FAIL": 0}
adapted_item = adapted_items[1]
assert adapted_item.status == "PASS"
assert adapted_item.validation_design_review_status == "approved"
assert adapted_item.evidence["accepted_interval_standard"] == "symmetric reference interval"
assert adapted_item.evidence["MC_mean"] == 102.0


with tempfile.TemporaryDirectory() as tmpdir:
    csv_path = Path(tmpdir) / "reference_rows.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Property,mean,sd,Source,cell_type,unit,n,source_file,source_location,source_url,extraction_method,note_ids,reported_value_raw",
                "Input Resistance,100,10,Synthetic Study,MC,MOhm,12,synthetic.csv,Table 1,https://example.com,curated,N_SYNTHETIC,100 +/- 10 MOhm",
            ]
        )
        + "\n"
    )
    rule = {
        "kind": "reference_band_rows",
        "loader": f"csv:{csv_path}",
        "reference_source": "Synthetic Study",
        "group_field": "cell_type",
        "property_metric_map": {"Input Resistance": "input_resistance_MOhm"},
        "property_band_modes": {"Input Resistance": "symmetric_sd"},
        "property_validation_design_review_statuses": {"Input Resistance": "approved"},
        "property_validation_design_review_notes": {"Input Resistance": "Manually reviewed."},
        "validation_design_review_reviewer": "qualified_domain_expert",
        "title": "Synthetic reference-band rows",
        "criterion": "unused prose fallback",
        "description": "Synthetic rule fixture.",
        "acceptable": "Synthetic acceptable text.",
        "acceptable_basis": "Synthetic acceptable basis.",
        "property_notes": {"Input Resistance": "Synthetic note."},
    }
    context = ValidationRuleContext(
        metrics=[{"cell_type": "MC", "input_resistance_MOhm": 105.0}],
        summary={"MC": {"input_resistance_MOhm": 105.0}},
        args=SimpleNamespace(reference_sigma_multiplier=2.0),
        config={},
        protocol_result=None,
    )
    items = build_rule_items([rule], context)
    assert len(items) == 2
    assert items[0].detail_level == "summary"
    assert items[0].summary_rollup_exempt is True
    item = items[1]
    assert item.status == "PASS"
    assert item.title == "MC input resistance stays within the uploaded reference band"
    assert item.note == "Synthetic note."
    assert item.validation_design_review_status == "approved"
    assert item.validation_design_review_note == "Manually reviewed."
    assert item.evidence["MC_mean"] == 105.0
    assert item.evidence["accepted_low"] == 80.0
    assert item.evidence["accepted_high"] == 120.0
    assert item.evidence["accepted_interval_mode"] == "symmetric_sd"
    assert item.evidence["accepted_interval_standard"] == "symmetric reference interval"
    assert item.evidence["reference_unit"] == "MOhm"
    assert "reference: 100.0 +/- 10.0 MOhm from Synthetic Study" in item.evidence["__reference_annotations__"]["MC_mean"]
    assert item.criterion_latex == r"\left|\bar{R}_{\mathrm{in}} - \mu_{\mathrm{ref}}\right| \leq 2\sigma_{\mathrm{ref}}"


print("neuronunit_reference_validation_suite: OK")
