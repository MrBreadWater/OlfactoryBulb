"""Unit tests for bounded and skew-aware reference-band validation rules."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import tempfile

from olfactorybulb.audit.reference_validation_rules import (
    ValidationRuleContext,
    build_rule_items,
    compute_reference_acceptance_band,
)


log_band = compute_reference_acceptance_band(
    reference_mean=0.45,
    reference_sd=0.29,
    sigma_multiplier=2.0,
    band_mode="lognormal_sd",
    lower_bound=0.0,
)
assert 0.0 < log_band.low < 0.2
assert 1.0 < log_band.high < 1.3
assert "lognormal distribution" in log_band.description

clipped_band = compute_reference_acceptance_band(
    reference_mean=0.4,
    reference_sd=0.5,
    sigma_multiplier=2.0,
    band_mode="symmetric_sd",
    lower_bound=0.0,
    upper_bound=1.0,
)
assert clipped_band.low == 0.0
assert clipped_band.high == 1.0

with tempfile.TemporaryDirectory() as tmpdir:
    csv_path = Path(tmpdir) / "reference_rows.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Property,Source,cell_type,mean,sd,unit",
                "ISI Coefficient of Variation,Burton & Urban (2014),MC,0.45,0.29,",
                "Firing Probability,Example et al. (2026),MC,0.40,0.50,",
            ]
        )
    )
    context = ValidationRuleContext(
        metrics=[],
        summary={"MC": {"cv_isi": 0.021, "firing_probability": 0.2}},
        args=Namespace(reference_sigma_multiplier=2.0),
        config={},
        protocol_result=None,
    )
    log_rule = {
        "kind": "reference_band_rows",
        "loader": f"csv:{csv_path}",
        "reference_source": "Burton & Urban (2014)",
        "group_field": "cell_type",
        "sigma_arg_name": "reference_sigma_multiplier",
        "property_metric_map": {"ISI Coefficient of Variation": "cv_isi"},
        "property_band_modes": {"ISI Coefficient of Variation": "lognormal_sd"},
        "property_lower_bounds": {"ISI Coefficient of Variation": 0.0},
        "check_id": "placeholder",
        "title": "placeholder",
        "criterion": "placeholder",
        "description": "placeholder",
        "acceptable": "placeholder",
        "acceptable_basis": "placeholder",
    }
    log_item = build_rule_items([log_rule], context)[0]
    assert log_item.evidence["accepted_interval_mode"] == "lognormal_sd"
    assert log_item.evidence["accepted_low"] > 0.0
    assert "lognormal distribution" in log_item.acceptable_basis
    assert "not a formal confidence interval" in log_item.acceptable_basis

    clipped_rule = {
        "kind": "reference_band_rows",
        "loader": f"csv:{csv_path}",
        "reference_source": "Example et al. (2026)",
        "group_field": "cell_type",
        "sigma_arg_name": "reference_sigma_multiplier",
        "property_metric_map": {"Firing Probability": "firing_probability"},
        "property_lower_bounds": {"Firing Probability": 0.0},
        "property_upper_bounds": {"Firing Probability": 1.0},
        "check_id": "placeholder",
        "title": "placeholder",
        "criterion": "placeholder",
        "description": "placeholder",
        "acceptable": "placeholder",
        "acceptable_basis": "placeholder",
    }
    clipped_item = build_rule_items([clipped_rule], context)[0]
    assert clipped_item.evidence["accepted_low"] == 0.0
    assert clipped_item.evidence["accepted_high"] == 1.0
    assert clipped_item.evidence["unbounded_low"] < 0.0
    assert clipped_item.evidence["unbounded_high"] > 1.0

print("reference_validation_band_rules: OK")
