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
assert "lognormal interval reconstructed" in log_band.description
assert log_band.standard_label == "lognormal reference interval"

beta_band = compute_reference_acceptance_band(
    reference_mean=0.4,
    reference_sd=0.15,
    sigma_multiplier=2.0,
    band_mode="beta_sd",
)
assert 0.0 < beta_band.low < beta_band.high < 1.0
assert beta_band.standard_label == "bounded probability interval"
assert "beta-distribution interval" in beta_band.description

quantile_band = compute_reference_acceptance_band(
    reference_mean=10.0,
    reference_sd=3.0,
    sigma_multiplier=2.0,
    band_mode="quantile_interval",
    quantile_low=7.0,
    quantile_high=15.0,
    quantile_low_label="25th percentile",
    quantile_high_label="75th percentile",
)
assert quantile_band.low == 7.0
assert quantile_band.high == 15.0
assert quantile_band.standard_label == "reported quantile interval"
assert "25th percentile" in quantile_band.description

binary_band = compute_reference_acceptance_band(
    reference_mean=1.0,
    reference_sd=1.0,
    sigma_multiplier=2.0,
    band_mode="binary_indicator",
)
assert binary_band.low == 1.0
assert binary_band.high == 1.0
assert binary_band.standard_label == "exact binary indicator"
assert "categorical rather than continuous" in binary_band.description

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
                "Property,Source,cell_type,mean,sd,unit,q_low,q_high,q_low_label,q_high_label",
                "ISI Coefficient of Variation,Burton & Urban (2014),MC,0.45,0.29,",
                "Firing Probability,Example et al. (2026),MC,0.40,0.15,",
                "Skewed Latency,Example et al. (2026),MC,10.0,3.0,ms,7.0,15.0,25th percentile,75th percentile",
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
    assert log_item.evidence["accepted_interval_standard"] == "lognormal reference interval"
    assert log_item.evidence["accepted_low"] > 0.0
    assert log_item.evidence["reference_mean"] == 0.45
    assert "lognormal interval reconstructed" in log_item.acceptable_basis
    assert "configured lognormal reference interval" in log_item.acceptable_basis
    assert (
        log_item.criterion_latex
        == r"\lvert \ln\!\left(\frac{\overline{\mathrm{CV}}_{\mathrm{ISI}}\sqrt{\mu^2 + \sigma^2}}{\mu^2}\right) \rvert \leq 2\sqrt{\ln(1 + (\sigma / \mu)^2)}"
    )
    assert log_item.criterion_definitions[0]["symbol"] == r"\overline{\mathrm{CV}}_{\mathrm{ISI}}"

    beta_rule = {
        "kind": "reference_band_rows",
        "loader": f"csv:{csv_path}",
        "reference_source": "Example et al. (2026)",
        "group_field": "cell_type",
        "sigma_arg_name": "reference_sigma_multiplier",
        "property_metric_map": {"Firing Probability": "firing_probability"},
        "property_band_modes": {"Firing Probability": "beta_sd"},
        "check_id": "placeholder",
        "title": "placeholder",
        "criterion": "placeholder",
        "description": "placeholder",
        "acceptable": "placeholder",
        "acceptable_basis": "placeholder",
    }
    beta_item = build_rule_items([beta_rule], context)[0]
    assert beta_item.evidence["accepted_interval_mode"] == "beta_sd"
    assert beta_item.evidence["accepted_interval_standard"] == "bounded probability interval"
    assert 0.0 < beta_item.evidence["accepted_low"] < beta_item.evidence["accepted_high"] < 1.0
    assert "beta-distribution interval" in beta_item.acceptable_basis

    quantile_rule = {
        "kind": "reference_band_rows",
        "loader": f"csv:{csv_path}",
        "reference_source": "Example et al. (2026)",
        "group_field": "cell_type",
        "sigma_arg_name": "reference_sigma_multiplier",
        "property_metric_map": {"Skewed Latency": "skewed_latency_ms"},
        "property_band_modes": {"Skewed Latency": "quantile_interval"},
        "check_id": "placeholder",
        "title": "placeholder",
        "criterion": "placeholder",
        "description": "placeholder",
        "acceptable": "placeholder",
        "acceptable_basis": "placeholder",
    }
    quantile_context = ValidationRuleContext(
        metrics=[],
        summary={"MC": {"skewed_latency_ms": 9.0}},
        args=Namespace(reference_sigma_multiplier=2.0),
        config={},
        protocol_result=None,
    )
    quantile_item = build_rule_items([quantile_rule], quantile_context)[0]
    assert quantile_item.evidence["accepted_interval_mode"] == "quantile_interval"
    assert quantile_item.evidence["accepted_interval_standard"] == "reported quantile interval"
    assert quantile_item.evidence["accepted_low"] == 7.0
    assert quantile_item.evidence["accepted_high"] == 15.0
    assert "25th percentile" in quantile_item.acceptable_basis

    binary_rule = {
        "kind": "reference_band_rows",
        "loader": f"csv:{csv_path}",
        "reference_source": "Example et al. (2026)",
        "group_field": "cell_type",
        "sigma_arg_name": "reference_sigma_multiplier",
        "property_metric_map": {"Firing Probability": "firing_probability"},
        "property_band_modes": {"Firing Probability": "binary_indicator"},
        "check_id": "placeholder",
        "title": "placeholder",
        "criterion": "placeholder",
        "description": "placeholder",
        "acceptable": "placeholder",
        "acceptable_basis": "placeholder",
    }
    binary_context = ValidationRuleContext(
        metrics=[],
        summary={"MC": {"firing_probability": 1.0}},
        args=Namespace(reference_sigma_multiplier=2.0),
        config={},
        protocol_result=None,
    )
    binary_csv_path = Path(tmpdir) / "binary_reference_rows.csv"
    binary_csv_path.write_text(
        "\n".join(
            [
                "Property,Source,cell_type,mean,sd,unit",
                "Firing Probability,Example et al. (2026),MC,1.0,1.0,",
            ]
        )
    )
    binary_rule["loader"] = f"csv:{binary_csv_path}"
    binary_item = build_rule_items([binary_rule], binary_context)[0]
    assert binary_item.evidence["accepted_interval_mode"] == "binary_indicator"
    assert binary_item.evidence["accepted_interval_standard"] == "exact binary indicator"
    assert "binary reference indicator exactly" in binary_item.criterion

    missing_mode_rule = {
        "kind": "reference_band_rows",
        "loader": f"csv:{csv_path}",
        "reference_source": "Burton & Urban (2014)",
        "group_field": "cell_type",
        "sigma_arg_name": "reference_sigma_multiplier",
        "property_metric_map": {"ISI Coefficient of Variation": "cv_isi"},
        "property_band_modes": {},
        "check_id": "placeholder",
        "title": "placeholder",
        "criterion": "placeholder",
        "description": "placeholder",
        "acceptable": "placeholder",
        "acceptable_basis": "placeholder",
    }
    try:
        build_rule_items([missing_mode_rule], context)
        raise AssertionError("Expected explicit property-band mode enforcement to fail")
    except ValueError as exc:
        assert "explicit band mode for every property" in str(exc)

print("reference_validation_band_rules: OK")
