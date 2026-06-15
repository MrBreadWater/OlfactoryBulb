"""Coverage for typed scalar/reference evidence payload helpers."""

from __future__ import annotations

import quantities as pq

from olfactorybulb.neuronunit.metric_quantities import resolve_metric_quantity
from olfactorybulb.neuronunit.scalar_observations import (
    ScalarGroupPair,
    ScalarMetricValue,
    ScalarMetricValueMap,
    ScalarStatusMapPolicy,
)
from olfactorybulb.neuronunit.validation_evidence import (
    ReferenceBandEvidencePayload,
    ScalarRuleEvidencePayload,
)


summary_payload = ScalarRuleEvidencePayload.summary_rule(
    observed=ScalarMetricValue(
        metric_quantity=resolve_metric_quantity("soma_diameter_um"),
        group="ungrouped",
        value=9.7 * pq.um,
    ),
    rule_kind="summary_metric_range",
    minimum=8.9,
    maximum=10.3,
    extra_metrics={"baseline_GCs_count": 41.0},
)
assert summary_payload.to_dict() == {
    "group": "ungrouped",
    "observed": 9.7,
    "minimum": 8.9,
    "maximum": 10.3,
    "baseline_GCs_count": 41.0,
    "metric_unit": "um",
    "metric_quantity_name": "Soma Diameter",
}

status_map_payload = ScalarRuleEvidencePayload.summary_rule(
    observed=ScalarMetricValue(
        metric_quantity=resolve_metric_quantity("epli_target_pattern_specificity_code"),
        group="ungrouped",
        value=1.0,
    ),
    rule_kind="summary_metric_status_map",
    status_map_policy=ScalarStatusMapPolicy(
        pass_values=(2.0,),
        warn_values=(1.0,),
        fail_values=(0.0,),
    ),
)
assert status_map_payload.to_dict()["warn_values"] == [1.0]
assert status_map_payload.to_dict()["default_status"] == "FAIL"

comparison_payload = ScalarRuleEvidencePayload.all_exact(
    prediction=ScalarMetricValueMap(
        metric_quantity=resolve_metric_quantity("zero_step_rate_Hz"),
        values={
            "MC1": 0.0 * pq.Hz,
            "TC1": 1.0 * pq.Hz,
        },
    ),
    expected=0.0,
    tolerance=1e-9,
    failing_values={"TC1": 1.0 * pq.Hz},
)
assert comparison_payload.to_dict() == {
    "metric_key": "zero_step_rate_Hz",
    "expected": 0.0,
    "tolerance": 0.0,
    "failing_values": {"TC1": 1.0},
    "metric_unit": "Hz",
    "metric_quantity_name": "Zero Step Rate",
}

pair_payload = ScalarRuleEvidencePayload.group_abs_diff_max(
    prediction=ScalarGroupPair(
        metric_quantity=resolve_metric_quantity("AP_onset_mV"),
        left_group="MC",
        right_group="TC",
        left_value=-42.0 * pq.mV,
        right_value=-42.5 * pq.mV,
    ),
    max_difference=5.0,
)
assert pair_payload.to_dict() == {
    "MC_mean": -42.0,
    "TC_mean": -42.5,
    "absolute_difference": 0.5,
    "max_difference": 5.0,
    "metric_unit": "mV",
    "metric_quantity_name": "AP Onset",
}

reference_payload = ReferenceBandEvidencePayload(
    observed_key="MC_mean",
    observed_value=102.0,
    reference_mean=100.0,
    reference_unit="MOhm",
    accepted_low=80.0,
    accepted_high=120.0,
    accepted_sigma_multiplier=2.0,
    accepted_interval_mode="symmetric_sd",
    accepted_interval_standard="symmetric reference interval",
    reference_annotation="reference: 100 +/- 10 MOhm from Synthetic Study (n=12)",
    accepted_lower_bound=50.0,
    unbounded_low=60.0,
)
assert reference_payload.to_dict() == {
    "MC_mean": 102.0,
    "reference_mean": 100.0,
    "reference_unit": "MOhm",
    "accepted_low": 80.0,
    "accepted_high": 120.0,
    "accepted_sigma_multiplier": 2.0,
    "accepted_interval_mode": "symmetric_sd",
    "accepted_interval_standard": "symmetric reference interval",
    "__reference_annotations__": {
        "MC_mean": "reference: 100 +/- 10 MOhm from Synthetic Study (n=12)",
    },
    "accepted_lower_bound": 50.0,
    "unbounded_low": 60.0,
}

print("neuronunit_validation_evidence: OK")
