"""Coverage for typed scalar observation and prediction helpers."""

from __future__ import annotations

import quantities as pq

from olfactorybulb.neuronunit.metric_quantities import resolve_metric_quantity
from olfactorybulb.neuronunit.scalar_observations import (
    ScalarGroupPair,
    ScalarGroupValueSet,
    ScalarMetricValue,
    ScalarMetricValueMap,
    is_finite_scalar,
)


metric_quantity = resolve_metric_quantity("AP_onset_mV")

value = ScalarMetricValue(
    metric_quantity=metric_quantity,
    group="MC",
    value=-42.125 * pq.mV,
)
assert value.metric_key == "AP_onset_mV"
assert value.unit_text == "mV"
assert value.quantity_name == "AP Onset"
assert value.observed_symbol == r"\bar{V}_{\mathrm{th}}"
assert value.numeric == -42.125
assert value.observation_payload()["observed"] == -42.125
assert value.observation_payload()["group"] == "MC"
assert value.observation_payload()["metric_observed_symbol"] == r"\bar{V}_{\mathrm{th}}"

value_map = ScalarMetricValueMap(
    metric_quantity=resolve_metric_quantity("zero_step_rate_Hz"),
    entity_key="cell_name",
    values={
        "MC1": 0.0 * pq.Hz,
        "TC1": 1.0 * pq.Hz,
        "TC2": float("nan"),
    },
)
assert value_map.entity_count == 3
assert sorted(value_map.failing_nonfinite()) == ["TC2"]
assert sorted(value_map.failing_not_equal(expected=0.0, tolerance=1e-9)) == ["TC1", "TC2"]
assert value_map.metadata()["metric_observed_symbol"] == r"\bar{f}_{0}"

pair = ScalarGroupPair(
    metric_quantity=resolve_metric_quantity("FWHM_ms"),
    left_group="MC",
    right_group="TC",
    left_value=1.1 * pq.ms,
    right_value=0.9 * pq.ms,
)
assert pair.unit_text == "ms"
assert pair.left_numeric == 1.1
assert pair.right_numeric == 0.9
assert round(pair.delta, 6) == -0.2
assert round(pair.absolute_difference, 6) == 0.2
assert pair.metadata()["left_group"] == "MC"
assert pair.metadata()["metric_observed_symbol"] == r"\overline{\mathrm{FWHM}}"

group_values = ScalarGroupValueSet(
    metric_quantity=resolve_metric_quantity("rheobase_pA"),
    values_by_group={
        "MC": 100.0 * pq.pA,
        "TC": 0.0 * pq.pA,
    },
)
assert group_values.numeric_group_values() == {"MC": 100.0, "TC": 0.0}
assert group_values.failing_positive_groups() == ["TC"]
assert group_values.metadata()["metric_observed_symbol"] == r"\bar{I}_{\mathrm{rh}}"

assert is_finite_scalar(5.0)
assert is_finite_scalar(2.0 * pq.ms)
assert not is_finite_scalar(float("nan"))
assert not is_finite_scalar(None)

print("neuronunit_scalar_observations: OK")
