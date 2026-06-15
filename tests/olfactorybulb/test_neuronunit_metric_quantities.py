"""Coverage for typed metric-quantity inference and overrides."""

from __future__ import annotations

from olfactorybulb.neuronunit.metric_quantities import (
    infer_metric_quantity_name,
    infer_metric_unit_text,
    metric_definition_text,
    resolve_metric_quantity,
)


assert infer_metric_unit_text("AP_onset_mV") == "mV"
assert infer_metric_unit_text("fi_slope_Hz_per_nA") == "Hz/nA"
assert infer_metric_unit_text("cv_isi") == ""

assert infer_metric_quantity_name("input_resistance_MOhm") == "Input Resistance"
assert infer_metric_quantity_name("GCs__MCs_entry_count") == "GCs MCs Entry Count"

ap_threshold = resolve_metric_quantity("AP_onset_mV")
assert ap_threshold.unit_text == "mV"
assert ap_threshold.resolved_quantity_name == "AP Onset"
assert ap_threshold.definition_label == "AP Onset (mV)"
assert metric_definition_text(ap_threshold, group="MC") == "MC mean AP Onset (mV)"

explicit = resolve_metric_quantity(
    "fi_gain_Hz_per_50pA",
    unit_text="Hz/50pA",
    quantity_name="FI gain",
)
assert explicit.unit_text == "Hz/50pA"
assert explicit.resolved_quantity_name == "FI gain"
assert explicit.definition_label == "FI gain (Hz/50pA)"
assert metric_definition_text(explicit, group="TC") == "TC mean FI gain (Hz/50pA)"

print("neuronunit_metric_quantities: OK")
