"""Targeted checks for maintained protocol-side metric helpers."""

from __future__ import annotations

import math
from argparse import Namespace

import numpy as np

from olfactorybulb.audit.reference_validation_config import load_reference_validation_config
from olfactorybulb.audit.reference_validation_protocols import (
    _afterdepolarization_depth_millivolts,
    _afterdepolarization_duration_milliseconds,
    _build_protocol_from_config,
)


resting_potential_mV = -65.0
synthetic_adp_trace = {
    "t": np.arange(0.0, 13.0, 1.0),
    "v_soma": np.array(
        [
            -65.0,
            -65.0,
            -65.0,
            10.0,
            40.0,
            20.0,
            0.0,
            -20.0,
            -40.0,
            -70.0,
            -68.0,
            -66.0,
            -65.0,
        ]
    ),
}

assert _afterdepolarization_duration_milliseconds(
    synthetic_adp_trace,
    resting_potential_millivolts=resting_potential_mV,
    step_delay_milliseconds=2.0,
) == 6.0
assert _afterdepolarization_depth_millivolts(
    synthetic_adp_trace,
    resting_potential_millivolts=resting_potential_mV,
    step_delay_milliseconds=2.0,
) == 5.0

no_spike_trace = {
    "t": np.arange(0.0, 8.0, 1.0),
    "v_soma": np.array([-65.0, -65.0, -64.0, -60.0, -58.0, -60.0, -63.0, -65.0]),
}
assert math.isnan(
    _afterdepolarization_duration_milliseconds(
        no_spike_trace,
        resting_potential_millivolts=resting_potential_mV,
        step_delay_milliseconds=2.0,
    )
)
assert math.isnan(
    _afterdepolarization_depth_millivolts(
        no_spike_trace,
        resting_potential_millivolts=resting_potential_mV,
        step_delay_milliseconds=2.0,
    )
)


gc_config = load_reference_validation_config(validation_id="gc_intrinsic_validation")
gc_protocol = _build_protocol_from_config(Namespace(dt_ms=0.1, bias_max_iterations=24), gc_config["protocol"])
assert gc_protocol.adp_enabled is True
assert gc_protocol.adp_current_duration_ms == 1.0
assert gc_protocol.adp_current_amplitude_nA == 1.0
assert gc_protocol.adp_tail_ms == 100.0
assert gc_protocol.adp_sampling_dt_ms == 0.125

epl_fsi_config = load_reference_validation_config(validation_id="epl_fsi_intrinsic_validation")
epl_fsi_protocol = _build_protocol_from_config(Namespace(dt_ms=0.1, bias_max_iterations=24), epl_fsi_config["protocol"])
assert epl_fsi_protocol.adp_enabled is False
assert epl_fsi_protocol.adp_current_duration_ms is None
assert epl_fsi_protocol.adp_current_amplitude_nA is None

print("reference_validation_protocol_metrics: OK")
