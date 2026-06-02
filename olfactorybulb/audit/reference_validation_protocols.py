"""Registered protocol runners for literature-backed single-cell validation."""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
from itertools import repeat
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from fi_curve_utils import find_spike_times_milliseconds
from prev_ob_models.cell_registry import get_cell_model_spec, load_cell_class


@dataclass(frozen=True)
class ProtocolRunResult:
    metrics: list[dict[str, Any]]
    protocol_evidence: dict[str, Any]
    group_field: str = "cell_type"


@dataclass(frozen=True)
class ValidationProtocolSpec:
    protocol_id: str
    title: str
    description: str
    add_cli_args: Callable[[argparse.ArgumentParser], None] | None
    run: Callable[[argparse.Namespace, dict[str, Any]], ProtocolRunResult]


PROTOCOL_SPECS: dict[str, ValidationProtocolSpec] = {}


def register_validation_protocol(spec: ValidationProtocolSpec) -> ValidationProtocolSpec:
    PROTOCOL_SPECS[spec.protocol_id] = spec
    return spec


def get_validation_protocol_spec(protocol_id: str) -> ValidationProtocolSpec:
    try:
        return PROTOCOL_SPECS[protocol_id]
    except KeyError as exc:
        known = ", ".join(sorted(PROTOCOL_SPECS))
        raise KeyError(f"Unknown reference validation protocol {protocol_id!r}. Known protocols: {known}") from exc


def iter_validation_protocol_specs() -> list[ValidationProtocolSpec]:
    return [PROTOCOL_SPECS[key] for key in sorted(PROTOCOL_SPECS)]


def _configure_parent_cache_dirs() -> None:
    user = os.environ.get("USER") or "obgpu"
    base_dir = os.path.join("/tmp", f"obgpu-audit-cache-{user}")
    mpl_dir = os.path.join(base_dir, "matplotlib")
    xdg_dir = os.path.join(base_dir, "xdg")
    font_dir = os.path.join(xdg_dir, "fontconfig")
    os.makedirs(mpl_dir, exist_ok=True)
    os.makedirs(font_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_dir)
    os.environ.setdefault("XDG_CACHE_HOME", xdg_dir)


@dataclass(frozen=True)
class BurtonUrbanProtocol:
    target_vm_mV: float | None = -58.0
    step_start_nA: float = 0.0
    step_stop_nA: float = 0.30
    step_increment_nA: float = 0.05
    step_duration_ms: float = 2000.0
    step_delay_ms: float = 200.0
    tail_ms: float = 200.0
    hyperpolarizing_start_nA: float = 0.0
    hyperpolarizing_stop_nA: float = -0.30
    hyperpolarizing_increment_nA: float = -0.05
    dt_ms: float = 0.1
    celsius: float = 35.0
    spike_threshold_mV: float = -20.0
    ap_derivative_threshold_mV_per_ms: float = 20.0
    cv_isi_target_rate_hz: float = 20.0
    bias_settle_ms: float = 1000.0
    bias_tolerance_mV: float = 0.1
    bias_max_iterations: int = 24

    @property
    def current_steps_nA(self) -> np.ndarray:
        return np.arange(
            self.step_start_nA,
            self.step_stop_nA + self.step_increment_nA * 0.5,
            self.step_increment_nA,
        )


def _cell_type_from_name(cell_name: str) -> str:
    return "".join(character for character in cell_name if character.isalpha())


def _cell_names(cell_types: Sequence[str], cell_count: int) -> list[str]:
    return [
        f"{cell_type}{cell_number}"
        for cell_type in cell_types
        for cell_number in range(1, cell_count + 1)
    ]


def _csv_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [token.strip() for token in value.replace(";", ",").split(",") if token.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [str(token).strip() for token in value if str(token).strip()]
    text = str(value).strip()
    return [text] if text else []


def _resolve_cell_specs(cell_models: Any, *, default_role: str | None = None) -> list[tuple[str, Any, str]]:
    resolved: list[tuple[str, Any, str]] = []
    for token in _csv_tokens(cell_models):
        if "." in token:
            spec = get_cell_model_spec(token)
            resolved.append((spec.class_name, load_cell_class(token), default_role or spec.role))
            continue
        resolved.append((token, token, default_role or _cell_type_from_name(token)))
    return resolved


def _resolved_jobs(cell_total: int, requested_jobs: int, *, use_gpu: bool = False) -> int:
    if cell_total <= 1:
        return 1
    if use_gpu:
        return 1
    if requested_jobs <= 0:
        requested_jobs = os.cpu_count() or 1
    return max(1, min(int(requested_jobs), int(cell_total)))


def _steady_state_voltage_millivolts(
    trace_result: dict[str, Any],
    *,
    step_delay_milliseconds: float,
    step_duration_milliseconds: float,
    window_milliseconds: float = 10.0,
) -> float:
    time_array = np.asarray(trace_result["t"], dtype=float)
    voltage_array = np.asarray(trace_result["v_soma"], dtype=float)
    step_end = step_delay_milliseconds + step_duration_milliseconds
    mask = (time_array >= max(step_delay_milliseconds, step_end - window_milliseconds)) & (time_array <= step_end)
    if not np.any(mask):
        return float("nan")
    return float(np.median(voltage_array[mask]))


def _trace_nearest_steady_state_voltage(
    traces_list: list[dict[str, Any]],
    *,
    target_voltage_millivolts: float,
    step_delay_milliseconds: float,
    step_duration_milliseconds: float,
) -> tuple[dict[str, Any] | None, float]:
    best_trace: dict[str, Any] | None = None
    best_voltage = float("nan")
    best_distance = float("inf")
    for trace_result in traces_list:
        steady_state_voltage = _steady_state_voltage_millivolts(
            trace_result,
            step_delay_milliseconds=step_delay_milliseconds,
            step_duration_milliseconds=step_duration_milliseconds,
        )
        if not np.isfinite(steady_state_voltage):
            continue
        distance = abs(steady_state_voltage - target_voltage_millivolts)
        if distance < best_distance:
            best_trace = trace_result
            best_voltage = steady_state_voltage
            best_distance = distance
    return best_trace, best_voltage


def _membrane_time_constant_milliseconds(
    trace_result: dict[str, Any],
    *,
    step_delay_milliseconds: float,
    step_duration_milliseconds: float,
) -> float:
    time_array = np.asarray(trace_result["t"], dtype=float)
    voltage_array = np.asarray(trace_result["v_soma"], dtype=float)
    step_end = step_delay_milliseconds + step_duration_milliseconds
    mask = time_array > step_end
    if np.count_nonzero(mask) < 2:
        return float("nan")
    post_time = time_array[mask] - step_end
    post_voltage = voltage_array[mask]
    start_voltage = float(post_voltage[0])
    end_voltage = float(post_voltage[-1])
    delta_voltage = end_voltage - start_voltage
    if not np.isfinite(delta_voltage) or delta_voltage <= 0.0:
        return float("nan")
    threshold_voltage = start_voltage + 0.6321206 * delta_voltage
    indices = np.where(post_voltage >= threshold_voltage)[0]
    if len(indices) == 0:
        return float("nan")
    return float(post_time[indices[0]])


def _sag_amplitude_millivolts(
    trace_result: dict[str, Any],
    *,
    step_delay_milliseconds: float,
    step_duration_milliseconds: float,
    sag_window_milliseconds: float = 100.0,
) -> float:
    time_array = np.asarray(trace_result["t"], dtype=float)
    voltage_array = np.asarray(trace_result["v_soma"], dtype=float)
    step_end = step_delay_milliseconds + step_duration_milliseconds
    mask = (time_array > step_delay_milliseconds) & (time_array <= step_end)
    if np.count_nonzero(mask) == 0:
        return float("nan")
    roi_voltage = voltage_array[mask]
    steady_state_voltage = float(roi_voltage[-1])
    minimum_voltage = float(np.min(roi_voltage))
    if minimum_voltage < steady_state_voltage:
        return -(steady_state_voltage - minimum_voltage)
    alt_mask = (
        (time_array > step_delay_milliseconds + sag_window_milliseconds - 1.0)
        & (time_array < step_delay_milliseconds + sag_window_milliseconds + 1.0)
    )
    if not np.any(alt_mask):
        return float("nan")
    return -(steady_state_voltage - float(np.median(voltage_array[alt_mask])))


def _rebound_potential_presence(
    trace_result: dict[str, Any],
    *,
    spike_threshold_millivolts: float,
    step_delay_milliseconds: float,
    step_duration_milliseconds: float,
) -> float:
    spike_times = find_spike_times_milliseconds(
        trace_result,
        spike_threshold_millivolts=spike_threshold_millivolts,
        step_onset_milliseconds=step_delay_milliseconds + step_duration_milliseconds,
    )
    return 1.0 if len(spike_times) > 0 else 0.0


def _trace_near_target_mean_rate(
    traces_list: list[dict[str, Any]],
    *,
    target_rate_hertz: float,
    spike_threshold_millivolts: float,
    step_delay_milliseconds: float,
    step_duration_milliseconds: float,
) -> tuple[dict[str, Any] | None, float]:
    step_end = step_delay_milliseconds + step_duration_milliseconds
    best_trace: dict[str, Any] | None = None
    best_rate = float("nan")
    best_distance = float("inf")
    for trace_result in traces_list:
        spike_count = len(
            find_spike_times_milliseconds(
                trace_result,
                spike_threshold_millivolts,
                step_delay_milliseconds,
                step_end,
            )
        )
        if spike_count < 2:
            continue
        mean_rate = spike_count / (step_duration_milliseconds * 1e-3)
        distance = abs(mean_rate - target_rate_hertz)
        if distance < best_distance:
            best_trace = trace_result
            best_rate = float(mean_rate)
            best_distance = distance
    return best_trace, best_rate


def _spike_accommodation_hertz(
    trace_result: dict[str, Any],
    *,
    spike_threshold_millivolts: float,
    step_delay_milliseconds: float,
    step_duration_milliseconds: float,
) -> float:
    spike_times = find_spike_times_milliseconds(
        trace_result,
        spike_threshold_millivolts,
        step_delay_milliseconds,
        step_delay_milliseconds + step_duration_milliseconds,
    )
    if len(spike_times) < 3:
        return 0.0
    isis_milliseconds = np.diff(spike_times)
    ifr_first = 1000.0 / float(isis_milliseconds[0])
    ifr_last = 1000.0 / float(isis_milliseconds[-1])
    return ifr_last - ifr_first


def _spike_accommodation_time_constant_milliseconds(
    trace_result: dict[str, Any],
    *,
    spike_threshold_millivolts: float,
    step_delay_milliseconds: float,
    step_duration_milliseconds: float,
) -> float:
    from scipy.optimize import curve_fit

    spike_times = find_spike_times_milliseconds(
        trace_result,
        spike_threshold_millivolts,
        step_delay_milliseconds,
        step_delay_milliseconds + step_duration_milliseconds,
    )
    if len(spike_times) < 4:
        return step_duration_milliseconds
    crossing_times = spike_times - spike_times[0]
    isis_milliseconds = np.diff(crossing_times)
    ifr_hertz = 1000.0 / isis_milliseconds
    ifr_time_milliseconds = (crossing_times - crossing_times[1])[1:]
    if len(ifr_hertz) < 3:
        return step_duration_milliseconds

    def ifr_curve(time_value: np.ndarray, start_value: float, finish_value: float, tau_value: float) -> np.ndarray:
        return (start_value - finish_value) * np.exp(-time_value / tau_value) + finish_value

    try:
        params, _ = curve_fit(
            ifr_curve,
            ifr_time_milliseconds,
            ifr_hertz,
            p0=(float(ifr_hertz[0]), float(ifr_hertz[-1]), 10.0),
            bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]),
            maxfev=10000,
        )
    except Exception:
        return step_duration_milliseconds
    return float(params[2])


def _spike_count_rate_hertz(
    trace_result: dict[str, Any],
    *,
    spike_threshold_millivolts: float,
    step_delay_milliseconds: float,
    step_duration_milliseconds: float,
) -> float:
    step_end = step_delay_milliseconds + step_duration_milliseconds
    spike_count = len(
        find_spike_times_milliseconds(
            trace_result,
            spike_threshold_millivolts,
            step_delay_milliseconds,
            step_end,
        )
    )
    return float(spike_count / (step_duration_milliseconds * 1e-3))


def _median_inverse_isi_hertz(
    trace_result: dict[str, Any],
    *,
    spike_threshold_millivolts: float,
    step_delay_milliseconds: float,
    step_duration_milliseconds: float,
) -> float:
    step_end = step_delay_milliseconds + step_duration_milliseconds
    spike_times = find_spike_times_milliseconds(
        trace_result,
        spike_threshold_millivolts,
        step_delay_milliseconds,
        step_end,
    )
    if len(spike_times) < 2:
        return 0.0
    isis_milliseconds = np.diff(spike_times)
    if len(isis_milliseconds) == 0:
        return 0.0
    return float(np.median(1000.0 / isis_milliseconds))


def _run_burton_urban_cell(
    cell_name: str,
    protocol: BurtonUrbanProtocol,
    use_coreneuron: bool = False,
    use_gpu: bool = False,
) -> dict[str, Any]:
    _configure_parent_cache_dirs()
    from single_cell_utils import (
        find_bias_current,
        run_current_clamp,
        run_current_clamp_series,
        run_hyperpolarizing_steps,
    )
    from fi_curve_utils import (
        compute_action_potential_properties,
        compute_fi_maximum_linear_slope,
        compute_input_resistance_megaohms,
        compute_isi_statistics_near_rate,
        compute_peak_instantaneous_firing_rate_hertz,
        compute_rheobase_nanoamps,
        compute_rheobase_spike_latency_milliseconds,
        traces_to_fi,
    )

    zero_current_trace = run_current_clamp(
        cell_name,
        amp_nA=0.0,
        duration_ms=500.0,
        delay_ms=0.0,
        tail_ms=0.0,
        dt=protocol.dt_ms,
        celsius=protocol.celsius,
        use_coreneuron=use_coreneuron,
        use_gpu=use_gpu,
    )
    resting_potential_mV = float(zero_current_trace["v_soma"][-1])

    bias_current_nA = find_bias_current(
        cell_name,
        target_membrane_potential_millivolts=protocol.target_vm_mV,
        settle_duration_milliseconds=protocol.bias_settle_ms,
        tolerance_millivolts=protocol.bias_tolerance_mV,
        max_iterations=protocol.bias_max_iterations,
        timestep_milliseconds=protocol.dt_ms,
        temperature_celsius=protocol.celsius,
    )

    step_traces = run_current_clamp_series(
        cell_name,
        amps_nA=protocol.current_steps_nA,
        duration_ms=protocol.step_duration_ms,
        delay_ms=protocol.step_delay_ms,
        tail_ms=protocol.tail_ms,
        dt=protocol.dt_ms,
        celsius=protocol.celsius,
        use_coreneuron=use_coreneuron,
        use_gpu=use_gpu,
        bias_current_nA=bias_current_nA,
        v_init_mV=protocol.target_vm_mV,
    )
    current_amplitudes_nA, firing_rates_hz = traces_to_fi(
        step_traces,
        protocol.step_duration_ms,
        threshold_mV=protocol.spike_threshold_mV,
        delay_ms=protocol.step_delay_ms,
    )
    fi_slope_hz_per_nA, _, _, gain_segment_index = compute_fi_maximum_linear_slope(
        current_amplitudes_nA,
        firing_rates_hz,
    )
    fi_gain_hz_per_50pA = fi_slope_hz_per_nA / 20.0 if np.isfinite(fi_slope_hz_per_nA) else float("nan")

    rheobase_nA = compute_rheobase_nanoamps(
        step_traces,
        protocol.spike_threshold_mV,
        step_delay_milliseconds=protocol.step_delay_ms,
        step_duration_milliseconds=protocol.step_duration_ms,
    )
    spike_latency_ms = compute_rheobase_spike_latency_milliseconds(
        step_traces,
        protocol.spike_threshold_mV,
        protocol.step_delay_ms,
        protocol.step_duration_ms,
    )
    peak_rate_hz = compute_peak_instantaneous_firing_rate_hertz(
        step_traces,
        protocol.spike_threshold_mV,
        protocol.step_delay_ms,
        protocol.step_duration_ms,
    )
    isi_stats = compute_isi_statistics_near_rate(
        step_traces,
        target_rate_hertz=protocol.cv_isi_target_rate_hz,
        spike_threshold_millivolts=protocol.spike_threshold_mV,
        step_delay_milliseconds=protocol.step_delay_ms,
        step_duration_milliseconds=protocol.step_duration_ms,
    )

    if np.isfinite(rheobase_nA):
        rheobase_trace = next(
            (trace for trace in step_traces if np.isclose(float(trace["amp_nA"]), rheobase_nA)),
            None,
        )
        ap_props = (
            compute_action_potential_properties(
                rheobase_trace,
                protocol.ap_derivative_threshold_mV_per_ms,
                protocol.step_delay_ms,
            )
            if rheobase_trace is not None
            else {}
        )
    else:
        ap_props = {}

    hyperpolarizing_traces = run_hyperpolarizing_steps(
        cell_name,
        current_start_nanoamps=protocol.hyperpolarizing_start_nA,
        current_stop_nanoamps=protocol.hyperpolarizing_stop_nA,
        current_step_nanoamps=protocol.hyperpolarizing_increment_nA,
        step_duration_milliseconds=protocol.step_duration_ms,
        delay_milliseconds=protocol.step_delay_ms,
        tail_duration_milliseconds=protocol.tail_ms,
        timestep_milliseconds=protocol.dt_ms,
        temperature_celsius=protocol.celsius,
        use_coreneuron=use_coreneuron,
        use_gpu=use_gpu,
        bias_current_nA=bias_current_nA,
        v_init_mV=protocol.target_vm_mV,
    )
    input_resistance_MOhm = compute_input_resistance_megaohms(
        hyperpolarizing_traces,
        step_duration_milliseconds=protocol.step_duration_ms,
        delay_milliseconds=protocol.step_delay_ms,
    )
    target_hyperpolarizing_trace, _ = _trace_nearest_steady_state_voltage(
        hyperpolarizing_traces,
        target_voltage_millivolts=-103.0,
        step_delay_milliseconds=protocol.step_delay_ms,
        step_duration_milliseconds=protocol.step_duration_ms,
    )
    membrane_time_constant_ms = (
        _membrane_time_constant_milliseconds(
            target_hyperpolarizing_trace,
            step_delay_milliseconds=protocol.step_delay_ms,
            step_duration_milliseconds=protocol.step_duration_ms,
        )
        if target_hyperpolarizing_trace is not None
        else float("nan")
    )
    cell_capacitance_pF = (
        membrane_time_constant_ms / input_resistance_MOhm * 1000.0
        if np.isfinite(membrane_time_constant_ms) and np.isfinite(input_resistance_MOhm) and input_resistance_MOhm > 0.0
        else float("nan")
    )
    sag_amplitude_mV = (
        _sag_amplitude_millivolts(
            target_hyperpolarizing_trace,
            step_delay_milliseconds=protocol.step_delay_ms,
            step_duration_milliseconds=protocol.step_duration_ms,
        )
        if target_hyperpolarizing_trace is not None
        else float("nan")
    )
    rebound_potential_presence = (
        _rebound_potential_presence(
            target_hyperpolarizing_trace,
            spike_threshold_millivolts=protocol.spike_threshold_mV,
            step_delay_milliseconds=protocol.step_delay_ms,
            step_duration_milliseconds=protocol.step_duration_ms,
        )
        if target_hyperpolarizing_trace is not None
        else float("nan")
    )
    target_rate_trace, _ = _trace_near_target_mean_rate(
        step_traces,
        target_rate_hertz=protocol.cv_isi_target_rate_hz,
        spike_threshold_millivolts=protocol.spike_threshold_mV,
        step_delay_milliseconds=protocol.step_delay_ms,
        step_duration_milliseconds=protocol.step_duration_ms,
    )
    spike_accommodation_hz = (
        _spike_accommodation_hertz(
            target_rate_trace,
            spike_threshold_millivolts=protocol.spike_threshold_mV,
            step_delay_milliseconds=protocol.step_delay_ms,
            step_duration_milliseconds=protocol.step_duration_ms,
        )
        if target_rate_trace is not None
        else float("nan")
    )
    spike_accommodation_time_constant_ms = (
        _spike_accommodation_time_constant_milliseconds(
            target_rate_trace,
            spike_threshold_millivolts=protocol.spike_threshold_mV,
            step_delay_milliseconds=protocol.step_delay_ms,
            step_duration_milliseconds=protocol.step_duration_ms,
        )
        if target_rate_trace is not None
        else float("nan")
    )

    zero_step_rate_hz = float(firing_rates_hz[0]) if len(firing_rates_hz) else float("nan")
    return {
        "cell_name": cell_name,
        "cell_type": _cell_type_from_name(cell_name),
        "resting_potential_mV": resting_potential_mV,
        "bias_current_pA": bias_current_nA * 1000.0,
        "zero_step_rate_Hz": zero_step_rate_hz,
        "membrane_time_constant_ms": membrane_time_constant_ms,
        "cell_capacitance_pF": cell_capacitance_pF,
        "sag_amplitude_mV": sag_amplitude_mV,
        "rebound_potential_presence": rebound_potential_presence,
        "rheobase_pA": rheobase_nA * 1000.0 if np.isfinite(rheobase_nA) else float("nan"),
        "spike_latency_ms": spike_latency_ms,
        "peak_rate_Hz": peak_rate_hz,
        "fi_gain_Hz_per_50pA": fi_gain_hz_per_50pA,
        "spike_accommodation_hz": spike_accommodation_hz,
        "spike_accommodation_time_constant_ms": spike_accommodation_time_constant_ms,
        "fi_gain_segment_start_index": gain_segment_index,
        "cv_isi": isi_stats["coefficient_of_variation_interspike_interval"],
        "cv_isi_step_pA": (
            isi_stats["selected_current_nanoamps"] * 1000.0
            if np.isfinite(isi_stats["selected_current_nanoamps"])
            else float("nan")
        ),
        "cv_isi_mean_rate_Hz": isi_stats["selected_mean_rate_hertz"],
        "input_resistance_MOhm": input_resistance_MOhm,
        "firing_rates_by_step_Hz": [float(value) for value in firing_rates_hz],
        "AP_onset_mV": ap_props.get("ap_onset_millivolts", float("nan")),
        "Amplitude_mV": ap_props.get("ap_amplitude_millivolts", float("nan")),
        "FWHM_ms": ap_props.get("ap_full_width_half_maximum_milliseconds", float("nan")),
        "Rise_slope_mV_per_ms": ap_props.get("ap_rise_slope_millivolts_per_millisecond", float("nan")),
        "Fall_slope_mV_per_ms": ap_props.get("ap_fall_slope_millivolts_per_millisecond", float("nan")),
        "AHP_amplitude_mV": ap_props.get("ahp_amplitude_millivolts", float("nan")),
        "T_AHP50_ms": ap_props.get("ahp_half_decay_time_milliseconds", float("nan")),
    }


def _run_intrinsic_current_clamp_cell(
    cell_spec: Any,
    cell_name: str,
    cell_type: str,
    protocol: BurtonUrbanProtocol,
    use_coreneuron: bool = False,
    use_gpu: bool = False,
    curve_rate_mode: str = "mean_step_rate",
) -> dict[str, Any]:
    _configure_parent_cache_dirs()
    from single_cell_utils import (
        find_bias_current,
        run_current_clamp,
        run_current_clamp_series,
        run_hyperpolarizing_steps,
    )
    from fi_curve_utils import (
        compute_action_potential_properties,
        compute_fi_maximum_linear_slope,
        compute_input_resistance_megaohms,
        compute_isi_statistics_near_rate,
        compute_peak_instantaneous_firing_rate_hertz,
        compute_rheobase_nanoamps,
        compute_rheobase_spike_latency_milliseconds,
        traces_to_fi,
    )

    zero_current_trace = run_current_clamp(
        cell_spec,
        amp_nA=0.0,
        duration_ms=500.0,
        delay_ms=0.0,
        tail_ms=0.0,
        dt=protocol.dt_ms,
        celsius=protocol.celsius,
        use_coreneuron=use_coreneuron,
        use_gpu=use_gpu,
    )
    resting_potential_mV = float(zero_current_trace["v_soma"][-1])
    spontaneous_spike_times = find_spike_times_milliseconds(
        zero_current_trace,
        protocol.spike_threshold_mV,
        0.0,
        500.0,
    )
    spontaneous_firing_rate_Hz = float(len(spontaneous_spike_times) / 0.5)

    if protocol.target_vm_mV is None:
        bias_current_nA = 0.0
        v_init_mV = resting_potential_mV
    else:
        bias_current_nA = find_bias_current(
            cell_spec,
            target_membrane_potential_millivolts=float(protocol.target_vm_mV),
            settle_duration_milliseconds=protocol.bias_settle_ms,
            tolerance_millivolts=protocol.bias_tolerance_mV,
            max_iterations=protocol.bias_max_iterations,
            timestep_milliseconds=protocol.dt_ms,
            temperature_celsius=protocol.celsius,
        )
        v_init_mV = float(protocol.target_vm_mV)

    step_traces = run_current_clamp_series(
        cell_spec,
        amps_nA=protocol.current_steps_nA,
        duration_ms=protocol.step_duration_ms,
        delay_ms=protocol.step_delay_ms,
        tail_ms=protocol.tail_ms,
        dt=protocol.dt_ms,
        celsius=protocol.celsius,
        use_coreneuron=use_coreneuron,
        use_gpu=use_gpu,
        bias_current_nA=bias_current_nA,
        v_init_mV=v_init_mV,
    )
    current_amplitudes_nA, firing_rates_hz = traces_to_fi(
        step_traces,
        protocol.step_duration_ms,
        threshold_mV=protocol.spike_threshold_mV,
        delay_ms=protocol.step_delay_ms,
    )
    fi_slope_hz_per_nA, _, _, gain_segment_index = compute_fi_maximum_linear_slope(
        current_amplitudes_nA,
        firing_rates_hz,
    )
    fi_gain_hz_per_50pA = fi_slope_hz_per_nA / 20.0 if np.isfinite(fi_slope_hz_per_nA) else float("nan")

    rheobase_nA = compute_rheobase_nanoamps(
        step_traces,
        protocol.spike_threshold_mV,
        step_delay_milliseconds=protocol.step_delay_ms,
        step_duration_milliseconds=protocol.step_duration_ms,
    )
    spike_latency_ms = compute_rheobase_spike_latency_milliseconds(
        step_traces,
        protocol.spike_threshold_mV,
        protocol.step_delay_ms,
        protocol.step_duration_ms,
    )
    peak_rate_hz = compute_peak_instantaneous_firing_rate_hertz(
        step_traces,
        protocol.spike_threshold_mV,
        protocol.step_delay_ms,
        protocol.step_duration_ms,
    )
    isi_stats = compute_isi_statistics_near_rate(
        step_traces,
        target_rate_hertz=protocol.cv_isi_target_rate_hz,
        spike_threshold_millivolts=protocol.spike_threshold_mV,
        step_delay_milliseconds=protocol.step_delay_ms,
        step_duration_milliseconds=protocol.step_duration_ms,
    )

    if np.isfinite(rheobase_nA):
        rheobase_trace = next(
            (trace for trace in step_traces if np.isclose(float(trace["amp_nA"]), rheobase_nA)),
            None,
        )
        ap_props = (
            compute_action_potential_properties(
                rheobase_trace,
                protocol.ap_derivative_threshold_mV_per_ms,
                protocol.step_delay_ms,
            )
            if rheobase_trace is not None
            else {}
        )
    else:
        ap_props = {}

    hyperpolarizing_traces = run_hyperpolarizing_steps(
        cell_spec,
        current_start_nanoamps=protocol.hyperpolarizing_start_nA,
        current_stop_nanoamps=protocol.hyperpolarizing_stop_nA,
        current_step_nanoamps=protocol.hyperpolarizing_increment_nA,
        step_duration_milliseconds=protocol.step_duration_ms,
        delay_milliseconds=protocol.step_delay_ms,
        tail_duration_milliseconds=protocol.tail_ms,
        timestep_milliseconds=protocol.dt_ms,
        temperature_celsius=protocol.celsius,
        use_coreneuron=use_coreneuron,
        use_gpu=use_gpu,
        bias_current_nA=bias_current_nA,
        v_init_mV=v_init_mV,
    )
    input_resistance_MOhm = compute_input_resistance_megaohms(
        hyperpolarizing_traces,
        step_duration_milliseconds=protocol.step_duration_ms,
        delay_milliseconds=protocol.step_delay_ms,
    )
    target_hyperpolarizing_trace, _ = _trace_nearest_steady_state_voltage(
        hyperpolarizing_traces,
        target_voltage_millivolts=-103.0,
        step_delay_milliseconds=protocol.step_delay_ms,
        step_duration_milliseconds=protocol.step_duration_ms,
    )
    membrane_time_constant_ms = (
        _membrane_time_constant_milliseconds(
            target_hyperpolarizing_trace,
            step_delay_milliseconds=protocol.step_delay_ms,
            step_duration_milliseconds=protocol.step_duration_ms,
        )
        if target_hyperpolarizing_trace is not None
        else float("nan")
    )
    cell_capacitance_pF = (
        membrane_time_constant_ms / input_resistance_MOhm * 1000.0
        if np.isfinite(membrane_time_constant_ms) and np.isfinite(input_resistance_MOhm) and input_resistance_MOhm > 0.0
        else float("nan")
    )
    sag_amplitude_mV = (
        _sag_amplitude_millivolts(
            target_hyperpolarizing_trace,
            step_delay_milliseconds=protocol.step_delay_ms,
            step_duration_milliseconds=protocol.step_duration_ms,
        )
        if target_hyperpolarizing_trace is not None
        else float("nan")
    )
    rebound_potential_presence = (
        _rebound_potential_presence(
            target_hyperpolarizing_trace,
            spike_threshold_millivolts=protocol.spike_threshold_mV,
            step_delay_milliseconds=protocol.step_delay_ms,
            step_duration_milliseconds=protocol.step_duration_ms,
        )
        if target_hyperpolarizing_trace is not None
        else float("nan")
    )
    target_rate_trace, _ = _trace_near_target_mean_rate(
        step_traces,
        target_rate_hertz=protocol.cv_isi_target_rate_hz,
        spike_threshold_millivolts=protocol.spike_threshold_mV,
        step_delay_milliseconds=protocol.step_delay_ms,
        step_duration_milliseconds=protocol.step_duration_ms,
    )
    spike_accommodation_hz = (
        _spike_accommodation_hertz(
            target_rate_trace,
            spike_threshold_millivolts=protocol.spike_threshold_mV,
            step_delay_milliseconds=protocol.step_delay_ms,
            step_duration_milliseconds=protocol.step_duration_ms,
        )
        if target_rate_trace is not None
        else float("nan")
    )
    spike_accommodation_time_constant_ms = (
        _spike_accommodation_time_constant_milliseconds(
            target_rate_trace,
            spike_threshold_millivolts=protocol.spike_threshold_mV,
            step_delay_milliseconds=protocol.step_delay_ms,
            step_duration_milliseconds=protocol.step_duration_ms,
        )
        if target_rate_trace is not None
        else float("nan")
    )

    if curve_rate_mode == "median_inverse_isi":
        curve_rates_hz = [
            _median_inverse_isi_hertz(
                trace,
                spike_threshold_millivolts=protocol.spike_threshold_mV,
                step_delay_milliseconds=protocol.step_delay_ms,
                step_duration_milliseconds=protocol.step_duration_ms,
            )
            for trace in step_traces
        ]
    else:
        curve_rates_hz = [float(value) for value in firing_rates_hz]

    fi_curve_rows = [
        {
            "cell_name": cell_name,
            "cell_type": cell_type,
            "current_pA": float(trace["amp_nA"]) * 1000.0,
            "firing_rate_Hz": float(rate),
            "step_duration_ms": protocol.step_duration_ms,
            "rate_definition": (
                "median_inverse_isi"
                if curve_rate_mode == "median_inverse_isi"
                else "mean_step_rate"
            ),
        }
        for trace, rate in zip(step_traces, curve_rates_hz)
    ]

    zero_step_rate_hz = float(firing_rates_hz[0]) if len(firing_rates_hz) else float("nan")
    max_fi_rate_hz = float(np.max(firing_rates_hz)) if len(firing_rates_hz) else float("nan")
    return {
        "cell_name": cell_name,
        "cell_type": cell_type,
        "resting_potential_mV": resting_potential_mV,
        "bias_current_pA": bias_current_nA * 1000.0,
        "zero_step_rate_Hz": zero_step_rate_hz,
        "spontaneous_firing_rate_Hz": spontaneous_firing_rate_Hz,
        "membrane_time_constant_ms": membrane_time_constant_ms,
        "cell_capacitance_pF": cell_capacitance_pF,
        "sag_amplitude_mV": sag_amplitude_mV,
        "rebound_potential_presence": rebound_potential_presence,
        "rheobase_pA": rheobase_nA * 1000.0 if np.isfinite(rheobase_nA) else float("nan"),
        "spike_latency_ms": spike_latency_ms,
        "first_spike_latency_ms": spike_latency_ms,
        "peak_rate_Hz": peak_rate_hz,
        "peak_instantaneous_rate_Hz": peak_rate_hz,
        "max_fi_rate_Hz": max_fi_rate_hz,
        "fi_slope_Hz_per_nA": fi_slope_hz_per_nA,
        "fi_gain_Hz_per_50pA": fi_gain_hz_per_50pA,
        "spike_accommodation_hz": spike_accommodation_hz,
        "spike_accommodation_time_constant_ms": spike_accommodation_time_constant_ms,
        "fi_gain_segment_start_index": gain_segment_index,
        "cv_isi": isi_stats["coefficient_of_variation_interspike_interval"],
        "cv_isi_step_pA": (
            isi_stats["selected_current_nanoamps"] * 1000.0
            if np.isfinite(isi_stats["selected_current_nanoamps"])
            else float("nan")
        ),
        "cv_isi_mean_rate_Hz": isi_stats["selected_mean_rate_hertz"],
        "input_resistance_MOhm": input_resistance_MOhm,
        "firing_rates_by_step_Hz": [float(value) for value in firing_rates_hz],
        "AP_onset_mV": ap_props.get("ap_onset_millivolts", float("nan")),
        "Amplitude_mV": ap_props.get("ap_amplitude_millivolts", float("nan")),
        "FWHM_ms": ap_props.get("ap_full_width_half_maximum_milliseconds", float("nan")),
        "Rise_slope_mV_per_ms": ap_props.get("ap_rise_slope_millivolts_per_millisecond", float("nan")),
        "Fall_slope_mV_per_ms": ap_props.get("ap_fall_slope_millivolts_per_millisecond", float("nan")),
        "AHP_amplitude_mV": ap_props.get("ahp_amplitude_millivolts", float("nan")),
        "T_AHP50_ms": ap_props.get("ahp_half_decay_time_milliseconds", float("nan")),
        "fi_curve_rows": fi_curve_rows,
    }


def run_burton_urban_protocol(
    *,
    cell_types: list[str],
    cell_count: int,
    protocol: BurtonUrbanProtocol,
    use_coreneuron: bool = False,
    use_gpu: bool = False,
    jobs: int = 0,
) -> list[dict[str, Any]]:
    cell_names = _cell_names(cell_types, cell_count)
    worker_count = _resolved_jobs(len(cell_names), jobs, use_gpu=use_gpu)
    if worker_count == 1:
        metrics = [
            _run_burton_urban_cell(
                cell_name,
                protocol,
                use_coreneuron=use_coreneuron,
                use_gpu=use_gpu,
            )
            for cell_name in cell_names
        ]
    else:
        context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
        ) as pool:
            metrics = list(
                pool.map(
                    _run_burton_urban_cell,
                    cell_names,
                    repeat(protocol),
                    repeat(use_coreneuron),
                    repeat(use_gpu),
                )
    )
    return sorted(metrics, key=lambda metric: metric["cell_name"])


def run_intrinsic_current_clamp_protocol(
    *,
    cell_specs: list[tuple[str, Any, str]],
    protocol: BurtonUrbanProtocol,
    use_coreneuron: bool = False,
    use_gpu: bool = False,
    jobs: int = 0,
    group_field: str,
    group_values: Sequence[str] | None = None,
    curve_rate_mode: str = "mean_step_rate",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    worker_count = _resolved_jobs(len(cell_specs), jobs, use_gpu=use_gpu)
    labels = [label for label, _cell_spec, _cell_type in cell_specs]
    if worker_count == 1:
        raw_metrics = [
            _run_intrinsic_current_clamp_cell(
                cell_spec,
                label,
                cell_type,
                protocol,
                use_coreneuron=use_coreneuron,
                use_gpu=use_gpu,
                curve_rate_mode=curve_rate_mode,
            )
            for label, cell_spec, cell_type in cell_specs
        ]
    else:
        context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
        ) as pool:
            raw_metrics = list(
                pool.map(
                    _run_intrinsic_current_clamp_cell,
                    [cell_spec for _label, cell_spec, _cell_type in cell_specs],
                    labels,
                    [cell_type for _label, _cell_spec, cell_type in cell_specs],
                    repeat(protocol),
                    repeat(use_coreneuron),
                    repeat(use_gpu),
                    repeat(curve_rate_mode),
                )
            )

    raw_metrics = sorted(raw_metrics, key=lambda metric: metric["cell_name"])
    target_groups = [str(value).strip() for value in (group_values or []) if str(value).strip()] or [""]
    metrics: list[dict[str, Any]] = []
    fi_curve_rows: list[dict[str, Any]] = []
    for raw_metric in raw_metrics:
        curve_rows = [dict(row) for row in raw_metric.pop("fi_curve_rows", [])]
        for target_group in target_groups:
            metric = dict(raw_metric)
            if target_group:
                metric[group_field] = target_group
            metrics.append(metric)
            for curve_row in curve_rows:
                row = dict(curve_row)
                if target_group:
                    row[group_field] = target_group
                fi_curve_rows.append(row)
    return metrics, fi_curve_rows


def _burton_protocol_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cell-count", type=int, default=5, help="Run models 1..N for each requested cell type.")
    parser.add_argument("--cell-types", default="MC,TC", help="Comma-separated cell type prefixes to audit.")
    parser.add_argument("--use-coreneuron", action="store_true", help="Run current-clamp sweeps with CoreNEURON.")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU mode when --use-coreneuron is set.")
    parser.add_argument("--dt-ms", type=float, default=0.1, help="Fixed integration time step in ms.")
    parser.add_argument("--bias-max-iterations", type=int, default=24, help="Binary-search iterations for held-voltage bias current.")
    parser.add_argument("--jobs", type=int, default=0, help="Worker processes. 0 uses all local CPU cores unless --use-gpu is set.")


def _run_registered_burton_protocol(args: argparse.Namespace, protocol_config: dict[str, Any]) -> ProtocolRunResult:
    protocol = BurtonUrbanProtocol(
        target_vm_mV=float(protocol_config.get("target_vm_mV", -58.0)),
        step_start_nA=float(protocol_config.get("step_start_nA", 0.0)),
        step_stop_nA=float(protocol_config.get("step_stop_nA", 0.30)),
        step_increment_nA=float(protocol_config.get("step_increment_nA", 0.05)),
        step_duration_ms=float(protocol_config.get("step_duration_ms", 2000.0)),
        step_delay_ms=float(protocol_config.get("step_delay_ms", 200.0)),
        tail_ms=float(protocol_config.get("tail_ms", 200.0)),
        hyperpolarizing_start_nA=float(protocol_config.get("hyperpolarizing_start_nA", 0.0)),
        hyperpolarizing_stop_nA=float(protocol_config.get("hyperpolarizing_stop_nA", -0.30)),
        hyperpolarizing_increment_nA=float(protocol_config.get("hyperpolarizing_increment_nA", -0.05)),
        dt_ms=float(getattr(args, "dt_ms", protocol_config.get("dt_ms", 0.1))),
        celsius=float(protocol_config.get("celsius", 35.0)),
        spike_threshold_mV=float(protocol_config.get("spike_threshold_mV", -20.0)),
        ap_derivative_threshold_mV_per_ms=float(protocol_config.get("ap_derivative_threshold_mV_per_ms", 20.0)),
        cv_isi_target_rate_hz=float(protocol_config.get("cv_isi_target_rate_hz", 20.0)),
        bias_settle_ms=float(protocol_config.get("bias_settle_ms", 1000.0)),
        bias_tolerance_mV=float(protocol_config.get("bias_tolerance_mV", 0.1)),
        bias_max_iterations=int(getattr(args, "bias_max_iterations", protocol_config.get("bias_max_iterations", 24))),
    )
    cell_types = [
        cell_type.strip().upper()
        for cell_type in str(getattr(args, "cell_types", protocol_config.get("cell_types", "MC,TC"))).split(",")
        if cell_type.strip()
    ]
    cell_count = int(getattr(args, "cell_count", protocol_config.get("cell_count", 5)))
    jobs = int(getattr(args, "jobs", protocol_config.get("jobs", 0)))
    use_coreneuron = bool(getattr(args, "use_coreneuron", False))
    use_gpu = bool(getattr(args, "use_gpu", False))
    metrics = run_burton_urban_protocol(
        cell_types=cell_types,
        cell_count=cell_count,
        protocol=protocol,
        use_coreneuron=use_coreneuron,
        use_gpu=use_gpu,
        jobs=jobs,
    )
    hyperpolarizing_currents = [
        value * 1000.0
        for value in np.arange(
            protocol.hyperpolarizing_start_nA,
            protocol.hyperpolarizing_stop_nA + protocol.hyperpolarizing_increment_nA * 0.5,
            protocol.hyperpolarizing_increment_nA,
        )
    ]
    protocol_evidence = {
        "target_vm_mV": protocol.target_vm_mV,
        "step_duration_ms": protocol.step_duration_ms,
        "step_currents_pA": [float(value * 1000.0) for value in protocol.current_steps_nA],
        "hyperpolarizing_currents_pA": [float(value) for value in hyperpolarizing_currents],
        "cell_count": len(metrics),
        "cell_names": [metric["cell_name"] for metric in metrics],
        "cell_types": ",".join(cell_types),
    }
    return ProtocolRunResult(metrics=metrics, protocol_evidence=protocol_evidence, group_field="cell_type")


register_validation_protocol(
    ValidationProtocolSpec(
        protocol_id="burton_urban_mctc_current_clamp",
        title="Burton and Urban 2014 mitral/tufted current clamp",
        description="Run the maintained MC/TC isolated cells through the Burton and Urban 2014 current-clamp protocol.",
        add_cli_args=_burton_protocol_cli_args,
        run=_run_registered_burton_protocol,
    )
)


def _add_intrinsic_cli_args(
    parser: argparse.ArgumentParser,
    *,
    default_cell_models: str,
) -> None:
    parser.add_argument("--cell-models", default=default_cell_models, help="Comma-separated cell models or registry keys to audit.")
    parser.add_argument("--use-coreneuron", action="store_true", help="Run current-clamp sweeps with CoreNEURON.")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU mode when --use-coreneuron is set.")
    parser.add_argument("--dt-ms", type=float, default=0.1, help="Fixed integration time step in ms.")
    parser.add_argument("--bias-max-iterations", type=int, default=24, help="Binary-search iterations for held-voltage bias current when normalization is enabled.")
    parser.add_argument("--jobs", type=int, default=0, help="Worker processes. 0 uses all local CPU cores unless --use-gpu is set.")


def _gc_protocol_cli_args(parser: argparse.ArgumentParser) -> None:
    _add_intrinsic_cli_args(parser, default_cell_models="GC1,GC2,GC3,GC4,GC5")
    parser.add_argument(
        "--reference-gc-subtypes",
        default="generic_or_unspecified",
        help="Comma-separated GC reference targets to compare against, such as generic_or_unspecified, sGC, or dGC.",
    )


def _build_protocol_from_config(args: argparse.Namespace, protocol_config: dict[str, Any]) -> BurtonUrbanProtocol:
    target_vm_raw = protocol_config.get("target_vm_mV", -58.0)
    target_vm = None if target_vm_raw in {"", None} else float(target_vm_raw)
    return BurtonUrbanProtocol(
        target_vm_mV=target_vm,
        step_start_nA=float(protocol_config.get("step_start_nA", 0.0)),
        step_stop_nA=float(protocol_config.get("step_stop_nA", 0.30)),
        step_increment_nA=float(protocol_config.get("step_increment_nA", 0.05)),
        step_duration_ms=float(protocol_config.get("step_duration_ms", 2000.0)),
        step_delay_ms=float(protocol_config.get("step_delay_ms", 200.0)),
        tail_ms=float(protocol_config.get("tail_ms", 200.0)),
        hyperpolarizing_start_nA=float(protocol_config.get("hyperpolarizing_start_nA", 0.0)),
        hyperpolarizing_stop_nA=float(protocol_config.get("hyperpolarizing_stop_nA", -0.30)),
        hyperpolarizing_increment_nA=float(protocol_config.get("hyperpolarizing_increment_nA", -0.05)),
        dt_ms=float(getattr(args, "dt_ms", protocol_config.get("dt_ms", 0.1))),
        celsius=float(protocol_config.get("celsius", 35.0)),
        spike_threshold_mV=float(protocol_config.get("spike_threshold_mV", -20.0)),
        ap_derivative_threshold_mV_per_ms=float(protocol_config.get("ap_derivative_threshold_mV_per_ms", 20.0)),
        cv_isi_target_rate_hz=float(protocol_config.get("cv_isi_target_rate_hz", 20.0)),
        bias_settle_ms=float(protocol_config.get("bias_settle_ms", 1000.0)),
        bias_tolerance_mV=float(protocol_config.get("bias_tolerance_mV", 0.1)),
        bias_max_iterations=int(getattr(args, "bias_max_iterations", protocol_config.get("bias_max_iterations", 24))),
    )


def _run_registered_gc_protocol(args: argparse.Namespace, protocol_config: dict[str, Any]) -> ProtocolRunResult:
    protocol = _build_protocol_from_config(args, protocol_config)
    cell_specs = _resolve_cell_specs(getattr(args, "cell_models", protocol_config.get("cell_models", "GC1,GC2,GC3,GC4,GC5")), default_role="GC")
    group_values = _csv_tokens(getattr(args, "reference_gc_subtypes", protocol_config.get("reference_gc_subtypes", "generic_or_unspecified")))
    metrics, fi_curve_rows = run_intrinsic_current_clamp_protocol(
        cell_specs=cell_specs,
        protocol=protocol,
        use_coreneuron=bool(getattr(args, "use_coreneuron", False)),
        use_gpu=bool(getattr(args, "use_gpu", False)),
        jobs=int(getattr(args, "jobs", 0)),
        group_field="gc_subtype",
        group_values=group_values,
        curve_rate_mode="mean_step_rate",
    )
    protocol_evidence = {
        "target_vm_mV": protocol.target_vm_mV,
        "step_duration_ms": protocol.step_duration_ms,
        "step_currents_pA": [float(value * 1000.0) for value in protocol.current_steps_nA],
        "cell_models": [label for label, _cell_spec, _cell_type in cell_specs],
        "reference_gc_subtypes": group_values,
        "fi_curve_rows": fi_curve_rows,
    }
    return ProtocolRunResult(metrics=metrics, protocol_evidence=protocol_evidence, group_field="gc_subtype")


def _epl_fsi_protocol_cli_args(parser: argparse.ArgumentParser) -> None:
    _add_intrinsic_cli_args(parser, default_cell_models="SyntheticEPL2026.PVCRH_FSI1")


def _run_registered_epl_fsi_protocol(args: argparse.Namespace, protocol_config: dict[str, Any]) -> ProtocolRunResult:
    protocol = _build_protocol_from_config(args, protocol_config)
    cell_specs = _resolve_cell_specs(
        getattr(args, "cell_models", protocol_config.get("cell_models", "SyntheticEPL2026.PVCRH_FSI1")),
        default_role="EPL-FSI",
    )
    metrics, fi_curve_rows = run_intrinsic_current_clamp_protocol(
        cell_specs=cell_specs,
        protocol=protocol,
        use_coreneuron=bool(getattr(args, "use_coreneuron", False)),
        use_gpu=bool(getattr(args, "use_gpu", False)),
        jobs=int(getattr(args, "jobs", 0)),
        group_field="cell_type",
        group_values=["EPL-FSI"],
        curve_rate_mode="median_inverse_isi",
    )
    protocol_evidence = {
        "target_vm_mV": protocol.target_vm_mV,
        "step_duration_ms": protocol.step_duration_ms,
        "step_currents_pA": [float(value * 1000.0) for value in protocol.current_steps_nA],
        "cell_models": [label for label, _cell_spec, _cell_type in cell_specs],
        "fi_curve_rows": fi_curve_rows,
    }
    return ProtocolRunResult(metrics=metrics, protocol_evidence=protocol_evidence, group_field="cell_type")


def _epli_correctness_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidate-slice", default=None, help="Optional exported EPLI slice directory or slice name to audit.")


def _run_registered_epli_correctness_protocol(args: argparse.Namespace, protocol_config: dict[str, Any]) -> ProtocolRunResult:
    from olfactorybulb.audit.epli_reference import (
        BRANCHING_ZONE_MAX_UM,
        FAST_SPIKING_REFERENCE,
    )
    from olfactorybulb.audit.neuron_protocols import monotonic_non_decreasing, sweep_soma_step_responses
    from olfactorybulb.epli import PRINCIPAL_PERISOMATIC_SELECTOR, default_slice_synapse_blueprints
    from olfactorybulb.slice_connectivity_optimizer import (
        load_slice_geometry,
        observed_metrics_for_synapse_set,
        resolve_slice_dir,
    )

    metric: dict[str, Any] = {"validation_group": "EPLI"}
    groups = load_slice_geometry("DorsalColumnSlice")
    counts = {
        "MCs": len(groups["MCs"].cell_names),
        "TCs": len(groups["TCs"].cell_names),
        "GCs": len(groups["GCs"].cell_names),
    }
    metric["baseline_population_min_count"] = float(min(counts.values()))
    metric["baseline_MCs_count"] = float(counts["MCs"])
    metric["baseline_TCs_count"] = float(counts["TCs"])
    metric["baseline_GCs_count"] = float(counts["GCs"])

    for synapse_set in ("GCs__MCs", "GCs__TCs"):
        observed = observed_metrics_for_synapse_set("DorsalColumnSlice", synapse_set, groups=groups)
        metric[f"{synapse_set}_entry_count"] = float(observed.entry_count)

    epli_blueprints = [
        blueprint
        for blueprint in default_slice_synapse_blueprints(include_epli=True)
        if blueprint["group_from"] == "EPLIs"
    ]
    reciprocal_ok = all(
        blueprint.get("is_reciprocal")
        and blueprint.get("synapse_name_source") == "AmpaNmdaSyn"
        and blueprint.get("synapse_name_dest") == "GabaSyn"
        for blueprint in epli_blueprints
    )
    metric["epli_blueprint_count"] = float(len(epli_blueprints))
    metric["epli_reciprocal_architecture_code"] = 2.0 if reciprocal_ok else 0.0

    selectors = [blueprint.get("section_pattern_dest") for blueprint in epli_blueprints]
    has_semantic_perisomatic_selector = all(selector == PRINCIPAL_PERISOMATIC_SELECTOR for selector in selectors)
    soma_only = all(selector == "*soma*" for selector in selectors)
    metric["epli_target_pattern_specificity_code"] = 2.0 if has_semantic_perisomatic_selector else (0.0 if soma_only else 1.0)
    metric["epli_default_contact_radius_code"] = (
        1.0 if any(float(blueprint.get("max_distance", 0)) >= 20.0 for blueprint in epli_blueprints) else 2.0
    )

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    blender_source = Path(repo_root, "olfactorybulb", "slicebuilder", "blender.py").read_text()
    has_opl_fallback = "self.epli_particles_name = epli_particles_object_name or tc_particles_object_name" in blender_source
    uses_slice_order_default = "epli_selection_strategy='slice_order'" in blender_source
    metric["epli_particle_cloud_source_code"] = 1.0 if has_opl_fallback else 2.0
    metric["epli_selection_strategy_default_code"] = 1.0 if uses_slice_order_default else 2.0

    if bool(getattr(args, "skip_neuron", False)):
        metric["synthetic_cell_geometry_status_code"] = 1.0
        metric["synthetic_cell_behavior_status_code"] = 1.0
    else:
        try:
            from prev_ob_models.SyntheticEPL2026.isolated_cells import PVCRH_FSI1

            PVCRH_FSI1._instance_counter = 0
            cell = PVCRH_FSI1()
            soma_diameter_um = float(cell.soma.diam)
            primary_count = len(cell.primary_dendrites)
            planar_span_um = float(cell.planar_dendritic_span_um)
            branch_root_distances = []
            for section in cell.branch_dendrites:
                x = cell.h.x3d(0, sec=section)
                y = cell.h.y3d(0, sec=section)
                z = cell.h.z3d(0, sec=section)
                branch_root_distances.append((x * x + y * y + z * z) ** 0.5)
            metric["synthetic_cell_geometry_status_code"] = 2.0
            metric["soma_diameter_um"] = soma_diameter_um
            metric["primary_process_count"] = float(primary_count)
            metric["planar_span_um"] = planar_span_um
            metric["max_branch_origin_um"] = max(branch_root_distances) if branch_root_distances else 0.0
            metric["has_axon_sections"] = 1.0 if (hasattr(cell, "axon") and bool(getattr(cell, "axon"))) else 0.0

            def cell_factory():
                PVCRH_FSI1._instance_counter = 0
                return PVCRH_FSI1()

            amplitudes = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, FAST_SPIKING_REFERENCE.audit_current_max_nA]
            responses = sweep_soma_step_responses(cell_factory, amplitudes)
            response_by_amp = {response.amp_nA: response for response in responses}
            response_rates = [response.step_rate_hz for response in responses if response.amp_nA > 0]
            rest = response_by_amp[0.0]
            moderate = response_by_amp[0.2]
            fast = response_by_amp[FAST_SPIKING_REFERENCE.audit_current_max_nA]
            max_rate = max(response_rates) if response_rates else 0.0
            metric["synthetic_cell_behavior_status_code"] = 2.0
            metric["synthetic_rest_stability_code"] = 2.0 if (not rest.has_nan and len(rest.step_spike_times_ms) == 0) else 0.0
            metric["synthetic_moderate_current_spiking_code"] = 2.0 if (not moderate.has_nan and len(moderate.step_spike_times_ms) > 0) else 0.0
            metric["synthetic_fast_spiking_capability_code"] = 2.0 if (not fast.has_nan and max_rate >= FAST_SPIKING_REFERENCE.minimum_fast_spiking_rate_hz) else 0.0
            metric["synthetic_fi_monotonicity_code"] = 2.0 if monotonic_non_decreasing(response_rates) else 1.0
            metric["synthetic_rest_spike_count"] = float(len(rest.step_spike_times_ms))
            metric["synthetic_rest_final_v_mV"] = float(rest.final_v_mV)
            metric["synthetic_moderate_step_rate_hz"] = float(moderate.step_rate_hz)
            metric["synthetic_fast_max_rate_hz"] = float(max_rate)
            metric["synthetic_fast_rate_at_max_current_hz"] = float(fast.step_rate_hz)
            metric["synthetic_fast_reference_min_rate_hz"] = float(FAST_SPIKING_REFERENCE.minimum_fast_spiking_rate_hz)
            metric["synthetic_branching_zone_limit_um"] = float(BRANCHING_ZONE_MAX_UM)
        except Exception:
            metric["synthetic_cell_geometry_status_code"] = 0.0
            metric["synthetic_cell_behavior_status_code"] = 0.0

    candidate_slice = getattr(args, "candidate_slice", None)
    metric["candidate_slice_requested"] = 1.0 if candidate_slice else 0.0
    if not candidate_slice:
        slice_dir = resolve_slice_dir("DorsalColumnSlice")
        has_epli_assets = (slice_dir / "EPLIs.json").exists()
        metric["canonical_epli_assets_present_code"] = 2.0 if has_epli_assets else 0.0
    else:
        slice_dir = resolve_slice_dir(candidate_slice)
        metric["candidate_slice_exists_code"] = 2.0 if slice_dir.exists() else 0.0
        if slice_dir.exists():
            group_presence = {name: (slice_dir / f"{name}.json").exists() for name in ("MCs", "TCs", "GCs", "EPLIs")}
            metric["candidate_slice_group_presence_code"] = 2.0 if all(group_presence.values()) else 0.0
            for synapse_set in ("EPLIs__MCs", "EPLIs__TCs"):
                syn_path = slice_dir / f"{synapse_set}.json"
                if syn_path.exists():
                    import json

                    entry_count = len(json.loads(syn_path.read_text()).get("entries", []))
                else:
                    entry_count = 0
                metric[f"candidate_{synapse_set}_entry_count"] = float(entry_count)
                metric[f"candidate_{synapse_set}_code"] = 2.0 if entry_count > 0 else 0.0
        else:
            metric["candidate_slice_group_presence_code"] = 0.0
            metric["candidate_EPLIs__MCs_code"] = 0.0
            metric["candidate_EPLIs__TCs_code"] = 0.0

    return ProtocolRunResult(metrics=[metric], protocol_evidence={"candidate_slice": candidate_slice or ""}, group_field="validation_group")


register_validation_protocol(
    ValidationProtocolSpec(
        protocol_id="gc_intrinsic_current_clamp",
        title="Granule-cell intrinsic current clamp",
        description="Run maintained granule-cell models through the configured intrinsic current-clamp protocol and compare them against reference GC summaries.",
        add_cli_args=_gc_protocol_cli_args,
        run=_run_registered_gc_protocol,
    )
)


register_validation_protocol(
    ValidationProtocolSpec(
        protocol_id="epl_fsi_current_clamp",
        title="External plexiform layer fast-spiking interneuron current clamp",
        description="Run the maintained synthetic EPL fast-spiking interneuron surrogate through the Burton, Malyshko, and Urban 2024 protocol family.",
        add_cli_args=_epl_fsi_protocol_cli_args,
        run=_run_registered_epl_fsi_protocol,
    )
)


register_validation_protocol(
    ValidationProtocolSpec(
        protocol_id="epli_correctness_structural",
        title="External plexiform layer interneuron structural/readiness audit protocol",
        description="Collect slice-readiness, source-code default, morphology, and fast-spiking scaffold metrics for the provisional EPLI path.",
        add_cli_args=_epli_correctness_cli_args,
        run=_run_registered_epli_correctness_protocol,
    )
)


__all__ = [
    "BurtonUrbanProtocol",
    "ProtocolRunResult",
    "PROTOCOL_SPECS",
    "ValidationProtocolSpec",
    "_resolved_jobs",
    "find_spike_times_milliseconds",
    "get_validation_protocol_spec",
    "iter_validation_protocol_specs",
    "register_validation_protocol",
    "run_burton_urban_protocol",
]
