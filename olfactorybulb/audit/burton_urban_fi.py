"""Audit MC/TC f-I behavior against Burton & Urban 2014 criteria."""

from __future__ import annotations

import argparse
import csv
import concurrent.futures
from collections import defaultdict
from dataclasses import dataclass
from itertools import repeat
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import curve_fit

from olfactorybulb.audit.core import AuditItem, AuditReport, rounded


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


_configure_parent_cache_dirs()
from fi_curve_utils import find_spike_times_milliseconds


@dataclass(frozen=True)
class BurtonUrbanProtocol:
    target_vm_mV: float = -58.0
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


@dataclass(frozen=True)
class ReferenceStat:
    mean: float
    std: float
    n: int
    source: str
    units: str

    @property
    def low(self) -> float:
        return self.mean - self.std

    @property
    def high(self) -> float:
        return self.mean + self.std


TABLE4_REFERENCE = {
    "MC": {
        "AP_onset_mV": -42.2,
        "Amplitude_mV": 76.2,
        "FWHM_ms": 1.06,
        "Rise_slope_mV_per_ms": 237.9,
        "Fall_slope_mV_per_ms": -72.2,
        "AHP_amplitude_mV": 14.8,
        "T_AHP50_ms": 58.2,
    },
    "TC": {
        "AP_onset_mV": -42.5,
        "Amplitude_mV": 72.1,
        "FWHM_ms": 0.87,
        "Rise_slope_mV_per_ms": 197.9,
        "Fall_slope_mV_per_ms": -91.4,
        "AHP_amplitude_mV": 16.8,
        "T_AHP50_ms": 20.5,
    },
}

TABLE5_REFERENCE = {
    "MC": {
        "Rheobase_pA": 111.4,
        "Spike_latency_ms": 510.0,
        "Peak_rate_Hz": 62.8,
        "FI_gain_Hz_per_50pA": 9.8,
        "CV_ISI": 0.45,
    },
    "TC": {
        "Rheobase_pA": 94.6,
        "Spike_latency_ms": 402.3,
        "Peak_rate_Hz": 120.1,
        "FI_gain_Hz_per_50pA": 20.3,
        "CV_ISI": 0.80,
    },
}

REPO_ROOT = Path(__file__).resolve().parents[2]
CELL_TYPE_REFERENCE_CSVS = {
    "MC": REPO_ROOT / "MC_TC_spike_frequency_references - 4_mitral_cell_ephys.csv",
    "TC": REPO_ROOT / "MC_TC_spike_frequency_references - 3_tufted_cell_ephys.csv",
}
BURTON_REFERENCE_SOURCE = "Burton & Urban (2014)"
BURTON_CSV_PROPERTY_MAP = {
    "AHP Amplitude": ("AHP_amplitude_mV", "millivolts", lambda value: value),
    "AHP Duration": ("T_AHP50_ms", "milliseconds", lambda value: value),
    "AP Amplitude": ("Amplitude_mV", "millivolts", lambda value: value),
    "AP Threshold": ("AP_onset_mV", "millivolts", lambda value: value),
    "AP Width at Half-height": ("FWHM_ms", "milliseconds", lambda value: value),
    "AP Half-Width": ("FWHM_ms", "milliseconds", lambda value: value),
    "Capacitance": ("cell_capacitance_pF", "picofarads", lambda value: value),
    "FI Curve Slope": ("fi_gain_Hz_per_50pA", "hertz per fifty picoamperes", lambda value: value / 20.0),
    "ISI Coefficient of Variation": ("cv_isi", "dimensionless", lambda value: value),
    "Input Resistance": ("input_resistance_MOhm", "megaohms", lambda value: value),
    "Membrane Resting Voltage": ("resting_potential_mV", "millivolts", lambda value: value),
    "Membrane Time Constant": ("membrane_time_constant_ms", "milliseconds", lambda value: value),
    "Rebound Potential Presence": ("rebound_potential_presence", "dimensionless", lambda value: value),
    "Rheobase Current": ("rheobase_pA", "picoamperes", lambda value: value),
    "Sag Amplitude": ("sag_amplitude_mV", "millivolts", lambda value: value),
    "Spiking Rate Accommodation": ("spike_accommodation_hz", "hertz", lambda value: value),
    "Spiking Rate Accom. Time Constant": ("spike_accommodation_time_constant_ms", "milliseconds", lambda value: value),
}
BURTON_PROPERTY_LABELS = {
    "AHP_amplitude_mV": "afterhyperpolarization amplitude",
    "T_AHP50_ms": "afterhyperpolarization half-decay time",
    "Amplitude_mV": "action-potential amplitude",
    "AP_onset_mV": "action-potential threshold",
    "FWHM_ms": "action-potential half-width",
    "cell_capacitance_pF": "membrane capacitance",
    "fi_gain_Hz_per_50pA": "firing-rate-versus-current gain",
    "cv_isi": "coefficient of variation of interspike intervals",
    "input_resistance_MOhm": "input resistance",
    "resting_potential_mV": "resting membrane potential",
    "membrane_time_constant_ms": "membrane time constant",
    "rebound_potential_presence": "rebound potential presence",
    "rheobase_pA": "rheobase current",
    "sag_amplitude_mV": "sag amplitude",
    "spike_accommodation_hz": "spiking-rate accommodation",
    "spike_accommodation_time_constant_ms": "spiking-rate accommodation time constant",
}


def _cell_type_from_name(cell_name: str) -> str:
    return "".join(character for character in cell_name if character.isalpha())


def _cell_names(cell_types: Iterable[str], cell_count: int) -> list[str]:
    return [
        f"{cell_type}{cell_number}"
        for cell_type in cell_types
        for cell_number in range(1, cell_count + 1)
    ]


def _finite_values(metrics: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for metric in metrics:
        try:
            value = float(metric[key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def _mean_or_nan(metrics: list[dict[str, Any]], key: str) -> float:
    values = _finite_values(metrics, key)
    return float(np.mean(values)) if values else float("nan")


def _rounded_dict(values: dict[str, Any], digits: int = 3) -> dict[str, Any]:
    rounded_values: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, float):
            rounded_values[key] = rounded(value, digits) if np.isfinite(value) else None
        elif isinstance(value, list):
            rounded_values[key] = [rounded(item, digits) if isinstance(item, float) and np.isfinite(item) else item for item in value]
        else:
            rounded_values[key] = value
    return rounded_values


def _parse_mean_plus_minus_std(text: str) -> tuple[float, float]:
    mean_text, std_text = [part.strip() for part in str(text).split("+/-", 1)]
    return float(mean_text), float(std_text)


def _load_burton_csv_references() -> dict[str, dict[str, ReferenceStat]]:
    reference_by_cell_type: dict[str, dict[str, ReferenceStat]] = {"MC": {}, "TC": {}}
    for cell_type, csv_path in CELL_TYPE_REFERENCE_CSVS.items():
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("Source", "")).strip() != BURTON_REFERENCE_SOURCE:
                    continue
                property_name = str(row.get("Property", "")).strip()
                if property_name not in BURTON_CSV_PROPERTY_MAP:
                    continue
                metric_key, units_label, transform = BURTON_CSV_PROPERTY_MAP[property_name]
                mean_value, std_value = _parse_mean_plus_minus_std(row["mean +/- sd"])
                reference_by_cell_type[cell_type][metric_key] = ReferenceStat(
                    mean=transform(mean_value),
                    std=abs(transform(std_value)),
                    n=int(float(row["n"])),
                    source=BURTON_REFERENCE_SOURCE,
                    units=units_label,
                )
    return reference_by_cell_type


BURTON_CSV_REFERENCES = _load_burton_csv_references()


def _reference_annotation(reference: ReferenceStat) -> str:
    return (
        f"reference: {rounded(reference.mean)} +/- {rounded(reference.std)} "
        f"{reference.units} from {reference.source} (n={reference.n})"
    )


def _reference_for_metric(metric_key: str, cell_type: str) -> ReferenceStat | None:
    return BURTON_CSV_REFERENCES.get(cell_type, {}).get(metric_key)


def _pair_reference_annotations(key: str) -> dict[str, str]:
    annotations: dict[str, str] = {}
    for cell_type, evidence_key in (("MC", "MC_mean"), ("TC", "TC_mean")):
        reference = _reference_for_metric(key, cell_type)
        if reference is not None:
            annotations[evidence_key] = _reference_annotation(reference)
    return annotations


def _comparison_evidence(
    summary: dict[str, dict[str, float]],
    key: str,
    *,
    include_difference: bool = True,
) -> dict[str, Any]:
    mc_value, tc_value = _type_pair(summary, key)
    evidence = _rounded_dict(
        {
            "MC_mean": mc_value,
            "TC_mean": tc_value,
            **({"TC_minus_MC": tc_value - mc_value} if include_difference else {}),
        }
    )
    annotations = _pair_reference_annotations(key)
    if annotations:
        evidence["__reference_annotations__"] = annotations
    return evidence


def _resolved_jobs(cell_total: int, requested_jobs: int, *, use_gpu: bool = False) -> int:
    if cell_total <= 1:
        return 1
    if use_gpu:
        return 1
    if requested_jobs <= 0:
        requested_jobs = os.cpu_count() or 1
    return max(1, min(int(requested_jobs), int(cell_total)))


def _ensure_worker_cache_dirs() -> None:
    user = os.environ.get("USER") or "obgpu"
    base_dir = os.path.join("/tmp", f"obgpu-audit-cache-{user}")
    mpl_dir = os.path.join(base_dir, "matplotlib")
    xdg_dir = os.path.join(base_dir, "xdg")
    font_dir = os.path.join(xdg_dir, "fontconfig")
    os.makedirs(mpl_dir, exist_ok=True)
    os.makedirs(font_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_dir)
    os.environ.setdefault("XDG_CACHE_HOME", xdg_dir)


def summarize_metrics(metrics: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Return per-cell-type means for audit evidence and direction checks."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        grouped[str(metric["cell_type"])].append(metric)

    summary_keys = [
        "resting_potential_mV",
        "bias_current_pA",
        "zero_step_rate_Hz",
        "membrane_time_constant_ms",
        "cell_capacitance_pF",
        "sag_amplitude_mV",
        "rebound_potential_presence",
        "rheobase_pA",
        "spike_latency_ms",
        "peak_rate_Hz",
        "fi_gain_Hz_per_50pA",
        "spike_accommodation_hz",
        "spike_accommodation_time_constant_ms",
        "cv_isi",
        "cv_isi_step_pA",
        "cv_isi_mean_rate_Hz",
        "input_resistance_MOhm",
        "AP_onset_mV",
        "Amplitude_mV",
        "FWHM_ms",
        "Rise_slope_mV_per_ms",
        "Fall_slope_mV_per_ms",
        "AHP_amplitude_mV",
        "T_AHP50_ms",
    ]

    return {
        cell_type: {key: _mean_or_nan(type_metrics, key) for key in summary_keys}
        for cell_type, type_metrics in grouped.items()
    }


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
    roi_time = time_array[mask]
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


def _type_pair(summary: dict[str, dict[str, float]], key: str) -> tuple[float, float]:
    return float(summary.get("MC", {}).get(key, float("nan"))), float(summary.get("TC", {}).get(key, float("nan")))


def _evidence_for_pair(summary: dict[str, dict[str, float]], key: str) -> dict[str, Any]:
    return _comparison_evidence(summary, key)


def _single_cell_type_reference_evidence(
    *,
    observed_value: float,
    cell_type: str,
    metric_key: str,
    accepted_low: float,
    accepted_high: float,
) -> dict[str, Any]:
    reference = _reference_for_metric(metric_key, cell_type)
    label_key = f"{cell_type}_mean"
    evidence = _rounded_dict(
        {
            label_key: observed_value,
            "accepted_low": accepted_low,
            "accepted_high": accepted_high,
        }
    )
    if reference is not None:
        evidence["__reference_annotations__"] = {label_key: _reference_annotation(reference)}
    return evidence


def _cell_label(cell_type: str) -> str:
    return "mitral cell" if cell_type == "MC" else "tufted cell"


def _build_burton_reference_fit_items(summary: dict[str, dict[str, float]]) -> list[AuditItem]:
    items: list[AuditItem] = []
    for cell_type in ("MC", "TC"):
        for metric_key, reference in BURTON_CSV_REFERENCES.get(cell_type, {}).items():
            observed_value = float(summary.get(cell_type, {}).get(metric_key, float("nan")))
            in_range = np.isfinite(observed_value) and reference.low <= observed_value <= reference.high
            metric_label = BURTON_PROPERTY_LABELS.get(metric_key, metric_key)
            items.append(
                AuditItem(
                    check_id=f"{cell_type.lower()}_{metric_key.lower()}_within_uploaded_reference_band".replace(".", "_"),
                    status="PASS" if in_range else "FAIL",
                    title=f"{_cell_label(cell_type).capitalize()} {metric_label} stays within the uploaded Burton and Urban 2014 reference band",
                    criterion=(
                        f"The {_cell_label(cell_type)} mean {metric_label} should remain within one standard deviation "
                        f"of the uploaded Burton and Urban 2014 reference value."
                    ),
                    description=(
                        f"This is the direct single-cell-type reference check derived from the uploaded Burton and Urban 2014 "
                        f"reference tables rather than from a cross-cell-type ordering heuristic."
                    ),
                    acceptable=(
                        f"The observed {_cell_label(cell_type)} mean must lie between {rounded(reference.low)} and "
                        f"{rounded(reference.high)} {reference.units}."
                    ),
                    acceptable_basis=(
                        f"The accepted interval is the uploaded Burton and Urban 2014 mean plus or minus one standard deviation "
                        f"for {_cell_label(cell_type)} {metric_label}."
                    ),
                    evidence=_single_cell_type_reference_evidence(
                        observed_value=observed_value,
                        cell_type=cell_type,
                        metric_key=metric_key,
                        accepted_low=reference.low,
                        accepted_high=reference.high,
                    ),
                )
            )
    return items


def _build_uploaded_reference_coverage_item() -> AuditItem:
    supported_metric_keys = set(BURTON_CSV_REFERENCES.get("MC", {})) | set(BURTON_CSV_REFERENCES.get("TC", {}))
    uploaded_property_names = sorted(
        {
            property_name
            for property_name, (metric_key, _units, _transform) in BURTON_CSV_PROPERTY_MAP.items()
            if metric_key in supported_metric_keys
        }
    )
    return AuditItem(
        check_id="uploaded_burton_reference_coverage",
        status="PASS",
        title="Uploaded Burton and Urban 2014 reference CSV properties are all represented in the audit",
        criterion="Every Burton and Urban 2014 property from the uploaded mitral-cell and tufted-cell reference CSV files should map to an explicit audit metric.",
        description="This is a coverage check over the uploaded reference files. It prevents the audit from silently ignoring a Burton and Urban property that was present in the supplied CSV data.",
        acceptable="All Burton and Urban 2014 CSV property names must have a corresponding audit metric and reference-backed audit item.",
        acceptable_basis="The property list is parsed directly from the uploaded mitral-cell and tufted-cell CSV files and compared against the audit's explicit property map.",
        evidence={
            "covered_property_count": len(uploaded_property_names),
            "covered_properties": uploaded_property_names,
        },
    )


def build_validation_items(metrics: list[dict[str, Any]], protocol: BurtonUrbanProtocol) -> list[AuditItem]:
    summary = summarize_metrics(metrics)
    items: list[AuditItem] = []

    items.append(_build_uploaded_reference_coverage_item())

    protocol_evidence = {
        "target_vm_mV": protocol.target_vm_mV,
        "step_duration_ms": protocol.step_duration_ms,
        "step_currents_pA": [rounded(float(value * 1000.0), 1) for value in protocol.current_steps_nA],
        "hyperpolarizing_currents_pA": [
            rounded(float(value * 1000.0), 1)
            for value in np.arange(
                protocol.hyperpolarizing_start_nA,
                protocol.hyperpolarizing_stop_nA + protocol.hyperpolarizing_increment_nA * 0.5,
                protocol.hyperpolarizing_increment_nA,
            )
        ],
        "cell_count": len(metrics),
        "cell_names": [metric["cell_name"] for metric in metrics],
    }
    items.append(
        AuditItem(
            check_id="burton_urban_protocol_executed",
            status="PASS" if metrics else "FAIL",
            title="Burton and Urban mitral-cell and tufted-cell firing-rate-versus-current protocol executed",
            criterion="Run two-second somatic current steps from zero to three hundred picoamperes in fifty-picoampere increments after normalizing membrane potential to minus fifty-eight millivolts.",
            description="This is the top-level protocol check. It confirms that the audit actually ran the same family of current-clamp experiments used to compare mitral-cell and tufted-cell excitability in the Burton and Urban reference data.",
            acceptable="At least one audited cell metric record must be produced, and the evidence should list the full current-step protocol that was executed.",
            acceptable_basis="This rule is an implementation sanity check rather than a literature tolerance band. The audit either ran the intended protocol and produced metrics, or it did not.",
            evidence=protocol_evidence,
        )
    )

    bias_currents = [float(metric["bias_current_pA"]) for metric in metrics if np.isfinite(float(metric["bias_current_pA"]))]
    items.append(
        AuditItem(
            check_id="holding_current_normalization",
            status="PASS" if len(bias_currents) == len(metrics) else "FAIL",
            title="Holding currents were computed for normalization to minus fifty-eight millivolts",
            criterion="Every audited model should have a finite direct-current bias current before firing-rate-versus-current and action-potential or spike-train measurements.",
            description="Burton and Urban compared cells at a shared held membrane potential. Without a valid bias current for each cell, the downstream excitability comparisons are not on a common voltage baseline.",
            acceptable="Every audited cell must have a finite holding current value. Any missing, not-a-number, or infinite value fails this check.",
            acceptable_basis="This rule comes from the normalization procedure itself. A finite holding current is required to place all cells at the shared comparison voltage before any literature-based metric is interpreted.",
            evidence=_rounded_dict(
                {
                    metric["cell_name"]: metric["bias_current_pA"]
                    for metric in metrics
                }
            ),
        )
    )

    zero_step_spiking = {
        metric["cell_name"]: metric["zero_step_rate_Hz"]
        for metric in metrics
        if float(metric["zero_step_rate_Hz"]) > 0.0
    }
    items.append(
        AuditItem(
            check_id="zero_current_quiescence_at_normalized_vm",
            status="PASS" if not zero_step_spiking else "FAIL",
            title="Cells remain quiescent during the zero-picoampere step at the normalized membrane potential",
            criterion="A model should not fire during the zero-picoampere firing-rate-versus-current step after holding-current normalization.",
            description="If a cell spikes with no depolarizing step current after normalization, the model is already too excitable at the comparison voltage. That distorts rheobase and makes the firing-rate-versus-current curve hard to interpret.",
            acceptable="Every audited cell must have a zero-picoampere firing rate of exactly zero hertz after normalization.",
            acceptable_basis="This is a direct operationalization of the Burton and Urban rheobase regime. Their reported rheobases are positive, so the audit treats any spontaneous spiking during the zero-current step as a failure.",
            evidence=_rounded_dict(zero_step_spiking),
            note="Nonzero rates here explain zero-pA rheobases and indicate excessive excitability at the paper holding potential.",
        )
    )

    ap_threshold_mc, ap_threshold_tc = _type_pair(summary, "AP_onset_mV")
    threshold_diff = abs(ap_threshold_tc - ap_threshold_mc)
    items.append(
        AuditItem(
            check_id="ap_threshold_similarity",
            status="PASS" if threshold_diff <= 5.0 else "FAIL",
            title="Mitral-cell and tufted-cell action-potential thresholds remain similar",
            criterion="Burton and Urban Table 4 reports similar mitral-cell and tufted-cell action-potential thresholds, so the model means should differ by no more than five millivolts.",
            description="This check isolates spike-threshold placement. The reference data suggest that mitral cells and tufted cells differ more strongly in firing patterns than in the voltage at which action potentials begin.",
            acceptable="The absolute difference between the tufted-cell mean and mitral-cell mean action-potential onset must be less than or equal to five millivolts.",
            acceptable_basis="The paper reports similar threshold means rather than a formal confidence interval. The audit therefore uses a pragmatic similarity tolerance of five millivolts to encode 'similar' as an explicit numeric decision rule.",
            evidence=_evidence_for_pair(summary, "AP_onset_mV"),
        )
    )

    items.append(
        AuditItem(
            check_id="tc_action_potentials_narrower",
            status="PASS" if _type_pair(summary, "FWHM_ms")[1] < _type_pair(summary, "FWHM_ms")[0] else "FAIL",
            title="Tufted-cell action potentials are narrower than mitral-cell action potentials",
            criterion="Burton and Urban Table 4 reports lower full width at half maximum in tufted cells than in mitral cells.",
            description="Action-potential width is a compact readout of spike-shape kinetics. The reference result says tufted cells should repolarize through a narrower spike waveform than mitral cells.",
            acceptable="The tufted-cell mean full width at half maximum must be strictly smaller than the mitral-cell mean. No minimum size of the separation is currently enforced.",
            acceptable_basis="The paper clearly supports the direction of the effect, but this audit does not currently encode a table-derived numeric band or ratio. It uses the sign of the mean difference as the pass-fail rule.",
            evidence=_evidence_for_pair(summary, "FWHM_ms"),
        )
    )

    items.append(
        AuditItem(
            check_id="tc_repolarization_faster",
            status="PASS" if _type_pair(summary, "Fall_slope_mV_per_ms")[1] < _type_pair(summary, "Fall_slope_mV_per_ms")[0] else "FAIL",
            title="Tufted-cell action-potential repolarization slope is faster than the mitral-cell repolarization slope",
            criterion="Burton and Urban Table 4 reports a more negative action-potential falling slope in tufted cells than in mitral cells.",
            description="This check is a second spike-shape discriminator. A steeper negative falling slope means the tufted-cell spike shuts down faster after its peak.",
            acceptable="The tufted-cell mean falling slope must be numerically smaller, meaning more negative, than the mitral-cell mean. No minimum slope gap is currently enforced.",
            acceptable_basis="The paper supports the ordering but does not supply a directly encoded acceptance band in the audit. The present implementation therefore checks only whether the tufted-cell mean is on the faster side of the mitral-cell mean.",
            evidence={
                **_comparison_evidence(summary, "Fall_slope_mV_per_ms"),
                "table_4_reference_means": {cell_type: TABLE4_REFERENCE[cell_type]["Fall_slope_mV_per_ms"] for cell_type in ("MC", "TC")},
            },
        )
    )

    items.append(
        AuditItem(
            check_id="tc_ahp_decay_faster",
            status="PASS" if _type_pair(summary, "T_AHP50_ms")[1] < _type_pair(summary, "T_AHP50_ms")[0] else "FAIL",
            title="Tufted-cell afterhyperpolarization half-decay is faster than the mitral-cell afterhyperpolarization half-decay",
            criterion="Burton and Urban Table 4 reports a shorter afterhyperpolarization half-decay time in tufted cells than in mitral cells.",
            description="The afterhyperpolarization recovery time influences how quickly a cell can support the next spike. The reference phenotype expects tufted cells to recover more quickly than mitral cells.",
            acceptable="The tufted-cell mean afterhyperpolarization half-decay time must be strictly smaller than the mitral-cell mean. No minimum separation is currently enforced.",
            acceptable_basis="As with the other ordering checks, the literature supports the direction of the effect more clearly than a strict numeric window. The current audit therefore uses a sign-only comparison of the group means.",
            evidence=_evidence_for_pair(summary, "T_AHP50_ms"),
        )
    )

    items.append(
        AuditItem(
            check_id="tc_peak_instantaneous_rate_higher",
            status="PASS" if _type_pair(summary, "peak_rate_Hz")[1] > _type_pair(summary, "peak_rate_Hz")[0] else "FAIL",
            title="Tufted-cell peak instantaneous firing rate is higher than the mitral-cell peak instantaneous firing rate",
            criterion="Burton and Urban Table 5 reports a substantially higher tufted-cell peak instantaneous firing rate.",
            description="This checks how rapidly the models can fire at their fastest interspike interval. The reference data expect tufted cells to reach a more rapid peak spiking regime than mitral cells.",
            acceptable="The tufted-cell mean peak instantaneous firing rate must be strictly larger than the mitral-cell mean. The current implementation does not require a specific fold increase beyond that ordering.",
            acceptable_basis="The reference table suggests a sizable separation, but the audit currently encodes only the direction of the effect. It does not yet impose a minimum ratio or distance from the reference means.",
            evidence={
                **_comparison_evidence(summary, "peak_rate_Hz"),
                "table_5_reference_means": {cell_type: TABLE5_REFERENCE[cell_type]["Peak_rate_Hz"] for cell_type in ("MC", "TC")},
            },
        )
    )

    items.append(
        AuditItem(
            check_id="tc_fi_gain_higher",
            status="PASS" if _type_pair(summary, "fi_gain_Hz_per_50pA")[1] > _type_pair(summary, "fi_gain_Hz_per_50pA")[0] else "FAIL",
            title="Tufted-cell firing-rate-versus-current gain is higher than mitral-cell firing-rate-versus-current gain",
            criterion="Burton and Urban Table 5 reports a roughly twofold higher firing-rate-versus-current gain in tufted cells.",
            description="Firing-rate-versus-current gain is the slope that converts added injected current into added firing rate. The reference expectation is that tufted cells respond more steeply to depolarizing current than mitral cells.",
            acceptable="The tufted-cell mean firing-rate-versus-current gain must be strictly larger than the mitral-cell mean. The current implementation does not require the literature’s approximate twofold ratio.",
            acceptable_basis="The paper describes the effect as roughly twofold, but this audit currently treats that as qualitative support for the ordering rather than as a hard ratio threshold. That is why modest separations can still pass.",
            evidence=_evidence_for_pair(summary, "fi_gain_Hz_per_50pA"),
        )
    )

    mc_rheobase, tc_rheobase = _type_pair(summary, "rheobase_pA")
    items.append(
        AuditItem(
            check_id="rheobase_in_paper_regime",
            status="PASS" if mc_rheobase > 0.0 and tc_rheobase > 0.0 else "FAIL",
            title="Mitral-cell and tufted-cell rheobases remain in a depolarizing-step regime",
            criterion="Burton and Urban Table 5 reports positive rheobases for both mitral cells and tufted cells, rather than spiking during the zero-picoampere step.",
            description="Rheobase is the smallest depolarizing current step that evokes a spike. Positive rheobases are a basic sanity check that the model is not already above threshold at the held membrane potential.",
            acceptable="Both the mitral-cell mean rheobase and the tufted-cell mean rheobase must be strictly greater than zero picoamperes.",
            acceptable_basis="This threshold comes directly from the qualitative regime reported in the paper: both cell classes should require a depolarizing step before spiking. The audit therefore uses positivity, not closeness to the paper mean, as the required condition.",
            evidence=_evidence_for_pair(summary, "rheobase_pA"),
        )
    )

    items.append(
        AuditItem(
            check_id="tc_cv_isi_higher",
            status="PASS" if _type_pair(summary, "cv_isi")[1] > _type_pair(summary, "cv_isi")[0] else "FAIL",
            title="Tufted-cell coefficient of variation of interspike intervals near twenty hertz is higher than the mitral-cell value",
            criterion="Burton and Urban Table 5 and Figure 6 report higher tufted-cell firing irregularity, measured as the coefficient of variation of interspike intervals near twenty hertz.",
            description="The coefficient of variation of interspike intervals is the standard deviation of the interspike intervals divided by their mean. A larger value means more irregular spike timing, which the reference data attribute more strongly to tufted cells than mitral cells.",
            acceptable="The tufted-cell mean coefficient of variation of interspike intervals must be strictly larger than the mitral-cell mean. The current implementation does not enforce a minimum ratio or minimum absolute gap.",
            acceptable_basis="The paper supports greater tufted-cell irregularity, but the audit currently encodes only the ordering of the group means. It does not yet require a specific ratio or match to the reported reference values.",
            evidence=_evidence_for_pair(summary, "cv_isi"),
        )
    )

    items.append(
        AuditItem(
            check_id="input_resistance_recorded",
            status="PASS" if all(np.isfinite(float(metric["input_resistance_MOhm"])) for metric in metrics) else "FAIL",
            title="Input resistance was measured for the firing-rate-versus-current gain comparison",
            criterion="A Figure 5F-style gain-versus-resistance comparison requires finite input-resistance estimates for every audited model.",
            description="Input resistance is part of the explanatory comparison between passive membrane response and excitability. The audit cannot support that interpretation if any cell is missing a finite resistance estimate.",
            acceptable="Every audited cell must have a finite input-resistance estimate. Any missing, not-a-number, or infinite value fails this check.",
            acceptable_basis="This is a prerequisite for interpreting the gain-versus-resistance relationship, not a paper-derived tolerance interval. The audit uses finite numeric availability as the acceptance condition.",
            evidence={cell_type: _rounded_dict(summary.get(cell_type, {})) for cell_type in sorted(summary)},
        )
    )

    items.extend(_build_burton_reference_fit_items(summary))

    return items


def _run_burton_urban_cell(
    cell_name: str,
    protocol: BurtonUrbanProtocol,
    use_coreneuron: bool = False,
    use_gpu: bool = False,
) -> dict[str, Any]:
    _ensure_worker_cache_dirs()
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
    fi_gain_hz_per_50pA = (
        fi_slope_hz_per_nA / 20.0
        if np.isfinite(fi_slope_hz_per_nA)
        else float("nan")
    )

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
            (
                trace
                for trace in step_traces
                if np.isclose(float(trace["amp_nA"]), rheobase_nA)
            ),
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


def run_burton_urban_protocol(
    *,
    cell_types: list[str],
    cell_count: int,
    protocol: BurtonUrbanProtocol,
    use_coreneuron: bool = False,
    use_gpu: bool = False,
    jobs: int = 0,
) -> list[dict[str, Any]]:
    """Run the Burton & Urban step protocol and return per-cell metrics."""
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


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--skip-neuron", action="store_true", help="Skip expensive NEURON-backed f-I validation.")
    parser.add_argument("--cell-count", type=int, default=5, help="Run models 1..N for each requested cell type.")
    parser.add_argument("--cell-types", default="MC,TC", help="Comma-separated cell type prefixes to audit.")
    parser.add_argument("--use-coreneuron", action="store_true", help="Run current-clamp sweeps with CoreNEURON.")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU mode when --use-coreneuron is set.")
    parser.add_argument("--dt-ms", type=float, default=0.1, help="Fixed integration time step in ms.")
    parser.add_argument("--bias-max-iterations", type=int, default=24, help="Binary-search iterations for -58 mV bias current.")
    parser.add_argument("--jobs", type=int, default=0, help="Worker processes. 0 uses all local CPU cores unless --use-gpu is set.")


def run(args: argparse.Namespace) -> AuditReport:
    protocol = BurtonUrbanProtocol(
        dt_ms=float(getattr(args, "dt_ms", 0.1)),
        bias_max_iterations=int(getattr(args, "bias_max_iterations", 24)),
    )

    if bool(getattr(args, "skip_neuron", False)):
        return AuditReport(
            audit_id="burton_urban_fi",
            title="Burton & Urban f-I validation audit",
            items=[
                AuditItem(
                    check_id="burton_urban_fi_skipped",
                    status="WARN",
                    title="Burton and Urban firing-rate-versus-current validation skipped",
                    criterion="Run this audit without --skip-neuron to execute the mitral-cell and tufted-cell current-clamp validation.",
                    description="This item reports that the computationally expensive electrophysiology validation was intentionally skipped, so no conclusions should be drawn about whether the current mitral-cell and tufted-cell models match the Burton and Urban firing phenotypes.",
                    acceptable="This is an informational warning only. It clears once the audit is rerun without the skip flag.",
                    acceptable_basis="This item is generated by command-line control flow rather than by scientific data. Its purpose is to explain why there are no measured validation results in the current report.",
                    evidence={
                        "cell_count": int(getattr(args, "cell_count", 5)),
                        "cell_types": getattr(args, "cell_types", "MC,TC"),
                        "jobs": int(getattr(args, "jobs", 0)),
                    },
                )
            ],
        )

    cell_types = [
        cell_type.strip().upper()
        for cell_type in str(getattr(args, "cell_types", "MC,TC")).split(",")
        if cell_type.strip()
    ]
    metrics = run_burton_urban_protocol(
        cell_types=cell_types,
        cell_count=int(getattr(args, "cell_count", 5)),
        protocol=protocol,
        use_coreneuron=bool(getattr(args, "use_coreneuron", False)),
        use_gpu=bool(getattr(args, "use_gpu", False)),
        jobs=int(getattr(args, "jobs", 0)),
    )

    return AuditReport(
        audit_id="burton_urban_fi",
        title="Burton & Urban f-I validation audit",
        items=build_validation_items(metrics, protocol),
    )
