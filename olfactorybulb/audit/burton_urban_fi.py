"""Audit MC/TC f-I behavior against Burton & Urban 2014 criteria."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from olfactorybulb.audit.core import AuditItem, AuditReport, rounded


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


def summarize_metrics(metrics: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Return per-cell-type means for audit evidence and direction checks."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        grouped[str(metric["cell_type"])].append(metric)

    summary_keys = [
        "resting_potential_mV",
        "bias_current_pA",
        "zero_step_rate_Hz",
        "rheobase_pA",
        "spike_latency_ms",
        "peak_rate_Hz",
        "fi_gain_Hz_per_50pA",
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


def _type_pair(summary: dict[str, dict[str, float]], key: str) -> tuple[float, float]:
    return float(summary.get("MC", {}).get(key, float("nan"))), float(summary.get("TC", {}).get(key, float("nan")))


def _evidence_for_pair(summary: dict[str, dict[str, float]], key: str) -> dict[str, Any]:
    mc_value, tc_value = _type_pair(summary, key)
    return _rounded_dict(
        {
            "MC_mean": mc_value,
            "TC_mean": tc_value,
            "TC_minus_MC": tc_value - mc_value,
        }
    )


def build_validation_items(metrics: list[dict[str, Any]], protocol: BurtonUrbanProtocol) -> list[AuditItem]:
    summary = summarize_metrics(metrics)
    items: list[AuditItem] = []

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
            title="Burton & Urban MC/TC f-I protocol executed",
            criterion="Run 2 s somatic current steps from 0 to 300 pA in 50 pA increments after normalizing Vm to -58 mV.",
            evidence=protocol_evidence,
        )
    )

    bias_currents = [float(metric["bias_current_pA"]) for metric in metrics if np.isfinite(float(metric["bias_current_pA"]))]
    items.append(
        AuditItem(
            check_id="holding_current_normalization",
            status="PASS" if len(bias_currents) == len(metrics) else "FAIL",
            title="Holding currents were computed for -58 mV normalization",
            criterion="Every audited model should have a finite DC bias current before f-I and AP/spike-train measurements.",
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
            title="Cells are quiescent during the 0 pA step at normalized Vm",
            criterion="A model should not fire during the 0 pA f-I step after holding-current normalization.",
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
            title="MC and TC AP thresholds remain similar",
            criterion="Burton & Urban Table 4 reports similar MC and TC AP thresholds; model means should differ by <= 5 mV.",
            evidence={
                **_evidence_for_pair(summary, "AP_onset_mV"),
                "reference_means": {cell_type: TABLE4_REFERENCE[cell_type]["AP_onset_mV"] for cell_type in ("MC", "TC")},
            },
        )
    )

    items.append(
        AuditItem(
            check_id="tc_action_potentials_narrower",
            status="PASS" if _type_pair(summary, "FWHM_ms")[1] < _type_pair(summary, "FWHM_ms")[0] else "FAIL",
            title="TC action potentials are narrower than MC action potentials",
            criterion="Burton & Urban Table 4 reports lower FWHM in TCs than MCs.",
            evidence={
                **_evidence_for_pair(summary, "FWHM_ms"),
                "reference_means": {cell_type: TABLE4_REFERENCE[cell_type]["FWHM_ms"] for cell_type in ("MC", "TC")},
            },
        )
    )

    items.append(
        AuditItem(
            check_id="tc_repolarization_faster",
            status="PASS" if _type_pair(summary, "Fall_slope_mV_per_ms")[1] < _type_pair(summary, "Fall_slope_mV_per_ms")[0] else "FAIL",
            title="TC AP falling slope is faster than MC falling slope",
            criterion="Burton & Urban Table 4 reports a more negative AP falling slope in TCs than MCs.",
            evidence={
                **_evidence_for_pair(summary, "Fall_slope_mV_per_ms"),
                "reference_means": {cell_type: TABLE4_REFERENCE[cell_type]["Fall_slope_mV_per_ms"] for cell_type in ("MC", "TC")},
            },
        )
    )

    items.append(
        AuditItem(
            check_id="tc_ahp_decay_faster",
            status="PASS" if _type_pair(summary, "T_AHP50_ms")[1] < _type_pair(summary, "T_AHP50_ms")[0] else "FAIL",
            title="TC AHP half-decay is faster than MC AHP half-decay",
            criterion="Burton & Urban Table 4 reports lower T_AHP50% in TCs than MCs.",
            evidence={
                **_evidence_for_pair(summary, "T_AHP50_ms"),
                "reference_means": {cell_type: TABLE4_REFERENCE[cell_type]["T_AHP50_ms"] for cell_type in ("MC", "TC")},
            },
        )
    )

    items.append(
        AuditItem(
            check_id="tc_peak_instantaneous_rate_higher",
            status="PASS" if _type_pair(summary, "peak_rate_Hz")[1] > _type_pair(summary, "peak_rate_Hz")[0] else "FAIL",
            title="TC peak instantaneous firing rate is higher than MC peak rate",
            criterion="Burton & Urban Table 5 reports substantially higher TC peak instantaneous rate.",
            evidence={
                **_evidence_for_pair(summary, "peak_rate_Hz"),
                "reference_means": {cell_type: TABLE5_REFERENCE[cell_type]["Peak_rate_Hz"] for cell_type in ("MC", "TC")},
            },
        )
    )

    items.append(
        AuditItem(
            check_id="tc_fi_gain_higher",
            status="PASS" if _type_pair(summary, "fi_gain_Hz_per_50pA")[1] > _type_pair(summary, "fi_gain_Hz_per_50pA")[0] else "FAIL",
            title="TC f-I gain is higher than MC f-I gain",
            criterion="Burton & Urban Table 5 reports roughly twofold higher TC f-I gain.",
            evidence={
                **_evidence_for_pair(summary, "fi_gain_Hz_per_50pA"),
                "reference_means": {cell_type: TABLE5_REFERENCE[cell_type]["FI_gain_Hz_per_50pA"] for cell_type in ("MC", "TC")},
            },
        )
    )

    mc_rheobase, tc_rheobase = _type_pair(summary, "rheobase_pA")
    items.append(
        AuditItem(
            check_id="rheobase_in_paper_regime",
            status="PASS" if mc_rheobase > 0.0 and tc_rheobase > 0.0 else "FAIL",
            title="MC and TC rheobases remain in a depolarizing-step regime",
            criterion="Burton & Urban Table 5 reports positive rheobases for both MCs and TCs, not firing at the 0 pA step.",
            evidence={
                **_evidence_for_pair(summary, "rheobase_pA"),
                "reference_means": {cell_type: TABLE5_REFERENCE[cell_type]["Rheobase_pA"] for cell_type in ("MC", "TC")},
            },
        )
    )

    items.append(
        AuditItem(
            check_id="tc_cv_isi_higher",
            status="PASS" if _type_pair(summary, "cv_isi")[1] > _type_pair(summary, "cv_isi")[0] else "FAIL",
            title="TC CV_ISI near 20 Hz is higher than MC CV_ISI",
            criterion="Burton & Urban Table 5 and Figure 6 report higher TC firing irregularity.",
            evidence={
                **_evidence_for_pair(summary, "cv_isi"),
                "reference_means": {cell_type: TABLE5_REFERENCE[cell_type]["CV_ISI"] for cell_type in ("MC", "TC")},
            },
        )
    )

    items.append(
        AuditItem(
            check_id="input_resistance_recorded",
            status="PASS" if all(np.isfinite(float(metric["input_resistance_MOhm"])) for metric in metrics) else "FAIL",
            title="Input resistance was measured for f-I gain comparison",
            criterion="Figure 5F-style gain/resistance comparison requires finite input resistance estimates for every audited model.",
            evidence={cell_type: _rounded_dict(summary.get(cell_type, {})) for cell_type in sorted(summary)},
        )
    )

    return items


def run_burton_urban_protocol(
    *,
    cell_types: list[str],
    cell_count: int,
    protocol: BurtonUrbanProtocol,
    use_coreneuron: bool = False,
    use_gpu: bool = False,
) -> list[dict[str, Any]]:
    """Run the Burton & Urban step protocol and return per-cell metrics."""
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

    metrics: list[dict[str, Any]] = []
    for cell_name in _cell_names(cell_types, cell_count):
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

        zero_step_rate_hz = float(firing_rates_hz[0]) if len(firing_rates_hz) else float("nan")
        metrics.append(
            {
                "cell_name": cell_name,
                "cell_type": _cell_type_from_name(cell_name),
                "resting_potential_mV": resting_potential_mV,
                "bias_current_pA": bias_current_nA * 1000.0,
                "zero_step_rate_Hz": zero_step_rate_hz,
                "rheobase_pA": rheobase_nA * 1000.0 if np.isfinite(rheobase_nA) else float("nan"),
                "spike_latency_ms": spike_latency_ms,
                "peak_rate_Hz": peak_rate_hz,
                "fi_gain_Hz_per_50pA": fi_gain_hz_per_50pA,
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
        )

    return metrics


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--skip-neuron", action="store_true", help="Skip expensive NEURON-backed f-I validation.")
    parser.add_argument("--cell-count", type=int, default=5, help="Run models 1..N for each requested cell type.")
    parser.add_argument("--cell-types", default="MC,TC", help="Comma-separated cell type prefixes to audit.")
    parser.add_argument("--use-coreneuron", action="store_true", help="Run current-clamp sweeps with CoreNEURON.")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU mode when --use-coreneuron is set.")
    parser.add_argument("--dt-ms", type=float, default=0.1, help="Fixed integration time step in ms.")
    parser.add_argument("--bias-max-iterations", type=int, default=24, help="Binary-search iterations for -58 mV bias current.")


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
                    title="Burton & Urban f-I validation skipped",
                    criterion="Run this audit without --skip-neuron to execute the MC/TC current-clamp validation.",
                    evidence={
                        "cell_count": int(getattr(args, "cell_count", 5)),
                        "cell_types": getattr(args, "cell_types", "MC,TC"),
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
    )

    return AuditReport(
        audit_id="burton_urban_fi",
        title="Burton & Urban f-I validation audit",
        items=build_validation_items(metrics, protocol),
    )
