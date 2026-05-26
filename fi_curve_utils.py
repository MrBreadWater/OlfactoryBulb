"""f-I curve characterization utilities for OB cell models.

Simulation is delegated to single_cell_utils; this module provides
spike-counting, f-I analysis, and plotting on top of those results.

All current values are in nA (NEURON native unit). Frequencies are in Hz.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from single_cell_utils import (
    _resolve_cell_class,
    run_current_clamp_series,
    run_fi_ramp,
)


# ---------------------------------------------------------------------------
# Spike counting and f-I conversion
# ---------------------------------------------------------------------------

def count_spikes(trace_dict: dict, threshold_mV: float = -20.0) -> int:
    """Count upward threshold crossings in a voltage trace dict.

    Parameters
    ----------
    trace_dict   : dict with a "v_soma" key (output of any single_cell_utils run)
    threshold_mV : spike detection threshold (mV)
    """
    v = trace_dict["v_soma"]
    above = v > threshold_mV
    return int(np.sum(np.diff(above.astype(int)) == 1))


def traces_to_fi(
    results: List[dict],
    step_dur_ms: float,
    threshold_mV: float = -20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert run_current_clamp_series output to (currents_nA, freqs_hz).

    Parameters
    ----------
    results      : list of dicts from single_cell_utils.run_current_clamp_series
    step_dur_ms  : current step duration used in the run (ms)
    threshold_mV : spike detection threshold (mV)

    Returns
    -------
    currents_nA : 1-D ndarray
    freqs_hz    : 1-D ndarray
    """
    currents = np.array([r["amp_nA"] for r in results])
    freqs = np.array([
        count_spikes(r, threshold_mV) / (step_dur_ms * 1e-3)
        for r in results
    ])
    return currents, freqs


def ramp_to_fi(
    ramp_result: dict,
    threshold_mV: float = -20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert single_cell_utils.run_fi_ramp output to (currents_nA, freqs_hz).

    Parameters
    ----------
    ramp_result  : dict from single_cell_utils.run_fi_ramp
    threshold_mV : spike detection threshold (mV)

    Returns
    -------
    currents_nA : 1-D ndarray
    freqs_hz    : 1-D ndarray
    """
    t           = ramp_result["t"]
    v           = ramp_result["v_soma"]
    currents_nA = ramp_result["currents_nA"]
    step_dur_ms = ramp_result["step_dur_ms"]
    delay_ms    = ramp_result["delay_ms"]

    above      = v > threshold_mV
    spike_mask = np.diff(above.astype(int)) == 1
    t_mid      = (t[:-1] + t[1:]) / 2
    t_spikes   = t_mid[spike_mask]

    freqs = []
    for i in range(len(currents_nA)):
        t0 = delay_ms + i * step_dur_ms
        t1 = t0 + step_dur_ms
        n_spikes = int(((t_spikes >= t0) & (t_spikes < t1)).sum())
        freqs.append(n_spikes / (step_dur_ms * 1e-3))

    return currents_nA, np.array(freqs)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_fi_slope(
    currents_nA: np.ndarray,
    freqs_hz: np.ndarray,
    freq_threshold_hz: float = 1.0,
) -> Tuple[float, float, float]:
    """Fit a line to the suprathreshold portion of an f-I curve.

    Only points where freqs_hz >= freq_threshold_hz are included in the fit.

    Returns
    -------
    slope_hz_per_nA, intercept_hz, r_squared
    All NaN when fewer than 2 suprathreshold points exist.
    """
    mask = freqs_hz >= freq_threshold_hz
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan
    slope, intercept, r, *_ = linregress(currents_nA[mask], freqs_hz[mask])
    return float(slope), float(intercept), float(r ** 2)


def _estimate_rheobase(currents_nA: np.ndarray, freqs_hz: np.ndarray) -> float:
    """Current (nA) at which the first spike epoch was observed."""
    firing = freqs_hz >= 1.0
    if not firing.any():
        return np.nan
    return float(currents_nA[firing][0])


# ---------------------------------------------------------------------------
# Convenience batch runner
# ---------------------------------------------------------------------------

def run_cell_type_fi(
    cell_type: str,
    protocol: str = "steps",
    n_cells: int = 5,
    use_coreneuron: bool = False,
    use_gpu: bool = False,
    i_start_nA: float = 0.0,
    i_stop_nA: float = 0.5,
    i_step_nA: float = 0.05,
    step_dur_ms: float = 500.0,
    delay_ms: float = 200.0,
    tail_ms: float = 50.0,
    dt: float = 0.1,
    celsius: float = 35.0,
    threshold_mV: float = -20.0,
) -> Dict[str, dict]:
    """Run the f-I protocol for all numbered models of one cell type.

    Parameters
    ----------
    cell_type  : 'MC', 'TC', or 'GC'
    protocol   : 'steps' (independent runs) or 'ramp' (staircase)
    use_coreneuron : enable CoreNEURON for this sweep (fast for many amplitudes)
    use_gpu    : enable GPU-accelerated CoreNEURON (requires GPU-capable build)
    n_cells    : run models 1 through n_cells

    Returns
    -------
    dict : {cell_name → {currents_nA, freqs_hz, slope, intercept, r2}}
    """
    if protocol not in {"steps", "ramp"}:
        raise ValueError("protocol must be 'steps' or 'ramp'")

    amps = np.arange(i_start_nA, i_stop_nA + i_step_nA * 0.5, i_step_nA)
    results = {}

    for i in range(1, n_cells + 1):
        name = f"{cell_type}{i}"
        print(f"  {name}...", end=" ", flush=True)

        if protocol == "steps":
            traces = run_current_clamp_series(
                name, amps_nA=amps,
                duration_ms=step_dur_ms, delay_ms=delay_ms, tail_ms=tail_ms,
                dt=dt, celsius=celsius,
                use_coreneuron=use_coreneuron, use_gpu=use_gpu,
            )
            currents, freqs = traces_to_fi(traces, step_dur_ms, threshold_mV)
        else:
            ramp = run_fi_ramp(
                name,
                i_start_nA=i_start_nA, i_stop_nA=i_stop_nA, i_step_nA=i_step_nA,
                step_dur_ms=step_dur_ms, delay_ms=delay_ms, tail_ms=tail_ms,
                dt=dt, celsius=celsius,
                use_coreneuron=use_coreneuron, use_gpu=use_gpu,
            )
            currents, freqs = ramp_to_fi(ramp, threshold_mV)

        slope, intercept, r2 = compute_fi_slope(currents, freqs)
        results[name] = dict(
            currents_nA=currents,
            freqs_hz=freqs,
            slope=slope,
            intercept=intercept,
            r2=r2,
        )
        if np.isnan(slope):
            print("no spikes detected")
        else:
            print(f"slope = {slope:.1f} Hz/nA,  R² = {r2:.3f}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_fi_curves(
    all_results: Dict[str, Dict[str, dict]],
    protocol_label: str = "",
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot f-I curves and linear fits for one or more cell types side-by-side.

    Parameters
    ----------
    all_results    : {cell_type: {cell_name: data_dict}}
                     as returned by run_cell_type_fi
    protocol_label : short string shown in each subplot title
    figsize        : override figure size
    save_path      : if given, save the figure to this path

    Returns
    -------
    matplotlib Figure
    """
    n_types = len(all_results)
    if figsize is None:
        figsize = (7 * n_types, 5)

    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, n_types, figsize=figsize, squeeze=False)

    for ax, (cell_type, type_results) in zip(axes[0], all_results.items()):
        all_currents = None

        for idx, (name, data) in enumerate(type_results.items()):
            c         = colors[idx % len(colors)]
            currents  = data["currents_nA"]
            freqs     = data["freqs_hz"]
            slope     = data["slope"]
            intercept = data["intercept"]
            r2        = data["r2"]

            if all_currents is None:
                all_currents = currents

            ax.plot(currents, freqs, "o-", color=c, label=name, markersize=5, lw=1.5)

            if not np.isnan(slope):
                mask      = freqs >= 1.0
                fit_label = f"{name}: {slope:.0f} Hz/nA  (R²={r2:.2f})"
                ax.plot(currents[mask], slope * currents[mask] + intercept,
                        "--", color=c, alpha=0.6, label=fit_label)

        title = f"{cell_type} f-I Curves"
        if protocol_label:
            title += f"  [{protocol_label}]"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Injected Current (nA)")
        ax.set_ylabel("Firing Frequency (Hz)")
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(True, alpha=0.3)
        if all_currents is not None:
            pad = (all_currents[-1] - all_currents[0]) * 0.03
            ax.set_xlim(all_currents[0] - pad, all_currents[-1] + pad)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_voltage_traces(
    cell_spec,
    currents_nA: List[float],
    step_dur_ms: float = 500.0,
    delay_ms: float = 200.0,
    tail_ms: float = 50.0,
    dt: float = 0.1,
    celsius: float = 35.0,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Record and plot soma voltage traces for a list of current amplitudes.

    Runs one simulation per amplitude via single_cell_utils, stacking traces
    vertically. The shaded region marks the current injection window.

    Parameters
    ----------
    cell_spec   : str, class, or instance accepted by single_cell_utils
    currents_nA : list of injected current amplitudes (nA)

    Returns
    -------
    matplotlib Figure
    """
    traces = run_current_clamp_series(
        cell_spec, amps_nA=currents_nA,
        duration_ms=step_dur_ms, delay_ms=delay_ms, tail_ms=tail_ms,
        dt=dt, celsius=celsius,
    )
    cell_name = _resolve_cell_class(cell_spec).__name__

    n = len(currents_nA)
    if figsize is None:
        figsize = (12, 2.5 * n)

    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, result in zip(axes, traces):
        amp = result["amp_nA"]
        ax.plot(result["t"], result["v_soma"], "k", lw=0.8)
        ax.axvspan(delay_ms, delay_ms + step_dur_ms,
                   alpha=0.08, color="steelblue", label=f"I = {amp:.3f} nA")
        ax.set_ylabel("V (mV)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(f"Voltage Traces — {cell_name}", fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def build_slope_table(
    results_by_type: Dict[str, Dict[str, dict]],
) -> "pd.DataFrame":
    """Build a summary DataFrame of f-I slopes, rheobase, and R² values.

    Parameters
    ----------
    results_by_type : {cell_type: {cell_name: data_dict}}
                      as returned by run_cell_type_fi

    Returns
    -------
    pandas DataFrame with one row per cell model
    """
    import pandas as pd

    rows = []
    for cell_type, type_results in results_by_type.items():
        for name, data in type_results.items():
            rows.append(dict(
                Cell=name,
                Type=cell_type,
                Slope_Hz_per_nA=round(data["slope"], 2) if not np.isnan(data["slope"]) else np.nan,
                Intercept_Hz=round(data["intercept"], 2) if not np.isnan(data["intercept"]) else np.nan,
                R2=round(data["r2"], 4) if not np.isnan(data["r2"]) else np.nan,
                Max_Freq_Hz=round(float(data["freqs_hz"].max()), 1),
                Rheobase_nA=_estimate_rheobase(data["currents_nA"], data["freqs_hz"]),
            ))
    return pd.DataFrame(rows).set_index("Cell")


# ---------------------------------------------------------------------------
# Burton & Urban (2014) comparison — spike and ISI analysis
# ---------------------------------------------------------------------------

def find_spike_times_milliseconds(
    trace_dict: dict,
    spike_threshold_millivolts: float = -20.0,
    step_onset_milliseconds: float = 0.0,
) -> np.ndarray:
    """Detect upward threshold crossings in a voltage trace and return spike times.

    Parameters
    ----------
    trace_dict : dict
        Output of any single_cell_utils run function. Must contain keys
        ``t`` (time array, ms) and ``v_soma`` (membrane potential, mV).
    spike_threshold_millivolts : float
        Voltage level used for spike detection (mV). Default −20.0.
    step_onset_milliseconds : float
        Only return spikes that occur at or after this time (ms). Used to
        exclude spontaneous activity before the current step. Default 0.0.

    Returns
    -------
    spike_times : np.ndarray
        1-D array of spike times (ms) in ascending order.
    """
    time_array_milliseconds = trace_dict["t"]
    soma_voltage_millivolts = trace_dict["v_soma"]

    above_threshold_mask = soma_voltage_millivolts > spike_threshold_millivolts
    upward_crossing_mask = np.diff(above_threshold_mask.astype(int)) == 1

    crossing_midpoint_times_milliseconds = (
        time_array_milliseconds[:-1] + time_array_milliseconds[1:]
    ) / 2.0

    all_spike_times_milliseconds = crossing_midpoint_times_milliseconds[upward_crossing_mask]
    return all_spike_times_milliseconds[all_spike_times_milliseconds >= step_onset_milliseconds]


def compute_interspike_interval_statistics(
    trace_dict: dict,
    spike_threshold_millivolts: float = -20.0,
    step_onset_milliseconds: float = 0.0,
) -> dict:
    """Compute ISI-based firing statistics from a single voltage trace.

    Requires at least 2 spikes; returns NaN-filled dict otherwise.

    Parameters
    ----------
    trace_dict : dict
        Output of any single_cell_utils run function (keys ``t``, ``v_soma``).
    spike_threshold_millivolts : float
        Voltage threshold for spike detection (mV). Default −20.0.
    step_onset_milliseconds : float
        Ignore spikes before this time (ms). Default 0.0.

    Returns
    -------
    dict with keys:
        mean_interspike_interval_milliseconds       : float
        std_interspike_interval_milliseconds        : float
        coefficient_of_variation_interspike_interval: float  (std / mean)
        peak_instantaneous_firing_rate_hertz        : float  (1 / min ISI, converted to Hz)
    All values are NaN when fewer than 2 spikes are present.
    """
    nan_result = dict(
        mean_interspike_interval_milliseconds=np.nan,
        std_interspike_interval_milliseconds=np.nan,
        coefficient_of_variation_interspike_interval=np.nan,
        peak_instantaneous_firing_rate_hertz=np.nan,
    )

    spike_times_milliseconds = find_spike_times_milliseconds(
        trace_dict, spike_threshold_millivolts, step_onset_milliseconds
    )

    if len(spike_times_milliseconds) < 2:
        return nan_result

    interspike_intervals_milliseconds = np.diff(spike_times_milliseconds)
    mean_interval_milliseconds = float(np.mean(interspike_intervals_milliseconds))
    std_interval_milliseconds = float(np.std(interspike_intervals_milliseconds))

    if mean_interval_milliseconds <= 0.0:
        return nan_result

    coefficient_of_variation = std_interval_milliseconds / mean_interval_milliseconds
    minimum_interval_milliseconds = float(np.min(interspike_intervals_milliseconds))
    peak_rate_hertz = 1000.0 / minimum_interval_milliseconds

    return dict(
        mean_interspike_interval_milliseconds=mean_interval_milliseconds,
        std_interspike_interval_milliseconds=std_interval_milliseconds,
        coefficient_of_variation_interspike_interval=coefficient_of_variation,
        peak_instantaneous_firing_rate_hertz=peak_rate_hertz,
    )


def compute_rheobase_nanoamps(
    traces_list: List[dict],
    spike_threshold_millivolts: float = -20.0,
) -> float:
    """Find the minimum current amplitude that evokes at least one spike.

    Iterates through traces_list in ascending current order (assumed) and
    returns the amplitude of the first trace that contains a spike.

    Parameters
    ----------
    traces_list : list of dict
        Output of single_cell_utils.run_current_clamp_series. Each dict must
        have keys ``amp_nA`` and ``v_soma``.
    spike_threshold_millivolts : float
        Voltage threshold for spike detection (mV). Default −20.0.

    Returns
    -------
    rheobase_current_nanoamps : float
        Current amplitude (nA) of the weakest suprathreshold step,
        or NaN if no step produced spikes.
    """
    sorted_traces = sorted(traces_list, key=lambda trace_result: trace_result["amp_nA"])
    for trace_result in sorted_traces:
        if count_spikes(trace_result, spike_threshold_millivolts) >= 1:
            return float(trace_result["amp_nA"])
    return np.nan


def compute_rheobase_spike_latency_milliseconds(
    traces_list: List[dict],
    spike_threshold_millivolts: float = -20.0,
    step_delay_milliseconds: float = 100.0,
) -> float:
    """Time from current step onset to the first spike at the rheobase amplitude.

    Parameters
    ----------
    traces_list : list of dict
        Output of single_cell_utils.run_current_clamp_series.
    spike_threshold_millivolts : float
        Voltage threshold for spike detection (mV). Default −20.0.
    step_delay_milliseconds : float
        Time of current step onset in the simulation (ms). Must match the
        ``delay_ms`` used in run_current_clamp_series. Default 100.0.

    Returns
    -------
    spike_latency_milliseconds : float
        Time (ms) from step onset to the first spike at rheobase, or NaN
        if rheobase could not be determined.
    """
    rheobase_current_nanoamps = compute_rheobase_nanoamps(
        traces_list, spike_threshold_millivolts
    )
    if np.isnan(rheobase_current_nanoamps):
        return np.nan

    for trace_result in traces_list:
        if np.isclose(float(trace_result["amp_nA"]), rheobase_current_nanoamps):
            spike_times_milliseconds = find_spike_times_milliseconds(
                trace_result,
                spike_threshold_millivolts,
                step_onset_milliseconds=step_delay_milliseconds,
            )
            if len(spike_times_milliseconds) > 0:
                return float(spike_times_milliseconds[0]) - step_delay_milliseconds
            break

    return np.nan


# ---------------------------------------------------------------------------
# Action potential shape analysis
# ---------------------------------------------------------------------------

def compute_action_potential_properties(
    trace_dict: dict,
    voltage_derivative_threshold_millivolts_per_millisecond: float = 20.0,
    step_onset_milliseconds: float = 0.0,
) -> dict:
    """Characterize the shape of the first action potential in a voltage trace.

    Implements the measurement criteria from Burton & Urban (2014):
    - AP onset: first time dV/dt exceeds the derivative threshold (paper: 20 mV/ms)
    - Amplitude: peak voltage − onset voltage
    - FWHM: time between half-amplitude crossing on rising and falling phases
    - Rise slope: maximum dV/dt during upstroke
    - Fall slope: minimum dV/dt during downstroke
    - AHP amplitude: minimum voltage within 10 ms after AP onset − onset voltage
    - AHP half-decay time: time from falling-phase return to onset voltage to
      50% recovery of AHP amplitude

    Parameters
    ----------
    trace_dict : dict
        Output of any single_cell_utils run function (keys ``t``, ``v_soma``).
    voltage_derivative_threshold_millivolts_per_millisecond : float
        dV/dt threshold that defines AP onset (mV/ms). Paper value is 20.0.
    step_onset_milliseconds : float
        Only analyze APs that begin at or after this time (ms). Default 0.0.

    Returns
    -------
    dict with keys (all float, all NaN when no AP is found):
        ap_onset_millivolts
        ap_amplitude_millivolts
        ap_full_width_half_maximum_milliseconds
        ap_rise_slope_millivolts_per_millisecond
        ap_fall_slope_millivolts_per_millisecond
        ahp_amplitude_millivolts
        ahp_half_decay_time_milliseconds
    """
    nan_result = dict(
        ap_onset_millivolts=np.nan,
        ap_amplitude_millivolts=np.nan,
        ap_full_width_half_maximum_milliseconds=np.nan,
        ap_rise_slope_millivolts_per_millisecond=np.nan,
        ap_fall_slope_millivolts_per_millisecond=np.nan,
        ahp_amplitude_millivolts=np.nan,
        ahp_half_decay_time_milliseconds=np.nan,
    )

    time_array_milliseconds = trace_dict["t"]
    soma_voltage_millivolts = trace_dict["v_soma"]

    post_onset_mask = time_array_milliseconds >= step_onset_milliseconds
    if not post_onset_mask.any():
        return nan_result

    time_post_onset = time_array_milliseconds[post_onset_mask]
    voltage_post_onset = soma_voltage_millivolts[post_onset_mask]

    if len(time_post_onset) < 3:
        return nan_result

    # Assume uniform timestep (true for fixed-step NEURON runs)
    timestep_milliseconds = float(time_post_onset[1] - time_post_onset[0])
    if timestep_milliseconds <= 0.0:
        return nan_result

    voltage_derivative_millivolts_per_millisecond = (
        np.diff(voltage_post_onset) / timestep_milliseconds
    )

    # AP onset: first index where dV/dt exceeds the threshold
    above_derivative_threshold = (
        voltage_derivative_millivolts_per_millisecond
        > voltage_derivative_threshold_millivolts_per_millisecond
    )
    if not above_derivative_threshold.any():
        return nan_result

    ap_onset_index = int(np.argmax(above_derivative_threshold))
    ap_onset_voltage_millivolts = float(voltage_post_onset[ap_onset_index])
    ap_onset_time_milliseconds = float(time_post_onset[ap_onset_index])

    # AP peak: maximum voltage within 5 ms of onset
    peak_search_end_index = min(
        ap_onset_index + max(1, round(5.0 / timestep_milliseconds)),
        len(voltage_post_onset),
    )
    local_peak_offset = int(np.argmax(voltage_post_onset[ap_onset_index:peak_search_end_index]))
    ap_peak_index = ap_onset_index + local_peak_offset
    ap_peak_voltage_millivolts = float(voltage_post_onset[ap_peak_index])
    ap_amplitude_millivolts = ap_peak_voltage_millivolts - ap_onset_voltage_millivolts

    if ap_amplitude_millivolts <= 0.0:
        return nan_result

    # FWHM: half-amplitude voltage level
    half_amplitude_voltage_millivolts = (
        ap_onset_voltage_millivolts + ap_amplitude_millivolts / 2.0
    )

    # Rising phase: first index at or above half-amplitude between onset and peak
    rising_phase_voltages = voltage_post_onset[ap_onset_index : ap_peak_index + 1]
    rising_half_amplitude_crossings = np.where(
        rising_phase_voltages >= half_amplitude_voltage_millivolts
    )[0]
    if len(rising_half_amplitude_crossings) == 0:
        return nan_result
    half_amplitude_rise_index = ap_onset_index + int(rising_half_amplitude_crossings[0])
    half_amplitude_rise_time_milliseconds = float(time_post_onset[half_amplitude_rise_index])

    # Falling phase: first index below half-amplitude after the peak
    falling_phase_end_index = min(
        ap_peak_index + max(1, round(10.0 / timestep_milliseconds)),
        len(voltage_post_onset),
    )
    falling_phase_voltages = voltage_post_onset[ap_peak_index:falling_phase_end_index]
    falling_half_amplitude_crossings = np.where(
        falling_phase_voltages < half_amplitude_voltage_millivolts
    )[0]
    if len(falling_half_amplitude_crossings) == 0:
        return nan_result
    half_amplitude_fall_index = ap_peak_index + int(falling_half_amplitude_crossings[0])
    half_amplitude_fall_time_milliseconds = float(time_post_onset[half_amplitude_fall_index])

    ap_full_width_half_maximum_milliseconds = (
        half_amplitude_fall_time_milliseconds - half_amplitude_rise_time_milliseconds
    )

    # Rise slope: maximum dV/dt between onset and peak
    ap_rise_slope_millivolts_per_millisecond = float(
        np.max(voltage_derivative_millivolts_per_millisecond[ap_onset_index:ap_peak_index])
    ) if ap_peak_index > ap_onset_index else np.nan

    # Fall slope: minimum dV/dt in window from peak to 5 ms after
    fall_slope_end_index = min(
        ap_peak_index + max(1, round(5.0 / timestep_milliseconds)),
        len(voltage_derivative_millivolts_per_millisecond),
    )
    ap_fall_slope_millivolts_per_millisecond = float(
        np.min(voltage_derivative_millivolts_per_millisecond[ap_peak_index:fall_slope_end_index])
    ) if fall_slope_end_index > ap_peak_index else np.nan

    # AHP: minimum voltage within 10 ms after AP onset
    ahp_window_end_index = min(
        ap_onset_index + max(1, round(10.0 / timestep_milliseconds)),
        len(voltage_post_onset),
    )
    ahp_window_voltages = voltage_post_onset[ap_onset_index:ahp_window_end_index]
    ahp_minimum_local_offset = int(np.argmin(ahp_window_voltages))
    ahp_minimum_voltage_millivolts = float(ahp_window_voltages[ahp_minimum_local_offset])
    ahp_amplitude_millivolts = ahp_minimum_voltage_millivolts - ap_onset_voltage_millivolts
    ahp_minimum_index = ap_onset_index + ahp_minimum_local_offset

    # T_AHP50%: time from falling-phase crossing of onset voltage to 50% recovery
    falling_phase_long_voltages = voltage_post_onset[ap_peak_index:]
    falling_phase_long_times = time_post_onset[ap_peak_index:]
    falling_crosses_onset_voltage = np.where(
        falling_phase_long_voltages <= ap_onset_voltage_millivolts
    )[0]

    if len(falling_crosses_onset_voltage) == 0:
        ahp_half_decay_time_milliseconds = np.nan
    else:
        ahp_start_time_milliseconds = float(
            falling_phase_long_times[int(falling_crosses_onset_voltage[0])]
        )
        # 50% recovery voltage: halfway between AHP minimum and onset voltage
        ahp_fifty_percent_recovery_voltage_millivolts = (
            ap_onset_voltage_millivolts + 0.5 * ahp_amplitude_millivolts
        )
        recovery_phase_voltages = voltage_post_onset[ahp_minimum_index:]
        recovery_phase_times = time_post_onset[ahp_minimum_index:]
        recovery_crosses_fifty_percent = np.where(
            recovery_phase_voltages >= ahp_fifty_percent_recovery_voltage_millivolts
        )[0]

        if len(recovery_crosses_fifty_percent) == 0:
            ahp_half_decay_time_milliseconds = np.nan
        else:
            fifty_percent_recovery_time_milliseconds = float(
                recovery_phase_times[int(recovery_crosses_fifty_percent[0])]
            )
            ahp_half_decay_time_milliseconds = (
                fifty_percent_recovery_time_milliseconds - ahp_start_time_milliseconds
            )

    return dict(
        ap_onset_millivolts=ap_onset_voltage_millivolts,
        ap_amplitude_millivolts=ap_amplitude_millivolts,
        ap_full_width_half_maximum_milliseconds=ap_full_width_half_maximum_milliseconds,
        ap_rise_slope_millivolts_per_millisecond=ap_rise_slope_millivolts_per_millisecond,
        ap_fall_slope_millivolts_per_millisecond=ap_fall_slope_millivolts_per_millisecond,
        ahp_amplitude_millivolts=ahp_amplitude_millivolts,
        ahp_half_decay_time_milliseconds=ahp_half_decay_time_milliseconds,
    )


# ---------------------------------------------------------------------------
# Input resistance
# ---------------------------------------------------------------------------

def compute_input_resistance_megaohms(
    hyperpolarizing_traces: List[dict],
    step_duration_milliseconds: float = 2000.0,
    delay_milliseconds: float = 200.0,
    steady_state_window_fraction: float = 0.2,
) -> float:
    """Estimate somatic input resistance from hyperpolarizing step responses.

    For each trace, averages the membrane potential over the last
    ``steady_state_window_fraction`` of the current-step window to obtain a
    steady-state voltage; then regresses delta-voltage against delta-current
    across all steps. The slope gives R_input (mV / nA = MΩ).

    Parameters
    ----------
    hyperpolarizing_traces : list of dict
        Output of single_cell_utils.run_hyperpolarizing_steps or
        run_current_clamp_series with negative amplitudes. Each dict must
        have keys ``t``, ``v_soma``, and ``amp_nA``.
    step_duration_milliseconds : float
        Duration of the current-injection epoch in each trace (ms). Used to
        locate the steady-state window within the trace. Default 2000.0.
    delay_milliseconds : float
        Time of current step onset (ms). Default 200.0.
    steady_state_window_fraction : float
        Fraction of the step duration to average for steady-state voltage.
        0.2 means the last 20% of the step window. Default 0.2.

    Returns
    -------
    input_resistance_megaohms : float
        Linear regression slope of ΔV (mV) vs ΔI (nA) across all steps.
        Units: mV / nA = MΩ. Returns NaN when fewer than 2 traces are provided.
    """
    if len(hyperpolarizing_traces) < 2:
        return np.nan

    step_end_milliseconds = delay_milliseconds + step_duration_milliseconds
    steady_state_window_start_milliseconds = (
        step_end_milliseconds
        - step_duration_milliseconds * steady_state_window_fraction
    )

    delta_currents_nanoamps = []
    delta_voltages_millivolts = []

    # Use the zero-current (or smallest absolute) trace as the voltage baseline
    baseline_trace = min(
        hyperpolarizing_traces,
        key=lambda trace_result: abs(trace_result["amp_nA"]),
    )
    time_baseline = baseline_trace["t"]
    baseline_window_mask = (
        (time_baseline >= steady_state_window_start_milliseconds)
        & (time_baseline <= step_end_milliseconds)
    )
    baseline_steady_state_voltage_millivolts = float(
        np.mean(baseline_trace["v_soma"][baseline_window_mask])
    ) if baseline_window_mask.any() else float(baseline_trace["v_soma"][-1])

    for trace_result in hyperpolarizing_traces:
        current_amplitude_nanoamps = float(trace_result["amp_nA"])
        time_array = trace_result["t"]
        steady_state_window_mask = (
            (time_array >= steady_state_window_start_milliseconds)
            & (time_array <= step_end_milliseconds)
        )
        if not steady_state_window_mask.any():
            continue

        steady_state_voltage_millivolts = float(
            np.mean(trace_result["v_soma"][steady_state_window_mask])
        )
        delta_voltage_millivolts = (
            steady_state_voltage_millivolts - baseline_steady_state_voltage_millivolts
        )
        delta_current_nanoamps = (
            current_amplitude_nanoamps - float(baseline_trace["amp_nA"])
        )
        delta_currents_nanoamps.append(delta_current_nanoamps)
        delta_voltages_millivolts.append(delta_voltage_millivolts)

    delta_currents_array = np.array(delta_currents_nanoamps)
    delta_voltages_array = np.array(delta_voltages_millivolts)

    nonzero_current_mask = delta_currents_array != 0.0
    if nonzero_current_mask.sum() < 2:
        return np.nan

    regression_slope, *_ = linregress(
        delta_currents_array[nonzero_current_mask],
        delta_voltages_array[nonzero_current_mask],
    )
    return float(regression_slope)


# ---------------------------------------------------------------------------
# Burton & Urban (2014) comparison — plot functions
# ---------------------------------------------------------------------------

def plot_voltage_traces_from_results(
    traces_list: List[dict],
    cell_name: str = "",
    step_duration_milliseconds: float = 500.0,
    delay_milliseconds: float = 200.0,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot stacked soma voltage traces from pre-run current-clamp results.

    Equivalent to Figure 5A/B in Burton & Urban (2014): one horizontal panel
    per current amplitude, stacked vertically, with the injection window shaded.
    No simulation is run — pass results from run_current_clamp_series directly.

    Parameters
    ----------
    traces_list : list of dict
        Output of single_cell_utils.run_current_clamp_series. Each dict must
        have keys ``t``, ``v_soma``, and ``amp_nA``.
    cell_name : str
        Label used in the figure title. Default empty string.
    step_duration_milliseconds : float
        Duration of the current-injection window (ms), used for shading. Default 500.0.
    delay_milliseconds : float
        Start time of the injection window (ms). Default 200.0.
    figsize : (float, float), optional
        Figure (width, height) in inches. Defaults to (10, 2.0 × n_traces).
    save_path : str, optional
        File path to save the figure. If None, the figure is only returned.

    Returns
    -------
    matplotlib Figure
    """
    number_of_traces = len(traces_list)
    if figsize is None:
        figsize = (10, 2.0 * number_of_traces)

    figure, axes_array = plt.subplots(
        number_of_traces, 1, figsize=figsize, sharex=True
    )
    if number_of_traces == 1:
        axes_array = [axes_array]

    injection_end_milliseconds = delay_milliseconds + step_duration_milliseconds

    for axis, trace_result in zip(axes_array, traces_list):
        current_amplitude_nanoamps = trace_result["amp_nA"]
        current_amplitude_picoamps = current_amplitude_nanoamps * 1000.0

        axis.plot(
            trace_result["t"],
            trace_result["v_soma"],
            color="black",
            linewidth=0.8,
        )
        axis.axvspan(
            delay_milliseconds,
            injection_end_milliseconds,
            alpha=0.07,
            color="steelblue",
        )
        axis.set_ylabel("Voltage (mV)", fontsize=8)
        axis.annotate(
            f"{current_amplitude_picoamps:.0f} pA",
            xy=(0.02, 0.80),
            xycoords="axes fraction",
            fontsize=8,
        )
        axis.grid(True, alpha=0.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes_array[-1].set_xlabel("Time (ms)")
    figure_title = "Soma Voltage Traces"
    if cell_name:
        figure_title += f" — {cell_name}"
    figure.suptitle(figure_title, fontsize=11)
    figure.tight_layout()

    if save_path is not None:
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return figure


def plot_fi_overlay_by_type(
    results_by_type: Dict[str, Dict[str, dict]],
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot all model f-I curves for each cell type overlaid on the same axes.

    Equivalent to Figure 5C/D in Burton & Urban (2014). One subplot per cell
    type with individual thin lines per model; the first model in each type is
    drawn with a thicker line to serve as the representative trace (matching
    the paper's thick-line convention). Current axis is displayed in picoamps
    to match the paper.

    Parameters
    ----------
    results_by_type : dict
        Nested dict: {cell_type_string: {cell_name_string: data_dict}}.
        Each data_dict must contain:
            ``current_amplitudes_nanoamps`` — 1-D ndarray of step currents (nA)
            ``firing_rates_hertz``          — 1-D ndarray of firing rates (Hz)
    figsize : (float, float), optional
        Figure (width, height) in inches. Defaults to (6 × n_types, 5).
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    matplotlib Figure
    """
    number_of_types = len(results_by_type)
    if figsize is None:
        figsize = (6 * number_of_types, 5)

    figure, axes_array = plt.subplots(1, number_of_types, figsize=figsize, squeeze=False)

    type_line_colors = {"MC": "black", "TC": "dimgray"}

    for column_index, (cell_type, type_results) in enumerate(results_by_type.items()):
        axis = axes_array[0][column_index]
        cell_names_list = list(type_results.keys())
        line_color = type_line_colors.get(cell_type, "black")

        for cell_index, cell_name in enumerate(cell_names_list):
            cell_data = type_results[cell_name]
            current_amplitudes_picoamps = cell_data["current_amplitudes_nanoamps"] * 1000.0
            firing_rates_hertz = cell_data["firing_rates_hertz"]

            is_representative = cell_index == 0
            axis.plot(
                current_amplitudes_picoamps,
                firing_rates_hertz,
                color=line_color,
                linewidth=2.0 if is_representative else 0.8,
                alpha=1.0 if is_representative else 0.55,
                label=cell_name if is_representative else None,
            )

        axis.set_title(
            f"{cell_type}  (n = {len(cell_names_list)} models)", fontsize=11
        )
        axis.set_xlabel("Injected Current (pA)")
        axis.set_ylabel("Firing Rate (Hz)")
        axis.set_xlim(left=0)
        axis.set_ylim(bottom=0)
        axis.legend(fontsize=8, loc="upper left")
        axis.grid(True, alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.tight_layout()
    if save_path is not None:
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return figure


def plot_fi_mean_sem_comparison(
    results_by_type: Dict[str, Dict[str, dict]],
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot mean ± SEM f-I curves for MC and TC on the same axes.

    Equivalent to Figure 5E in Burton & Urban (2014). Firing rates for each
    cell type are interpolated to a shared current grid, then the mean and
    standard error of the mean (SEM) are plotted as a line with a shaded band.

    Parameters
    ----------
    results_by_type : dict
        Same structure as plot_fi_overlay_by_type: each cell data_dict must
        contain ``current_amplitudes_nanoamps`` and ``firing_rates_hertz``.
    figsize : (float, float), optional
        Figure (width, height) in inches. Default (6, 5).
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    matplotlib Figure
    """
    if figsize is None:
        figsize = (6, 5)

    figure, axis = plt.subplots(figsize=figsize)

    type_colors = {"MC": "black", "TC": "dimgray"}

    for cell_type, type_results in results_by_type.items():
        if not type_results:
            continue

        first_cell_data = next(iter(type_results.values()))
        shared_current_grid_picoamps = (
            first_cell_data["current_amplitudes_nanoamps"] * 1000.0
        )

        stacked_firing_rates_matrix = np.array([
            np.interp(
                shared_current_grid_picoamps,
                cell_data["current_amplitudes_nanoamps"] * 1000.0,
                cell_data["firing_rates_hertz"],
            )
            for cell_data in type_results.values()
        ])

        mean_firing_rates_hertz = np.mean(stacked_firing_rates_matrix, axis=0)
        number_of_cells = stacked_firing_rates_matrix.shape[0]
        sem_firing_rates_hertz = (
            np.std(stacked_firing_rates_matrix, axis=0) / np.sqrt(number_of_cells)
        )

        line_color = type_colors.get(cell_type, "steelblue")

        axis.plot(
            shared_current_grid_picoamps,
            mean_firing_rates_hertz,
            color=line_color,
            linewidth=2.0,
            label=f"{cell_type}  (n = {number_of_cells})",
        )
        axis.fill_between(
            shared_current_grid_picoamps,
            mean_firing_rates_hertz - sem_firing_rates_hertz,
            mean_firing_rates_hertz + sem_firing_rates_hertz,
            color=line_color,
            alpha=0.18,
        )

    axis.set_xlabel("Injected Current (pA)")
    axis.set_ylabel("Firing Rate (Hz)")
    axis.set_title("Average f-I Relationships: MC vs TC", fontsize=11)
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.legend(fontsize=9)
    axis.grid(True, alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()

    if save_path is not None:
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return figure


def plot_gain_vs_input_resistance_scatter(
    cell_metrics_list: List[dict],
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter plot of f-I gain vs input resistance for MC and TC models.

    Equivalent to Figure 5F in Burton & Urban (2014). MC models are plotted
    with '+' markers and TC models with downward triangles ('v'). The first
    model of each type is drawn as a filled symbol to indicate the representative.

    Parameters
    ----------
    cell_metrics_list : list of dict
        One dict per cell model, each containing:
            ``cell_name``                    — string label
            ``cell_type``                    — "MC" or "TC"
            ``fi_gain_hertz_per_50_picoamps`` — float (Hz / 50 pA)
            ``input_resistance_megaohms``     — float (MΩ)
            ``is_representative_cell``        — bool (True → thicker marker)
    figsize : (float, float), optional
        Figure (width, height) in inches. Default (6, 5).
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    matplotlib Figure
    """
    if figsize is None:
        figsize = (6, 5)

    figure, axis = plt.subplots(figsize=figsize)

    type_marker_styles = {"MC": "+", "TC": "v"}
    type_colors = {"MC": "black", "TC": "dimgray"}
    type_label_already_added = {"MC": False, "TC": False}

    for cell_metric_dict in cell_metrics_list:
        cell_type = cell_metric_dict["cell_type"]
        gain_hertz_per_50_picoamps = cell_metric_dict["fi_gain_hertz_per_50_picoamps"]
        resistance_megaohms = cell_metric_dict["input_resistance_megaohms"]
        is_representative = cell_metric_dict.get("is_representative_cell", False)

        if np.isnan(gain_hertz_per_50_picoamps) or np.isnan(resistance_megaohms):
            continue

        marker_style = type_marker_styles.get(cell_type, "o")
        point_color = type_colors.get(cell_type, "steelblue")
        marker_size_points_squared = (14 if is_representative else 8) ** 2
        edge_linewidth = 2.0 if is_representative else 1.0

        label_text = cell_type if not type_label_already_added.get(cell_type, False) else None
        type_label_already_added[cell_type] = True

        axis.scatter(
            resistance_megaohms,
            gain_hertz_per_50_picoamps,
            marker=marker_style,
            color=point_color,
            s=marker_size_points_squared,
            linewidths=edge_linewidth,
            label=label_text,
        )

    axis.set_xlabel("Input Resistance (MΩ)")
    axis.set_ylabel("f-I Gain (Hz / 50 pA)")
    axis.set_title("f-I Gain vs Input Resistance", fontsize=11)
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.legend(fontsize=9)
    axis.grid(True, alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()

    if save_path is not None:
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return figure
