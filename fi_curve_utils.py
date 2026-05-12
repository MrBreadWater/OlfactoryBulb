"""f-I curve characterization utilities for OB cell models.

All current values are in nA throughout this module (NEURON's native unit).
Frequencies are in Hz.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_cell_class(cell_name: str):
    """Resolve a string like 'MC1' or 'TC3' to its class object."""
    from prev_ob_models.Birgiolas2020.isolated_cells import (
        MC1, MC2, MC3, MC4, MC5,
        TC1, TC2, TC3, TC4, TC5,
        GC1, GC2, GC3, GC4, GC5,
    )
    registry = dict(
        MC1=MC1, MC2=MC2, MC3=MC3, MC4=MC4, MC5=MC5,
        TC1=TC1, TC2=TC2, TC3=TC3, TC4=TC4, TC5=TC5,
        GC1=GC1, GC2=GC2, GC3=GC3, GC4=GC4, GC5=GC5,
    )
    if cell_name not in registry:
        raise ValueError(f"Unknown cell {cell_name!r}. Options: {sorted(registry)}")
    return registry[cell_name]


def _make_currents(i_start_nA: float, i_stop_nA: float, i_step_nA: float) -> np.ndarray:
    """Inclusive array from i_start to i_stop with spacing i_step (all nA)."""
    return np.arange(i_start_nA, i_stop_nA + i_step_nA * 0.5, i_step_nA)


def _configure_neuron(dt: float, celsius: float) -> None:
    """Switch to fixed-step integration with the given dt and temperature."""
    from neuron import h
    h.cvode_active(0)
    h.celsius = celsius
    h.steps_per_ms = round(1.0 / dt)
    h.dt = dt


# ---------------------------------------------------------------------------
# Protocol 1 — multiple independent runs, one per current level
# ---------------------------------------------------------------------------

def run_fi_steps(
    cell_class,
    i_start_nA: float = 0.0,
    i_stop_nA: float = 0.5,
    i_step_nA: float = 0.05,
    step_dur_ms: float = 500.0,
    delay_ms: float = 200.0,
    tail_ms: float = 50.0,
    dt: float = 0.1,
    celsius: float = 35.0,
    threshold_mV: float = -20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Classic f-I step protocol: one independent h.run() per current level.

    For each amplitude the cell is re-initialized from h.v_init and driven
    with a constant IClamp for step_dur_ms. Spike count is read from
    h.APCount immediately after each run.

    Parameters
    ----------
    cell_class   : class from isolated_cells (e.g. MC1, TC2)
    i_start_nA   : first current step (nA)
    i_stop_nA    : last current step, inclusive (nA)
    i_step_nA    : increment between steps (nA)
    step_dur_ms  : duration of each constant-current epoch (ms)
    delay_ms     : silent period before the current onset (ms)
    tail_ms      : silent period after the current offset (ms)
    dt           : fixed integration timestep (ms)
    celsius      : simulation temperature (°C)
    threshold_mV : APCount spike-detection threshold (mV)

    Returns
    -------
    currents_nA : 1-D ndarray of injected currents (nA)
    freqs_hz    : 1-D ndarray of mean firing frequencies (Hz)
    """
    from neuron import h

    currents_nA = _make_currents(i_start_nA, i_stop_nA, i_step_nA)

    cell = cell_class()
    _configure_neuron(dt, celsius)
    h.tstop = delay_ms + step_dur_ms + tail_ms

    ic = h.IClamp(cell.soma(0.5))
    ic.delay = delay_ms
    ic.dur = step_dur_ms

    apc = h.APCount(cell.soma(0.5))
    apc.thresh = threshold_mV

    freqs = []
    for amp in currents_nA:
        ic.amp = float(amp)
        h.run()  # calls finitialize → resets state and apc.n
        freqs.append(float(apc.n) / (step_dur_ms * 1e-3))

    return currents_nA, np.array(freqs)


# ---------------------------------------------------------------------------
# Protocol 2 — single staircase simulation
# ---------------------------------------------------------------------------

def run_fi_ramp(
    cell_class,
    i_start_nA: float = 0.0,
    i_stop_nA: float = 0.5,
    i_step_nA: float = 0.05,
    step_dur_ms: float = 500.0,
    delay_ms: float = 200.0,
    tail_ms: float = 50.0,
    dt: float = 0.1,
    celsius: float = 35.0,
    threshold_mV: float = -20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """f-I staircase protocol: all current levels in a single h.run().

    A Vector drives the IClamp amplitude at every dt, stepping up by
    i_step_nA every step_dur_ms ms. All spike times are recorded into a
    Vector and then binned per epoch to compute per-step firing frequency.

    Parameters
    ----------
    (same as run_fi_steps)

    Returns
    -------
    currents_nA : 1-D ndarray of per-epoch injected currents (nA)
    freqs_hz    : 1-D ndarray of per-epoch firing frequencies (Hz)
    """
    from neuron import h

    currents_nA = _make_currents(i_start_nA, i_stop_nA, i_step_nA)
    n_steps = len(currents_nA)

    cell = cell_class()
    _configure_neuron(dt, celsius)
    h.tstop = delay_ms + n_steps * step_dur_ms + tail_ms

    # IClamp active for the whole simulation; amplitude is vector-driven
    ic = h.IClamp(cell.soma(0.5))
    ic.delay = 0.0
    ic.dur = h.tstop

    # Build staircase array sampled at dt — one value per timestep
    n_delay    = round(delay_ms    / dt)
    n_per_step = round(step_dur_ms / dt)
    n_tail     = round(tail_ms     / dt) + 2   # +2 guards against rounding at tstop

    staircase = np.concatenate([
        np.zeros(n_delay),
        *[np.full(n_per_step, float(amp)) for amp in currents_nA],
        np.zeros(n_tail),
    ])

    # amp_vec must stay alive until after h.run()
    amp_vec = h.Vector(staircase)
    amp_vec.play(ic._ref_amp, dt)

    # Record every spike time
    spike_times = h.Vector()
    apc = h.APCount(cell.soma(0.5))
    apc.thresh = threshold_mV
    apc.record(spike_times)

    h.run()

    t_spikes = np.array(spike_times)

    freqs = []
    for i in range(n_steps):
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
    **protocol_kwargs,
) -> Dict[str, dict]:
    """Run the f-I protocol for all numbered models of one cell type.

    Parameters
    ----------
    cell_type      : 'MC', 'TC', or 'GC'
    protocol       : 'steps' (independent runs) or 'ramp' (staircase)
    n_cells        : run models 1 through n_cells
    **protocol_kwargs : forwarded to run_fi_steps or run_fi_ramp

    Returns
    -------
    dict : {cell_name → {currents_nA, freqs_hz, slope, intercept, r2}}
    """
    fn = run_fi_steps if protocol == "steps" else run_fi_ramp
    results = {}

    for i in range(1, n_cells + 1):
        name = f"{cell_type}{i}"
        print(f"  {name}...", end=" ", flush=True)
        cls = _get_cell_class(name)
        currents, freqs = fn(cls, **protocol_kwargs)
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
            c = colors[idx % len(colors)]
            currents = data["currents_nA"]
            freqs    = data["freqs_hz"]
            slope    = data["slope"]
            intercept = data["intercept"]
            r2       = data["r2"]

            if all_currents is None:
                all_currents = currents

            ax.plot(currents, freqs, "o-", color=c, label=name, markersize=5, lw=1.5)

            if not np.isnan(slope):
                mask = freqs >= 1.0
                fit_x = currents[mask]
                fit_label = f"{name}: {slope:.0f} Hz/nA  (R²={r2:.2f})"
                ax.plot(fit_x, slope * fit_x + intercept, "--",
                        color=c, alpha=0.6, label=fit_label)

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
    cell_class,
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

    Runs one simulation per amplitude, stacking traces vertically. The shaded
    region marks the current injection window. All currents in nA.

    Returns
    -------
    matplotlib Figure
    """
    from neuron import h

    n = len(currents_nA)
    if figsize is None:
        figsize = (12, 2.5 * n)

    _configure_neuron(dt, celsius)
    h.tstop = delay_ms + step_dur_ms + tail_ms

    cell = cell_class()
    ic = h.IClamp(cell.soma(0.5))
    ic.delay = delay_ms
    ic.dur = step_dur_ms

    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(cell.soma(0.5)._ref_v)

    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, amp in zip(axes, currents_nA):
        ic.amp = float(amp)
        h.run()
        ax.plot(np.array(t_vec), np.array(v_vec), "k", lw=0.8)
        ax.axvspan(delay_ms, delay_ms + step_dur_ms,
                   alpha=0.08, color="steelblue", label=f"I = {amp:.3f} nA")
        ax.set_ylabel("V (mV)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(f"Voltage Traces — {cell_class.__name__}", fontsize=11)
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
