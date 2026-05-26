"""Single-cell simulation utilities for OB mitral and tufted cell models.

Provides current-clamp, voltage-clamp, and staircase protocols for isolated
cell validation. fi_curve_utils.py consumes output from this module for f-I
analysis and plotting.

All functions return numpy arrays / dicts in memory; no file I/O.
Units: current in nA, voltage in mV, time in ms throughout.
"""

from __future__ import annotations

import inspect
from typing import List

import numpy as np
from neuron import h
from prev_ob_models.Birgiolas2020.isolated_cells import (
    MC1, MC2, MC3, MC4, MC5,
    TC1, TC2, TC3, TC4, TC5,
    GC1, GC2, GC3, GC4, GC5,
)

h.load_file("stdrun.hoc")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_cell_class(cell_spec):
    """Return the class for a cell_spec that is a string, class, or instance."""
    registry = dict(
        MC1=MC1, MC2=MC2, MC3=MC3, MC4=MC4, MC5=MC5,
        TC1=TC1, TC2=TC2, TC3=TC3, TC4=TC4, TC5=TC5,
        GC1=GC1, GC2=GC2, GC3=GC3, GC4=GC4, GC5=GC5,
    )
    if isinstance(cell_spec, str):
        if cell_spec not in registry:
            raise ValueError(f"Unknown cell {cell_spec!r}. Options: {sorted(registry)}")
        return registry[cell_spec]
    if inspect.isclass(cell_spec):
        return cell_spec
    return type(cell_spec)


def _configure_neuron(dt: float, celsius: float, use_coreneuron: bool = False) -> None:
    """Switch to fixed-step integration with the given dt and temperature.

    usetable_LCa and usetable_Ih are disabled for plain h.run() because the
    compiled lookup tables are not populated under fixed-step h.run() on
    NEURON 9 / ARM (stau=0 forces s→0, suppressing all spiking). They must
    NOT be disabled for CoreNEURON runs: CoreNEURON hashes mechanism state and
    rejects a run if usetable differs from the compiled default.
    """
    h.cvode_active(0)
    h.celsius = celsius
    h.steps_per_ms = round(1.0 / dt)
    h.dt = dt
    h.setdt()
    if not use_coreneuron:
        try:
            h.usetable_LCa = 0
        except LookupError:
            pass
        try:
            h.usetable_Ih = 0
        except LookupError:
            pass
    h.cvode.cache_efficient(1)


def _configure_coreneuron(use_gpu: bool) -> None:
    from neuron import coreneuron
    coreneuron.enable      = True
    coreneuron.gpu         = use_gpu
    coreneuron.file_mode   = False
    coreneuron.verbose     = 0
    coreneuron.cell_permute = 2 if use_gpu else 0


def _psolve_series(cell, ic, amps_nA, tstop, dt, use_gpu, t_vec, v_vec) -> List[dict]:
    """Run one CoreNEURON psolve per amplitude; share a single ParallelContext."""
    _configure_coreneuron(use_gpu)
    pc = h.ParallelContext()
    gid = 1
    pc.set_gid2node(gid, 0)
    _nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
    _nc.threshold = -20.0
    pc.cell(gid, _nc)
    pc.setup_transfer()
    h.cvode_active(0)
    h.dt = dt
    h.steps_per_ms = round(1.0 / dt)
    h.setdt()
    pc.set_maxstep(10)

    results = []
    for amp in amps_nA:
        ic.amp = float(amp)
        h.stdinit()
        pc.psolve(tstop)
        results.append({
            "t":      np.array(t_vec),
            "v_soma": np.array(v_vec),
            "amp_nA": float(amp),
        })
    return results


# ---------------------------------------------------------------------------
# Cell factory and introspection
# ---------------------------------------------------------------------------

def build_cell(cell_spec, param_values=None):
    """Instantiate a Birgiolas cell, optionally applying parameter overrides.

    Parameters
    ----------
    cell_spec    : str ("MC1"), class (MC1), or existing instance
    param_values : list of floats matching the cell's params index order,
                   as returned by get_default_param_values(). When None the
                   fitted defaults from isolated_cells.py are used.

    Returns
    -------
    Instantiated cell object with .soma accessible.
    """
    cls = _resolve_cell_class(cell_spec)
    cell = cls()
    if param_values is not None:
        cell.set_model_params(list(param_values))
    return cell


def get_default_param_values(cell_spec) -> list:
    """Return a copy of the fitted parameter values for a cell model.

    Use this to read the defaults, modify specific indices, and pass the result
    as param_values to any protocol function.
    """
    cell = build_cell(cell_spec)
    return list(cell.param_values)


def describe_params(cell_spec) -> None:
    """Print a table of parameter index, attribute name, sections, and value."""
    cell = build_cell(cell_spec)
    cls = _resolve_cell_class(cell_spec)

    header = f"{'Idx':>4}  {'Attr':<16}  {'Lists':<30}  Value"
    print(f"\n{cls.__name__} parameters")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for i, (param, value) in enumerate(zip(cell.params, cell.param_values)):
        lists_str = ", ".join(param["lists"])
        print(f"{i:>4}  {param['attr']:<16}  {lists_str:<30}  {value:.6g}")
    print()


# ---------------------------------------------------------------------------
# Current-clamp protocols
# ---------------------------------------------------------------------------

def run_current_clamp(
    cell_spec,
    amp_nA: float,
    duration_ms: float = 500.0,
    delay_ms: float = 100.0,
    tail_ms: float = 50.0,
    dt: float = 0.1,
    celsius: float = 35.0,
    param_values=None,
    use_coreneuron: bool = False,
    use_gpu: bool = False,
) -> dict:
    """Single constant-amplitude IClamp pulse on the soma.

    Parameters
    ----------
    cell_spec   : str, class, or instance
    amp_nA      : injected current amplitude (nA)
    duration_ms : duration of the current step (ms)
    delay_ms    : silent period before the step (ms)
    tail_ms     : silent period after the step (ms)
    dt          : fixed integration timestep (ms)
    celsius     : simulation temperature (°C)
    param_values: optional list of parameter overrides (see get_default_param_values)

    Returns
    -------
    dict with keys:
        t        — time array (ms)
        v_soma   — soma membrane potential (mV)
        amp_nA   — injected current amplitude (nA)
    """
    cell = build_cell(cell_spec, param_values)
    _configure_neuron(dt, celsius, use_coreneuron=locals().get('use_coreneuron', False))
    h.tstop = delay_ms + duration_ms + tail_ms

    ic = h.IClamp(cell.soma(0.5))
    ic.delay = delay_ms
    ic.dur = duration_ms
    ic.amp = float(amp_nA)

    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(cell.soma(0.5)._ref_v)

    if use_coreneuron:
        return _psolve_series(cell, ic, [amp_nA], h.tstop, dt, use_gpu, t_vec, v_vec)[0]

    h.run()

    return {
        "t": np.array(t_vec),
        "v_soma": np.array(v_vec),
        "amp_nA": float(amp_nA),
    }


def run_current_clamp_series(
    cell_spec,
    amps_nA: List[float],
    duration_ms: float = 500.0,
    delay_ms: float = 100.0,
    tail_ms: float = 50.0,
    dt: float = 0.1,
    celsius: float = 35.0,
    param_values=None,
    use_coreneuron: bool = False,
    use_gpu: bool = False,
) -> List[dict]:
    """One IClamp run per amplitude; cell is built once and reused.

    h.run() calls h.finitialize() each iteration, so each trace is independent.

    Parameters
    ----------
    cell_spec   : str, class, or instance
    amps_nA     : list of current amplitudes (nA)
    (other args same as run_current_clamp)

    Returns
    -------
    list of dicts, one per amplitude, each with keys t, v_soma, amp_nA
    """
    cell = build_cell(cell_spec, param_values)
    _configure_neuron(dt, celsius, use_coreneuron=locals().get('use_coreneuron', False))
    h.tstop = delay_ms + duration_ms + tail_ms

    ic = h.IClamp(cell.soma(0.5))
    ic.delay = delay_ms
    ic.dur = duration_ms

    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(cell.soma(0.5)._ref_v)

    if use_coreneuron:
        return _psolve_series(cell, ic, amps_nA, h.tstop, dt, use_gpu, t_vec, v_vec)

    results = []
    for amp in amps_nA:
        ic.amp = float(amp)
        h.run()
        results.append({
            "t": np.array(t_vec),
            "v_soma": np.array(v_vec),
            "amp_nA": float(amp),
        })

    return results


def run_current_clamp_batch(
    cell_specs,
    amps_nA: List[float],
    duration_ms: float = 500.0,
    delay_ms: float = 100.0,
    tail_ms: float = 50.0,
    dt: float = 0.1,
    celsius: float = 35.0,
    threshold_mV: float = -20.0,
    param_values=None,
    use_coreneuron: bool = False,
    use_gpu: bool = False,
    gid_start: int = 1,
) -> List[dict]:
    """Run independent current-clamp conditions in one batched simulation.

    This instantiates one isolated cell clone per (cell_spec, amp_nA) pair,
    applies a constant IClamp to each clone, and counts soma spikes with a
    NetCon threshold detector. It is equivalent to independent f-I step runs
    for spike-count/frequency analysis, but avoids launching one simulation per
    amplitude and gives CoreNEURON/GPU a useful batch of cells.

    Parameters
    ----------
    cell_specs   : cell identifier or list of identifiers accepted by build_cell
    amps_nA      : current amplitudes to test for every cell (nA)
    duration_ms  : duration of each current step (ms)
    delay_ms     : silent period before each step (ms)
    tail_ms      : silent period after each step (ms)
    dt           : fixed integration timestep (ms)
    celsius      : simulation temperature (C)
    threshold_mV : spike detection threshold (mV)
    param_values : optional parameter overrides applied to every clone

    Returns
    -------
    list of dicts with keys:
        cell_name, cell_index, amp_nA, spike_count, freq_hz, spike_times_ms
    """
    if isinstance(cell_specs, (str, bytes)) or inspect.isclass(cell_specs):
        cell_specs = [cell_specs]
    else:
        cell_specs = list(cell_specs)

    amps_nA = np.asarray(amps_nA, dtype=float)
    h.tstop = delay_ms + duration_ms + tail_ms

    conditions = []
    for cell_index, cell_spec in enumerate(cell_specs):
        cell_name = _resolve_cell_class(cell_spec).__name__
        for amp_index, amp_nA in enumerate(amps_nA):
            conditions.append(dict(
                condition_index=len(conditions),
                cell_index=cell_index,
                cell_spec=cell_spec,
                cell_name=cell_name,
                amp_index=amp_index,
                amp_nA=float(amp_nA),
            ))

    cells = []
    clamps = []
    netcons = []
    spike_vectors = {}
    pc = None
    rank = 0
    nhost = 1

    if use_coreneuron:
        _configure_coreneuron(use_gpu)
        pc = h.ParallelContext()
        rank = int(pc.id())
        nhost = int(pc.nhost())
        for condition in conditions:
            gid = gid_start + condition["condition_index"]
            pc.set_gid2node(gid, condition["condition_index"] % nhost)

    for condition in conditions:
        owner_rank = condition["condition_index"] % nhost
        if use_coreneuron and owner_rank != rank:
            continue

        cell = build_cell(condition["cell_spec"], param_values)
        clamp = h.IClamp(cell.soma(0.5))
        clamp.delay = delay_ms
        clamp.dur = duration_ms
        clamp.amp = condition["amp_nA"]

        spike_vector = h.Vector()
        netcon = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
        netcon.threshold = threshold_mV
        netcon.record(spike_vector)

        if use_coreneuron:
            gid = gid_start + condition["condition_index"]
            pc.cell(gid, netcon)

        cells.append(cell)
        clamps.append(clamp)
        netcons.append(netcon)
        spike_vectors[condition["condition_index"]] = spike_vector

    _configure_neuron(dt, celsius, use_coreneuron=use_coreneuron)

    if use_coreneuron:
        pc.setup_transfer()
        h.cvode_active(0)
        h.dt = dt
        h.steps_per_ms = round(1.0 / dt)
        h.setdt()
        pc.set_maxstep(10)
        h.stdinit()
        pc.psolve(h.tstop)
    else:
        h.run()

    step_start_ms = delay_ms
    step_end_ms = delay_ms + duration_ms
    local_results = []
    local_condition_indexes = set(spike_vectors)

    for condition in conditions:
        if condition["condition_index"] not in local_condition_indexes:
            continue

        spike_times_ms = np.asarray(spike_vectors[condition["condition_index"]])
        spike_times_ms = spike_times_ms[
            (spike_times_ms >= step_start_ms) & (spike_times_ms < step_end_ms)
        ]
        spike_count = int(len(spike_times_ms))
        local_results.append(dict(
            condition_index=condition["condition_index"],
            cell_name=condition["cell_name"],
            cell_index=condition["cell_index"],
            amp_index=condition["amp_index"],
            amp_nA=condition["amp_nA"],
            spike_count=spike_count,
            freq_hz=spike_count / (duration_ms * 1e-3),
            spike_times_ms=spike_times_ms,
        ))

    if use_coreneuron and nhost > 1:
        gathered_results = pc.py_allgather(local_results)
        results = [
            result
            for rank_results in gathered_results
            for result in rank_results
        ]
    else:
        results = local_results

    return sorted(results, key=lambda result: result["condition_index"])


def run_fi_ramp(
    cell_spec,
    i_start_nA: float = 0.0,
    i_stop_nA: float = 0.5,
    i_step_nA: float = 0.05,
    step_dur_ms: float = 500.0,
    delay_ms: float = 200.0,
    tail_ms: float = 50.0,
    dt: float = 0.1,
    celsius: float = 35.0,
    param_values=None,
    use_coreneuron: bool = False,
    use_gpu: bool = False,
) -> dict:
    """Staircase current protocol in a single run.

    A Vector drives the IClamp amplitude, stepping up by i_step_nA every
    step_dur_ms ms. Returns the full voltage trace plus staircase metadata
    so fi_curve_utils.ramp_to_fi can bin spikes per epoch.

    Parameters
    ----------
    cell_spec   : str, class, or instance
    i_start_nA  : first current step (nA)
    i_stop_nA   : last current step, inclusive (nA)
    i_step_nA   : increment between steps (nA)
    step_dur_ms : duration of each constant-current epoch (ms)
    delay_ms    : silent lead-in before the first step (ms)
    tail_ms     : silent tail after the last step (ms)
    dt          : fixed integration timestep (ms)
    celsius     : simulation temperature (°C)
    param_values: optional list of parameter overrides

    Returns
    -------
    dict with keys:
        t           — time array (ms)
        v_soma      — soma membrane potential (mV)
        currents_nA — 1-D array of per-epoch current amplitudes (nA)
        step_dur_ms — epoch duration (ms)
        delay_ms    — silent lead-in duration (ms)
    """
    currents_nA = np.arange(i_start_nA, i_stop_nA + i_step_nA * 0.5, i_step_nA)
    cell = build_cell(cell_spec, param_values)
    _configure_neuron(dt, celsius, use_coreneuron=locals().get('use_coreneuron', False))
    h.tstop = delay_ms + len(currents_nA) * step_dur_ms + tail_ms

    ic = h.IClamp(cell.soma(0.5))
    ic.delay = 0.0
    ic.dur = h.tstop

    n_delay    = round(delay_ms    / dt)
    n_per_step = round(step_dur_ms / dt)
    n_tail     = round(tail_ms     / dt) + 2

    staircase = np.concatenate([
        np.zeros(n_delay),
        *[np.full(n_per_step, float(amp)) for amp in currents_nA],
        np.zeros(n_tail),
    ])

    amp_vec = h.Vector(staircase)
    amp_vec.play(ic._ref_amp, dt)

    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(cell.soma(0.5)._ref_v)

    if use_coreneuron:
        # Single run — use a throw-away amp list of length 1; ic.amp is driven by
        # the Vector.play so _psolve_series' ic.amp assignment is overwritten, but
        # we still need the psolve infrastructure.
        _configure_coreneuron(use_gpu)
        pc = h.ParallelContext()
        gid = 1
        pc.set_gid2node(gid, 0)
        _nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
        _nc.threshold = -20.0
        pc.cell(gid, _nc)
        pc.setup_transfer()
        h.cvode_active(0)
        h.dt = dt
        h.steps_per_ms = round(1.0 / dt)
        h.setdt()
        pc.set_maxstep(10)
        h.stdinit()
        pc.psolve(h.tstop)
    else:
        h.run()

    return {
        "t": np.array(t_vec),
        "v_soma": np.array(v_vec),
        "currents_nA": currents_nA,
        "step_dur_ms": step_dur_ms,
        "delay_ms": delay_ms,
    }


# ---------------------------------------------------------------------------
# Voltage-clamp protocols
# ---------------------------------------------------------------------------

def run_voltage_clamp(
    cell_spec,
    v_hold_mV: float,
    v_step_mV: float,
    step_dur_ms: float = 200.0,
    delay_ms: float = 50.0,
    tail_ms: float = 100.0,
    dt: float = 0.025,
    celsius: float = 35.0,
    rs_mohm: float = 0.001,
    param_values=None,
) -> dict:
    """Two-pulse SEClamp: hold → step → hold.

    Epoch 1 (delay_ms):    hold at v_hold_mV
    Epoch 2 (step_dur_ms): step to v_step_mV
    Epoch 3 (tail_ms):     return to v_hold_mV

    Parameters
    ----------
    cell_spec   : str, class, or instance
    v_hold_mV   : holding potential (mV)
    v_step_mV   : command step potential (mV)
    step_dur_ms : duration of the voltage step (ms)
    delay_ms    : holding epoch before the step (ms)
    tail_ms     : tail epoch after the step (ms)
    dt          : fixed timestep — 0.025 ms default to resolve channel kinetics
    celsius     : simulation temperature (°C)
    rs_mohm     : series resistance (MΩ). 0.001 ≈ near-perfect clamp.
    param_values: optional list of parameter overrides

    Returns
    -------
    dict with keys:
        t          — time array (ms)
        i_clamp_nA — SEClamp current (nA)
        v_soma     — soma membrane potential (mV)
        v_step_mV  — commanded step voltage (mV)
        v_hold_mV  — holding voltage (mV)
    """
    cell = build_cell(cell_spec, param_values)
    _configure_neuron(dt, celsius, use_coreneuron=locals().get('use_coreneuron', False))
    h.tstop = delay_ms + step_dur_ms + tail_ms

    sevc = h.SEClamp(cell.soma(0.5))
    sevc.rs   = rs_mohm
    sevc.dur1 = delay_ms
    sevc.amp1 = float(v_hold_mV)
    sevc.dur2 = step_dur_ms
    sevc.amp2 = float(v_step_mV)
    sevc.dur3 = tail_ms
    sevc.amp3 = float(v_hold_mV)

    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(cell.soma(0.5)._ref_v)
    i_vec = h.Vector().record(sevc._ref_i)

    h.run()

    return {
        "t": np.array(t_vec),
        "i_clamp_nA": np.array(i_vec),
        "v_soma": np.array(v_vec),
        "v_step_mV": float(v_step_mV),
        "v_hold_mV": float(v_hold_mV),
    }


# ---------------------------------------------------------------------------
# Bias current and hyperpolarizing protocols
# ---------------------------------------------------------------------------

def find_bias_current(
    cell_spec,
    target_membrane_potential_millivolts: float = -58.0,
    search_lower_bound_nanoamps: float = -0.5,
    search_upper_bound_nanoamps: float = 0.5,
    settle_duration_milliseconds: float = 1000.0,
    tolerance_millivolts: float = 0.1,
    max_iterations: int = 30,
    timestep_milliseconds: float = 0.1,
    temperature_celsius: float = 35.0,
    param_values=None,
) -> float:
    """Find the DC holding current that clamps the soma at a target membrane potential.

    Uses binary search over a constant IClamp amplitude, running a short
    settle simulation each iteration and reading the final membrane potential.
    Matches the paper protocol step: "current was injected to normalize V_m to −58 mV"
    (Burton & Urban 2014).

    Parameters
    ----------
    cell_spec : str, class, or instance
        Cell identifier accepted by build_cell (e.g. "MC1", MC1 class, or instance).
    target_membrane_potential_millivolts : float
        Desired steady-state soma membrane potential in mV. Paper value is −58.0.
    search_lower_bound_nanoamps : float
        Lower bound of the binary-search current range (nA). Default −0.5.
    search_upper_bound_nanoamps : float
        Upper bound of the binary-search current range (nA). Default +0.5.
    settle_duration_milliseconds : float
        Duration of each test simulation used to measure steady-state potential (ms).
        Default 1000.0 ms allows passive settling to equilibrium.
    tolerance_millivolts : float
        Search stops when |measured − target| < tolerance (mV). Default 0.1.
    max_iterations : int
        Maximum binary-search iterations before returning best estimate. Default 30.
    timestep_milliseconds : float
        Fixed integration timestep (ms). Default 0.1.
    temperature_celsius : float
        Simulation temperature (°C). Default 35.0.
    param_values : list of float, optional
        Parameter overrides applied via set_model_params (see get_default_param_values).

    Returns
    -------
    bias_current_nanoamps : float
        The DC current amplitude (nA) that holds the soma at
        target_membrane_potential_millivolts at steady state.
        Returns the best estimate found even if tolerance was not met.
    """
    cell = build_cell(cell_spec, param_values)
    _configure_neuron(timestep_milliseconds, temperature_celsius)
    h.tstop = settle_duration_milliseconds

    bias_clamp = h.IClamp(cell.soma(0.5))
    bias_clamp.delay = 0.0
    bias_clamp.dur = settle_duration_milliseconds

    soma_voltage_vector = h.Vector().record(cell.soma(0.5)._ref_v)

    lower_bound_nanoamps = search_lower_bound_nanoamps
    upper_bound_nanoamps = search_upper_bound_nanoamps
    best_midpoint_nanoamps = (lower_bound_nanoamps + upper_bound_nanoamps) / 2.0

    for _iteration_number in range(max_iterations):
        midpoint_current_nanoamps = (lower_bound_nanoamps + upper_bound_nanoamps) / 2.0
        best_midpoint_nanoamps = midpoint_current_nanoamps

        bias_clamp.amp = midpoint_current_nanoamps
        h.run()
        measured_membrane_potential_millivolts = float(soma_voltage_vector[-1])

        potential_error_millivolts = (
            measured_membrane_potential_millivolts - target_membrane_potential_millivolts
        )
        if abs(potential_error_millivolts) < tolerance_millivolts:
            return midpoint_current_nanoamps

        if measured_membrane_potential_millivolts < target_membrane_potential_millivolts:
            lower_bound_nanoamps = midpoint_current_nanoamps
        else:
            upper_bound_nanoamps = midpoint_current_nanoamps

    return best_midpoint_nanoamps


def run_hyperpolarizing_steps(
    cell_spec,
    current_start_nanoamps: float = 0.0,
    current_stop_nanoamps: float = -0.3,
    current_step_nanoamps: float = -0.05,
    step_duration_milliseconds: float = 2000.0,
    delay_milliseconds: float = 200.0,
    tail_duration_milliseconds: float = 200.0,
    timestep_milliseconds: float = 0.1,
    temperature_celsius: float = 35.0,
    param_values=None,
    use_coreneuron: bool = False,
    use_gpu: bool = False,
) -> List[dict]:
    """Hyperpolarizing current step series for sag amplitude and input resistance.

    A convenience wrapper over run_current_clamp_series that generates a sequence
    of negative (hyperpolarizing) current steps. Defaults match the Burton & Urban
    (2014) sag/input-resistance protocol: 0 to −300 pA in −50 pA steps, 2 s each.

    Parameters
    ----------
    cell_spec : str, class, or instance
        Cell identifier accepted by build_cell.
    current_start_nanoamps : float
        First step amplitude, typically 0 (no current) or a small negative value.
        Default 0.0 nA.
    current_stop_nanoamps : float
        Most hyperpolarizing step amplitude (nA). Must be < current_start_nanoamps.
        Default −0.3 nA (= −300 pA).
    current_step_nanoamps : float
        Increment between steps (nA). Must be negative. Default −0.05 nA (= −50 pA).
    step_duration_milliseconds : float
        Duration of each constant-current epoch (ms). Default 2000.0.
    delay_milliseconds : float
        Silent lead-in before the first step (ms). Default 200.0.
    tail_duration_milliseconds : float
        Silent tail period after the last step (ms). Default 200.0.
    timestep_milliseconds : float
        Fixed integration timestep (ms). Default 0.1.
    temperature_celsius : float
        Simulation temperature (°C). Default 35.0.
    param_values : list of float, optional
        Parameter overrides applied via set_model_params.

    Returns
    -------
    list of dict
        One dict per current step, identical in structure to run_current_clamp_series
        output: keys are ``t``, ``v_soma``, ``amp_nA``.
    """
    number_of_steps = round(
        abs(current_stop_nanoamps - current_start_nanoamps) / abs(current_step_nanoamps)
    ) + 1
    hyperpolarizing_amplitudes_nanoamps = np.linspace(
        current_start_nanoamps, current_stop_nanoamps, number_of_steps
    )

    return run_current_clamp_series(
        cell_spec,
        amps_nA=hyperpolarizing_amplitudes_nanoamps,
        duration_ms=step_duration_milliseconds,
        delay_ms=delay_milliseconds,
        tail_ms=tail_duration_milliseconds,
        dt=timestep_milliseconds,
        celsius=temperature_celsius,
        param_values=param_values,
        use_coreneuron=use_coreneuron,
        use_gpu=use_gpu,
    )


# ---------------------------------------------------------------------------
# Voltage-clamp protocols
# ---------------------------------------------------------------------------

def run_voltage_clamp_series(
    cell_spec,
    v_hold_mV: float,
    v_steps_mV: List[float],
    step_dur_ms: float = 200.0,
    delay_ms: float = 50.0,
    tail_ms: float = 100.0,
    dt: float = 0.025,
    celsius: float = 35.0,
    rs_mohm: float = 0.001,
    param_values=None,
) -> List[dict]:
    """One voltage-clamp run per step voltage; cell is built once and reused.

    Parameters
    ----------
    cell_spec   : str, class, or instance
    v_hold_mV   : holding potential (mV)
    v_steps_mV  : list of command step potentials (mV)
    (other args same as run_voltage_clamp)

    Returns
    -------
    list of dicts, one per step voltage, each with keys
    t, i_clamp_nA, v_soma, v_step_mV, v_hold_mV
    """
    cell = build_cell(cell_spec, param_values)
    _configure_neuron(dt, celsius, use_coreneuron=locals().get('use_coreneuron', False))
    h.tstop = delay_ms + step_dur_ms + tail_ms

    sevc = h.SEClamp(cell.soma(0.5))
    sevc.rs   = rs_mohm
    sevc.dur1 = delay_ms
    sevc.amp1 = float(v_hold_mV)
    sevc.dur2 = step_dur_ms
    sevc.amp3 = float(v_hold_mV)
    sevc.dur3 = tail_ms

    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(cell.soma(0.5)._ref_v)
    i_vec = h.Vector().record(sevc._ref_i)

    results = []
    for v_step in v_steps_mV:
        sevc.amp2 = float(v_step)
        h.run()
        results.append({
            "t": np.array(t_vec),
            "i_clamp_nA": np.array(i_vec),
            "v_soma": np.array(v_vec),
            "v_step_mV": float(v_step),
            "v_hold_mV": float(v_hold_mV),
        })

    return results
