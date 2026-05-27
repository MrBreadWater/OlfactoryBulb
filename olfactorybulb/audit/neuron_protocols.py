"""Reusable NEURON step-protocol helpers for repository audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class StepResponse:
    amp_nA: float
    delay_ms: float
    dur_ms: float
    tstop_ms: float
    threshold_mV: float
    spike_times_ms: tuple[float, ...]
    step_spike_times_ms: tuple[float, ...]
    step_rate_hz: float
    max_v_mV: float
    min_v_mV: float
    final_v_mV: float
    has_nan: bool


def simulate_soma_step_response(
    cell,
    *,
    amp_nA: float,
    delay_ms: float = 100.0,
    dur_ms: float = 300.0,
    tstop_ms: float | None = None,
    dt_ms: float = 0.025,
    threshold_mV: float = 0.0,
    initial_v_mV: float = -68.0,
) -> StepResponse:
    """Run one fixed-step current step and return compact spike/voltage metrics."""
    h = cell.h
    h.cvode_active(0)
    h.dt = float(dt_ms)

    stim = h.IClamp(cell.soma(0.5))
    stim.delay = float(delay_ms)
    stim.dur = float(dur_ms)
    stim.amp = float(amp_nA)

    v = h.Vector().record(cell.soma(0.5)._ref_v)
    spikes = h.Vector()
    nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
    nc.threshold = float(threshold_mV)
    nc.record(spikes)

    h.finitialize(float(initial_v_mV))
    h.tstop = float(tstop_ms if tstop_ms is not None else delay_ms + dur_ms + 100.0)
    h.run()

    voltage = [float(value) for value in v]
    spike_times = tuple(float(value) for value in spikes)
    step_spike_times = tuple(time for time in spike_times if delay_ms <= time <= delay_ms + dur_ms)
    has_nan = any(value != value for value in voltage)
    if has_nan:
        max_v = float("nan")
        min_v = float("nan")
        final_v = float("nan")
        step_rate_hz = 0.0
    else:
        max_v = max(voltage)
        min_v = min(voltage)
        final_v = voltage[-1]
        step_rate_hz = float(len(step_spike_times) / (dur_ms / 1000.0)) if dur_ms > 0 else 0.0

    return StepResponse(
        amp_nA=float(amp_nA),
        delay_ms=float(delay_ms),
        dur_ms=float(dur_ms),
        tstop_ms=float(h.tstop),
        threshold_mV=float(threshold_mV),
        spike_times_ms=spike_times,
        step_spike_times_ms=step_spike_times,
        step_rate_hz=step_rate_hz,
        max_v_mV=max_v,
        min_v_mV=min_v,
        final_v_mV=final_v,
        has_nan=has_nan,
    )


def sweep_soma_step_responses(
    cell_factory: Callable[[], object],
    amps_nA: Sequence[float],
    *,
    cell_configurer: Callable[[object], None] | None = None,
    delay_ms: float = 100.0,
    dur_ms: float = 300.0,
    tstop_ms: float | None = None,
    dt_ms: float = 0.025,
    threshold_mV: float = 0.0,
    initial_v_mV: float = -68.0,
) -> list[StepResponse]:
    """Instantiate a fresh cell per current step and collect response metrics."""
    responses: list[StepResponse] = []
    for amp_nA in amps_nA:
        cell = cell_factory()
        if cell_configurer is not None:
            cell_configurer(cell)
        responses.append(
            simulate_soma_step_response(
                cell,
                amp_nA=amp_nA,
                delay_ms=delay_ms,
                dur_ms=dur_ms,
                tstop_ms=tstop_ms,
                dt_ms=dt_ms,
                threshold_mV=threshold_mV,
                initial_v_mV=initial_v_mV,
            )
        )
    return responses


def monotonic_non_decreasing(values: Sequence[float], *, tolerance: float = 1e-9) -> bool:
    return all(float(right) + tolerance >= float(left) for left, right in zip(values, values[1:]))


__all__ = [
    "StepResponse",
    "monotonic_non_decreasing",
    "simulate_soma_step_response",
    "sweep_soma_step_responses",
]
