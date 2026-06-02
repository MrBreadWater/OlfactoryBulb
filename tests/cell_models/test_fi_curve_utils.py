import numpy as np

from fi_curve_utils import (
    compute_action_potential_properties,
    compute_fi_maximum_linear_slope,
    compute_isi_statistics_near_rate,
    compute_peak_instantaneous_firing_rate_hertz,
    compute_rheobase_nanoamps,
    traces_to_fi,
)


def _trace(amp_nA, spike_times_ms, duration_ms=500.0, dt_ms=1.0):
    t = np.arange(0.0, duration_ms + dt_ms, dt_ms)
    v = np.full_like(t, -65.0)
    for spike_time_ms in spike_times_ms:
        index = int(round(spike_time_ms / dt_ms))
        if 0 <= index < len(v):
            v[index] = 30.0
    return {"t": t, "v_soma": v, "amp_nA": amp_nA}


def test_traces_to_fi_counts_only_current_step_window():
    traces = [
        _trace(0.0, [50.0, 150.0, 450.0]),
        _trace(0.1, [120.0, 200.0, 480.0]),
    ]

    currents, freqs = traces_to_fi(
        traces,
        step_dur_ms=300.0,
        threshold_mV=-20.0,
        delay_ms=100.0,
    )

    assert np.allclose(currents, [0.0, 0.1])
    assert np.allclose(freqs, [1 / 0.3, 2 / 0.3])


def test_rheobase_ignores_pre_step_spikes():
    traces = [
        _trace(0.0, [50.0]),
        _trace(0.05, [80.0]),
        _trace(0.1, [150.0]),
    ]

    rheobase = compute_rheobase_nanoamps(
        traces,
        spike_threshold_millivolts=-20.0,
        step_delay_milliseconds=100.0,
        step_duration_milliseconds=300.0,
    )

    assert rheobase == 0.1


def test_maximum_fi_slope_uses_steepest_adjacent_segment():
    currents = np.array([0.0, 0.05, 0.10, 0.15])
    freqs = np.array([0.0, 5.0, 30.0, 35.0])

    slope, intercept, r2, segment_index = compute_fi_maximum_linear_slope(currents, freqs)

    assert slope == 500.0
    assert intercept == -20.0
    assert r2 == 1.0
    assert segment_index == 1


def test_peak_rate_and_cv_isi_follow_paper_selection_rules():
    traces = [
        _trace(0.05, [150.0, 300.0]),          # 4 Hz, min ISI 150 ms
        _trace(0.10, [150.0, 200.0, 250.0]),   # 6 Hz, min ISI 50 ms
        _trace(0.20, np.arange(125.0, 425.0, 50.0)),  # 12 Hz, min ISI 50 ms
    ]

    peak_rate = compute_peak_instantaneous_firing_rate_hertz(
        traces,
        spike_threshold_millivolts=-20.0,
        step_delay_milliseconds=100.0,
        step_duration_milliseconds=500.0,
    )
    isi_stats = compute_isi_statistics_near_rate(
        traces,
        target_rate_hertz=10.0,
        spike_threshold_millivolts=-20.0,
        step_delay_milliseconds=100.0,
        step_duration_milliseconds=500.0,
    )

    assert peak_rate == 20.0
    assert isi_stats["selected_current_nanoamps"] == 0.20
    assert isi_stats["selected_mean_rate_hertz"] == 12.0
    assert isi_stats["coefficient_of_variation_interspike_interval"] == 0.0


def test_action_potential_ahp_amplitude_is_positive():
    t = np.arange(0.0, 20.1, 0.1)
    v = np.full_like(t, -60.0)
    v[(t >= 5.0) & (t < 5.3)] = np.linspace(-40.0, 30.0, 3)
    v[(t >= 5.3) & (t < 5.8)] = np.linspace(30.0, -75.0, 5)
    v[(t >= 5.8) & (t < 8.0)] = -75.0
    v[t >= 8.0] = -40.0

    props = compute_action_potential_properties(
        {"t": t, "v_soma": v},
        voltage_derivative_threshold_millivolts_per_millisecond=20.0,
        step_onset_milliseconds=0.0,
    )

    assert props["ahp_amplitude_millivolts"] > 0
