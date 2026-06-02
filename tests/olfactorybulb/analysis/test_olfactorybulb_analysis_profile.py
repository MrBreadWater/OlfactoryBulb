"""Smoke tests for the concrete olfactory-bulb analysis profile builder."""

from __future__ import annotations

import numpy as np

from olfactorybulb.analysis_profile import (
    OlfactoryBulbAnalysisProfileHooks,
    build_olfactorybulb_analysis_profile,
)


def main() -> None:
    hooks = OlfactoryBulbAnalysisProfileHooks(
        plot_voltage_traces_fn=lambda result, **_kwargs: result.get("soma_vs"),
        plot_spike_raster_fn=lambda result, **_kwargs: result.get("soma_spikes"),
        plot_hfo_power_summary_fn=lambda result, **_kwargs: result.get("lfp"),
        plot_named_signal_fn=lambda result, **_kwargs: result.get("lfp"),
        plot_spectrogram_fn=lambda result, **_kwargs: result.get("lfp"),
        plot_wavelet_fn=lambda result, **_kwargs: result.get("lfp"),
        plot_wavelet_band_power_fn=lambda result, **_kwargs: result.get("lfp"),
    )
    profile = build_olfactorybulb_analysis_profile(hooks)

    result = {
        "lfp_t": [0.0, 1.0, 2.0],
        "lfp": [1.0, 2.0, 3.0],
        "input_times": [("MC0.soma", np.asarray([10.0, 20.0], dtype=float))],
        "gc_output_events": [
            {
                "source_section": "GC0.soma",
                "dest_section": "MC0.soma",
                "times": np.asarray([5.0, 25.0], dtype=float),
            }
        ],
        "soma_vs": [("MC0.soma", [0.0, 1.0], [-65.0, -60.0])],
        "soma_spikes": {
            "labels": ["MC0.soma"],
            "spike_times": [np.asarray([10.0, 20.0], dtype=float)],
        },
        "voltage_summary": {
            "t_by_type": {"MC": [0.0, 1.0]},
            "mean_by_type": {"MC": [-64.0, -61.0]},
        },
    }

    assert profile.list_available_signals(result) == [
        "lfp",
        "gc_output_rate",
        "gc_output_rate_MC",
        "gc_output_rate_TC",
        "input_rate",
        "input_rate_MC",
        "input_rate_TC",
        "mean_MC_voltage",
    ]
    signal_t, signal_y = profile.resolve_signal(result, "lfp", dt_ms=0.1)
    np.testing.assert_allclose(signal_t[[0, -1]], [0.0, 2.0])
    np.testing.assert_allclose(signal_y[[0, -1]], [1.0, 3.0])
    assert len(signal_t) == 21
    assert len(signal_y) == 21

    gc_output_suite = profile.event_family("gc_output")
    assert len(gc_output_suite.filter_rows(result, include_prefixes=("MC",))) == 1
    spike_suite = profile.frequency_plot_suite("spike")
    assert spike_suite.family.title_1d == "Soma Spike Frequency Distribution"
    assert profile.list_builtin_sweep_plot_names() == [
        "gc_output_frequency_kde_1d",
        "gc_output_frequency_kde_2d",
        "gc_output_frequency_time_binned",
        "hfo_power_summary",
        "named_signal",
        "spectrogram",
        "spike_frequency_kde_1d",
        "spike_frequency_kde_2d",
        "spike_frequency_time_binned",
        "spike_raster",
        "voltage_traces",
        "wavelet",
        "wavelet_band_power",
    ]
    assert profile.resolve_builtin_sweep_plot("named_signal")(result) == result["lfp"]

    print("olfactorybulb analysis profile: OK")


if __name__ == "__main__":
    main()
