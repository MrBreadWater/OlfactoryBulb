"""Smoke tests for the concrete olfactory-bulb analysis profile builder."""

from __future__ import annotations

import numpy as np

from neuroinfra.analysis import CategoryCatalogHooks
from olfactorybulb.analysis_profile import (
    OlfactoryBulbAnalysisProfileHooks,
    build_olfactorybulb_analysis_profile,
)


def main() -> None:
    hooks = OlfactoryBulbAnalysisProfileHooks(
        category_hooks=CategoryCatalogHooks(
            categorize_label_fn=lambda label: "MC" if str(label).startswith("MC") else "other",
            order_categories_fn=lambda names: sorted(set(str(name) for name in names)),
        ),
        uniform_trace_fn=lambda t, values, dt_ms=None: (
            np.asarray(t, dtype=float),
            np.asarray(values, dtype=float),
        ),
        split_traces_by_type_fn=lambda result: {
            "MC": [
                (
                    "MC0.soma",
                    np.asarray(result["soma_vs"][0][1], dtype=float),
                    np.asarray(result["soma_vs"][0][2], dtype=float),
                )
            ]
        },
        list_available_cell_types_fn=lambda _result: ["MC"],
        list_available_soma_labels_fn=lambda _result: ["MC0.soma"],
        saved_voltage_summary_signal_fn=lambda result, *, cell_type, moment="mean", dt_ms=None: (
            np.asarray(result["voltage_summary"]["t_by_type"][cell_type], dtype=float),
            np.asarray(result["voltage_summary"][f"{moment}_by_type"][cell_type], dtype=float),
        ),
        collect_spike_frequency_samples_fn=lambda result, **_kwargs: {
            "times": np.asarray(result.get("sample_t", []), dtype=float),
            "freqs": np.asarray(result.get("sample_f", []), dtype=float),
            "labels": ["MC0.soma"],
            "n_traces": 1,
            "cell_types": ["MC"],
        },
        normalize_cell_name_fn=lambda name: str(name).split(".", 1)[0],
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
        "voltage_summary": {
            "t_by_type": {"MC": [0.0, 1.0]},
            "mean_by_type": {"MC": [-64.0, -61.0]},
        },
        "sample_t": [10.0, 20.0],
        "sample_f": [90.0, 100.0],
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
    np.testing.assert_allclose(signal_t, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(signal_y, [1.0, 2.0, 3.0])

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
