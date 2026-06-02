"""Smoke tests for extracted result-analysis profiles."""

from __future__ import annotations

import numpy as np

from neuroinfra.analysis.events import ResultEventFamilySpec, ResultEventFamilySuite
from neuroinfra.analysis.frequency_plots import ResultFrequencyPlotFamily, ResultFrequencyPlotSuite
from neuroinfra.analysis.profiles import ResultAnalysisProfile
from neuroinfra.analysis.signals import ResultSignalRegistry, keyed_trace_signal_provider
from neuroinfra.analysis.sweeps import SweepPlotRegistry


def main() -> None:
    signal_registry = ResultSignalRegistry(
        providers=(
            keyed_trace_signal_provider(
                "lfp",
                time_key="lfp_t",
                value_key="lfp",
                uniform_trace_fn=lambda t, values, dt_ms=None: (
                    np.asarray(t, dtype=float),
                    np.asarray(values, dtype=float),
                ),
            ),
        ),
    )
    event_family = ResultEventFamilySuite(
        spec=ResultEventFamilySpec(
            rows_from_result_fn=lambda result: list(result.get("input_times", [])),
            filter_label_fn=lambda row: row[0],
            times_fn=lambda row: row[1],
        ),
        infer_t_stop_fn=lambda _result, rows: max(
            (float(np.asarray(row[1], dtype=float)[-1]) for row in rows if len(row[1]) > 0),
            default=0.0,
        ),
    )
    frequency_suite = ResultFrequencyPlotSuite(
        ResultFrequencyPlotFamily(
            collect_samples_fn=lambda result, **_kwargs: {
                "times": np.asarray(result.get("sample_t", []), dtype=float),
                "freqs": np.asarray(result.get("sample_f", []), dtype=float),
            },
            selection_label_fn=lambda selection: str(selection),
            title_1d="Example 1D",
            title_2d="Example 2D",
            title_time_binned="Example Time Binned",
        )
    )
    sweep_plot = lambda result, **_kwargs: result.get("lfp")
    profile = ResultAnalysisProfile(
        signal_registry=signal_registry,
        event_families={"input": event_family},
        event_plots={},
        frequency_plots={"spike": frequency_suite},
        sweep_plot_registry=SweepPlotRegistry({"trace": sweep_plot}),
    )

    result = {
        "lfp_t": [0.0, 1.0, 2.0],
        "lfp": [1.0, 2.0, 3.0],
        "input_times": [("MC0.soma", np.asarray([5.0, 15.0], dtype=float))],
        "sample_t": [10.0, 20.0],
        "sample_f": [80.0, 90.0],
    }

    assert profile.list_available_signals(result) == ["lfp"]
    signal_t, signal_y = profile.resolve_signal(result, "lfp", dt_ms=0.1)
    np.testing.assert_allclose(signal_t, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(signal_y, [1.0, 2.0, 3.0])

    assert profile.event_family("input") is event_family
    assert profile.frequency_plot_suite("spike") is frequency_suite
    assert profile.list_builtin_sweep_plot_names() == ["trace"]
    assert profile.resolve_builtin_sweep_plot("trace") is sweep_plot

    try:
        profile.event_plot_suite("missing")
        raise AssertionError("expected missing event plot suite to raise KeyError")
    except KeyError as exc:
        assert "Unknown event plot suite" in str(exc)

    print("analysis profiles: OK")


if __name__ == "__main__":
    main()
