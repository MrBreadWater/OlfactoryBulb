"""Smoke tests for concrete olfactory-bulb notebook presentation helpers."""

from __future__ import annotations

import numpy as np

from olfactorybulb.analysis_presentations import (
    StandardOutputHooks,
    SweepAnimationHooks,
    animate_lfp_sweep,
    animate_sniff_average_sweep,
    animate_spectrogram_sweep,
    animate_wavelet_sweep,
    show_all_outputs,
)


class _Recorder:
    def __init__(self) -> None:
        self.calls = []

    def record(self, name: str, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        return name


class _FakeAxes:
    def __init__(self) -> None:
        self.calls = []

    def contourf(self, *args, **kwargs):
        self.calls.append(("contourf", args, kwargs))

    def set_ylim(self, value):
        self.calls.append(("set_ylim", (value,), {}))

    def set_xlabel(self, value):
        self.calls.append(("set_xlabel", (value,), {}))

    def set_ylabel(self, value):
        self.calls.append(("set_ylabel", (value,), {}))


class _FakeFigure:
    pass


class _FakePlt:
    def __init__(self) -> None:
        self.subplots_calls = []
        self.show_count = 0

    def subplots(self, **kwargs):
        self.subplots_calls.append(kwargs)
        return _FakeFigure(), _FakeAxes()

    def show(self):
        self.show_count += 1


def main() -> None:
    recorder = _Recorder()
    fake_plt = _FakePlt()

    sweep_hooks = SweepAnimationHooks(
        animate_sweep_fn=lambda sweep, plot_fn, **kwargs: {
            "sweep": sweep,
            "plot_fn": plot_fn,
            "kwargs": kwargs,
        },
        plot_named_signal_fn=lambda result, **kwargs: ("named", result, kwargs),
        plot_lfp_overview_fn=lambda result, **kwargs: ("lfp", result, kwargs),
        plot_spectrogram_fn=lambda result, **kwargs: ("spectrogram", result, kwargs),
        plot_wavelet_fn=lambda result, **kwargs: ("wavelet", result, kwargs),
        get_named_signal_fn=lambda result, **kwargs: (
            np.asarray(result["t"], dtype=float),
            np.asarray(result["y"], dtype=float),
        ),
        compute_wavelet_map_fn=lambda t, y, **kwargs: (
            t,
            y,
            np.asarray([20.0, 40.0]),
            np.ones((2, len(t))),
        ),
        plt_module=fake_plt,
    )

    sweep = {"items": [{"result": {"t": [0.0, 1.0], "y": [1.0, 2.0]}}]}
    lfp_anim = animate_lfp_sweep(sweep_hooks, sweep, signal="lfp", dt_ms=0.1, interval=10)
    assert lfp_anim["kwargs"]["figsize"] == (12, 7)
    assert lfp_anim["kwargs"]["interval"] == 10
    assert lfp_anim["plot_fn"]({"demo": True})[0] == "lfp"

    named_anim = animate_lfp_sweep(sweep_hooks, sweep, signal="mean_MC_voltage", dt_ms=0.2, interval=11)
    assert named_anim["kwargs"]["figsize"] == (12, 4)
    assert named_anim["plot_fn"]({"demo": True})[0] == "named"

    spectrogram_anim = animate_spectrogram_sweep(
        sweep_hooks,
        sweep,
        signal="lfp",
        dt_ms=0.1,
        max_freq_hz=200.0,
        nperseg=64,
        noverlap=32,
        interval=12,
    )
    assert spectrogram_anim["kwargs"]["figsize"] == (12, 4)
    assert spectrogram_anim["plot_fn"]({"demo": True})[0] == "spectrogram"

    wavelet_anim = animate_wavelet_sweep(sweep_hooks, sweep, signal="lfp", dt_ms=0.1, interval=13)
    assert wavelet_anim["kwargs"]["figsize"] == (12, 4)
    assert wavelet_anim["plot_fn"]({"demo": True})[0] == "wavelet"

    sniff_anim = animate_sniff_average_sweep(sweep_hooks, sweep, dt_ms=100.0, sniff_count=3, interval=14)
    assert sniff_anim["kwargs"]["figsize"] == (5, 5)
    sniff_fig = sniff_anim["plot_fn"]({"t": [0.0, 100.0, 200.0], "y": [1.0, 2.0, 3.0]})
    assert isinstance(sniff_fig, _FakeFigure)
    assert fake_plt.subplots_calls[-1]["figsize"] == (5, 5)

    output_hooks = StandardOutputHooks(
        plot_input_overview_fn=lambda *args, **kwargs: recorder.record("input", *args, **kwargs),
        plot_voltage_traces_fn=lambda *args, **kwargs: recorder.record("voltage", *args, **kwargs),
        plot_spike_raster_fn=lambda *args, **kwargs: recorder.record("spike", *args, **kwargs),
        plot_gc_output_overview_fn=lambda *args, **kwargs: recorder.record("gc", *args, **kwargs),
        plot_lfp_overview_fn=lambda *args, **kwargs: recorder.record("lfp", *args, **kwargs),
        plot_spectrogram_fn=lambda *args, **kwargs: recorder.record("spectrogram", *args, **kwargs),
        plot_wavelet_fn=lambda *args, **kwargs: recorder.record("wavelet", *args, **kwargs),
        plot_wavelet_band_power_fn=lambda *args, **kwargs: recorder.record("wavelet_band", *args, **kwargs),
        plt_show_fn=fake_plt.show,
    )

    config = {
        "analysis_dt_ms": 0.2,
        "input_bin_ms": 7.0,
        "input_smooth_sigma_ms": 11.0,
        "input_max_segments": 33,
        "input_rate_normalization": "total",
        "show_voltage_traces": True,
        "max_voltage_traces_per_type": 5,
        "max_spike_raster_cells_per_type": 12,
        "gc_output_bin_ms": 9.0,
        "gc_output_smooth_sigma_ms": 13.0,
        "gc_output_max_connections": 44,
        "gc_output_rate_normalization": "per_connection",
        "lfp_show_psd_target_template": False,
        "lfp_psd_template_fit_band_hz": [90.0, 150.0],
        "lfp_psd_template_floor": "0.25",
        "lfp_psd_xlim_hz": [0.0, 180.0],
        "spectrogram_signal": "mean_MC_voltage",
        "spectrogram_max_freq_hz": 180.0,
        "spectrogram_nperseg": 64,
        "spectrogram_noverlap": 32,
        "wavelet_signal": "mean_MC_voltage",
    }
    show_all_outputs(output_hooks, {"result": True}, config)
    names = [name for name, _args, _kwargs in recorder.calls]
    assert names == ["input", "voltage", "spike", "gc", "lfp", "spectrogram", "wavelet", "wavelet_band"]
    lfp_kwargs = next(kwargs for name, _args, kwargs in recorder.calls if name == "lfp")
    assert lfp_kwargs["dt_ms"] == 0.2
    assert lfp_kwargs["show_psd_target_template"] is False
    assert lfp_kwargs["psd_template_fit_band_hz"] == (90.0, 150.0)
    assert lfp_kwargs["psd_template_floor"] == 0.25
    assert lfp_kwargs["psd_xlim_hz"] == (0.0, 180.0)
    spectrogram_kwargs = next(kwargs for name, _args, kwargs in recorder.calls if name == "spectrogram")
    assert spectrogram_kwargs["signal"] == "mean_MC_voltage"
    assert spectrogram_kwargs["max_freq_hz"] == 180.0
    wavelet_kwargs = next(kwargs for name, _args, kwargs in recorder.calls if name == "wavelet")
    assert wavelet_kwargs["signal"] == "mean_MC_voltage"
    assert fake_plt.show_count == 8

    print("olfactorybulb analysis presentations: OK")


if __name__ == "__main__":
    main()
