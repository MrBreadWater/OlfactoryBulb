"""Focused tests for olfactory-bulb notebook presentation adapters."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from olfactorybulb.notebook_presentations import (
    NotebookPresentationHooks,
    print_run_summary,
    save_figure,
    show_all_outputs,
)


class _FakeFigure:
    pass


class _FakePlt:
    def __init__(self) -> None:
        self._current = _FakeFigure()
        self.show_count = 0
        self.closed = []

    def gcf(self):
        return self._current

    def close(self, fig):
        self.closed.append(fig)

    def show(self):
        self.show_count += 1


def _build_hooks():
    fake_plt = _FakePlt()
    save_calls = []
    plot_calls = []
    writes = []

    def _record_plot(name: str):
        def _inner(*args, **kwargs):
            plot_calls.append((name, args, kwargs))
            return name

        return _inner

    hooks = NotebookPresentationHooks(
        default_results_base="/tmp/notebook_runs",
        make_timestamp_fn=lambda: "20260602_120000",
        safe_name_fn=lambda value: str(value).replace(" ", "_"),
        plt_module=fake_plt,
        save_figure_fn=lambda *args, **kwargs: (
            save_calls.append((args, kwargs)),
            kwargs["default_output_dir_factory"]() / f"{kwargs['safe_name_fn'](args[0])}.png",
        )[-1],
        plot_input_overview_fn=_record_plot("input"),
        plot_voltage_traces_fn=_record_plot("voltage"),
        plot_spike_raster_fn=_record_plot("spike"),
        plot_gc_output_overview_fn=_record_plot("gc"),
        plot_lfp_overview_fn=_record_plot("lfp"),
        plot_spectrogram_fn=_record_plot("spectrogram"),
        plot_wavelet_fn=_record_plot("wavelet"),
        plot_wavelet_band_power_fn=_record_plot("wavelet_band"),
        result_overview_fn=lambda result: {"label": result.get("label", "demo")},
        build_run_config_fn=lambda **config: {"paramset": "GammaSignature", **config},
        resolve_effective_params_fn=lambda config: {
            "input_odors_source": "config",
            "n_odor_presentations": 1,
            "odor_names": ["Apple"],
            "input_odors": {0: {"name": "Apple", "rel_conc": 0.1}},
            "max_firing_rate_hz": 150.0,
            "inhale_duration_ms": 125.0,
            "mc_input_weight": 1.0,
            "tc_input_weight": 1.0,
            "full_param_snapshot": {"gaba_tau2_ms": config.get("gaba_tau2_ms", 36.0)},
        },
        resolve_paramset_defaults_fn=lambda name: {"gaba_tau2_ms": 36.0},
        diff_values_fn=lambda before, after: [] if before == after else [{"path": "gaba_tau2_ms", "before": before, "after": after}],
        extract_runtime_control_snapshot_fn=lambda config: {"analysis_dt_ms": config.get("analysis_dt_ms", 0.1)},
        print_diff_section_fn=lambda title, changes, max_items=None: writes.append(f"{title}:{len(changes)}"),
        write_fn=writes.append,
    )
    return hooks, fake_plt, save_calls, plot_calls, writes


def main() -> None:
    hooks, fake_plt, save_calls, plot_calls, writes = _build_hooks()

    saved = save_figure(hooks, "Notebook Summary")
    assert saved == Path("/tmp/notebook_runs/figures/20260602_120000/Notebook_Summary.png")
    assert save_calls[-1][1]["fig"] is fake_plt.gcf()
    assert save_calls[-1][1]["close_figure_fn"].__self__ is fake_plt

    result = {"label": "demo"}
    config = {
        "show_voltage_traces": True,
        "analysis_dt_ms": 0.2,
        "spectrogram_signal": "mean_MC_voltage",
        "wavelet_signal": "mean_MC_voltage",
    }
    show_all_outputs(hooks, result, config)
    assert [name for name, _args, _kwargs in plot_calls] == [
        "input",
        "voltage",
        "spike",
        "gc",
        "lfp",
        "spectrogram",
        "wavelet",
        "wavelet_band",
    ]
    assert fake_plt.show_count == 8

    run = SimpleNamespace(
        result_dir="/tmp/notebook_runs/demo",
        command=["python", "demo.py"],
        config={"paramset": "GammaSignature", "gaba_tau2_ms": 36.0},
    )
    summary_result = {
        "label": "demo",
        "run_info": {"config": run.config},
        "input_times": [],
        "gc_output_events": [],
        "lfp": [],
    }
    print_run_summary(hooks, run, summary_result)
    assert any("Result directory: /tmp/notebook_runs/demo" in line for line in writes)
    assert any("Command: python demo.py" in line for line in writes)
    assert any(line.startswith("Requested/effective param changes vs clean paramset") for line in writes)

    print("olfactorybulb notebook presentations: OK")


if __name__ == "__main__":
    main()
