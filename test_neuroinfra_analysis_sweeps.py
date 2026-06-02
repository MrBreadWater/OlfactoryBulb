"""Smoke tests for extracted sweep-analysis helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from neuroinfra.analysis.sweeps import (
    SweepPlotSpec,
    build_sweep_plot_callable,
    describe_unavailable_sweep_item,
    format_sweep_frame_title,
    format_sweep_progress_label,
    format_sweep_value,
    format_sweep_value_label,
    is_deprecated_sweep_animation_spec,
    make_sweep_placeholder_figure,
    make_sweep_plot_spec,
    normalize_sweep_plot_spec,
    render_sweep_frame,
)


def _simple_plot(result):
    fig, ax = plt.subplots(figsize=(2, 1))
    ax.plot(result["x"], result["y"], linewidth=1.0)
    ax.set_title("simple")
    return fig


def main() -> None:
    spec = make_sweep_plot_spec("trace", filename="trace_anim")
    assert isinstance(spec, SweepPlotSpec)
    assert spec.name == "trace"
    assert spec.filename == "trace_anim"

    normalized = normalize_sweep_plot_spec({"plot": "trace", "interval": 200, "fps": 12})
    assert isinstance(normalized, SweepPlotSpec)
    assert normalized.interval == 200
    assert normalized.fps == 12

    wrapped = build_sweep_plot_callable(
        spec,
        plot_resolver=lambda name: _simple_plot if name == "trace" else None,
    )
    fig = wrapped({"x": [0, 1], "y": [1, 2]})
    try:
        assert hasattr(fig, "savefig")
    finally:
        plt.close(fig)

    assert is_deprecated_sweep_animation_spec(spec, deprecated_names={"trace_anim"}) is True
    assert format_sweep_value(1.23456) == "1.235"
    assert format_sweep_value_label({"path": "gaba_tau2_ms"}, 50.0) == "gaba_tau2_ms = 50"
    assert format_sweep_progress_label(1, 4) == "2/4 (50.0%)"
    assert "2/4" in format_sweep_frame_title({"path": "gaba_tau2_ms"}, 50.0, 1, 4)

    unavailable_reason = describe_unavailable_sweep_item({"status": {"state": "FAILED"}})
    assert unavailable_reason == "state: FAILED"

    placeholder = make_sweep_placeholder_figure(
        {"path": "gaba_tau2_ms"},
        {"value": 50.0, "label": "item_001"},
        0,
        3,
        reason="missing result",
        figsize=(3.0, 2.0),
    )
    try:
        assert hasattr(placeholder, "savefig")
    finally:
        plt.close(placeholder)

    frame_rgb, title = render_sweep_frame(
        {"path": "gaba_tau2_ms"},
        {"result": {"x": [0, 1], "y": [1, 2]}, "value": 50.0, "label": "item_001"},
        0,
        3,
        _simple_plot,
        figsize=(3.0, 2.0),
    )
    assert isinstance(frame_rgb, np.ndarray)
    assert frame_rgb.ndim == 3
    assert frame_rgb.shape[2] == 3
    assert "gaba_tau2_ms" in title

    placeholder_rgb, placeholder_title = render_sweep_frame(
        {"path": "gaba_tau2_ms"},
        {"result": None, "value": 55.0, "label": "item_002", "status": {"ok": False}},
        1,
        3,
        _simple_plot,
        figsize=(3.0, 2.0),
    )
    assert isinstance(placeholder_rgb, np.ndarray)
    assert placeholder_rgb.ndim == 3
    assert "2/3" in placeholder_title

    print("analysis sweep helpers: OK")


if __name__ == "__main__":
    main()
