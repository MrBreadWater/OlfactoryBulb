"""Smoke tests for extracted sweep-analysis helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tempfile
from types import SimpleNamespace

from neuroinfra.analysis.sweeps import (
    SweepPlotRegistry,
    SweepPlotSpec,
    animate_sweep,
    animate_sweep_plots,
    build_sweep_plot_callable,
    compose_sweep_display_frame,
    default_sweep_animation_worker_count,
    describe_unavailable_sweep_item,
    format_sweep_frame_title,
    format_sweep_progress_label,
    format_sweep_value,
    format_sweep_value_label,
    iter_parallel_sweep_display_frames,
    is_deprecated_sweep_animation_spec,
    list_registry_plot_names,
    list_sweeps,
    load_sweep,
    make_sweep_placeholder_figure,
    make_sweep_plot_spec,
    normalize_sweep_plot_spec,
    render_sweep_frame,
    resolve_registry_plot,
    save_animation,
    save_sweep,
    save_sweep_animation_stream,
    write_sweep_info,
)


def _simple_plot(result):
    fig, ax = plt.subplots(figsize=(2, 1))
    ax.plot(result["x"], result["y"], linewidth=1.0)
    ax.set_title("simple")
    return fig


def _safe_name(value: object) -> str:
    return str(value).replace(" ", "_").replace("/", "_")


def main() -> None:
    spec = make_sweep_plot_spec("trace", filename="trace_anim")
    assert isinstance(spec, SweepPlotSpec)
    assert spec.name == "trace"
    assert spec.filename == "trace_anim"

    registry = SweepPlotRegistry(
        plots={"trace": _simple_plot, "other": _simple_plot},
        deprecated_names=frozenset({"trace_anim"}),
    )
    assert list_registry_plot_names(registry) == ["other", "trace"]
    assert resolve_registry_plot(registry, "trace") is _simple_plot
    try:
        resolve_registry_plot(registry, "trace_anim")
        raise AssertionError("Expected deprecated registry plot lookup to fail")
    except KeyError as exc:
        assert "deprecated" in str(exc)
    try:
        resolve_registry_plot(registry, "missing")
        raise AssertionError("Expected missing registry plot lookup to fail")
    except KeyError as exc:
        assert "Available:" in str(exc)

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

    display_frame = compose_sweep_display_frame(
        np.zeros((8, 10, 3), dtype=np.uint8),
        "demo title",
        figsize=(2.0, 1.0),
        frame_index=1,
        total_frames=3,
    )
    assert display_frame.ndim == 3
    assert default_sweep_animation_worker_count(2) == 1

    with tempfile.TemporaryDirectory() as tmp_dir_text:
        tmp_dir = Path(tmp_dir_text)
        good_result_dir = tmp_dir / "good-run"
        good_result_dir.mkdir(parents=True, exist_ok=True)
        (good_result_dir / "run_info.json").write_text("{}")

        sweep = {
            "path": "gaba_tau2_ms",
            "values": [50.0, 55.0],
            "paramset": "GammaSignature",
            "partial": True,
            "missing_labels": ["item_001"],
            "items": [
                {
                    "label": "item_000",
                    "value": 50.0,
                    "run": SimpleNamespace(result_dir=good_result_dir),
                    "result": {"result_dir": good_result_dir},
                    "status": {"ok": True},
                },
                {
                    "label": "item_001",
                    "value": 55.0,
                    "run": None,
                    "result": None,
                    "status": {"ok": False},
                },
            ],
        }
        sweep_dir = save_sweep(
            sweep,
            name="demo_sweep",
            base_dir=tmp_dir / "sweeps",
            timestamp_factory=lambda: "20260601_220000",
            safe_name=_safe_name,
            json_ready=lambda value: value,
            resolve_git_head=lambda: "deadbeef",
        )
        assert (sweep_dir / "sweep_info.json").exists()
        assert (sweep_dir / "runs" / "00_50.0" / "result_dir.txt").exists()

        listed = list_sweeps(base_dir=tmp_dir / "sweeps")
        assert listed == [sweep_dir]

        def _load_result(result_dir: Path) -> dict[str, object]:
            return {"result_dir": result_dir, "loaded": True}

        reloaded = load_sweep(
            sweep_dir,
            load_result_fn=_load_result,
            safe_name=_safe_name,
        )
        assert len(reloaded["items"]) == 2
        assert reloaded["items"][0]["result"]["loaded"] is True
        assert reloaded["items"][1]["result"] is None
        assert reloaded["partial"] is True
        assert reloaded["missing_labels"] == ["item_001"]

        metadata_only_sweep = {
            "path": "gaba_tau2_ms",
            "values": [50.0],
            "items": [{"label": "item_000", "value": 50.0, "status": {"ok": True}}],
        }
        metadata_dir = tmp_dir / "metadata_only"
        write_sweep_info(
            metadata_only_sweep,
            sweep_dir=metadata_dir,
            timestamp="20260601_220500",
            json_ready=lambda value: value,
            resolve_git_head=lambda: "cafebabe",
        )
        assert (metadata_dir / "sweep_info.json").exists()

        animation_sweep = {
            "path": "gaba_tau2_ms",
            "sweep_dir": tmp_dir / "anim-sweep",
            "items": [
                {"label": "item_000", "value": 50.0, "result": {"x": [0, 1], "y": [1, 2]}},
                {"label": "item_001", "value": 55.0, "result": {"x": [0, 1], "y": [2, 3]}},
            ],
        }
        animation_sweep["sweep_dir"].mkdir(parents=True, exist_ok=True)

        frames = list(
            iter_parallel_sweep_display_frames(
                animation_sweep,
                _simple_plot,
                figsize=(2.0, 1.5),
                workers=1,
            )
        )
        assert len(frames) == 2
        assert all(frame.ndim == 3 for frame in frames)

        anim = animate_sweep(animation_sweep, _simple_plot, figsize=(2.0, 1.5), interval=10)
        gif_path = save_animation(
            anim,
            "anim_demo",
            safe_name=_safe_name,
            output_dir=tmp_dir / "manual-gifs",
            fps=2,
        )
        assert gif_path.exists()

        stream_gif = save_sweep_animation_stream(
            animation_sweep,
            _simple_plot,
            "stream_demo",
            safe_name=_safe_name,
            output_dir=tmp_dir / "stream-gifs",
            figsize=(2.0, 1.5),
            fps=2,
            workers=1,
        )
        assert stream_gif.exists()

        animated = animate_sweep_plots(
            animation_sweep,
            [make_sweep_plot_spec(_simple_plot, name="simple", fps=2)],
            plot_builder=lambda spec: (
                build_sweep_plot_callable(spec, plot_resolver=lambda _name: _simple_plot),
                spec.name,
            ),
            safe_name=_safe_name,
            output_dir=tmp_dir / "auto-gifs",
            workers=1,
        )
        assert set(animated.keys()) == {"simple"}
        assert next(iter(animated.values())).exists()

    print("analysis sweep helpers: OK")


if __name__ == "__main__":
    main()
