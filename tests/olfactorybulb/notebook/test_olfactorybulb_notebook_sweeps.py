"""Focused tests for olfactory-bulb notebook sweep adapters."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from olfactorybulb.notebook_sweeps import (
    NotebookSweepHooks,
    animate_sweep_plots,
    list_sweeps,
    load_sweep,
    run_sweep_with_animations,
    save_animation,
    save_sweep,
    save_sweep_animation_stream,
)


class _FakeProgress:
    def __init__(self) -> None:
        self.updates: list[int] = []
        self.closed = False

    def update_to(self, value: int) -> None:
        self.updates.append(int(value))

    def close(self) -> None:
        self.closed = True


def _hooks() -> tuple[NotebookSweepHooks, list[str], list[tuple[int, str]], list[tuple[tuple, dict]]]:
    messages: list[str] = []
    progress_calls: list[tuple[int, str]] = []
    generic_calls: list[tuple[tuple, dict]] = []

    hooks = NotebookSweepHooks(
        sweeps_base="/tmp/sweeps",
        default_results_base="/tmp/notebook_runs",
        make_timestamp_fn=lambda: "20260602_120000",
        safe_name_fn=lambda value: str(value).replace(" ", "_"),
        json_ready_fn=lambda value: value,
        resolve_git_head_fn=lambda: "deadbeef",
        load_result_fn=lambda result_dir, *, progress=False: {"result_dir": str(result_dir), "progress": progress},
        save_sweep_fn=lambda *args, **kwargs: generic_calls.append((args, kwargs)) or Path(kwargs["base_dir"]) / str(kwargs["name"] or "auto"),
        load_sweep_fn=lambda *args, **kwargs: generic_calls.append((args, kwargs)) or {"loaded": True, "path": str(args[0]), "result": kwargs["load_result_fn"]("/tmp/run")},
        list_sweeps_fn=lambda *args, **kwargs: generic_calls.append((args, kwargs)) or [Path(kwargs["base_dir"]) / "demo_sweep"],
        save_animation_fn=lambda *args, **kwargs: generic_calls.append((args, kwargs)) or kwargs["default_output_dir_factory"]() / "demo.gif",
        save_sweep_animation_stream_fn=lambda *args, **kwargs: (
            kwargs["progress_callback"](1, 2) if kwargs.get("progress_callback") else None,
            kwargs["progress_callback"](2, 2) if kwargs.get("progress_callback") else None,
            generic_calls.append((args, kwargs)),
            kwargs["default_output_dir_factory"]() / "stream.gif",
        )[-1],
        animate_sweep_plots_fn=lambda *args, **kwargs: generic_calls.append((args, kwargs)) or {"artifact": kwargs["default_output_dir_factory"]() / "artifact.gif"},
        build_sweep_plot_callable_fn=lambda spec: (lambda result: result, f"artifact_{spec.name}"),
        normalize_sweep_plot_spec_fn=lambda raw: raw if isinstance(raw, SimpleNamespace) else SimpleNamespace(name=str(raw)),
        is_deprecated_sweep_animation_spec_fn=lambda spec: spec.name == "deprecated",
        deprecated_plot_names=("deprecated",),
        progress_factory_fn=lambda total, desc: progress_calls.append((int(total), desc)) or _FakeProgress(),
        progress_write_fn=messages.append,
        run_parameter_sweep_fn=lambda base_config, sweep_path, values=None: {
            "mode": "param",
            "base_config": dict(base_config),
            "path": sweep_path,
            "values": list(values or []),
            "items": [{"label": "item0"}],
        },
        run_grid_sweep_fn=lambda base_config, sweep_path: {
            "mode": "grid",
            "base_config": dict(base_config),
            "path": sweep_path,
            "values": [],
            "items": [{"label": "item0"}],
        },
    )
    return hooks, messages, progress_calls, generic_calls


def main() -> None:
    hooks, messages, progress_calls, generic_calls = _hooks()

    saved = save_sweep(hooks, {"items": []}, name="demo")
    assert saved == Path("/tmp/sweeps/demo")
    assert generic_calls[-1][1]["base_dir"] == "/tmp/sweeps"

    loaded = load_sweep(hooks, "/tmp/sweeps/demo")
    assert loaded["loaded"] is True
    assert loaded["result"]["progress"] is False

    listed = list_sweeps(hooks)
    assert listed == [Path("/tmp/sweeps/demo_sweep")]

    gif = save_animation(hooks, object(), "demo gif")
    assert gif == Path("/tmp/notebook_runs/animations/20260602_120000/demo.gif")

    stream_path = save_sweep_animation_stream(
        hooks,
        {"items": [{}, {}]},
        lambda result: result,
        "stream demo",
    )
    assert stream_path == Path("/tmp/notebook_runs/animations/20260602_120000/stream.gif")
    assert progress_calls[-1] == (2, "[OBGPU load] Render stream demo")

    artifacts = animate_sweep_plots(
        hooks,
        {"items": [{"label": "item0"}]},
        [SimpleNamespace(name="deprecated"), SimpleNamespace(name="keep")],
    )
    assert artifacts["artifact"] == Path("/tmp/notebook_runs/animations/20260602_120000/artifact.gif")
    assert messages == ["[OBGPU load] Skipping deprecated sweep animation plot 'deprecated'."]
    assert generic_calls[-1][1]["deprecated_names"] == {"deprecated"}

    sweep, sweep_artifacts = run_sweep_with_animations(
        hooks,
        {"paramset": "GammaSignature"},
        "gaba_tau2_ms",
        [36.0, 50.0],
        plots=[SimpleNamespace(name="keep")],
    )
    assert sweep["mode"] == "param"
    assert "artifact" in sweep_artifacts

    grid_sweep, grid_artifacts = run_sweep_with_animations(
        hooks,
        {"paramset": "GammaSignature"},
        {"gaba_tau2_ms": [36.0, 50.0]},
        use_grid=True,
    )
    assert grid_sweep["mode"] == "grid"
    assert grid_artifacts == {}

    try:
        run_sweep_with_animations(
            hooks,
            {"paramset": "GammaSignature"},
            "gaba_tau2_ms",
            use_grid=True,
        )
        raise AssertionError("expected invalid grid sweep to raise")
    except TypeError as exc:
        assert "Grid sweeps require sweep_path to be a dict" in str(exc)

    print("olfactorybulb notebook sweeps: OK")


if __name__ == "__main__":
    main()
