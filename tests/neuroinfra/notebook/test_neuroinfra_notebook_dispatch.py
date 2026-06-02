"""Focused tests for generic notebook dispatch helpers."""

from __future__ import annotations

from pathlib import Path

from neuroinfra.notebooks.dispatch import (
    NotebookRunDispatchHooks,
    NotebookSweepDispatchHooks,
    dispatch_grid_sweep,
    dispatch_parameter_sweep,
    dispatch_run,
)


def main() -> None:
    local_run_calls = []
    remote_run_calls = []

    hooks = NotebookRunDispatchHooks(
        normalize_config_fn=lambda config: {"paramset": "GammaSignature", **(config or {})},
        make_timestamp_fn=lambda: "2026-06-02T12-00-00",
        make_label_fn=lambda config, timestamp: f"{config['paramset']}_{timestamp}",
        execute_local_run_fn=lambda config, label, timestamp, result_dir: local_run_calls.append(
            (dict(config), label, timestamp, Path(result_dir))
        )
        or "LOCAL-RUN",
        execute_remote_run_fn=lambda config, label, timestamp, result_dir: remote_run_calls.append(
            (dict(config), label, timestamp, Path(result_dir))
        )
        or "REMOTE-RUN",
        default_results_base="/tmp/notebook-runs",
    )

    local_result = dispatch_run(hooks, {"results_base": "/tmp/demo-local"}, label="local_demo")
    assert local_result == "LOCAL-RUN"
    assert local_run_calls == [
        (
            {"paramset": "GammaSignature", "results_base": "/tmp/demo-local"},
            "local_demo",
            "2026-06-02T12-00-00",
            Path("/tmp/demo-local/local_demo"),
        )
    ]

    remote_result = dispatch_run(
        hooks,
        {"runner_backend": "slurm_remote", "results_base": "/tmp/demo-remote"},
        label="remote_demo",
    )
    assert remote_result == "REMOTE-RUN"
    assert remote_run_calls == [
        (
            {
                "paramset": "GammaSignature",
                "runner_backend": "slurm_remote",
                "results_base": "/tmp/demo-remote",
            },
            "remote_demo",
            "2026-06-02T12-00-00",
            Path("/tmp/demo-remote/remote_demo"),
        )
    ]

    try:
        dispatch_run(hooks, {"runner_backend": "bogus"})
        raise AssertionError("expected unsupported backend to raise")
    except ValueError as exc:
        assert "Unsupported runner_backend='bogus'" in str(exc)

    local_sweep_calls = []
    remote_sweep_calls = []

    sweep_hooks = NotebookSweepDispatchHooks(
        prepare_sweep_plan_fn=lambda base_config, sweep_path, values=None, *, grid=False: {
            "path": sweep_path,
            "values": list(values or []),
            "grid": {"path": sweep_path} if grid else None,
            "base_config": dict(base_config),
            "sweep_label": "demo_sweep",
            "items": [],
            "paramset": base_config.get("paramset"),
        },
        uses_remote_batch_engine_fn=lambda config: bool(config.get("remote")),
        execute_local_sweep_fn=lambda sweep_plan: local_sweep_calls.append(sweep_plan) or {"mode": "local", **sweep_plan},
        execute_remote_sweep_fn=lambda sweep_plan: remote_sweep_calls.append(sweep_plan) or {"mode": "remote", **sweep_plan},
    )

    local_sweep = dispatch_parameter_sweep(
        sweep_hooks,
        {"paramset": "GammaSignature"},
        "gaba_tau2_ms",
        [36.0, 50.0],
    )
    assert local_sweep["mode"] == "local"
    assert local_sweep_calls[0]["path"] == "gaba_tau2_ms"

    remote_grid = dispatch_grid_sweep(
        sweep_hooks,
        {"paramset": "GammaSignature", "remote": True},
        {"gaba_tau2_ms": [36.0, 50.0]},
    )
    assert remote_grid["mode"] == "remote"
    assert remote_sweep_calls[0]["grid"] == {"path": {"gaba_tau2_ms": [36.0, 50.0]}}

    print("neuroinfra notebook dispatch: OK")


if __name__ == "__main__":
    main()
