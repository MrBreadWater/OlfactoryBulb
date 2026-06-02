"""Focused tests for olfactory-bulb notebook entrypoint adapters."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from olfactorybulb.notebook_dispatch import (
    NotebookRunDispatchAdapterHooks,
    NotebookSweepDispatchAdapterHooks,
    run_notebook_grid_sweep,
    run_notebook_parameter_sweep,
    run_notebook_simulation,
)


def main() -> None:
    local_execute_calls = []
    remote_execute_calls = []

    run_hooks = NotebookRunDispatchAdapterHooks(
        normalize_config_fn=lambda config: {"paramset": "GammaSignature", **(config or {})},
        make_timestamp_fn=lambda: "2026-06-02T12-00-00",
        make_label_fn=lambda config, timestamp: f"{config['paramset']}_{timestamp}",
        build_local_run_payload_fn=lambda payload_hooks, config, **kwargs: SimpleNamespace(
            result_dir=Path(config["results_base"]) / kwargs["label"],
            env={"PYTHONPATH": str(kwargs["repo_root"])},
            command=["python", "demo.py", kwargs["label"]],
        ),
        local_run_payload_hooks_fn=lambda: "PAYLOAD-HOOKS",
        build_local_run_hooks_fn=lambda hook_builder_hooks: f"LOCAL-RUN-HOOKS<{hook_builder_hooks}>",
        local_run_hook_builder_hooks_fn=lambda: "HOOK-BUILDER",
        execute_local_run_fn=lambda **kwargs: local_execute_calls.append(kwargs) or "LOCAL-RUN",
        execute_remote_run_fn=lambda config, *, label, timestamp, local_result_dir: remote_execute_calls.append(
            (dict(config), label, timestamp, Path(local_result_dir))
        )
        or "REMOTE-RUN",
        default_results_base="/tmp/default-runs",
    )

    local_result = run_notebook_simulation(
        run_hooks,
        {"results_base": "/tmp/local-runs"},
        label="local_demo",
    )
    assert local_result == "LOCAL-RUN"
    assert local_execute_calls and local_execute_calls[0]["runner_name"] == "obgpu_experiment_helpers.run_simulation"
    assert local_execute_calls[0]["result_dir"] == Path("/tmp/local-runs/local_demo")
    assert local_execute_calls[0]["command"] == ["python", "demo.py", "local_demo"]
    assert local_execute_calls[0]["hooks"] == "LOCAL-RUN-HOOKS<HOOK-BUILDER>"

    remote_result = run_notebook_simulation(
        run_hooks,
        {"runner_backend": "slurm_remote", "results_base": "/tmp/remote-runs"},
        label="remote_demo",
    )
    assert remote_result == "REMOTE-RUN"
    assert remote_execute_calls == [
        (
            {
                "paramset": "GammaSignature",
                "runner_backend": "slurm_remote",
                "results_base": "/tmp/remote-runs",
            },
            "remote_demo",
            "2026-06-02T12-00-00",
            Path("/tmp/remote-runs/remote_demo"),
        )
    ]

    local_sweep_calls = []
    remote_sweep_calls = []

    sweep_hooks = NotebookSweepDispatchAdapterHooks(
        prepare_sweep_plan_fn=lambda base_config, sweep_path, values=None, *, grid=False: {
            "path": sweep_path,
            "values": list(values or []),
            "base_config": dict(base_config),
            "grid": {"path": sweep_path} if grid else None,
            "sweep_label": "demo_sweep",
            "items": [],
            "paramset": base_config.get("paramset"),
        },
        uses_remote_batch_engine_fn=lambda config: bool(config.get("remote")),
        build_local_sweep_hooks_fn=lambda workflow_hooks: f"LOCAL-SWEEP-HOOKS<{workflow_hooks}>",
        notebook_workflow_adapter_hooks_fn=lambda: "WORKFLOW-HOOKS",
        execute_local_sweep_plan_fn=lambda hooks, sweep_plan: local_sweep_calls.append((hooks, sweep_plan))
        or {"mode": "local", **sweep_plan},
        execute_remote_sweep_fn=lambda sweep_plan: remote_sweep_calls.append(sweep_plan) or {"mode": "remote", **sweep_plan},
    )

    local_sweep = run_notebook_parameter_sweep(
        sweep_hooks,
        {"paramset": "GammaSignature"},
        "gaba_tau2_ms",
        [36.0, 50.0],
    )
    assert local_sweep["mode"] == "local"
    assert local_sweep_calls[0][0] == "LOCAL-SWEEP-HOOKS<WORKFLOW-HOOKS>"

    remote_grid = run_notebook_grid_sweep(
        sweep_hooks,
        {"paramset": "GammaSignature", "remote": True},
        {"gaba_tau2_ms": [36.0, 50.0]},
    )
    assert remote_grid["mode"] == "remote"
    assert remote_sweep_calls[0]["grid"] == {"path": {"gaba_tau2_ms": [36.0, 50.0]}}

    print("olfactorybulb notebook dispatch: OK")


if __name__ == "__main__":
    main()
