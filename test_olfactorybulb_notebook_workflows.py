"""Focused tests for olfactory-bulb notebook workflow adapters."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

from neuroinfra.notebooks.workflows import (
    load_run_pair,
    run_and_load,
    run_local_sweep_plan,
)

from olfactorybulb.notebook_workflows import (
    NotebookWorkflowAdapterHooks,
    build_load_run_pair_hooks,
    build_local_sweep_hooks,
    build_result_merge_payload,
    build_run_and_load_hooks,
)


def main() -> None:
    merge_calls = []
    save_calls = []
    pair_calls = []

    hooks = NotebookWorkflowAdapterHooks(
        load_run_record_fn=lambda **kwargs: pair_calls.append(kwargs) or SimpleNamespace(result_dir=Path("/tmp/pair"), label="pair"),
        load_result_fn=lambda run: {
            "result_dir": str(run.result_dir),
            "artifact_sizes": {"lfp.pkl": 12},
            "load_timing_seconds": {"lfp.pkl": 0.1},
            "load_total_seconds": 0.2,
        },
        run_simulation_fn=lambda config=None, *, label=None: SimpleNamespace(
            result_dir=Path(config["results_base"]) / str(label) if config and "results_base" in config else Path("/tmp/run"),
            label=label,
            config=config,
        ),
        merge_run_info_payload_fn=lambda result_dir, payload: merge_calls.append((Path(result_dir), payload)),
        save_sweep_fn=lambda sweep, **kwargs: save_calls.append((sweep, kwargs)) or (Path(kwargs["base_dir"]) / kwargs["name"]),
        sweep_item_runs_dir_fn=lambda config, label: Path(config["results_base"]) / "sweeps" / label / "item_runs",
        sweep_dir_fn=lambda config, label: Path(config["results_base"]) / "sweeps" / label,
    )

    payload = build_result_merge_payload(
        {
            "artifact_sizes": {"lfp.pkl": 12},
            "load_timing_seconds": {"lfp.pkl": 0.1},
            "load_total_seconds": 0.2,
        }
    )
    assert payload == {
        "artifact_sizes": {"lfp.pkl": 12},
        "load_timing_seconds": {"lfp.pkl": 0.1},
        "load_total_seconds": 0.2,
    }

    run_result, loaded_result = run_and_load(
        build_run_and_load_hooks(hooks),
        {"results_base": "/tmp/demo-results"},
        label="demo",
    )
    assert run_result.label == "demo"
    assert loaded_result["artifact_sizes"]["lfp.pkl"] == 12
    assert merge_calls == [
        (
            Path("/tmp/demo-results/demo"),
            {
                "artifact_sizes": {"lfp.pkl": 12},
                "load_timing_seconds": {"lfp.pkl": 0.1},
                "load_total_seconds": 0.2,
            },
        )
    ]

    pair_run, pair_result = load_run_pair(
        build_load_run_pair_hooks(hooks),
        prefix="demo",
        index=-1,
        results_base="/tmp/results",
    )
    assert pair_run.result_dir == Path("/tmp/pair")
    assert pair_result["result_dir"] == "/tmp/pair"
    assert pair_calls[0]["prefix"] == "demo"

    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        sweep_plan = {
            "path": "gaba_tau2_ms",
            "values": [36.0, 50.0],
            "items": [
                {"label": "item0", "value": 36.0, "config": {"paramset": "GammaSignature", "gaba_tau2_ms": 36.0}},
                {"label": "item1", "value": 50.0, "config": {"paramset": "GammaSignature", "gaba_tau2_ms": 50.0}},
            ],
            "paramset": "GammaSignature",
            "sweep_label": "demo_sweep",
            "base_config": {"results_base": str(base / "runs"), "paramset": "GammaSignature"},
            "grid": None,
        }
        local_sweep = run_local_sweep_plan(
            build_local_sweep_hooks(
                NotebookWorkflowAdapterHooks(
                    load_run_record_fn=hooks.load_run_record_fn,
                    load_result_fn=hooks.load_result_fn,
                    run_simulation_fn=lambda config=None, *, label=None: SimpleNamespace(
                        result_dir=Path(config["results_base"]) / str(label),
                        label=label,
                        config=config,
                    ),
                    merge_run_info_payload_fn=lambda result_dir, payload: merge_calls.append((Path(result_dir), payload)),
                    save_sweep_fn=hooks.save_sweep_fn,
                    sweep_item_runs_dir_fn=lambda config, label: base / "item_runs" / label,
                    sweep_dir_fn=lambda config, label: base / "sweeps" / label,
                )
            ),
            sweep_plan,
        )
        assert [item["run"].label for item in local_sweep["items"]] == ["item0", "item1"]
        assert local_sweep["items"][0]["config"]["results_base"] == str(base / "item_runs" / "demo_sweep")
        assert save_calls[-1][1]["name"] == "demo_sweep"
        assert save_calls[-1][1]["base_dir"] == base / "sweeps"
        assert merge_calls[-1][0] == base / "item_runs" / "demo_sweep" / "item1"

    print("olfactorybulb notebook workflows: OK")


if __name__ == "__main__":
    main()
