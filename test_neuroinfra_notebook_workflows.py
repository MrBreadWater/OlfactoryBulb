"""Focused tests for generic notebook workflow helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

from neuroinfra.notebooks.workflows import (
    LoadRunPairHooks,
    LocalSweepHooks,
    RunAndLoadHooks,
    load_run_pair,
    run_and_load,
    run_local_sweep_plan,
)


def main() -> None:
    run_calls = []
    load_calls = []
    merge_calls = []

    def _run_simulation(config=None, *, label=None):
        run_calls.append((config, label))
        return SimpleNamespace(result_dir=Path("/tmp/demo-run"), label=label)

    def _load_result(run):
        load_calls.append(run)
        return {"artifact_sizes": {"lfp.pkl": 12}, "load_timing_seconds": {"lfp.pkl": 0.1}, "load_total_seconds": 0.2}

    run_result, loaded_result = run_and_load(
        RunAndLoadHooks(
            run_simulation_fn=_run_simulation,
            load_result_fn=_load_result,
            merge_run_info_payload_fn=lambda result_dir, payload: merge_calls.append((Path(result_dir), payload)),
            build_merge_payload_fn=lambda result: {"artifact_sizes": result["artifact_sizes"]},
        ),
        {"paramset": "GammaSignature"},
        label="demo",
    )
    assert run_result.label == "demo"
    assert loaded_result["artifact_sizes"]["lfp.pkl"] == 12
    assert merge_calls == [(Path("/tmp/demo-run"), {"artifact_sizes": {"lfp.pkl": 12}})]

    pair_calls = []
    pair_run, pair_result = load_run_pair(
        LoadRunPairHooks(
            load_run_record_fn=lambda **kwargs: pair_calls.append(kwargs) or SimpleNamespace(result_dir=Path("/tmp/pair")),
            load_result_fn=lambda run: {"result_dir": str(run.result_dir)},
        ),
        prefix="demo",
        index=-1,
        results_base="/tmp/results",
    )
    assert pair_run.result_dir == Path("/tmp/pair")
    assert pair_result["result_dir"] == "/tmp/pair"
    assert pair_calls[0]["prefix"] == "demo"

    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        saved = []

        def _run_and_load(config, label):
            result_dir = Path(config["results_base"]) / str(label)
            return (
                SimpleNamespace(result_dir=result_dir, label=label),
                {"result_dir": str(result_dir)},
            )

        sweep_plan = {
            "path": "gaba_tau2_ms",
            "values": [36.0, 50.0],
            "items": [
                {"label": "item0", "value": 36.0, "config": {"paramset": "GammaSignature", "gaba_tau2_ms": 36.0}},
                {"label": "item1", "value": 50.0, "config": {"paramset": "GammaSignature", "gaba_tau2_ms": 50.0}},
            ],
            "paramset": "GammaSignature",
            "sweep_label": "demo_sweep",
            "base_config": {"paramset": "GammaSignature"},
            "grid": None,
        }
        sweep = run_local_sweep_plan(
            LocalSweepHooks(
                run_and_load_fn=_run_and_load,
                save_sweep_fn=lambda sweep, **kwargs: saved.append((sweep, kwargs)) or (Path(kwargs["base_dir"]) / kwargs["name"]),
                item_runs_dir_fn=lambda plan: base / "item_runs",
                sweep_base_dir_fn=lambda plan: base / "sweeps",
            ),
            sweep_plan,
        )
        assert [item["run"].label for item in sweep["items"]] == ["item0", "item1"]
        assert sweep["items"][0]["config"]["results_base"] == str(base / "item_runs")
        assert saved[0][1]["name"] == "demo_sweep"
        assert saved[0][1]["base_dir"] == base / "sweeps"

    print("neuroinfra notebook workflows: OK")


if __name__ == "__main__":
    main()
