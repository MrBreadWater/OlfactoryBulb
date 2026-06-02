"""Focused tests for generic notebook run-info helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from neuroinfra.notebooks.run_info import (
    RunInfoHooks,
    build_run_info_payload,
    env_subset,
    load_run_info_payload,
    merge_run_info_payload,
    persist_run_info,
)


def main() -> None:
    hooks = RunInfoHooks(
        json_ready_fn=lambda value: value,
        build_overrides_fn=lambda config: {"paramset": config["paramset"]},
        resolve_execution_mode_fn=lambda config: {"backend": config.get("runner_backend", "local")},
        resolve_effective_params_fn=lambda config: {"full_param_snapshot": {"paramset": config["paramset"]}},
        env_keys=("OB_RUN_TIMESTAMP", "OB_RESULTS_BASE"),
    )

    completed = SimpleNamespace(returncode=0)
    payload = build_run_info_payload(
        hooks,
        config={"paramset": "GammaSignature"},
        label="demo",
        timestamp="2026-06-02T12-00-00",
        command=["python", "demo.py"],
        env={"OB_RUN_TIMESTAMP": "stamp", "OB_RESULTS_BASE": "/tmp/results", "OTHER": "ignore"},
        completed=completed,
        runner="demo.runner",
        summary={"label": "demo"},
        extra_payload={"remote": None},
        existing_payload={"previous": True},
    )
    assert payload["previous"] is True
    assert payload["env"] == {"OB_RUN_TIMESTAMP": "stamp", "OB_RESULTS_BASE": "/tmp/results"}
    assert payload["resolved_execution_mode"] == {"backend": "local"}
    assert payload["effective_params"]["full_param_snapshot"]["paramset"] == "GammaSignature"
    assert payload["remote"] is None

    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        persisted = persist_run_info(
            base,
            hooks,
            config={"paramset": "GammaSignature"},
            label="demo",
            timestamp="2026-06-02T12-00-00",
            command=["python", "demo.py"],
            env={"OB_RUN_TIMESTAMP": "stamp"},
            completed=completed,
            runner="demo.runner",
        )
        assert persisted.name == "run_info.json"
        loaded = load_run_info_payload(base)
        assert loaded["label"] == "demo"
        merge_run_info_payload(base, extra_payload={"artifact_sizes": {"lfp.pkl": 12}}, json_ready_fn=lambda value: value)
        merged = json.loads((base / "run_info.json").read_text())
        assert merged["artifact_sizes"]["lfp.pkl"] == 12

    failing_hooks = RunInfoHooks(
        json_ready_fn=lambda value: value,
        build_overrides_fn=lambda config: {},
        resolve_effective_params_fn=lambda config: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    failed_payload = build_run_info_payload(
        failing_hooks,
        config={"paramset": "GammaSignature"},
        label="demo",
        timestamp="2026-06-02T12-00-00",
        command=["python", "demo.py"],
        env={},
        completed=completed,
        runner="demo.runner",
    )
    assert failed_payload["effective_params_error"] == "RuntimeError: boom"

    assert env_subset({"A": 1, "B": 2}, ("B", "A")) == {"B": 2, "A": 1}

    print("neuroinfra notebook run info: OK")


if __name__ == "__main__":
    main()
