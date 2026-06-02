"""Focused tests for olfactory-bulb notebook run-info helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from olfactorybulb.notebook_run_info import NotebookRunInfoHooks, merge_extra_run_info, write_run_info


def main() -> None:
    hooks = NotebookRunInfoHooks(
        json_ready_fn=lambda value: value,
        build_param_overrides_fn=lambda config: {"paramset": config["paramset"], "gaba_tau2_ms": config["gaba_tau2_ms"]},
        resolve_execution_mode_fn=lambda config: {"backend": config.get("runner_backend", "local")},
        resolve_effective_params_fn=lambda config: {"full_param_snapshot": {"gaba_tau2_ms": config["gaba_tau2_ms"]}},
    )
    completed = SimpleNamespace(returncode=0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        write_run_info(
            hooks,
            base,
            config={"paramset": "GammaSignature", "gaba_tau2_ms": 36.0},
            label="demo",
            timestamp="2026-06-02T12-00-00",
            command=["python", "demo.py"],
            env={"OB_RUN_TIMESTAMP": "stamp", "OB_RESULTS_BASE": "/tmp/results"},
            completed=completed,
            runner="demo.runner",
            summary={"label": "demo"},
            extra_payload={"remote": None},
        )
        payload = json.loads((base / "run_info.json").read_text())
        assert payload["overrides"]["gaba_tau2_ms"] == 36.0
        assert payload["resolved_execution_mode"] == {"backend": "local"}
        assert payload["effective_params"]["full_param_snapshot"]["gaba_tau2_ms"] == 36.0
        assert payload["env"]["OB_RUN_TIMESTAMP"] == "stamp"

        merge_extra_run_info(hooks, base, extra_payload={"artifact_sizes": {"lfp.pkl": 42}})
        merged = json.loads((base / "run_info.json").read_text())
        assert merged["artifact_sizes"]["lfp.pkl"] == 42

    print("olfactorybulb notebook run info: OK")


if __name__ == "__main__":
    main()
