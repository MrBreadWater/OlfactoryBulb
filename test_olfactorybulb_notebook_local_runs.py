"""Focused tests for olfactory-bulb local notebook-run adapters."""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from neuroinfra.notebooks.local_runs import execute_local_run

from olfactorybulb.notebook_local_runs import (
    NotebookLocalRunHookBuilderHooks,
    LocalRunPayloadHooks,
    build_local_run_hooks,
    build_local_run_payload,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        overrides_written = []

        def _write_overrides(path, payload):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            overrides_written.append(path)

        payload = build_local_run_payload(
            LocalRunPayloadHooks(
                benchmark_param_overrides_payload_fn=lambda config: (
                    {"gaba_tau2_ms": config["gaba_tau2_ms"]},
                    None,
                ),
                write_benchmark_overrides_file_fn=_write_overrides,
                build_run_command_fn=lambda config, label, **kwargs: [
                    "python",
                    "demo.py",
                    str(kwargs["overrides_file"]),
                ],
            ),
            {
                "paramset": "GammaSignature",
                "gaba_tau2_ms": 36.0,
                "results_base": str(tmp / "results"),
                "cell_permute": 5,
            },
            label="demo_local",
            timestamp="2026-06-02T12-00-00",
            repo_root="/repo/OlfactoryBulb",
            default_results_base="/repo/results",
        )
        assert payload.result_dir == tmp / "results" / "demo_local"
        assert payload.command[0] == "python"
        assert payload.command[2].endswith("/.obgpu-wrapper/demo_local/overrides.json")
        assert payload.env["PYTHONPATH"].startswith("/repo/OlfactoryBulb")
        assert payload.env["OB_RUN_TIMESTAMP"] == "2026-06-02T12-00-00"
        assert payload.env["OB_RESULT_LABEL"] == "demo_local"
        assert payload.env["OB_RESULTS_BASE"] == str(tmp / "results")
        assert payload.env["OB_CORENRN_CELL_PERMUTE"] == "5"
        assert overrides_written and json.loads(overrides_written[0].read_text()) == {"gaba_tau2_ms": 36.0}

        run_info_calls = []
        local_hooks = build_local_run_hooks(
            NotebookLocalRunHookBuilderHooks(
                read_summary_fn=lambda path: json.loads(Path(path).read_text()),
                write_run_info_fn=lambda *args, **kwargs: run_info_calls.append((args, kwargs)),
                build_param_overrides_fn=lambda config: {"gaba_tau2_ms": config["gaba_tau2_ms"]},
                run_record_factory_fn=lambda **kwargs: SimpleNamespace(**kwargs),
            )
        )

        result_dir = tmp / "results" / "local_run"

        def _run_subprocess(command, *, cwd, env, capture_output, text, check):
            Path(cwd).mkdir(parents=True, exist_ok=True)
            (Path(cwd) / "summary.json").write_text(json.dumps({"label": "local_run", "ok": True}))
            return SimpleNamespace(returncode=0, stdout="stdout\n", stderr="")

        delegated = execute_local_run(
            config={"paramset": "GammaSignature", "gaba_tau2_ms": 36.0},
            label="local_run",
            timestamp="2026-06-02T12-00-00",
            result_dir=result_dir,
            env={"PYTHONPATH": "/repo/OlfactoryBulb"},
            command=["python", "demo.py"],
            runner_name="demo.runner",
            hooks=replace(local_hooks, run_subprocess_fn=_run_subprocess),
            success_extra_payload={"remote": None},
        )
        assert delegated.label == "local_run"
        assert delegated.overrides == {"gaba_tau2_ms": 36.0}
        assert delegated.stdout == "stdout\n"
        assert run_info_calls and run_info_calls[0][1]["runner"] == "demo.runner"

    print("olfactorybulb notebook local runs: OK")


if __name__ == "__main__":
    main()
