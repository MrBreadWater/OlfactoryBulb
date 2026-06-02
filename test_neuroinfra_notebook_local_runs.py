"""Focused tests for generic local notebook-run execution helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import subprocess

from neuroinfra.notebooks.local_runs import LocalRunHooks, execute_local_run


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        result_dir = base / "demo"
        writes = []

        def _run_success(command, **kwargs):
            cwd = Path(kwargs["cwd"])
            (cwd / "summary.json").write_text(json.dumps({"label": "demo"}))
            return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

        result = execute_local_run(
            config={"paramset": "GammaSignature"},
            label="demo",
            timestamp="2026-06-02T12-00-00",
            result_dir=result_dir,
            env={"OB_RESULTS_BASE": str(base)},
            command=["python", "demo.py"],
            runner_name="demo.runner",
            hooks=LocalRunHooks(
                read_summary_fn=lambda path: json.loads(Path(path).read_text()),
                write_run_info_fn=lambda *args, **kwargs: writes.append((args, kwargs)),
                build_return_value_fn=lambda **kwargs: kwargs,
                run_subprocess_fn=_run_success,
            ),
            success_extra_payload={"remote": None},
        )
        assert result["summary"] == {"label": "demo"}
        assert (result_dir / "command.txt").read_text().strip() == "python demo.py"
        assert (result_dir / "stdout.txt").read_text() == "ok\n"
        assert (result_dir / "stderr.txt").read_text() == ""
        assert writes[0][1]["runner"] == "demo.runner"
        assert writes[0][1]["summary"] == {"label": "demo"}
        assert writes[0][1]["extra_payload"] == {"remote": None}

    with tempfile.TemporaryDirectory() as tmp_dir:
        result_dir = Path(tmp_dir) / "failed"
        writes = []

        def _run_failure(command, **kwargs):
            return subprocess.CompletedProcess(command, 2, stdout="hello\n", stderr="bad things\n")

        try:
            execute_local_run(
                config={"paramset": "GammaSignature"},
                label="failed",
                timestamp="2026-06-02T12-00-00",
                result_dir=result_dir,
                env={},
                command=["python", "fail.py"],
                runner_name="demo.runner",
                hooks=LocalRunHooks(
                    read_summary_fn=lambda path: json.loads(Path(path).read_text()),
                    write_run_info_fn=lambda *args, **kwargs: writes.append((args, kwargs)),
                    build_return_value_fn=lambda **kwargs: kwargs,
                    run_subprocess_fn=_run_failure,
                ),
            )
            raise AssertionError("expected local run failure to raise")
        except RuntimeError as exc:
            assert "Simulation failed." in str(exc)
            assert "python fail.py" in str(exc)
        assert "summary" not in writes[0][1]
        assert writes[0][1]["completed"].returncode == 2

    with tempfile.TemporaryDirectory() as tmp_dir:
        result_dir = Path(tmp_dir) / "missing_summary"

        def _run_without_summary(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        try:
            execute_local_run(
                config={"paramset": "GammaSignature"},
                label="missing",
                timestamp="2026-06-02T12-00-00",
                result_dir=result_dir,
                env={},
                command=["python", "missing.py"],
                runner_name="demo.runner",
                hooks=LocalRunHooks(
                    read_summary_fn=lambda path: json.loads(Path(path).read_text()),
                    write_run_info_fn=lambda *args, **kwargs: None,
                    build_return_value_fn=lambda **kwargs: kwargs,
                    run_subprocess_fn=_run_without_summary,
                ),
            )
            raise AssertionError("expected missing summary to raise")
        except FileNotFoundError as exc:
            assert "summary.json" in str(exc)

    print("neuroinfra notebook local runs: OK")


if __name__ == "__main__":
    main()
