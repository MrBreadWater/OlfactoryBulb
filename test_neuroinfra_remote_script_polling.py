"""Smoke tests for the extracted remote-safe polling helpers."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import neuroinfra.remote_script_polling as remote_script_polling


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _completed(stdout: str = "", *, stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["cmd"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    poll_module = _load_module(repo_root / "tools" / "remote" / "poll_sol_run.py", "remote_poll_sol_run_repo")

    assert remote_script_polling.normalize_state("running+") == "RUNNING"
    assert poll_module.normalize_state("failed+") == "FAILED"
    assert "COMPLETED" in remote_script_polling.TERMINAL_OK
    assert "FAILED" in poll_module.TERMINAL_FAIL

    pending = remote_script_polling.query_state(
        "12345",
        run_command_fn=lambda command: (
            _completed("PENDING|Priority\n") if command[:1] == ["squeue"] else _completed("")
        ),
    )
    assert pending == {"state": "PENDING", "reason": "Priority", "location": ""}

    running = remote_script_polling.query_state(
        "12345",
        run_command_fn=lambda command: (
            _completed("RUNNING|pcc080\n") if command[:1] == ["squeue"] else _completed("12345|RUNNING\n")
        ),
    )
    assert running == {"state": "RUNNING", "reason": "", "location": "pcc080"}

    with TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        result_dir = tmp / "result"
        result_dir.mkdir()
        wrapper_dir = result_dir.parent / ".obgpu-wrapper" / result_dir.name
        wrapper_dir.mkdir(parents=True)
        (result_dir / "sim_progress.json").write_text(json.dumps({"current_ms": 100, "total_ms": 200, "percent": 50.0}))
        (wrapper_dir / "stdout.txt").write_text("stdout payload\n")
        (wrapper_dir / "stderr.txt").write_text("stderr payload\n")
        (wrapper_dir / "bootstrap.log").write_text("bootstrap payload\n")
        (wrapper_dir / "command.txt").write_text("command payload\n")
        (wrapper_dir / "slurm-12345.out").write_text("slurm tail payload\n")

        in_progress_payload = remote_script_polling.poll_result_payload(
            job_id="12345",
            result_dir=result_dir,
            wrapper_dir=wrapper_dir,
            include_sacct=False,
            include_tails=True,
            run_command_fn=lambda command: _completed("RUNNING|pcc081\n") if command[:1] == ["squeue"] else _completed(""),
        )
        assert in_progress_payload["state"] == "RUNNING"
        assert in_progress_payload["done"] is False
        assert in_progress_payload["stdout_exists"] is True
        assert in_progress_payload["bootstrap_tail"] == "bootstrap payload\n"
        assert in_progress_payload["slurm_tail"] == "slurm tail payload\n"
        assert in_progress_payload["progress_current_ms"] == 100

        cleanup_calls: list[list[str]] = []
        cleanup_dirs: list[str] = []
        repo_dir = tmp / "repo"
        worktree_dir = tmp / "worktree"
        repo_dir.mkdir()
        worktree_dir.mkdir()
        (result_dir / "summary.json").write_text("{}")

        def _cleanup_run(command: list[str]) -> subprocess.CompletedProcess[str]:
            cleanup_calls.append(list(command))
            if command[:1] == ["git"]:
                return _completed("")
            if command[:1] == ["squeue"]:
                return _completed("RUNNING|pcc081\n")
            return _completed("12345|RUNNING\n")

        completed_payload = remote_script_polling.poll_result_payload(
            job_id="12345",
            result_dir=result_dir,
            wrapper_dir=wrapper_dir,
            repo_root=str(repo_dir),
            worktree_path=str(worktree_dir),
            include_sacct=True,
            include_tails=False,
            run_command_fn=_cleanup_run,
            remove_tree_fn=lambda path_text: cleanup_dirs.append(path_text),
        )
        assert completed_payload["done"] is True
        assert completed_payload["ok"] is True
        assert completed_payload["stdout_tail"] == ""
        assert completed_payload["cleanup"]["attempted"] is True
        assert any(command[:4] == ["git", "-C", str(repo_dir.resolve()), "worktree"] for command in cleanup_calls)
        assert any(command[:4] == ["git", "-C", str(repo_dir.resolve()), "worktree"] and "remove" in command for command in cleanup_calls)
        assert any(command[:4] == ["git", "-C", str(repo_dir.resolve()), "worktree"] and "prune" in command for command in cleanup_calls)
        assert cleanup_dirs == [str(worktree_dir.resolve())]

    with TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        (tmp / "neuroinfra").mkdir(parents=True, exist_ok=True)
        (tmp / "poll_sol_run.py").write_text((repo_root / "tools" / "remote" / "poll_sol_run.py").read_text())
        (tmp / "neuroinfra" / "__init__.py").write_text((repo_root / "neuroinfra" / "__init__.py").read_text())
        (tmp / "neuroinfra" / "inventory.py").write_text((repo_root / "neuroinfra" / "inventory.py").read_text())
        (tmp / "neuroinfra" / "remote_script_common.py").write_text(
            (repo_root / "neuroinfra" / "remote_script_common.py").read_text()
        )
        (tmp / "neuroinfra" / "remote_script_polling.py").write_text(
            (repo_root / "neuroinfra" / "remote_script_polling.py").read_text()
        )
        probe_path = tmp / "probe.py"
        probe_path.write_text(
            "import importlib.util, json\n"
            "from pathlib import Path\n"
            "spec = importlib.util.spec_from_file_location('bundle_poll', Path('poll_sol_run.py').resolve())\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            "payload = {\n"
            "  'state': module.normalize_state('running+'),\n"
            "  'completed_known': 'COMPLETED' in module.TERMINAL_OK,\n"
            "}\n"
            "print(json.dumps(payload, sort_keys=True))\n"
        )
        completed = subprocess.run(
            [sys.executable, str(probe_path)],
            cwd=tmp,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Helper-bundle poll_sol_run probe failed.\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        payload = json.loads((completed.stdout or "").strip())
        assert payload == {"completed_known": True, "state": "RUNNING"}

    print("neuroinfra remote script polling smoke test: OK")


if __name__ == "__main__":
    main()
