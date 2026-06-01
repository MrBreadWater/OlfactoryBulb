"""Smoke tests for the extracted remote-safe submit helpers."""

from __future__ import annotations

import base64
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import neuroinfra.remote_script_submit as remote_script_submit


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
    submit_module = _load_module(repo_root / "tools" / "remote" / "submit_sol_run.py", "remote_submit_sol_run_repo")

    payload_b64 = base64.b64encode(json.dumps(["nrniv", "-python", "bench.py"]).encode("utf-8")).decode("ascii")
    assert remote_script_submit.decode_command(payload_b64) == ["nrniv", "-python", "bench.py"]
    assert submit_module.decode_command(payload_b64) == ["nrniv", "-python", "bench.py"]

    relocated = remote_script_submit.relocate_benchmark_command(
        [
            "/repo/OlfactoryBulb/tools/bench.py",
            "/results/notebook_runs/keep.json",
            "--flag",
        ],
        repo_root=Path("/repo/OlfactoryBulb"),
        worktree_root=Path("/repo/.obgpu-worktrees/run1"),
        preserved_roots=[Path("/results/notebook_runs")],
    )
    assert relocated == [
        "/repo/.obgpu-worktrees/run1/tools/bench.py",
        "/results/notebook_runs/keep.json",
        "--flag",
    ]

    preflight = remote_script_submit.neuron_mpi_preflight_suffix(
        ["nrniv", "-python", "bench.py"]
    )
    assert preflight is not None
    assert preflight[:2] == ["nrniv", "-mpi"]
    assert "OBGPU_EXPECTED_NRANKS" in preflight[-1]
    assert submit_module.neuron_mpi_preflight_suffix(["nrniv", "-python", "bench.py"]) == preflight

    with TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        repo_root_path = tmp / "repo"
        repo_root_path.mkdir()
        mechanisms_dir = repo_root_path / "prev_ob_models" / "Birgiolas2020" / "Mechanisms"
        mechanisms_dir.mkdir(parents=True)
        (mechanisms_dir / "Na.mod").write_text("NEURON {}\n")
        result_dir = tmp / "results" / "run1"
        worktree_root = tmp / "worktrees" / "run1"
        args = SimpleNamespace(
            partition="debug",
            account="lab",
            time="00:30:00",
            gpus=1,
            cpus_per_task=8,
            mem="64G",
            sbatch_arg=["--constraint=cascadelake"],
            heartbeat_timeout_s=120,
        )
        batch_path = remote_script_submit.write_batch_script(
            repo_root=repo_root_path,
            result_dir=result_dir,
            label="run1",
            repo_mode="shared",
            worktree_root=worktree_root,
            conda_activate_cmd="source activate OBGPU",
            runtime_profiles_b64="",
            fallback_conda_activate_cmd="",
            fast_node_feature="",
            mechanism_profile="default",
            fallback_mechanism_profile="portable",
            benchmark_command=["srun", "-n", "4", "nrniv", "-mpi", "-python", "bench.py"],
            mpi_exec="srun",
            git_ref="deadbeef",
            git_fetch=True,
            git_remote="origin",
            args=args,
        )
        batch_source = batch_path.read_text()
        assert "selected_conda_activate_cmd" in batch_source
        assert "OBGPU_EXPECTED_NRANKS=4" in batch_source
        assert "NEURON MPI preflight" in batch_source

        with patch.object(remote_script_submit.subprocess, "run", return_value=_completed("12345;cluster\n")):
            assert remote_script_submit.submit_batch(batch_path) == "12345"
            assert submit_module.submit_batch(batch_path) == "12345"

        wrapper_dir = result_dir.parent / ".obgpu-wrapper" / "run1"

        def _fake_step_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
            assert command[:2] == ["bash", "-lc"]
            shell_script = command[2]
            assert '--nodes=1 --ntasks="$step_ntasks"' in shell_script
            assert 'step_ntasks=5' in shell_script
            return _completed("14537854.7\n")

        with patch.object(remote_script_submit.subprocess, "run", side_effect=_fake_step_run):
            assert remote_script_submit.submit_allocation_step(batch_path, "14537854", wrapper_dir, step_ntasks=5) == "14537854.7"
            assert submit_module.submit_allocation_step(batch_path, "14537854", wrapper_dir, step_ntasks=5) == "14537854.7"

    with TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        (tmp / "neuroinfra").mkdir(parents=True, exist_ok=True)
        (tmp / "submit_sol_run.py").write_text((repo_root / "tools" / "remote" / "submit_sol_run.py").read_text())
        for relative in (
            "__init__.py",
            "inventory.py",
            "remote_script_common.py",
            "remote_script_submit.py",
        ):
            (tmp / "neuroinfra" / relative).write_text((repo_root / "neuroinfra" / relative).read_text())
        probe_path = tmp / "probe.py"
        probe_path.write_text(
            "import importlib.util, json\n"
            "from pathlib import Path\n"
            "spec = importlib.util.spec_from_file_location('bundle_submit', Path('submit_sol_run.py').resolve())\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            "payload = {\n"
            "  'has_write_batch_script': hasattr(module, 'write_batch_script'),\n"
            "  'decode_ok': module.decode_command('WyJucm5pdiJd') == ['nrniv'],\n"
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
                "Helper-bundle submit_sol_run probe failed.\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        payload = json.loads((completed.stdout or "").strip())
        assert payload == {"decode_ok": True, "has_write_batch_script": True}

    print("neuroinfra remote script submit smoke test: OK")


if __name__ == "__main__":
    main()
