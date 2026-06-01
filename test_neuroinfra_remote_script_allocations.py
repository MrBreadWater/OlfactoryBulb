"""Smoke tests for the extracted remote-safe allocation lifecycle helpers."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import neuroinfra.remote_script_allocations as remote_script_allocations


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
    submit_module = _load_module(
        repo_root / "tools" / "remote" / "submit_slurm_allocation.py",
        "remote_submit_slurm_allocation_repo",
    )
    cleanup_module = _load_module(
        repo_root / "tools" / "remote" / "cleanup_stale_allocations.py",
        "remote_cleanup_allocations_repo",
    )

    args = SimpleNamespace(
        name="alloc_job",
        partition="debug",
        account="lab",
        time="01:00:00",
        gpus=1,
        cpus_per_task=8,
        mem="64G",
        heartbeat_timeout_s=120,
        sbatch_arg=["--constraint", "cascadelake"],
    )

    with TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        lines = remote_script_allocations.holder_script_lines(args, tmp)
        assert lines[0] == "#!/usr/bin/env bash"
        assert any("#SBATCH --job-name=alloc_job" == line for line in lines)
        assert any("lease-expired.txt" in line for line in lines)

        script_path, slurm_log_path, heartbeat_path = remote_script_allocations.write_holder_script(args, tmp)
        assert script_path.exists()
        assert heartbeat_path.exists()
        assert script_path.read_text().startswith("#!/usr/bin/env bash\n")
        assert slurm_log_path.name == "allocation-%j.out"

        payload = remote_script_allocations.allocation_payload(
            job_id="12345",
            name="alloc_job",
            allocation_root=tmp,
            batch_script=script_path,
            heartbeat_path=heartbeat_path,
            heartbeat_timeout_s=120,
            slurm_log_pattern=slurm_log_path,
        )
        assert payload["job_id"] == "12345"
        assert payload["heartbeat_timeout_s"] == 120
        assert payload["allocation_root"] == str(tmp.resolve())

        parsed_job_id = remote_script_allocations.submit_batch(
            script_path,
            run_command_fn=lambda command: _completed("12345;cluster\n"),
        )
        assert parsed_job_id == "12345"
        parsed_job_id_wrapper = submit_module.submit_batch(
            script_path,
            run_command_fn=lambda command: _completed("67890\n"),
        )
        assert parsed_job_id_wrapper == "67890"

        now_s = time.time()
        heartbeat_path.write_text("")
        assert remote_script_allocations.determine_stale_reason(
            {"heartbeat_path": "", "heartbeat_timeout_s": 120},
            default_timeout_s=120,
            now_s=now_s,
        ) == "legacy_no_heartbeat"
        assert remote_script_allocations.determine_stale_reason(
            {"heartbeat_path": str(tmp / "missing.txt"), "heartbeat_timeout_s": 120},
            default_timeout_s=120,
            now_s=now_s,
        ) == "missing_heartbeat"
        heartbeat_path.touch()
        old_time = now_s - 300
        os.utime(heartbeat_path, (old_time, old_time))
        assert remote_script_allocations.determine_stale_reason(
            {"heartbeat_path": str(heartbeat_path), "heartbeat_timeout_s": 120},
            default_timeout_s=120,
            now_s=now_s,
        ) == "expired_heartbeat"
        os.utime(heartbeat_path, None)
        assert remote_script_allocations.determine_stale_reason(
            {"heartbeat_path": str(heartbeat_path), "heartbeat_timeout_s": 120},
            default_timeout_s=120,
            now_s=time.time(),
        ) == ""

        root = tmp / "allocations"
        fresh_dir = root / "fresh"
        stale_dir = root / "stale"
        invalid_dir = root / "invalid"
        fresh_dir.mkdir(parents=True)
        stale_dir.mkdir(parents=True)
        invalid_dir.mkdir(parents=True)
        fresh_heartbeat = fresh_dir / "heartbeat.txt"
        stale_heartbeat = stale_dir / "heartbeat.txt"
        fresh_heartbeat.write_text("")
        stale_heartbeat.write_text("")
        os.utime(stale_heartbeat, (old_time, old_time))
        (fresh_dir / "allocation.json").write_text(
            json.dumps({"job_id": "111", "heartbeat_path": str(fresh_heartbeat), "heartbeat_timeout_s": 120})
        )
        (stale_dir / "allocation.json").write_text(
            json.dumps({"job_id": "222", "heartbeat_path": str(stale_heartbeat), "heartbeat_timeout_s": 120})
        )
        (invalid_dir / "allocation.json").write_text("{broken")

        actions = remote_script_allocations.stale_allocation_actions(
            root,
            default_timeout_s=120,
            now_s=now_s,
            cancel_job_fn=lambda job_id: _completed("", stderr=f"cancel {job_id}", returncode=0),
        )
        assert any(
            action.get("job_id") == "222" and action["reason"] == "expired_heartbeat"
            for action in actions
        )
        assert any(action["reason"] == "invalid_json" for action in actions)
        assert not any(action.get("job_id") == "111" for action in actions)
        wrapper_actions = cleanup_module.stale_allocation_actions(
            root,
            default_timeout_s=120,
            now_s=now_s,
            cancel_job_fn=lambda job_id: _completed("", stderr=f"cancel {job_id}", returncode=0),
        )
        assert wrapper_actions == actions

    with TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        (tmp / "neuroinfra").mkdir(parents=True, exist_ok=True)
        for relative in (
            "submit_slurm_allocation.py",
            "cleanup_stale_allocations.py",
            "slurm_common.py",
        ):
            (tmp / relative).write_text((repo_root / "tools" / "remote" / relative).read_text())
        for relative in (
            "__init__.py",
            "inventory.py",
            "remote_script_common.py",
            "remote_script_allocations.py",
        ):
            (tmp / "neuroinfra" / relative).write_text((repo_root / "neuroinfra" / relative).read_text())
        probe_path = tmp / "probe.py"
        probe_path.write_text(
            "import importlib.util, json\n"
            "from pathlib import Path\n"
            "def load(name, path):\n"
            "    spec = importlib.util.spec_from_file_location(name, path)\n"
            "    module = importlib.util.module_from_spec(spec)\n"
            "    spec.loader.exec_module(module)\n"
            "    return module\n"
            "submit = load('bundle_submit', Path('submit_slurm_allocation.py').resolve())\n"
            "cleanup = load('bundle_cleanup', Path('cleanup_stale_allocations.py').resolve())\n"
            "payload = {\n"
            "  'submit_has': hasattr(submit, 'write_holder_script'),\n"
            "  'cleanup_has': hasattr(cleanup, 'stale_allocation_actions'),\n"
            "  'legacy_reason': cleanup.determine_stale_reason({'heartbeat_path': ''}, default_timeout_s=120, now_s=0.0),\n"
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
                "Helper-bundle allocation wrapper probe failed.\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        payload = json.loads((completed.stdout or "").strip())
        assert payload == {
            "cleanup_has": True,
            "legacy_reason": "legacy_no_heartbeat",
            "submit_has": True,
        }

    print("neuroinfra remote script allocations smoke test: OK")


if __name__ == "__main__":
    main()
