"""Smoke tests for the extracted remote-safe sweep runner helpers."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import neuroinfra.remote_script_sweeps as remote_script_sweeps


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    sweep_module = _load_module(
        repo_root / "tools" / "remote" / "remote_sweep_driver.py",
        "remote_sweep_driver_repo",
    )

    items = remote_script_sweeps.normalize_items(
        [
            {
                "index": 0,
                "label": "item_000",
                "value": 1.0,
                "result_dir": "/remote/item_000",
                "command": ["python", "bench.py"],
                "overrides_file": "",
                "overrides": {"a": 1},
            }
        ]
    )
    assert items[0]["overrides_file"] is None
    assert items[0]["command"] == ["python", "bench.py"]
    assert sweep_module.normalize_items(
        [
            {
                "index": 1,
                "label": "item_001",
                "value": 2.0,
                "result_dir": "/remote/item_001",
                "command": ["python", "bench.py"],
            }
        ]
    )[0]["index"] == 1

    assert remote_script_sweeps.relocate_repo_paths(
        ["/shared/repo/a.py", "/elsewhere/b.py"],
        shared_repo_root="/shared/repo",
        repo_root="/worktree/repo",
    ) == ["/worktree/repo/a.py", "/elsewhere/b.py"]

    assert remote_script_sweeps.add_srun_parallel_step_flags(
        ["srun", "--mpi=pmix_v4", "-n", "15", "nrniv"]
    ) == ["srun", "--exclusive", "--exact", "--mpi=pmix_v4", "-n", "15", "nrniv"]
    assert sweep_module.add_srun_parallel_step_flags(
        ["srun", "--exclusive", "--exact", "-n", "15", "nrniv"]
    ) == ["srun", "--exclusive", "--exact", "-n", "15", "nrniv"]

    with TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        requested_dir = tmp / "requested"
        requested_dir.mkdir(parents=True, exist_ok=True)
        payload_dir = tmp / "requested_20260525_120000"
        payload_dir.mkdir(parents=True, exist_ok=True)
        (payload_dir / "summary.json").write_text(
            json.dumps(
                {
                    "label": payload_dir.name,
                    "requested_label": requested_dir.name,
                    "timestamp": "20260525_120000",
                }
            )
        )
        assert remote_script_sweeps.resolve_completed_result_dir(requested_dir, requested_dir.name) == payload_dir

        progress = remote_script_sweeps.progress_payload(
            sweep_label="sweep",
            total_items=2,
            pending_items=[{"label": "item_001"}],
            running_items=[{"label": "item_000", "result_dir": "/remote/item_000"}],
            finished_items=[{"label": "item_002", "ok": True}],
        )
        assert progress["pending_labels"] == ["item_001"]
        assert progress["completed_labels"] == ["item_002"]
        assert progress["failed_labels"] == []

    with TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        (tmp / "neuroinfra").mkdir(parents=True, exist_ok=True)
        (tmp / "remote_sweep_driver.py").write_text((repo_root / "tools" / "remote" / "remote_sweep_driver.py").read_text())
        for relative in ("__init__.py", "inventory.py", "remote_script_common.py", "remote_script_sweeps.py"):
            (tmp / "neuroinfra" / relative).write_text((repo_root / "neuroinfra" / relative).read_text())
        probe_path = tmp / "probe.py"
        probe_path.write_text(
            "import importlib.util, json\n"
            "from pathlib import Path\n"
            "spec = importlib.util.spec_from_file_location('bundle_sweep', Path('remote_sweep_driver.py').resolve())\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            "payload = {\n"
            "  'flags': module.add_srun_parallel_step_flags(['srun', '-n', '4', 'nrniv']),\n"
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
                "Remote sweep wrapper probe failed.\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        payload = json.loads((completed.stdout or "").strip())
        assert payload == {"flags": ["srun", "--exclusive", "--exact", "-n", "4", "nrniv"]}

    print("neuroinfra remote script sweeps smoke test: OK")


if __name__ == "__main__":
    main()
