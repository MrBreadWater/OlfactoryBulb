"""Smoke tests for the extracted remote-safe Slurm script common helpers."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import neuroinfra.remote_script_common as remote_script_common


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    slurm_common_module = _load_module(repo_root / "tools" / "remote" / "slurm_common.py", "remote_slurm_common_repo")

    args = SimpleNamespace(
        partition="debug",
        account="lab",
        time="01:00:00",
        gpus=1,
        cpus_per_task=8,
        mem="64G",
        sbatch_arg=["--constraint", "cascadelake", "--qos=fast"],
    )

    expected_directives = [
        "#SBATCH --job-name=test_job",
        "#SBATCH --partition=debug",
        "#SBATCH --account=lab",
        "#SBATCH --time=01:00:00",
        "#SBATCH --gpus=1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=64G",
        "#SBATCH --constraint cascadelake",
        "#SBATCH --qos=fast",
    ]
    assert remote_script_common.slurm_directives(args, "test_job") == expected_directives
    assert slurm_common_module.slurm_directives(args, "test_job") == expected_directives

    assert remote_script_common.shell_join(["python", "a b.py"]) == "python 'a b.py'"
    assert slurm_common_module.shell_join(["python", "a b.py"]) == "python 'a b.py'"
    assert remote_script_common.path_is_within("/root/a/b", "/root/a")
    assert not remote_script_common.path_is_within("/root/ab", "/root/a")
    assert remote_script_common.normalize_sbatch_args(["--constraint", "cascadelake", "--qos=fast"]) == [
        "--constraint cascadelake",
        "--qos=fast",
    ]
    assert remote_script_common.requested_mpi_rank_count(["srun", "-n", "16", "nrniv"]) == 16
    assert slurm_common_module.requested_mpi_rank_count(["mpiexec", "--ntasks=8", "nrniv"]) == 8

    with TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        (tmp / "neuroinfra").mkdir(parents=True, exist_ok=True)
        (tmp / "slurm_common.py").write_text((repo_root / "tools" / "remote" / "slurm_common.py").read_text())
        (tmp / "neuroinfra" / "__init__.py").write_text((repo_root / "neuroinfra" / "__init__.py").read_text())
        (tmp / "neuroinfra" / "inventory.py").write_text((repo_root / "neuroinfra" / "inventory.py").read_text())
        (tmp / "neuroinfra" / "remote_script_common.py").write_text(
            (repo_root / "neuroinfra" / "remote_script_common.py").read_text()
        )
        probe_path = tmp / "probe.py"
        probe_path.write_text(
            "import importlib.util, json\n"
            "from pathlib import Path\n"
            "spec = importlib.util.spec_from_file_location('bundle_slurm_common', Path('slurm_common.py').resolve())\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            "payload = {\n"
            "  'shell_join': module.shell_join(['python', 'a b.py']),\n"
            "  'rank_count': module.requested_mpi_rank_count(['srun', '-n', '12', 'nrniv']),\n"
            "  'within': module.path_is_within('/root/a/b', '/root/a'),\n"
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
                "Helper-bundle slurm_common probe failed.\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        payload = json.loads((completed.stdout or "").strip())
        assert payload == {
            "rank_count": 12,
            "shell_join": "python 'a b.py'",
            "within": True,
        }

    print("neuroinfra remote script common smoke test: OK")


if __name__ == "__main__":
    main()
