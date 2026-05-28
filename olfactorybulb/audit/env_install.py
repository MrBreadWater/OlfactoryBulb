"""Audit whether the local machine has a usable OBGPU install."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import importlib
import importlib.util
import io
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Iterable
import warnings

from olfactorybulb.audit.core import AuditItem, AuditReport, collect_items


REPO_ROOT = Path(__file__).resolve().parents[2]
MECHANISM_SOURCE_DIR = REPO_ROOT / "prev_ob_models" / "Birgiolas2020" / "Mechanisms"
VERIFY_IMPORTS_SCRIPT = REPO_ROOT / "tools" / "setup" / "verify_obgpu_python_imports.py"

REQUIRED_REPO_FILES = [
    "install-obgpu.sh",
    "tools/setup/setup_ob_modern.sh",
    "tools/setup/activate_obgpu.sh",
    "tools/setup/activate_sol_obgpu.sh",
    "tools/setup/verify_obgpu_python_imports.py",
    "tools/benchmarks/benchmark_ob.py",
    "environments/environment-modern.yml",
    "third_party_patches/nrn/manifest.json",
    "prev_ob_models/Birgiolas2020/Mechanisms",
]

REQUIRED_COMMANDS = ["nrniv", "nrnivmodl"]
OPTIONAL_GPU_COMMANDS = ["nvc", "nvc++", "nvcc"]


def _status(ok: bool, failure_status: str = "FAIL") -> str:
    return "PASS" if ok else failure_status


def _resolved_path(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return str(Path(value).expanduser().resolve())
    except OSError:
        return value


def _path_contains(raw_path: str | None, expected: Path) -> bool:
    if not raw_path:
        return False
    expected_text = str(expected.resolve())
    return any(_resolved_path(entry) == expected_text for entry in raw_path.split(os.pathsep) if entry)


def _first_command(command: str | None) -> str | None:
    if not command:
        return None
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    return parts[0] if parts else None


def _command_paths(commands: Iterable[str]) -> dict[str, str | None]:
    return {command: shutil.which(command) for command in commands}


def _import_module(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        importlib.import_module(name)


def _import_module_from_path(name: str, path: Path) -> None:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(module)


def _call_with_captured_output(callback: Callable[[], None]) -> tuple[Exception | None, str, str]:
    stdout_text = io.StringIO()
    stderr_text = io.StringIO()
    exc: Exception | None = None
    sys.stdout.flush()
    sys.stderr.flush()
    original_stdout_fd = os.dup(1)
    original_stderr_fd = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as stdout_fd, tempfile.TemporaryFile(mode="w+b") as stderr_fd:
        try:
            os.dup2(stdout_fd.fileno(), 1)
            os.dup2(stderr_fd.fileno(), 2)
            with redirect_stdout(stdout_text), redirect_stderr(stderr_text):
                callback()
        except Exception as caught:
            exc = caught
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)

        stdout_fd.seek(0)
        stderr_fd.seek(0)
        stdout = stdout_text.getvalue() + stdout_fd.read().decode(errors="replace")
        stderr = stderr_text.getvalue() + stderr_fd.read().decode(errors="replace")

    return exc, stdout, stderr


def _verify_import_lists() -> tuple[list[str], list[str]]:
    spec = importlib.util.spec_from_file_location("verify_obgpu_python_imports", VERIFY_IMPORTS_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {VERIFY_IMPORTS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.THIRD_PARTY_IMPORTS), list(module.REPO_IMPORTS)


@contextmanager
def _temporarily_unset(name: str):
    marker = object()
    previous = os.environ.pop(name, marker)
    try:
        yield
    finally:
        if previous is not marker:
            os.environ[name] = str(previous)


def audit_repo_layout() -> list[AuditItem]:
    missing = [relative for relative in REQUIRED_REPO_FILES if not (REPO_ROOT / relative).exists()]
    cwd_matches = Path.cwd().resolve() == REPO_ROOT.resolve()
    return [
        AuditItem(
            check_id="repo_layout",
            status=_status(not missing),
            title="Maintained OBGPU install files are present",
            criterion="A machine install should use the maintained setup, activation, environment, patch, mechanism, and benchmark files.",
            evidence={
                "repo_root": str(REPO_ROOT),
                "missing": missing,
                "checked": REQUIRED_REPO_FILES,
            },
        ),
        AuditItem(
            check_id="repo_root_cwd",
            status=_status(cwd_matches, "WARN"),
            title="Audit is running from the repository root",
            criterion="The maintained workflow expects commands to run from the repository root.",
            evidence={"cwd": str(Path.cwd()), "repo_root": str(REPO_ROOT)},
        ),
    ]


def audit_python_environment() -> list[AuditItem]:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    env_name = conda_env or Path(sys.prefix).name
    env_ok = bool(env_name and env_name.startswith("OBGPU"))
    return [
        AuditItem(
            check_id="python_environment",
            status=_status(env_ok),
            title="Python is running inside an OBGPU conda environment",
            criterion="The maintained install should run from an OBGPU or OBGPU-portable conda environment, not base/system Python.",
            evidence={
                "python_executable": sys.executable,
                "sys_prefix": sys.prefix,
                "conda_default_env": conda_env,
                "conda_prefix": os.environ.get("CONDA_PREFIX"),
            },
        )
    ]


def audit_activation_hooks() -> list[AuditItem]:
    repo_root_env = _resolved_path(os.environ.get("OBGPU_SHARED_REPO_ROOT"))
    mechanism_root = Path(os.environ.get("OBGPU_MECHANISM_ROOT") or REPO_ROOT).expanduser().resolve()
    arch_dir = mechanism_root / platform.machine()
    coreneuronlib = _resolved_path(os.environ.get("CORENEURONLIB"))
    expected_corenrn = arch_dir / "libcorenrnmech.so"
    conda_prefix = os.environ.get("CONDA_PREFIX")

    repo_root_ok = repo_root_env == str(REPO_ROOT.resolve())
    mechanism_root_ok = mechanism_root.exists()
    corenrn_ok = coreneuronlib == str(expected_corenrn.resolve()) if expected_corenrn.exists() else coreneuronlib is None
    ld_path = os.environ.get("LD_LIBRARY_PATH")
    ld_has_arch = _path_contains(ld_path, arch_dir)
    ld_has_conda = _path_contains(ld_path, Path(conda_prefix) / "lib") if conda_prefix else False

    return [
        AuditItem(
            check_id="activation_runtime_hooks",
            status=_status(repo_root_ok and mechanism_root_ok and corenrn_ok and ld_has_arch and ld_has_conda),
            title="Repo activation hooks are active",
            criterion="source tools/setup/activate_obgpu.sh should export repo, mechanism, CoreNEURON, and library-path settings.",
            evidence={
                "OBGPU_SHARED_REPO_ROOT": os.environ.get("OBGPU_SHARED_REPO_ROOT"),
                "OBGPU_MECHANISM_ROOT": os.environ.get("OBGPU_MECHANISM_ROOT"),
                "CORENEURONLIB": os.environ.get("CORENEURONLIB"),
                "expected_arch_dir": str(arch_dir),
                "ld_library_path_has_arch_dir": ld_has_arch,
                "ld_library_path_has_conda_lib": ld_has_conda,
            },
        )
    ]


def audit_command_line_tools(*, require_gpu: bool = False) -> list[AuditItem]:
    command_paths = _command_paths(REQUIRED_COMMANDS)
    mpi_exec = os.environ.get("OB_MPIEXEC")
    mpi_command = _first_command(mpi_exec)
    mpi_command_path = shutil.which(mpi_command) if mpi_command else None
    fallback_mpi_paths = _command_paths(["mpiexec", "srun"])
    mpi_ok = bool(mpi_command_path or any(fallback_mpi_paths.values()))
    required_ok = all(command_paths.values())

    items = [
        AuditItem(
            check_id="command_line_tools",
            status=_status(required_ok and mpi_ok),
            title="NEURON and MPI launcher commands are available",
            criterion="A runnable machine install needs nrniv, nrnivmodl, and either OB_MPIEXEC, mpiexec, or srun on PATH.",
            evidence={
                "required": command_paths,
                "OB_MPIEXEC": mpi_exec,
                "OB_MPIEXEC_command_path": mpi_command_path,
                "fallback_mpi_launchers": fallback_mpi_paths,
            },
        )
    ]

    if require_gpu:
        gpu_paths = _command_paths(OPTIONAL_GPU_COMMANDS)
        items.append(
            AuditItem(
                check_id="gpu_build_tools",
                status=_status(all(gpu_paths.values())),
                title="GPU build tools are available",
                criterion="GPU-enabled OBGPU installs require NVIDIA HPC SDK compilers and nvcc.",
                evidence=gpu_paths,
            )
        )

    return items


def audit_mechanism_outputs() -> list[AuditItem]:
    mechanism_root = Path(os.environ.get("OBGPU_MECHANISM_ROOT") or REPO_ROOT).expanduser().resolve()
    arch_dir = mechanism_root / platform.machine()
    source_mods = sorted(path.stem for path in MECHANISM_SOURCE_DIR.glob("*.mod"))
    compiled_cpp = sorted(path.stem for path in arch_dir.glob("*.cpp"))
    missing_compiled = sorted(set(source_mods) - set(compiled_cpp))
    required_outputs = {
        "special": arch_dir / "special",
        "libnrnmech": arch_dir / "libnrnmech.so",
        "libcorenrnmech": arch_dir / "libcorenrnmech.so",
    }
    output_presence = {name: path.exists() for name, path in required_outputs.items()}

    return [
        AuditItem(
            check_id="mechanism_build_outputs",
            status=_status(bool(source_mods) and all(output_presence.values()) and not missing_compiled),
            title="Birgiolas mechanism outputs are compiled for this architecture",
            criterion="The active mechanism root should contain special, libnrnmech.so, libcorenrnmech.so, and generated code for every maintained .mod file.",
            evidence={
                "mechanism_source_dir": str(MECHANISM_SOURCE_DIR),
                "mechanism_root": str(mechanism_root),
                "arch": platform.machine(),
                "arch_dir": str(arch_dir),
                "source_mod_count": len(source_mods),
                "missing_compiled_mechanisms": missing_compiled,
                "required_outputs": {name: str(path) for name, path in required_outputs.items()},
                "output_presence": output_presence,
            },
        )
    ]


def audit_legacy_nmodl_path() -> list[AuditItem]:
    value = os.environ.get("NRN_NMODL_PATH")
    return [
        AuditItem(
            check_id="legacy_nrn_nmodl_path",
            status=_status(not value, "WARN"),
            title="Legacy NRN_NMODL_PATH is not forcing stale mechanism autoloads",
            criterion="The maintained runtime should not rely on NRN_NMODL_PATH auto-loading repo-root mechanisms.",
            evidence={"NRN_NMODL_PATH": value},
            note="Older activation hooks used this variable; it can collide with the explicit Birgiolas mechanism load path.",
        )
    ]


def audit_python_imports(*, skip_imports: bool = False) -> list[AuditItem]:
    if skip_imports:
        return [
            AuditItem(
                check_id="python_import_surface_skipped",
                status="WARN",
                title="Maintained Python import surface check skipped",
                criterion="Run without --skip-imports to verify third-party and repo imports.",
            )
        ]

    failures: list[dict[str, str]] = []
    import_messages: list[dict[str, str]] = []
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    try:
        third_party_imports, repo_imports = _verify_import_lists()
    except Exception as exc:
        return [
            AuditItem(
                check_id="python_import_surface",
                status="FAIL",
                title="Maintained Python import list could not be loaded",
                criterion="tools/setup/verify_obgpu_python_imports.py should be importable and define the maintained import surface.",
                evidence={"error": repr(exc), "script": str(VERIFY_IMPORTS_SCRIPT)},
            )
        ]

    with _temporarily_unset("NRN_NMODL_PATH"):
        for name in third_party_imports:
            exc, stdout, stderr = _call_with_captured_output(lambda name=name: _import_module(name))
            if exc is not None:
                failure: dict[str, str] = {"kind": "third_party", "target": name, "error": repr(exc)}
                if stdout.strip():
                    failure["stdout"] = stdout.strip()[-500:]
                if stderr.strip():
                    failure["stderr"] = stderr.strip()[-500:]
                failures.append(failure)
            elif stdout.strip() or stderr.strip():
                import_messages.append(
                    {
                        "kind": "third_party",
                        "target": name,
                        "stdout": stdout.strip()[-500:],
                        "stderr": stderr.strip()[-500:],
                    }
                )

        for name in repo_imports:
            exc, stdout, stderr = _call_with_captured_output(lambda name=name: _import_module(name))
            if exc is not None:
                failure = {"kind": "repo", "target": name, "error": repr(exc)}
                if stdout.strip():
                    failure["stdout"] = stdout.strip()[-500:]
                if stderr.strip():
                    failure["stderr"] = stderr.strip()[-500:]
                failures.append(failure)
            elif stdout.strip() or stderr.strip():
                import_messages.append(
                    {
                        "kind": "repo",
                        "target": name,
                        "stdout": stdout.strip()[-500:],
                        "stderr": stderr.strip()[-500:],
                    }
                )

        benchmark_path = REPO_ROOT / "tools" / "benchmarks" / "benchmark_ob.py"
        exc, stdout, stderr = _call_with_captured_output(lambda: _import_module_from_path("benchmark_ob", benchmark_path))
        if exc is not None:
            failure = {"kind": "repo_file", "target": str(benchmark_path), "error": repr(exc)}
            if stdout.strip():
                failure["stdout"] = stdout.strip()[-500:]
            if stderr.strip():
                failure["stderr"] = stderr.strip()[-500:]
            failures.append(failure)
        elif stdout.strip() or stderr.strip():
            import_messages.append(
                {
                    "kind": "repo_file",
                    "target": str(benchmark_path),
                    "stdout": stdout.strip()[-500:],
                    "stderr": stderr.strip()[-500:],
                }
            )

    return [
        AuditItem(
            check_id="python_import_surface",
            status=_status(not failures),
            title="Maintained OBGPU Python import surface loads",
            criterion="The same Python import surface checked by tools/setup/verify_obgpu_python_imports.py should import cleanly.",
            evidence={
                "third_party_checked": third_party_imports,
                "repo_checked": repo_imports,
                "repo_file_checked": str(REPO_ROOT / "tools" / "benchmarks" / "benchmark_ob.py"),
                "failure_count": len(failures),
                "failures": failures,
                "import_messages": import_messages,
            },
        )
    ]


def audit_launcher_smoke(*, run_launcher_smoke: bool = False, timeout_seconds: float = 20.0) -> list[AuditItem]:
    if not run_launcher_smoke:
        return [
            AuditItem(
                check_id="nrniv_launcher_smoke_skipped",
                status="WARN",
                title="nrniv launcher smoke skipped",
                criterion="Run with --run-launcher-smoke to execute a cheap nrniv subprocess.",
            )
        ]

    nrniv = shutil.which("nrniv")
    if not nrniv:
        return [
            AuditItem(
                check_id="nrniv_launcher_smoke",
                status="FAIL",
                title="nrniv launcher is unavailable",
                criterion="nrniv must be on PATH before a launcher smoke can run.",
            )
        ]

    command = [nrniv, "-python", "-c", "from neuron import h; print(h.nrnversion())"]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return [
            AuditItem(
                check_id="nrniv_launcher_smoke",
                status="FAIL",
                title="nrniv launcher smoke timed out",
                criterion="A cheap nrniv version probe should complete quickly.",
                evidence={"command": command, "timeout_seconds": timeout_seconds, "error": repr(exc)},
            )
        ]

    stderr = result.stderr.strip()
    has_native_warning = "dlopen failed" in stderr.lower()
    status = "FAIL" if result.returncode != 0 else ("WARN" if has_native_warning else "PASS")
    return [
        AuditItem(
            check_id="nrniv_launcher_smoke",
            status=status,
            title="nrniv can launch a NEURON Python probe",
            criterion="The nrniv executable should be able to start NEURON and report its version.",
            evidence={
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout.strip()[-500:],
                "stderr": stderr[-500:],
            },
            note="nrniv returned 0 but emitted a native dlopen warning." if has_native_warning else "",
        )
    ]


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--skip-neuron", action="store_true", help="Accepted for new_sweep compatibility; no effect.")
    parser.add_argument("--skip-imports", action="store_true", help="Skip the maintained Python import surface check.")
    parser.add_argument("--require-gpu", action="store_true", help="Require GPU build tools such as nvc, nvc++, and nvcc.")
    parser.add_argument("--run-launcher-smoke", action="store_true", help="Run a cheap nrniv subprocess smoke test.")
    parser.add_argument("--launcher-timeout-seconds", type=float, default=20.0, help="Timeout for --run-launcher-smoke.")


def run(args: argparse.Namespace) -> AuditReport:
    items = collect_items(
        audit_repo_layout(),
        audit_python_environment(),
        audit_activation_hooks(),
        audit_command_line_tools(require_gpu=bool(getattr(args, "require_gpu", False))),
        audit_mechanism_outputs(),
        audit_legacy_nmodl_path(),
        audit_python_imports(skip_imports=bool(getattr(args, "skip_imports", False))),
        audit_launcher_smoke(
            run_launcher_smoke=bool(getattr(args, "run_launcher_smoke", False)),
            timeout_seconds=float(getattr(args, "launcher_timeout_seconds", 20.0)),
        ),
    )
    return AuditReport(
        audit_id="env_install",
        title="Environment/install audit",
        items=items,
    )
