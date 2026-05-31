"""Audit whether the local machine has a usable OBGPU install."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from typing import Iterable

from olfactorybulb.audit.core import AuditItem, AuditReport, collect_items


REPO_ROOT = Path(__file__).resolve().parents[2]
MECHANISM_SOURCE_DIR = REPO_ROOT / "prev_ob_models" / "Birgiolas2020" / "Mechanisms"
VERIFY_IMPORTS_SCRIPT = REPO_ROOT / "tools" / "setup" / "verify_obgpu_python_imports.py"
FIX_NVHPC_LIBNRNMECH = REPO_ROOT / "tools" / "setup" / "fix_nvhpc_libnrnmech.sh"

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


def _env_item(
    *,
    check_id: str,
    status: str,
    title: str,
    criterion: str,
    description: str,
    acceptable: str,
    acceptable_basis: str,
    evidence: dict[str, object] | None = None,
    note: str = "",
) -> AuditItem:
    return AuditItem(
        check_id=check_id,
        status=status,
        title=title,
        criterion=criterion,
        description=description,
        acceptable=acceptable,
        acceptable_basis=acceptable_basis,
        evidence=evidence or {},
        note=note,
    )


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


def _print_needed(shared_object: Path) -> tuple[str, list[str]]:
    patchelf = shutil.which("patchelf")
    if not patchelf:
        return ("WARN", [])
    if not shared_object.exists():
        return ("FAIL", [])
    result = subprocess.run(
        [patchelf, "--print-needed", str(shared_object)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ("FAIL", [])
    needed = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return ("PASS", needed)


def audit_repo_layout() -> list[AuditItem]:
    missing = [relative for relative in REQUIRED_REPO_FILES if not (REPO_ROOT / relative).exists()]
    cwd_matches = Path.cwd().resolve() == REPO_ROOT.resolve()
    return [
        _env_item(
            check_id="repo_layout",
            status=_status(not missing),
            title="Maintained OBGPU install files are present",
            criterion="A machine install should use the maintained setup, activation, environment, patch, mechanism, and benchmark files.",
            description="This check verifies that the repository contains the required installation, activation, benchmark, environment, and mechanism files that define the maintained OBGPU workflow.",
            acceptable="Every path in the required-repository-file list exists under the repository root. The missing-file list must be empty.",
            acceptable_basis="The acceptance list is a curated set of files that the maintained OBGPU installation and benchmark workflow depends on directly.",
            evidence={
                "repo_root": str(REPO_ROOT),
                "missing": missing,
                "checked": REQUIRED_REPO_FILES,
            },
        ),
        _env_item(
            check_id="repo_root_cwd",
            status=_status(cwd_matches, "WARN"),
            title="Audit is running from the repository root",
            criterion="The maintained workflow expects commands to run from the repository root.",
            description="This check confirms that the current working directory matches the maintained repository root, which reduces path-resolution mistakes in setup and benchmark commands.",
            acceptable="The current working directory resolves to the same absolute path as the repository root. A mismatch is a warning rather than a hard failure.",
            acceptable_basis="The rule is a direct path comparison between Path.cwd() and the repository root resolved from the audit module location.",
            evidence={"cwd": str(Path.cwd()), "repo_root": str(REPO_ROOT)},
        ),
    ]


def audit_python_environment() -> list[AuditItem]:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    env_name = conda_env or Path(sys.prefix).name
    env_ok = bool(env_name and env_name.startswith("OBGPU"))
    return [
        _env_item(
            check_id="python_environment",
            status=_status(env_ok),
            title="Python is running inside an OBGPU conda environment",
            criterion="The maintained install should run from an OBGPU or OBGPU-portable conda environment, not base/system Python.",
            description="This check confirms that the active Python interpreter comes from an OBGPU-prefixed conda environment, which is the supported runtime for repository-local scripts and audits.",
            acceptable="The active environment name begins with OBGPU. Base or unrelated Python environments do not satisfy this requirement.",
            acceptable_basis="The rule is derived from CONDA_DEFAULT_ENV when available, otherwise from the interpreter prefix name, because those are the canonical markers of the maintained environment choice.",
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
        _env_item(
            check_id="activation_runtime_hooks",
            status=_status(repo_root_ok and mechanism_root_ok and corenrn_ok and ld_has_arch and ld_has_conda),
            title="Repo activation hooks are active",
            criterion="source tools/setup/activate_obgpu.sh should export repo, mechanism, CoreNEURON, and library-path settings.",
            description="This check validates that the activation script has populated the runtime environment variables and library paths that the maintained NEURON and CoreNEURON workflows expect.",
            acceptable="The shared repo root matches this checkout, the mechanism root exists, the CoreNEURON mechanism library matches the expected architecture path when present, and LD_LIBRARY_PATH contains both the architecture directory and the active conda lib directory.",
            acceptable_basis="The rule is taken from the maintained activate_obgpu.sh contract: these variables and path entries are the minimum environment hooks needed for the repo-local NEURON runtime to resolve the correct mechanism libraries.",
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
        _env_item(
            check_id="command_line_tools",
            status=_status(required_ok and mpi_ok),
            title="NEURON and MPI launcher commands are available",
            criterion="A runnable machine install needs nrniv, nrnivmodl, and either OB_MPIEXEC, mpiexec, or srun on PATH.",
            description="This check verifies that the core NEURON executables and at least one usable MPI launcher are available on the active PATH.",
            acceptable="Both nrniv and nrnivmodl resolve on PATH, and at least one of OB_MPIEXEC, mpiexec, or srun resolves to an executable command.",
            acceptable_basis="The rule comes from the maintained simulation entrypoints, which depend on the NEURON executables plus an MPI launcher for distributed or remote benchmark runs.",
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
            _env_item(
                check_id="gpu_build_tools",
                status=_status(all(gpu_paths.values())),
                title="GPU build tools are available",
                criterion="GPU-enabled OBGPU installs require NVIDIA HPC SDK compilers and nvcc.",
                description="This check verifies that the GPU compiler toolchain required for GPU-enabled builds is installed and reachable on PATH.",
                acceptable="nvc, nvc++, and nvcc all resolve on PATH when GPU capability is required.",
                acceptable_basis="The rule follows the maintained GPU build workflow, which depends on the NVIDIA HPC SDK compilers plus nvcc for compatible mechanism compilation and runtime support.",
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
        _env_item(
            check_id="mechanism_build_outputs",
            status=_status(bool(source_mods) and all(output_presence.values()) and not missing_compiled),
            title="Birgiolas mechanism outputs are compiled for this architecture",
            criterion="The active mechanism root should contain special, libnrnmech.so, libcorenrnmech.so, and generated code for every maintained .mod file.",
            description="This check confirms that the maintained Birgiolas mechanism source set has been compiled for the current architecture and produced the expected NEURON and CoreNEURON outputs.",
            acceptable="The special executable, libnrnmech.so, and libcorenrnmech.so all exist, and every maintained .mod source file has a generated compiled-code counterpart for the active architecture.",
            acceptable_basis="The rule is derived from the maintained nrnivmodl build products and the full set of .mod files under prev_ob_models/Birgiolas2020/Mechanisms.",
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


def audit_nvhpc_transient_dependencies() -> list[AuditItem]:
    mechanism_root = Path(os.environ.get("OBGPU_MECHANISM_ROOT") or REPO_ROOT).expanduser().resolve()
    arch_dir = mechanism_root / platform.machine()
    targets = {
        "libnrnmech": arch_dir / "libnrnmech.so",
        "libcorenrnmech": arch_dir / "libcorenrnmech.so",
    }
    evidence: dict[str, object] = {
        "arch_dir": str(arch_dir),
        "repair_script": str(FIX_NVHPC_LIBNRNMECH),
        "targets": {name: str(path) for name, path in targets.items()},
    }
    bad_needed: dict[str, list[str]] = {}
    unavailable = False
    failed = False

    for name, path in targets.items():
        state, needed = _print_needed(path)
        evidence[f"{name}_needed"] = needed
        bogus = [entry for entry in needed if entry.startswith("/tmp/pgcudafat")]
        if bogus:
            bad_needed[name] = bogus
        if state == "WARN":
            unavailable = True
        elif state == "FAIL":
            failed = True

    if bad_needed:
        status = "FAIL"
        note = (
            "Run tools/setup/fix_nvhpc_libnrnmech.sh on the affected library to "
            "remove stale /tmp/pgcudafat NEEDED entries."
        )
    elif failed:
        status = "FAIL"
        note = "Could not inspect shared-library dependencies with patchelf."
    elif unavailable:
        status = "WARN"
        note = "Install patchelf to inspect libnrnmech/libcorenrnmech NEEDED entries directly."
    else:
        status = "PASS"
        note = ""

    evidence["bad_needed"] = bad_needed
    return [
        _env_item(
            check_id="nvhpc_transient_dependencies",
            status=status,
            title="Mechanism libraries are free of stale NVHPC temp-object dependencies",
            criterion="libnrnmech.so and libcorenrnmech.so should not embed /tmp/pgcudafat* loader paths.",
            description="This check scans the built mechanism libraries for stale NVIDIA HPC SDK temporary object dependencies that can break dlopen at runtime.",
            acceptable="The ideal result is that neither shared library contains a /tmp/pgcudafat* dependency. If patchelf is unavailable the result is downgraded to a warning because the check cannot be completed.",
            acceptable_basis="The rule is based on patchelf --print-needed output. A stale /tmp/pgcudafat dependency is a known broken build artifact in this repository's OBGPU workflow.",
            evidence=evidence,
            note=note,
        )
    ]


def audit_legacy_nmodl_path() -> list[AuditItem]:
    value = os.environ.get("NRN_NMODL_PATH")
    return [
        _env_item(
            check_id="legacy_nrn_nmodl_path",
            status=_status(not value, "WARN"),
            title="Legacy NRN_NMODL_PATH is not forcing stale mechanism autoloads",
            criterion="The maintained runtime should not rely on NRN_NMODL_PATH auto-loading repo-root mechanisms.",
            description="This check warns when the legacy NRN_NMODL_PATH variable is still set, because it can autoload the wrong mechanisms and mask the maintained explicit mechanism path.",
            acceptable="The preferred state is that NRN_NMODL_PATH is unset. If it is set, the audit warns rather than fails because the runtime may still work but is more fragile.",
            acceptable_basis="The rule comes from earlier repository activation behavior that used NRN_NMODL_PATH, which can now conflict with the explicit Birgiolas mechanism loading strategy.",
            evidence={"NRN_NMODL_PATH": value},
            note="Older activation hooks used this variable; it can collide with the explicit Birgiolas mechanism load path.",
        )
    ]


def audit_python_imports(*, skip_imports: bool = False, timeout_seconds: float = 120.0) -> list[AuditItem]:
    if skip_imports:
        return [
            _env_item(
                check_id="python_import_surface_skipped",
                status="WARN",
                title="Maintained Python import surface check skipped",
                criterion="Run without --skip-imports to verify third-party and repo imports.",
                description="This item records that the maintained Python import smoke test was intentionally skipped, so the report does not currently confirm third-party and repository import health.",
                acceptable="This is an informational warning only. It clears once the audit is rerun without the skip flag.",
                acceptable_basis="The item is emitted by command-line control flow rather than by scientific or runtime measurements. It exists to explain the missing import verification step.",
            )
        ]

    command = [sys.executable, str(VERIFY_IMPORTS_SCRIPT), "--repo-root", str(REPO_ROOT)]
    env = os.environ.copy()
    env.pop("NRN_NMODL_PATH", None)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return [
            _env_item(
                check_id="python_import_surface",
                status="FAIL",
                title="Maintained Python import surface check timed out",
                criterion="tools/setup/verify_obgpu_python_imports.py should complete and report import failures without hanging.",
                description="This check failed because the maintained import verification subprocess did not finish within the configured timeout.",
                acceptable="The verification subprocess completes within the timeout and returns a structured result instead of hanging.",
                acceptable_basis="The rule comes from the dedicated verify_obgpu_python_imports.py probe, which is the maintained import-health smoke test for this repository.",
                evidence={"command": command, "timeout_seconds": timeout_seconds, "error": repr(exc)},
            )
        ]

    payload = None
    if result.stdout.strip():
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            payload = None

    failures = payload.get("failures", []) if isinstance(payload, dict) else []
    ok = result.returncode == 0 and isinstance(payload, dict) and payload.get("ok") is True
    evidence = {
        "command": command,
        "returncode": result.returncode,
        "ok": payload.get("ok") if isinstance(payload, dict) else None,
        "third_party_checked": payload.get("third_party_checked", []) if isinstance(payload, dict) else [],
        "repo_checked": payload.get("repo_checked", []) if isinstance(payload, dict) else [],
        "repo_file_checked": payload.get("repo_file_checked") if isinstance(payload, dict) else None,
        "failure_count": len(failures),
        "failures": failures,
    }
    if result.stdout.strip() and payload is None:
        evidence["stdout_tail"] = result.stdout.strip()[-2000:]
    if result.stderr.strip():
        evidence["stderr_tail"] = result.stderr.strip()[-2000:]

    return [
        _env_item(
            check_id="python_import_surface",
            status=_status(ok),
            title="Maintained OBGPU Python import surface loads",
            criterion="The same Python import surface checked by tools/setup/verify_obgpu_python_imports.py should import cleanly.",
            description="This check runs the maintained import-health probe and records whether the third-party and repository import surface loads successfully in the active OBGPU environment.",
            acceptable="The subprocess returns successfully, reports ok=true, and records no import failures.",
            acceptable_basis="The rule is the structured JSON result produced by verify_obgpu_python_imports.py, which is the maintained import smoke test for this repository.",
            evidence=evidence,
        )
    ]


def audit_launcher_smoke(*, run_launcher_smoke: bool = False, timeout_seconds: float = 20.0) -> list[AuditItem]:
    if not run_launcher_smoke:
        return [
            _env_item(
                check_id="nrniv_launcher_smoke_skipped",
                status="WARN",
                title="nrniv launcher smoke skipped",
                criterion="Run with --run-launcher-smoke to execute a cheap nrniv subprocess.",
                description="This item records that the cheap nrniv subprocess smoke test was intentionally skipped, so the report does not currently confirm that the NEURON launcher itself starts cleanly.",
                acceptable="This is an informational warning only. It clears once the audit is rerun with --run-launcher-smoke.",
                acceptable_basis="The item is emitted by command-line control flow to explain why launcher health was not measured during this audit invocation.",
            )
        ]

    nrniv = shutil.which("nrniv")
    if not nrniv:
        return [
            _env_item(
                check_id="nrniv_launcher_smoke",
                status="FAIL",
                title="nrniv launcher is unavailable",
                criterion="nrniv must be on PATH before a launcher smoke can run.",
                description="This check failed before running the smoke test because the nrniv executable could not be resolved on PATH.",
                acceptable="The nrniv executable resolves on PATH so the launcher smoke test can actually run.",
                acceptable_basis="The maintained runtime enters NEURON through nrniv, so its absence is an immediate hard failure for this smoke path.",
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
            _env_item(
                check_id="nrniv_launcher_smoke",
                status="FAIL",
                title="nrniv launcher smoke timed out",
                criterion="A cheap nrniv version probe should complete quickly.",
                description="This check failed because a minimal nrniv subprocess did not finish within the configured timeout, indicating a launcher or runtime hang.",
                acceptable="A minimal nrniv invocation completes within the timeout and returns a version probe result.",
                acceptable_basis="The rule is based on a cheap NEURON version command that should finish quickly on any healthy install.",
                evidence={"command": command, "timeout_seconds": timeout_seconds, "error": repr(exc)},
            )
        ]

    stderr = result.stderr.strip()
    has_native_warning = "dlopen failed" in stderr.lower()
    status = "FAIL" if result.returncode != 0 else ("WARN" if has_native_warning else "PASS")
    return [
        _env_item(
            check_id="nrniv_launcher_smoke",
            status=status,
            title="nrniv can launch a NEURON Python probe",
            criterion="The nrniv executable should be able to start NEURON and report its version.",
            description="This check launches a minimal NEURON Python probe through nrniv to confirm that the launcher itself can start the runtime successfully.",
            acceptable="The subprocess exits with return code 0. Native dlopen warnings are downgraded to a warning because they indicate a compromised but still starting runtime.",
            acceptable_basis="The rule is based on the observed return code and stderr of a minimal 'from neuron import h; print(h.nrnversion())' probe launched through nrniv.",
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
    parser.add_argument("--import-timeout-seconds", type=float, default=120.0, help="Timeout for the import surface subprocess.")
    parser.add_argument("--launcher-timeout-seconds", type=float, default=20.0, help="Timeout for --run-launcher-smoke.")


def run(args: argparse.Namespace) -> AuditReport:
    items = collect_items(
        audit_repo_layout(),
        audit_python_environment(),
        audit_activation_hooks(),
        audit_command_line_tools(require_gpu=bool(getattr(args, "require_gpu", False))),
        audit_mechanism_outputs(),
        audit_nvhpc_transient_dependencies(),
        audit_legacy_nmodl_path(),
        audit_python_imports(
            skip_imports=bool(getattr(args, "skip_imports", False)),
            timeout_seconds=float(getattr(args, "import_timeout_seconds", 120.0)),
        ),
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
