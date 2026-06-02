"""Run curated maintained-surface repo-health checks through the audit system."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys

from olfactorybulb.audit.core import AuditItem, AuditReport


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RepoHealthCheck:
    check_id: str
    title: str
    command: tuple[str, ...]


def _py(*args: str) -> tuple[str, ...]:
    return (sys.executable, *args)


_PROFILES: dict[str, tuple[RepoHealthCheck, ...]] = {
    "quick": (
        RepoHealthCheck(
            check_id="env_install",
            title="Maintained OBGPU environment audit with launcher smoke",
            command=_py("tools/run_audit.py", "env_install", "--run-launcher-smoke"),
        ),
    ),
    "maintained": (
        RepoHealthCheck(
            check_id="env_install",
            title="Maintained OBGPU environment audit with launcher smoke",
            command=_py("tools/run_audit.py", "env_install", "--run-launcher-smoke"),
        ),
        RepoHealthCheck(
            check_id="human_review_status",
            title="Human-review coverage audit",
            command=_py("tools/run_audit.py", "human_review_status"),
        ),
        RepoHealthCheck(
            check_id="hfo_feature_contracts",
            title="HFO feature/visual contract audit",
            command=_py("tools/run_audit.py", "hfo_feature_contracts"),
        ),
        RepoHealthCheck(
            check_id="config_helpers",
            title="Notebook facade and delegation smoke tests",
            command=_py("test_config_helpers.py"),
        ),
        RepoHealthCheck(
            check_id="reference_validation_engine",
            title="Declarative validation engine smoke tests",
            command=_py("test_reference_validation_engine.py"),
        ),
        RepoHealthCheck(
            check_id="reference_data_sanity",
            title="Reference-data sanity heuristics",
            command=_py("test_reference_data_sanity.py"),
        ),
    ),
    "reference": (
        RepoHealthCheck(
            check_id="reference_dataset_engine",
            title="Declarative reference-dataset engine smoke tests",
            command=_py("test_reference_dataset_engine.py"),
        ),
        RepoHealthCheck(
            check_id="epl_fsi_reference_data",
            title="EPL-FSI reference-data tests",
            command=_py("test_pv_crh_epl_fsi_reference_data.py"),
        ),
        RepoHealthCheck(
            check_id="gc_reference_data",
            title="Granule-cell reference-data tests",
            command=_py("test_gc_reference_data.py"),
        ),
        RepoHealthCheck(
            check_id="verify_epl_fsi_reference_data",
            title="EPL-FSI generated bundle verifier",
            command=_py("tools/verify_pv_crh_epl_fsi_reference_data.py"),
        ),
        RepoHealthCheck(
            check_id="verify_gc_reference_data",
            title="Granule-cell generated bundle verifier",
            command=_py("tools/verify_gc_reference_data.py"),
        ),
    ),
}
_PROFILES["full"] = _PROFILES["maintained"] + _PROFILES["reference"]


def list_repo_health_profiles() -> tuple[str, ...]:
    return tuple(_PROFILES.keys())


def repo_health_checks(profile: str) -> tuple[RepoHealthCheck, ...]:
    try:
        return _PROFILES[profile]
    except KeyError as exc:  # pragma: no cover - parser choices prevent this in normal use
        raise ValueError(f"Unknown repo-health profile {profile!r}") from exc


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = __doc__
    parser.add_argument(
        "--profile",
        default="maintained",
        choices=list_repo_health_profiles(),
        help="Curated repo-health profile to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the maintained command plan without executing it.",
    )


def _health_item(
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
        human_review_status="not_applicable",
    )


def _run_subprocess(command: tuple[str, ...]) -> tuple[int, str, str]:
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return int(completed.returncode), completed.stdout, completed.stderr


def run(args: argparse.Namespace) -> AuditReport:
    profile = str(args.profile)
    checks = repo_health_checks(profile)
    items: list[AuditItem] = []
    dry_run = bool(getattr(args, "dry_run", False))

    for check in checks:
        command_text = " ".join(shlex.quote(part) for part in check.command)
        if dry_run:
            items.append(
                _health_item(
                    check_id=check.check_id,
                    status="PASS",
                    title=check.title,
                    criterion="The repo-health audit should expose one maintained command plan for each curated check.",
                    description="This dry-run mode verifies discoverability of the curated maintained-surface check plan without executing the subprocess.",
                    acceptable="Each planned repo-health check resolves to one concrete command rooted at the repository root.",
                    acceptable_basis="The command plan comes from the centralized repo-health audit profile definitions rather than ad hoc shell notes.",
                    evidence={"profile": profile, "command": command_text},
                    note="dry-run only; command not executed",
                )
            )
            continue

        returncode, stdout, stderr = _run_subprocess(check.command)
        items.append(
            _health_item(
                check_id=check.check_id,
                status="PASS" if returncode == 0 else "FAIL",
                title=check.title,
                criterion="Each curated maintained-surface health check should complete successfully when the repository is in a healthy maintained state.",
                description="This meta-audit runs a curated command from the maintained environment, audit, notebook-facade, or reference-data health surface and records whether it returned success.",
                acceptable="The subprocess exits with code 0.",
                acceptable_basis="The accepted command list is centralized in the repo-health audit so the maintained health surface is reproducible and does not depend on ad hoc shell recipes.",
                evidence={
                    "profile": profile,
                    "command": command_text,
                    "exit_code": returncode,
                    "stdout_tail": stdout[-4000:],
                    "stderr_tail": stderr[-4000:],
                },
            )
        )

    return AuditReport(
        audit_id="repo_health",
        title="Repo health audit",
        items=items,
    )
