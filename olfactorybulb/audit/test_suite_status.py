"""Run structured Python test suites through the audit system."""

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
class TestModuleSpec:
    module_name: str
    title: str


@dataclass(frozen=True)
class TestSuiteSpec:
    suite_id: str
    title: str
    description: str
    modules: tuple[TestModuleSpec, ...]


SUITES: dict[str, TestSuiteSpec] = {
    "maintained_core": TestSuiteSpec(
        suite_id="maintained_core",
        title="Maintained core smoke-test suite",
        description="Notebook-facade, reference-validation, and reference-sanity smoke tests for the maintained surface.",
        modules=(
            TestModuleSpec("tests.integration.test_config_helpers", "Notebook facade and delegation smoke tests"),
            TestModuleSpec("tests.reference.test_reference_validation_engine", "Declarative reference-validation engine smoke tests"),
            TestModuleSpec("tests.reference.test_reference_data_sanity", "Reference-data sanity heuristics"),
        ),
    ),
    "reference_bundles": TestSuiteSpec(
        suite_id="reference_bundles",
        title="Reference bundle smoke-test suite",
        description="Reference-dataset extraction, downloader, and normalization smoke tests for maintained datasets.",
        modules=(
            TestModuleSpec("tests.reference.test_reference_dataset_engine", "Declarative reference-dataset engine smoke tests"),
            TestModuleSpec("tests.reference.test_download_epl_fsi_reference_sources", "EPL-FSI source downloader smoke tests"),
            TestModuleSpec("tests.reference.test_pv_crh_epl_fsi_reference_data", "EPL-FSI reference-data smoke tests"),
            TestModuleSpec("tests.reference.test_download_gc_reference_sources", "Granule-cell source downloader smoke tests"),
            TestModuleSpec("tests.reference.test_gc_reference_data", "Granule-cell reference-data smoke tests"),
        ),
    ),
    "audit_surface": TestSuiteSpec(
        suite_id="audit_surface",
        title="Audit and dashboard surface smoke-test suite",
        description="CLI, grouped-output, audit-dashboard, and control-center smoke tests for the maintained presentation surface.",
        modules=(
            TestModuleSpec("tests.audit.test_repo_health", "Repo-health audit profile smoke tests"),
            TestModuleSpec("tests.audit.test_audit_cli_output", "Audit CLI output smoke tests"),
            TestModuleSpec("tests.audit.test_audit_dashboard", "Audit HTML dashboard smoke tests"),
            TestModuleSpec("tests.audit.test_audit_style_contracts", "Audit metadata/style smoke tests"),
            TestModuleSpec("tests.neuroinfra.dashboard.test_neuroinfra_dashboard_shell", "Shared dashboard shell smoke tests"),
            TestModuleSpec("tests.integration.test_control_center_dashboard", "Unified control-center dashboard smoke tests"),
        ),
    ),
}


def list_test_suites() -> tuple[str, ...]:
    return tuple(SUITES)


def get_test_suite(suite_id: str) -> TestSuiteSpec:
    try:
        return SUITES[suite_id]
    except KeyError as exc:
        known = ", ".join(list_test_suites())
        raise ValueError(f"Unknown test suite {suite_id!r}. Known suites: {known}") from exc


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = __doc__
    parser.add_argument(
        "--suite",
        default="maintained_core",
        help="Structured Python test suite to run.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Emit one audit item per underlying Python test module in addition to the suite summary.",
    )
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="List available structured test suites through the audit surface.",
    )


def _item(
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
    group_id: str = "",
    group_title: str = "",
    detail_level: str = "detail",
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
        validation_design_review_status="not_applicable",
        group_id=group_id,
        group_title=group_title,
        detail_level=detail_level,
    )


def _run_test_module(module_name: str) -> tuple[int, str, str]:
    completed = subprocess.run(
        (sys.executable, "-m", module_name),
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return int(completed.returncode), completed.stdout, completed.stderr


def _list_report() -> AuditReport:
    items = [
        _item(
            check_id=suite_id,
            status="PASS",
            title=spec.title,
            criterion="Structured test suites should be centrally discoverable through one maintained registry.",
            description="This surfaces the current maintained test-suite registry through the audit system without running the underlying tests.",
            acceptable="Each suite has a stable identifier, description, and module list.",
            acceptable_basis="The suite registry is the maintained source of truth for audit-visible grouped Python test execution.",
            evidence={
                "suite_id": suite_id,
                "description": spec.description,
                "module_count": len(spec.modules),
                "modules": [module.module_name for module in spec.modules],
            },
        )
        for suite_id, spec in SUITES.items()
    ]
    return AuditReport(
        audit_id="test_suite_status",
        title="Structured test-suite registry audit",
        items=items,
    )


def run(args: argparse.Namespace) -> AuditReport:
    if bool(getattr(args, "list_suites", False)):
        return _list_report()

    suite = get_test_suite(str(args.suite))
    module_results: list[tuple[TestModuleSpec, int, str, str]] = []
    for module in suite.modules:
        module_results.append((module, *_run_test_module(module.module_name)))

    failed = [module.module_name for module, returncode, _stdout, _stderr in module_results if returncode != 0]
    passed = [module.module_name for module, returncode, _stdout, _stderr in module_results if returncode == 0]
    summary_status = "PASS" if not failed else "FAIL"

    items = [
        _item(
            check_id=f"{suite.suite_id}_summary",
            status=summary_status,
            title=suite.title,
            criterion="Every Python test module in the selected structured suite should exit successfully.",
            description="This grouped audit presents developer-facing unit and smoke tests through the maintained audit surface without requiring one first-class audit per tiny test file.",
            acceptable="All test modules in the selected suite exit with code 0.",
            acceptable_basis="The accepted module list comes from the centralized structured test-suite registry so maintained health profiles can rely on grouped, reproducible test execution.",
            evidence={
                "suite_id": suite.suite_id,
                "suite_description": suite.description,
                "module_count": len(suite.modules),
                "passed_modules": passed,
                "failed_modules": failed,
            },
            group_id=suite.suite_id,
            group_title=suite.title,
            detail_level="summary",
        )
    ]

    if bool(getattr(args, "details", False)):
        for module, returncode, stdout, stderr in module_results:
            command = " ".join(shlex.quote(part) for part in (sys.executable, "-m", module.module_name))
            items.append(
                _item(
                    check_id=module.module_name.rsplit(".", 1)[-1],
                    status="PASS" if returncode == 0 else "FAIL",
                    title=module.title,
                    criterion="Each grouped Python test module should exit successfully.",
                    description="This is the per-module detail view for the structured test-suite audit.",
                    acceptable="The Python test module exits with code 0.",
                    acceptable_basis="The per-module detail comes from the same centralized suite registry as the summary item.",
                    evidence={
                        "suite_id": suite.suite_id,
                        "module": module.module_name,
                        "command": command,
                        "exit_code": returncode,
                        "stdout_tail": stdout[-4000:],
                        "stderr_tail": stderr[-4000:],
                    },
                    group_id=suite.suite_id,
                    group_title=suite.title,
                )
            )

    return AuditReport(
        audit_id="test_suite_status",
        title=f"Structured test-suite audit ({suite.title})",
        items=items,
    )
