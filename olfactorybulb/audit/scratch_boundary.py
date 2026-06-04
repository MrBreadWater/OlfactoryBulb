"""Audit that local scratch and generated-artifact boundaries stay explicit and protected."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from olfactorybulb.audit.core import AuditItem, AuditReport


REPO_ROOT = Path(__file__).resolve().parents[2]
GITIGNORE_PATH = REPO_ROOT / ".gitignore"
RESEARCH_README_PATH = REPO_ROOT / "research_context/README.md"
REQUIRED_GITIGNORE_PATTERNS = (
    "notebooks/*-Copy*.ipynb",
    "research_context/source_data/",
    "results/sweeps/",
)
SCRATCH_TRACKED_PATHS = (
    "research_context/source_data",
    "results/sweeps",
)
SCRATCH_TRACKED_GLOBS = (
    "notebooks/*-Copy*.ipynb",
    "notebooks/*Copy*.ipynb",
)
ROOT_TEST_GLOB = "test_*.py"


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = __doc__


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
        validation_design_review_status="not_applicable",
    )


def _tracked_files(pathspec: str) -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "--", pathspec],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def run(args: argparse.Namespace) -> AuditReport:
    gitignore_text = GITIGNORE_PATH.read_text()
    missing_patterns = [pattern for pattern in REQUIRED_GITIGNORE_PATTERNS if pattern not in gitignore_text]

    tracked_scratch: dict[str, list[str]] = {}
    for pathspec in SCRATCH_TRACKED_PATHS + SCRATCH_TRACKED_GLOBS:
        matches = _tracked_files(pathspec)
        if matches:
            tracked_scratch[pathspec] = matches

    research_readme = RESEARCH_README_PATH.read_text()
    root_test_files = sorted(_tracked_files(ROOT_TEST_GLOB))

    items = [
        _item(
            check_id="gitignore_contains_required_scratch_patterns",
            status="PASS" if not missing_patterns else "FAIL",
            title="Git ignore rules cover the maintained scratch boundaries",
            criterion="Known local scratch and downloaded-source paths should be ignored explicitly so they do not drift into the tracked repo surface by accident.",
            description="This protects the maintained repo from notebook copies, local source corpora, and ad hoc sweep outputs that are useful locally but not meant to become canonical history.",
            acceptable="The maintained ignore list contains the required scratch-boundary patterns.",
            acceptable_basis="These patterns correspond to the explicit local-boundary rules documented in the maintained docs and used by the active workflows.",
            evidence={"missing_patterns": missing_patterns},
        ),
        _item(
            check_id="scratch_paths_not_tracked",
            status="PASS" if not tracked_scratch else "FAIL",
            title="Scratch-only paths are not tracked by git",
            criterion="Paths declared as local scratch or downloaded-source boundaries should not already contain tracked files.",
            description="Ignoring a scratch path is not enough if files inside it are already tracked, because that would keep the ambiguous boundary alive indefinitely.",
            acceptable="No tracked files exist under the declared scratch-only pathspecs.",
            acceptable_basis="The maintained repo boundary now treats these locations as local working areas rather than source-of-truth content.",
            evidence={"tracked_matches": tracked_scratch},
        ),
        _item(
            check_id="research_context_readme_marks_generated_boundaries",
            status="PASS"
            if "generated canonical outputs" in research_readme and "Do not hand-edit generated canonical outputs" in research_readme and "Raw/source files" in research_readme
            else "FAIL",
            title="research_context README explains the generated-versus-raw boundary",
            criterion="The research-context boundary doc should make it obvious which files are generated outputs, which are raw sources, and which should not be hand-edited.",
            description="This keeps dataset configs, raw source files, manual intake, and generated outputs from collapsing back into one ambiguous directory surface.",
            acceptable="The README explicitly distinguishes raw/source files, generated canonical outputs, and the hand-edit boundary.",
            acceptable_basis="The research_context directory now holds both source material and canonical generated validation artifacts, so the boundary must remain explicit.",
            evidence={
                "mentions_raw_source_files": "Raw/source files" in research_readme,
                "mentions_generated_outputs": "generated canonical outputs" in research_readme,
                "mentions_no_hand_edits": "Do not hand-edit generated canonical outputs" in research_readme,
            },
        ),
        _item(
            check_id="root_test_files_moved_under_tests_package",
            status="PASS" if not root_test_files else "FAIL",
            title="Developer test files live under the structured tests package instead of repo root",
            criterion="Tracked low-level Python test modules should live under tests/ so the repo root stays focused on maintained runtime and configuration surfaces.",
            description="This protects the maintained tree from drifting back to a flat test-file namespace where audit surfaces, runtime entrypoints, and one-off regressions are mixed together.",
            acceptable="No tracked root-level files matching test_*.py exist in the repository root.",
            acceptable_basis="The structured tests/ tree is now the maintained home for developer-facing tests, while grouped suite audits present selected families through the user-facing audit surface.",
            evidence={"root_test_files": root_test_files},
        ),
    ]

    return AuditReport(
        audit_id="scratch_boundary",
        title="Scratch boundary audit",
        items=items,
    )
