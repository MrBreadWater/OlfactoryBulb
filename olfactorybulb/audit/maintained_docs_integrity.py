"""Audit that maintained docs point at current entrypoints and expose the maintained docs portal."""

from __future__ import annotations

import argparse
from pathlib import Path

from olfactorybulb.audit.core import AuditItem, AuditReport


REPO_ROOT = Path(__file__).resolve().parents[2]
MAINTAINED_DOCS = (
    REPO_ROOT / "readme.md",
    REPO_ROOT / "INSTALL.md",
    REPO_ROOT / "tools/README.md",
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "tests/README.md",
    REPO_ROOT / "notes/DOCS_OWNERSHIP_MAP.md",
    REPO_ROOT / "notes/REFERENCE_DATASET_HOWTO.md",
    REPO_ROOT / "notes/REFERENCE_VALIDATION_HOWTO.md",
    REPO_ROOT / "notes/REFERENCE_VALIDATION_SYSTEM_OVERVIEW.md",
    REPO_ROOT / "research_context/README.md",
)
DOCS_PORTAL_HTML = REPO_ROOT / "docs/index.html"
DOCS_PORTAL_SOURCE = REPO_ROOT / "docs-source/index.rst"
DOCS_PORTAL_SOURCE_TRACKED = REPO_ROOT / "docs/_sources/index.rst.txt"
BANNED_DOC_REFERENCES = (
    "initslice.py",
    "runbatch.py",
    "tools/verify_pv_crh_epl_fsi_reference_data.py",
    "tools/verify_gc_reference_data.py",
    "tools/extract_pv_crh_epl_fsi_reference_data.py",
    "tools/extract_gc_reference_data.py",
    "tools/download_epl_fsi_reference_sources.py",
    "tools/download_gc_reference_sources.py",
    "tools/audit_burton_urban_fi.py",
    "tools/audit_env_install.py",
    "tools/audit_epli_correctness.py",
    "tools/audit_gc_intrinsic_validation.py",
    "tools/audit_epl_fsi_intrinsic_validation.py",
)
REQUIRED_PORTAL_LINKS = (
    "../readme.md",
    "../INSTALL.md",
    "../tools/README.md",
    "../tests/README.md",
    "../notes/REFERENCE_DATASET_HOWTO.md",
    "../notes/REFERENCE_VALIDATION_HOWTO.md",
    "../research_context/README.md",
)


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
        human_review_status="not_applicable",
    )


def _text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def run(args: argparse.Namespace) -> AuditReport:
    hits: dict[str, list[str]] = {}
    for path in MAINTAINED_DOCS:
        text = _text(path)
        matched = [token for token in BANNED_DOC_REFERENCES if token in text]
        if matched:
            hits[str(path.relative_to(REPO_ROOT))] = matched

    portal_text = _text(DOCS_PORTAL_HTML)
    portal_source_text = _text(DOCS_PORTAL_SOURCE)
    portal_source_tracked_text = _text(DOCS_PORTAL_SOURCE_TRACKED)
    ownership_map_text = _text(REPO_ROOT / "notes/DOCS_OWNERSHIP_MAP.md")

    items = [
        _item(
            check_id="maintained_docs_avoid_removed_entrypoints",
            status="PASS" if not hits else "FAIL",
            title="Maintained docs avoid removed scripts and entrypoints",
            criterion="The maintained user-facing docs should not instruct people to use deleted or compatibility-removed entrypoints.",
            description="Once the maintained workflow is centralized, stale command references in the maintained docs become a reproducibility and support burden.",
            acceptable="No maintained doc contains references to removed scripts or entrypoints.",
            acceptable_basis="The maintained command surface is now the audit CLI, generic dataset CLI, benchmark runner, and notebook/helper path documented elsewhere in the repo.",
            evidence={"doc_hits": hits},
        ),
        _item(
            check_id="docs_portal_exists",
            status="PASS" if DOCS_PORTAL_HTML.exists() and DOCS_PORTAL_SOURCE.exists() and DOCS_PORTAL_SOURCE_TRACKED.exists() else "FAIL",
            title="Docs portal artifacts exist",
            criterion="The repurposed docs surface should keep one maintained landing page in both the tracked HTML portal and its source files.",
            description="This ensures local file-based access remains available even though the old docs surface is no longer the source of truth for maintained procedures.",
            acceptable="The maintained docs portal HTML and its paired source files all exist.",
            acceptable_basis="The repository keeps a tracked docs portal for linkable local navigation to current maintained markdown docs.",
            evidence={
                "docs_index_html": DOCS_PORTAL_HTML.exists(),
                "docs_source_index": DOCS_PORTAL_SOURCE.exists(),
                "docs_sources_index_txt": DOCS_PORTAL_SOURCE_TRACKED.exists(),
            },
        ),
        _item(
            check_id="docs_portal_exposes_maintained_markdown",
            status="PASS" if all(link in portal_text for link in REQUIRED_PORTAL_LINKS) else "FAIL",
            title="Docs portal links to maintained markdown docs",
            criterion="The docs landing page should expose the current maintained markdown docs rather than sending users into historical pages first.",
            description="This repurposes the historical docs site into a small maintained navigation surface while keeping old generated pages available only as secondary reference.",
            acceptable="The docs portal links to the maintained README, install guide, tooling map, and reference-data docs.",
            acceptable_basis="The maintained docs now live primarily in markdown files under the repository root and notes directories.",
            evidence={
                "required_links": list(REQUIRED_PORTAL_LINKS),
                "missing_links": [link for link in REQUIRED_PORTAL_LINKS if link not in portal_text],
            },
        ),
        _item(
            check_id="docs_portal_marks_historical_reference",
            status="PASS"
            if "Historical reference pages" in portal_text and "Historical reference pages" in portal_source_text and "Historical reference pages" in portal_source_tracked_text
            else "FAIL",
            title="Docs portal explicitly marks historical reference pages as secondary",
            criterion="The repurposed docs surface should clearly distinguish maintained docs from historical generated pages.",
            description="This prevents the old Sphinx pages from silently regaining ownership of current procedures while still leaving them accessible when historical context is useful.",
            acceptable="The portal HTML and paired source files both contain a visible historical-reference section.",
            acceptable_basis="The docs ownership map treats generated docs as historical unless they are explicitly serving the maintained portal.",
            evidence={
                "html_mentions_historical": "Historical reference pages" in portal_text,
                "source_mentions_historical": "Historical reference pages" in portal_source_text,
                "tracked_source_mentions_historical": "Historical reference pages" in portal_source_tracked_text,
            },
        ),
        _item(
            check_id="docs_ownership_map_covers_portal",
            status="PASS"
            if "docs-source/" in ownership_map_text and "tracked generated `docs/`" in ownership_map_text
            else "FAIL",
            title="Docs ownership map still covers the repurposed docs surface",
            criterion="The docs ownership map should explain how the maintained portal relates to the historical docs tree.",
            description="This keeps future cleanups from accidentally duplicating the same documentation policy in multiple places.",
            acceptable="The ownership map names docs-source and tracked generated docs explicitly.",
            acceptable_basis="The docs portal is intentionally a thin maintained navigation layer sitting on top of a largely historical docs tree.",
            evidence={
                "ownership_map_mentions_docs_source": "docs-source/" in ownership_map_text,
                "ownership_map_mentions_tracked_docs": "tracked generated `docs/`" in ownership_map_text,
            },
        ),
        _item(
            check_id="docs_ownership_map_covers_tests_readme",
            status="PASS" if "tests/README.md" in ownership_map_text else "FAIL",
            title="Docs ownership map covers the structured test tree guide",
            criterion="The docs ownership map should explain where the maintained tests layout and execution rules live.",
            description="Once the root test-file sprawl is replaced with a structured tests package, the ownership map should tell future agents where that policy is documented.",
            acceptable="The ownership map explicitly names tests/README.md.",
            acceptable_basis="The structured tests tree is now a maintained repo contract and should not rely on tribal knowledge.",
            evidence={"ownership_map_mentions_tests_readme": "tests/README.md" in ownership_map_text},
        ),
    ]

    return AuditReport(
        audit_id="maintained_docs_integrity",
        title="Maintained docs integrity audit",
        items=items,
    )
