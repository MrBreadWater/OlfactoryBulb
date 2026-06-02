"""CLI entrypoint for repository audits."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from olfactorybulb.audit import AuditItem, AuditReport, format_report, get_audit_spec, iter_audit_specs
from olfactorybulb.audit.core import _expand_terms
from olfactorybulb.audit.registry import iter_new_sweep_audit_specs

DEFAULT_AUDIT_ALIAS = "default"
DEFAULT_AUDIT_TARGET = "repo_health"
DEFAULT_AUDIT_TARGET_ARGS = ["--profile", "maintained"]
ALL_AUDIT_ALIASES = {"new_sweep", "new-sweep", "all"}


def _build_root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audit_id",
        nargs="?",
        help=(
            "Audit id to run. Omit, or pass 'all', to start a new sweep "
            "across every registered audit. Pass 'default' for the maintained "
            "repo-health profile. Use --list to inspect available audits."
        ),
    )
    parser.add_argument("--list", action="store_true", help="List available audits and exit.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color output for text reports.")
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Expand grouped output instead of collapsing passing groups in multi-audit reports.",
    )
    parser.add_argument(
        "--failures-only",
        action="store_true",
        help="Render only warning/failure items in text output.",
    )
    return parser


def _paint(text: str, code: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


def audit_alias_entries() -> list[dict[str, Any]]:
    return [
        {
            "audit_id": DEFAULT_AUDIT_ALIAS,
            "title": "Default audit profile",
            "description": (
                "Run the maintained repo-health profile. If no explicit args are "
                "provided, this resolves to repo_health --profile maintained."
            ),
            "default_args": list(DEFAULT_AUDIT_TARGET_ARGS),
        },
        {
            "audit_id": "all",
            "title": "All registered audits",
            "description": "Run a new sweep across every registered audit.",
            "default_args": [],
        },
    ]


def available_audit_entries() -> list[dict[str, Any]]:
    entries = list(audit_alias_entries())
    entries.extend(
        {
            "audit_id": spec.audit_id,
            "title": spec.title,
            "description": spec.description,
            "default_args": list(DEFAULT_AUDIT_TARGET_ARGS) if spec.audit_id == DEFAULT_AUDIT_TARGET else [],
        }
        for spec in iter_audit_specs()
    )
    return entries


def list_audits(*, color: bool = True) -> int:
    entries = available_audit_entries()
    id_width = max(len(str(entry["audit_id"])) for entry in entries)
    title_width = max(len(str(entry["title"])) for entry in entries)
    print(_paint("Available audits", "1;96", enabled=color))
    print(_paint("=" * (id_width + title_width + 5), "2", enabled=color))
    for entry in entries:
        audit_id = _paint(str(entry["audit_id"]).ljust(id_width), "1;36", enabled=color)
        title = _paint(_expand_terms(str(entry["title"]), sentence_case=True).ljust(title_width), "1", enabled=color)
        print(f"{audit_id}  {title}  {_expand_terms(str(entry['description']), sentence_case=True)}")
    return 0


def _run_one_audit(spec, argv: list[str], *, allow_unknown: bool = False) -> AuditReport:
    module = spec.load_module()

    audit_parser = argparse.ArgumentParser(description=spec.description)
    audit_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    if hasattr(module, "configure_parser"):
        module.configure_parser(audit_parser)
    if allow_unknown:
        audit_args, _ignored = audit_parser.parse_known_args(argv)
    else:
        audit_args = audit_parser.parse_args(argv)
    return module.run(audit_args)


def _prefixed_items(report: AuditReport) -> list[AuditItem]:
    return [
        AuditItem(
            check_id=f"{report.audit_id}.{item.check_id}",
            status=item.status,
            title=f"{report.title}: {item.title}",
            criterion=item.criterion,
            description=item.description,
            acceptable=item.acceptable,
            acceptable_basis=item.acceptable_basis,
            evidence={"audit_id": report.audit_id, **item.evidence},
            note=item.note,
            human_review_status=item.human_review_status,
            human_review_note=item.human_review_note,
            human_review_reviewer=item.human_review_reviewer,
            group_id=report.audit_id,
            group_title=report.title,
            detail_level=item.detail_level,
        )
        for item in report.items
    ]


def run_new_sweep(argv: list[str]) -> AuditReport:
    reports = [_run_one_audit(spec, argv, allow_unknown=True) for spec in iter_new_sweep_audit_specs()]
    items = [item for report in reports for item in _prefixed_items(report)]
    return AuditReport(
        audit_id="new_sweep",
        title="New sweep",
        items=items,
    )


def _resolve_audit_request(audit_id: str | None, argv: list[str]) -> tuple[str | None, list[str]]:
    normalized = str(audit_id or "").strip()
    args = list(argv)
    if not normalized or normalized in ALL_AUDIT_ALIASES:
        return None, args
    if normalized == DEFAULT_AUDIT_ALIAS:
        return DEFAULT_AUDIT_TARGET, (args if args else list(DEFAULT_AUDIT_TARGET_ARGS))
    return normalized, args


def run_audit_by_id(audit_id: str | None, argv: list[str]) -> AuditReport:
    resolved_audit_id, resolved_args = _resolve_audit_request(audit_id, argv)
    if resolved_audit_id is None:
        return run_new_sweep(resolved_args)
    spec = get_audit_spec(resolved_audit_id)
    return _run_one_audit(spec, resolved_args)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root_parser = _build_root_parser()
    root_args, remainder = root_parser.parse_known_args(argv)
    use_color = not bool(root_args.no_color)

    if root_args.list:
        return list_audits(color=use_color)

    report = run_audit_by_id(root_args.audit_id, remainder)
    if root_args.json:
        print(report.to_json())
    else:
        print(
            format_report(
                report,
                color=use_color,
                expand=bool(root_args.expand),
                failures_only=bool(root_args.failures_only),
            ),
            end="",
        )
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
