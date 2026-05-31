"""CLI entrypoint for repository audits."""

from __future__ import annotations

import argparse
import sys

from olfactorybulb.audit import AuditItem, AuditReport, format_report, get_audit_spec, iter_audit_specs
from olfactorybulb.audit.core import _expand_terms


def _build_root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audit_id",
        nargs="?",
        help=(
            "Audit id to run. Omit, or pass 'new_sweep', to start a new sweep "
            "across every registered audit. Use --list to inspect available audits."
        ),
    )
    parser.add_argument("--list", action="store_true", help="List available audits and exit.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color output for text reports.")
    return parser


def _paint(text: str, code: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


def list_audits(*, color: bool = True) -> int:
    specs = list(iter_audit_specs())
    id_width = max(len(spec.audit_id) for spec in specs)
    title_width = max(len(spec.title) for spec in specs)
    print(_paint("Available audits", "1;96", enabled=color))
    print(_paint("=" * (id_width + title_width + 5), "2", enabled=color))
    for spec in specs:
        audit_id = _paint(spec.audit_id.ljust(id_width), "1;36", enabled=color)
        title = _paint(_expand_terms(spec.title, sentence_case=True).ljust(title_width), "1", enabled=color)
        print(f"{audit_id}  {title}  {_expand_terms(spec.description, sentence_case=True)}")
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
            title=f"{report.audit_id}: {item.title}",
            criterion=item.criterion,
            evidence={"audit_id": report.audit_id, **item.evidence},
            note=item.note,
        )
        for item in report.items
    ]


def run_new_sweep(argv: list[str]) -> AuditReport:
    reports = [_run_one_audit(spec, argv, allow_unknown=True) for spec in iter_audit_specs()]
    items = [item for report in reports for item in _prefixed_items(report)]
    return AuditReport(
        audit_id="new_sweep",
        title="New sweep",
        items=items,
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root_parser = _build_root_parser()
    root_args, remainder = root_parser.parse_known_args(argv)
    use_color = not bool(root_args.no_color)

    if root_args.list:
        return list_audits(color=use_color)

    if not root_args.audit_id or root_args.audit_id in {"new_sweep", "new-sweep", "all"}:
        report = run_new_sweep(remainder)
        if root_args.json:
            print(report.to_json())
        else:
            print(format_report(report, color=use_color), end="")
        return report.exit_code

    spec = get_audit_spec(root_args.audit_id)
    report = _run_one_audit(spec, remainder)
    if root_args.json:
        print(report.to_json())
    else:
        print(format_report(report, color=use_color), end="")
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
