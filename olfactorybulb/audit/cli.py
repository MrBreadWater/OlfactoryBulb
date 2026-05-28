"""CLI entrypoint for repository audits."""

from __future__ import annotations

import argparse
import sys

from olfactorybulb.audit import AuditItem, AuditReport, format_report, get_audit_spec, iter_audit_specs


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
    return parser


def list_audits() -> int:
    for spec in iter_audit_specs():
        print(f"{spec.audit_id}\t{spec.title}\t{spec.description}")
    return 0


def _run_one_audit(spec, argv: list[str]) -> AuditReport:
    module = spec.load_module()

    audit_parser = argparse.ArgumentParser(description=spec.description)
    audit_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    if hasattr(module, "configure_parser"):
        module.configure_parser(audit_parser)
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
    reports = [_run_one_audit(spec, argv) for spec in iter_audit_specs()]
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

    if root_args.list:
        return list_audits()

    if not root_args.audit_id or root_args.audit_id in {"new_sweep", "new-sweep", "all"}:
        report = run_new_sweep(remainder)
        if root_args.json:
            print(report.to_json())
        else:
            print(format_report(report), end="")
        return report.exit_code

    spec = get_audit_spec(root_args.audit_id)
    report = _run_one_audit(spec, remainder)
    if root_args.json:
        print(report.to_json())
    else:
        print(format_report(report), end="")
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
