"""CLI entrypoint for repository audits."""

from __future__ import annotations

import argparse
import sys

from olfactorybulb.audit import format_report, get_audit_spec, iter_audit_specs


def _build_root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_id", nargs="?", help="Audit id to run. Use --list to inspect available audits.")
    parser.add_argument("--list", action="store_true", help="List available audits and exit.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def list_audits() -> int:
    for spec in iter_audit_specs():
        print(f"{spec.audit_id}\t{spec.title}\t{spec.description}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root_parser = _build_root_parser()
    root_args, remainder = root_parser.parse_known_args(argv)

    if root_args.list:
        return list_audits()

    if not root_args.audit_id:
        root_parser.error("audit_id is required unless --list is used")

    spec = get_audit_spec(root_args.audit_id)
    module = spec.load_module()

    audit_parser = argparse.ArgumentParser(description=spec.description)
    audit_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    if hasattr(module, "configure_parser"):
        module.configure_parser(audit_parser)
    audit_args = audit_parser.parse_args(remainder)

    report = module.run(audit_args)
    if audit_args.json or root_args.json:
        print(report.to_json())
    else:
        print(format_report(report), end="")
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
