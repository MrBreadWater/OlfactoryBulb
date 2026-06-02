"""Run a declarative literature-validation configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from olfactorybulb.audit import format_report  # noqa: E402
from olfactorybulb.audit.reference_validation_config import (  # noqa: E402
    DEFAULT_REFERENCE_VALIDATION_ID,
    list_reference_validation_ids,
    load_validation_extensions,
    load_reference_validation_config,
    validation_title,
)
from olfactorybulb.audit.reference_validation_engine import (  # noqa: E402
    add_reference_validation_common_args,
    add_reference_validation_protocol_args,
    run_reference_validation,
)
from olfactorybulb.audit.reference_validation_protocols import iter_validation_protocol_specs  # noqa: E402


def _root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-id", default=DEFAULT_REFERENCE_VALIDATION_ID, help="Validation config id to run.")
    parser.add_argument("--config-path", type=Path, default=None, help="Optional explicit validation config path.")
    parser.add_argument("--list-validations", action="store_true", help="List available validation configs and exit.")
    parser.add_argument("--list-protocols", action="store_true", help="List registered protocol runners and exit.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color output for text reports.")
    return parser


def _list_validations() -> int:
    print("Available reference validations")
    print("=============================")
    for validation_id in list_reference_validation_ids():
        config = load_reference_validation_config(validation_id=validation_id)
        print(f"- {validation_id}: {validation_title(config)}")
    return 0


def _list_protocols(root_args: argparse.Namespace) -> int:
    if root_args.config_path is not None or root_args.validation_id:
        config = load_reference_validation_config(validation_id=root_args.validation_id, path=root_args.config_path)
        load_validation_extensions(config)
    print("Registered validation protocols")
    print("==============================")
    for spec in iter_validation_protocol_specs():
        print(f"- {spec.protocol_id}: {spec.title}")
        print(f"  {spec.description}")
    return 0


def main(argv: list[str] | None = None) -> int:
    root = _root_parser()
    root_args, remainder = root.parse_known_args(argv)
    if root_args.list_validations:
        return _list_validations()
    if root_args.list_protocols:
        return _list_protocols(root_args)

    config = load_reference_validation_config(validation_id=root_args.validation_id, path=root_args.config_path)
    load_validation_extensions(config)
    parser = argparse.ArgumentParser(description=validation_title(config))
    add_reference_validation_common_args(parser)
    add_reference_validation_protocol_args(parser, config=config)
    args = parser.parse_args(remainder)
    report = run_reference_validation(
        args=args,
        config=config,
        audit_id=str(config.get("validation_id")),
        title=validation_title(config),
    )
    if root_args.json:
        print(report.to_json())
    else:
        print(format_report(report, color=not root_args.no_color), end="")
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
