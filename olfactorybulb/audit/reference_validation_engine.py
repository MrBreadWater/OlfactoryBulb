"""Shared engine for declarative literature-validation audits."""

from __future__ import annotations

import argparse
from typing import Any, Iterable

from olfactorybulb.audit import AuditItem, AuditReport
from olfactorybulb.audit.reference_validation_plan import (
    ReferenceValidationPlan,
    load_reference_validation_plan,
)


def add_reference_validation_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--skip-neuron", action="store_true", help="Skip expensive NEURON-backed protocol execution.")
    parser.add_argument(
        "--reference-sigma-multiplier",
        type=float,
        default=2.0,
        help="Width of configurable reference acceptance bands in standard deviations. Default: 2.0.",
    )


def add_reference_validation_protocol_args(
    parser: argparse.ArgumentParser,
    *,
    validation: ReferenceValidationPlan,
) -> None:
    validation.add_protocol_args(parser)


def apply_validation_defaults(args: argparse.Namespace, *, validation: ReferenceValidationPlan) -> argparse.Namespace:
    return validation.apply_defaults(args)


def build_reference_validation_items(
    *,
    metrics: list[dict[str, Any]],
    args: argparse.Namespace,
    validation: ReferenceValidationPlan,
    protocol_result: Any | None,
) -> list:
    return validation.build_rule_items(
        metrics=metrics,
        args=args,
        protocol_result=protocol_result,
    )


def build_configured_skip_item(
    *,
    args: argparse.Namespace,
    validation: ReferenceValidationPlan,
) -> AuditItem | None:
    return validation.build_skip_item(args=args)


def run_reference_validation(
    *,
    args: argparse.Namespace,
    validation: ReferenceValidationPlan,
    audit_id: str,
    title: str | None = None,
    pre_items: Iterable | None = None,
    skip_item=None,
) -> AuditReport:
    args = apply_validation_defaults(args, validation=validation)
    items = list(pre_items or [])
    if bool(getattr(args, "skip_neuron", False)) and validation.skip_neuron_mode != "protocol_handles_skip":
        configured_skip_item = build_configured_skip_item(args=args, validation=validation)
        if skip_item is not None:
            items.append(skip_item)
        elif configured_skip_item is not None:
            items.append(configured_skip_item)
        return AuditReport(
            audit_id=audit_id,
            title=title or validation.title,
            items=items,
        )

    protocol_result = validation.protocol_spec.run(args, dict(validation.protocol_defaults))
    items.extend(
        build_reference_validation_items(
            metrics=protocol_result.metrics,
            args=args,
            validation=validation,
            protocol_result=protocol_result,
        )
    )
    return AuditReport(
        audit_id=audit_id,
        title=title or validation.title,
        items=items,
    )


def load_validation_and_protocol(
    *,
    validation_id: str | None = None,
    config_path=None,
) -> tuple[ReferenceValidationPlan, Any]:
    validation = load_reference_validation_plan(validation_id=validation_id, path=config_path)
    return validation, validation.protocol_spec


__all__ = [
    "add_reference_validation_common_args",
    "add_reference_validation_protocol_args",
    "apply_validation_defaults",
    "build_configured_skip_item",
    "build_reference_validation_items",
    "load_reference_validation_plan",
    "load_validation_and_protocol",
    "run_reference_validation",
]
