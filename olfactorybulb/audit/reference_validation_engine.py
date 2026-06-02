"""Shared engine for declarative literature-validation audits."""

from __future__ import annotations

import argparse
from typing import Any, Iterable

from olfactorybulb.audit import AuditItem, AuditReport
from olfactorybulb.audit.reference_validation_config import (
    load_validation_extensions,
    load_reference_validation_config,
    validation_defaults,
    validation_human_review_defaults,
    validation_protocol_defaults,
    validation_protocol_runner_id,
    validation_rule_specs,
    validation_skip_item,
    validation_skip_neuron_mode,
    validation_title,
)
from olfactorybulb.audit.reference_validation_protocols import get_validation_protocol_spec
from olfactorybulb.audit.reference_validation_rules import ValidationRuleContext, build_rule_items, summarize_numeric_metrics


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
    config: dict[str, Any],
) -> None:
    load_validation_extensions(config)
    protocol_id = validation_protocol_runner_id(config)
    if not protocol_id:
        return
    spec = get_validation_protocol_spec(protocol_id)
    if spec.add_cli_args is not None:
        spec.add_cli_args(parser)


def apply_validation_defaults(args: argparse.Namespace, *, config: dict[str, Any]) -> argparse.Namespace:
    defaults = validation_defaults(config)
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
            continue
        current = getattr(args, key)
        if current is None:
            setattr(args, key, value)
    return args


def build_reference_validation_items(
    *,
    metrics: list[dict[str, Any]],
    args: argparse.Namespace,
    config: dict[str, Any],
    protocol_result: Any | None,
) -> list:
    group_field = str(config.get("metric_group_field", getattr(protocol_result, "group_field", "cell_type")))
    summary = summarize_numeric_metrics(metrics, group_field=group_field)
    context = ValidationRuleContext(
        metrics=metrics,
        summary=summary,
        args=args,
        config=config,
        protocol_result=protocol_result,
    )
    return build_rule_items(validation_rule_specs(config), context)


def build_configured_skip_item(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> AuditItem | None:
    spec = validation_skip_item(config)
    if not spec:
        return None
    evidence = dict(spec.get("evidence", {}))
    for key in list(spec.get("evidence_arg_keys", [])):
        evidence[str(key)] = getattr(args, str(key), None)
    review_defaults = validation_human_review_defaults(config)
    return AuditItem(
        check_id=str(spec["check_id"]),
        status=str(spec.get("status", "WARN")),
        title=str(spec["title"]),
        criterion=str(spec["criterion"]),
        description=str(spec.get("description", "")),
        acceptable=str(spec.get("acceptable", "")),
        acceptable_basis=str(spec.get("acceptable_basis", "")),
        evidence=evidence,
        note=str(spec.get("note", "")),
        human_review_status=str(spec.get("human_review_status", review_defaults.get("default_status", ""))),
        human_review_note=str(spec.get("human_review_note", review_defaults.get("default_note", ""))),
        human_review_reviewer=str(spec.get("human_review_reviewer", review_defaults.get("default_reviewer", ""))),
    )


def run_reference_validation(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    audit_id: str,
    title: str | None = None,
    pre_items: Iterable | None = None,
    skip_item=None,
) -> AuditReport:
    args = apply_validation_defaults(args, config=config)
    items = list(pre_items or [])
    if bool(getattr(args, "skip_neuron", False)) and validation_skip_neuron_mode(config) != "protocol_handles_skip":
        configured_skip_item = build_configured_skip_item(args=args, config=config)
        if skip_item is not None:
            items.append(skip_item)
        elif configured_skip_item is not None:
            items.append(configured_skip_item)
        return AuditReport(
            audit_id=audit_id,
            title=title or validation_title(config),
            items=items,
        )

    load_validation_extensions(config)
    protocol_id = validation_protocol_runner_id(config)
    protocol_spec = get_validation_protocol_spec(protocol_id)
    protocol_result = protocol_spec.run(args, validation_protocol_defaults(config))
    items.extend(
        build_reference_validation_items(
            metrics=protocol_result.metrics,
            args=args,
            config=config,
            protocol_result=protocol_result,
        )
    )
    return AuditReport(
        audit_id=audit_id,
        title=title or validation_title(config),
        items=items,
    )


def load_validation_and_protocol(
    *,
    validation_id: str | None = None,
    config_path=None,
) -> tuple[dict[str, Any], Any]:
    config = load_reference_validation_config(validation_id=validation_id, path=config_path)
    load_validation_extensions(config)
    protocol_spec = get_validation_protocol_spec(validation_protocol_runner_id(config))
    return config, protocol_spec


__all__ = [
    "add_reference_validation_common_args",
    "add_reference_validation_protocol_args",
    "apply_validation_defaults",
    "build_configured_skip_item",
    "build_reference_validation_items",
    "load_reference_validation_config",
    "load_validation_and_protocol",
    "run_reference_validation",
]
