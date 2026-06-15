"""Typed top-level runtime plan for declarative reference validations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from olfactorybulb.audit import AuditItem
from olfactorybulb.audit.reference_validation_config import load_validation_extensions
from olfactorybulb.audit.reference_validation_document import (
    ReferenceValidationDocument,
    ReferenceValidationSkipItemSpec,
    ValidationDesignReviewDefaultsSpec,
    load_reference_validation_document,
)
from olfactorybulb.audit.reference_validation_protocols import (
    ValidationProtocolSpec,
    get_validation_protocol_spec,
)
from olfactorybulb.audit.reference_validation_rules import (
    ValidationRuleDispatch,
    ValidationRuleContext,
    build_rule_items,
    compile_rule_dispatches,
    summarize_numeric_metrics,
)


def _skip_item_to_audit_item(
    skip_item: ReferenceValidationSkipItemSpec,
    *,
    args: argparse.Namespace,
    review_defaults: ValidationDesignReviewDefaultsSpec,
) -> AuditItem:
    evidence = dict(skip_item.evidence)
    for key in skip_item.evidence_arg_keys:
        evidence[key] = getattr(args, key, None)
    return AuditItem(
        check_id=skip_item.check_id,
        status=skip_item.status,
        title=skip_item.title,
        criterion=skip_item.criterion,
        criterion_latex=skip_item.criterion_latex,
        criterion_formulae=list(skip_item.criterion_formulae),
        criterion_definitions=[dict(value) for value in skip_item.criterion_definitions],
        description=skip_item.description,
        acceptable=skip_item.acceptable,
        acceptable_basis=skip_item.acceptable_basis,
        evidence=evidence,
        note=skip_item.note,
        validation_design_review_status=skip_item.validation_design_review_status or review_defaults.status,
        validation_design_review_note=skip_item.validation_design_review_note or review_defaults.note,
        validation_design_review_reviewer=skip_item.validation_design_review_reviewer or review_defaults.reviewer,
        validation_design_review_required_expertise=(
            skip_item.validation_design_review_required_expertise or review_defaults.required_expertise
        ),
        validation_design_review_focus=skip_item.validation_design_review_focus or review_defaults.focus,
    )


@dataclass(frozen=True)
class ReferenceValidationPlan:
    validation_id: str
    title: str
    config_path: str
    extension_specs: tuple[str, ...]
    metric_group_field: str
    default_group: str
    notes_path: str
    skip_neuron_mode: str
    defaults: dict[str, Any]
    protocol_defaults: dict[str, Any]
    rules: tuple[dict[str, Any], ...]
    rule_dispatches: tuple[ValidationRuleDispatch, ...]
    protocol_runner_id: str
    protocol_spec: ValidationProtocolSpec
    design_review_defaults: ValidationDesignReviewDefaultsSpec
    skip_item: ReferenceValidationSkipItemSpec | None = None

    @classmethod
    def from_document(cls, document: ReferenceValidationDocument) -> "ReferenceValidationPlan":
        load_validation_extensions(document.extension_specs)
        return cls(
            validation_id=document.validation_id,
            title=document.title,
            config_path=document.config_path,
            extension_specs=document.extension_specs,
            metric_group_field=document.metric_group_field,
            default_group=document.default_group,
            notes_path=document.notes_path,
            skip_neuron_mode=document.skip_neuron_mode,
            defaults=dict(document.defaults),
            protocol_defaults=dict(document.protocol_defaults),
            rules=tuple(dict(rule) for rule in document.rules),
            rule_dispatches=compile_rule_dispatches(document.rules),
            protocol_runner_id=document.protocol_runner_id,
            protocol_spec=get_validation_protocol_spec(document.protocol_runner_id),
            design_review_defaults=document.design_review_defaults,
            skip_item=document.skip_item,
        )

    def add_protocol_args(self, parser: argparse.ArgumentParser) -> None:
        if self.protocol_spec.add_cli_args is not None:
            self.protocol_spec.add_cli_args(parser)

    def apply_defaults(self, args: argparse.Namespace) -> argparse.Namespace:
        for key, value in self.defaults.items():
            if not hasattr(args, key):
                setattr(args, key, value)
                continue
            if getattr(args, key) is None:
                setattr(args, key, value)
        return args

    def resolved_group_field(self, protocol_result: Any | None) -> str:
        if self.metric_group_field:
            return self.metric_group_field
        return str(getattr(protocol_result, "group_field", "cell_type"))

    def build_rule_items(
        self,
        *,
        metrics: list[dict[str, Any]],
        args: argparse.Namespace,
        protocol_result: Any | None,
    ) -> list[AuditItem]:
        summary = summarize_numeric_metrics(
            metrics,
            group_field=self.resolved_group_field(protocol_result),
        )
        context = ValidationRuleContext(
            metrics=metrics,
            summary=summary,
            args=args,
            validation_id=self.validation_id,
            default_group=self.default_group,
            notes_path=self.notes_path,
            design_review_defaults=self.design_review_defaults,
            protocol_result=protocol_result,
        )
        return build_rule_items(self.rule_dispatches, context)

    def build_skip_item(self, *, args: argparse.Namespace) -> AuditItem | None:
        if self.skip_item is None:
            return None
        return _skip_item_to_audit_item(
            self.skip_item,
            args=args,
            review_defaults=self.design_review_defaults,
        )


def load_reference_validation_plan(
    *,
    validation_id: str | None = None,
    path: Path | None = None,
) -> ReferenceValidationPlan:
    return ReferenceValidationPlan.from_document(
        load_reference_validation_document(validation_id=validation_id, path=path)
    )


__all__ = [
    "ReferenceValidationPlan",
    "load_reference_validation_plan",
]
