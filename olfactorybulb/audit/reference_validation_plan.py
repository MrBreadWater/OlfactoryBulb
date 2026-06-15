"""Typed top-level runtime plan for declarative reference validations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from olfactorybulb.audit import AuditItem
from olfactorybulb.audit.reference_validation_config import (
    load_reference_validation_config,
    load_validation_extensions,
    validation_defaults,
    validation_design_review_defaults,
    validation_extension_specs,
    validation_protocol_defaults,
    validation_protocol_runner_id,
    validation_rule_specs,
    validation_skip_item,
    validation_skip_neuron_mode,
    validation_title,
)
from olfactorybulb.audit.reference_validation_protocols import (
    ValidationProtocolSpec,
    get_validation_protocol_spec,
)
from olfactorybulb.audit.reference_validation_rules import (
    ValidationRuleContext,
    build_rule_items,
    summarize_numeric_metrics,
)


@dataclass(frozen=True)
class ValidationDesignReviewDefaultsSpec:
    status: str = ""
    note: str = ""
    reviewer: str = ""
    required_expertise: str = ""
    focus: str = ""

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ValidationDesignReviewDefaultsSpec":
        defaults = validation_design_review_defaults(config)
        return cls(
            status=str(defaults.get("default_status", "")).strip(),
            note=str(defaults.get("default_note", "")).strip(),
            reviewer=str(defaults.get("default_reviewer", "")).strip(),
            required_expertise=str(defaults.get("default_required_expertise", "")).strip(),
            focus=str(defaults.get("default_focus", "")).strip(),
        )


@dataclass(frozen=True)
class ReferenceValidationSkipItemSpec:
    check_id: str
    status: str
    title: str
    criterion: str
    criterion_latex: str = ""
    criterion_formulae: tuple[str, ...] = ()
    criterion_definitions: tuple[dict[str, Any], ...] = ()
    description: str = ""
    acceptable: str = ""
    acceptable_basis: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    evidence_arg_keys: tuple[str, ...] = ()
    note: str = ""
    validation_design_review_status: str = ""
    validation_design_review_note: str = ""
    validation_design_review_reviewer: str = ""
    validation_design_review_required_expertise: str = ""
    validation_design_review_focus: str = ""

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ReferenceValidationSkipItemSpec | None":
        spec = validation_skip_item(config)
        if spec is None:
            return None
        return cls(
            check_id=str(spec["check_id"]),
            status=str(spec.get("status", "WARN")),
            title=str(spec["title"]),
            criterion=str(spec["criterion"]),
            criterion_latex=str(spec.get("criterion_latex", "")),
            criterion_formulae=tuple(str(value) for value in spec.get("criterion_formulae", [])),
            criterion_definitions=tuple(
                dict(value) for value in spec.get("criterion_definitions", []) if isinstance(value, dict)
            ),
            description=str(spec.get("description", "")),
            acceptable=str(spec.get("acceptable", "")),
            acceptable_basis=str(spec.get("acceptable_basis", "")),
            evidence=dict(spec.get("evidence", {})),
            evidence_arg_keys=tuple(str(key) for key in spec.get("evidence_arg_keys", [])),
            note=str(spec.get("note", "")),
            validation_design_review_status=str(spec.get("validation_design_review_status", "")).strip(),
            validation_design_review_note=str(spec.get("validation_design_review_note", "")).strip(),
            validation_design_review_reviewer=str(spec.get("validation_design_review_reviewer", "")).strip(),
            validation_design_review_required_expertise=str(
                spec.get("validation_design_review_required_expertise", "")
            ).strip(),
            validation_design_review_focus=str(spec.get("validation_design_review_focus", "")).strip(),
        )

    def to_audit_item(
        self,
        *,
        args: argparse.Namespace,
        review_defaults: ValidationDesignReviewDefaultsSpec,
    ) -> AuditItem:
        evidence = dict(self.evidence)
        for key in self.evidence_arg_keys:
            evidence[key] = getattr(args, key, None)
        return AuditItem(
            check_id=self.check_id,
            status=self.status,
            title=self.title,
            criterion=self.criterion,
            criterion_latex=self.criterion_latex,
            criterion_formulae=list(self.criterion_formulae),
            criterion_definitions=[dict(value) for value in self.criterion_definitions],
            description=self.description,
            acceptable=self.acceptable,
            acceptable_basis=self.acceptable_basis,
            evidence=evidence,
            note=self.note,
            validation_design_review_status=self.validation_design_review_status or review_defaults.status,
            validation_design_review_note=self.validation_design_review_note or review_defaults.note,
            validation_design_review_reviewer=self.validation_design_review_reviewer or review_defaults.reviewer,
            validation_design_review_required_expertise=(
                self.validation_design_review_required_expertise or review_defaults.required_expertise
            ),
            validation_design_review_focus=self.validation_design_review_focus or review_defaults.focus,
        )


@dataclass(frozen=True)
class ReferenceValidationPlan:
    validation_id: str
    title: str
    config_path: str
    extension_specs: tuple[str, ...]
    metric_group_field: str
    skip_neuron_mode: str
    defaults: dict[str, Any]
    protocol_defaults: dict[str, Any]
    rules: tuple[dict[str, Any], ...]
    rule_context_config: dict[str, Any]
    protocol_runner_id: str
    protocol_spec: ValidationProtocolSpec
    design_review_defaults: ValidationDesignReviewDefaultsSpec
    skip_item: ReferenceValidationSkipItemSpec | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ReferenceValidationPlan":
        load_validation_extensions(config)
        protocol_runner_id = validation_protocol_runner_id(config)
        return cls(
            validation_id=str(config.get("validation_id", "")).strip(),
            title=validation_title(config),
            config_path=str(config.get("__path__", "")).strip(),
            extension_specs=tuple(validation_extension_specs(config)),
            metric_group_field=str(config.get("metric_group_field", "")).strip(),
            skip_neuron_mode=validation_skip_neuron_mode(config),
            defaults=validation_defaults(config),
            protocol_defaults=validation_protocol_defaults(config),
            rules=tuple(dict(rule) for rule in validation_rule_specs(config)),
            rule_context_config={
                "validation_id": str(config.get("validation_id", "")).strip(),
                "notes_path": str(config.get("notes_path", "")).strip(),
                "default_group": str(config.get("default_group", "")).strip(),
                "validation_design_review": dict(validation_design_review_defaults(config)),
            },
            protocol_runner_id=protocol_runner_id,
            protocol_spec=get_validation_protocol_spec(protocol_runner_id),
            design_review_defaults=ValidationDesignReviewDefaultsSpec.from_config(config),
            skip_item=ReferenceValidationSkipItemSpec.from_config(config),
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
            config=dict(self.rule_context_config),
            protocol_result=protocol_result,
        )
        return build_rule_items([dict(rule) for rule in self.rules], context)

    def build_skip_item(self, *, args: argparse.Namespace) -> AuditItem | None:
        if self.skip_item is None:
            return None
        return self.skip_item.to_audit_item(args=args, review_defaults=self.design_review_defaults)


def load_reference_validation_plan(
    *,
    validation_id: str | None = None,
    path: Path | None = None,
) -> ReferenceValidationPlan:
    return ReferenceValidationPlan.from_config(
        load_reference_validation_config(validation_id=validation_id, path=path)
    )


__all__ = [
    "ReferenceValidationPlan",
    "ReferenceValidationSkipItemSpec",
    "ValidationDesignReviewDefaultsSpec",
    "load_reference_validation_plan",
]
