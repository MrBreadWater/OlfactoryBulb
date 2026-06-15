"""Typed raw-document layer for declarative reference-validation configs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from olfactorybulb.audit.reference_validation_config import (
    load_reference_validation_config,
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


@dataclass(frozen=True)
class ReferenceValidationDocument:
    validation_id: str
    title: str
    config_path: str
    extension_specs: tuple[str, ...]
    protocol_runner_id: str
    metric_group_field: str
    default_group: str
    notes_path: str
    skip_neuron_mode: str
    defaults: dict[str, Any]
    protocol_defaults: dict[str, Any]
    rules: tuple[dict[str, Any], ...]
    design_review_defaults: ValidationDesignReviewDefaultsSpec
    skip_item: ReferenceValidationSkipItemSpec | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ReferenceValidationDocument":
        return cls(
            validation_id=str(config.get("validation_id", "")).strip(),
            title=validation_title(config),
            config_path=str(config.get("__path__", "")).strip(),
            extension_specs=tuple(validation_extension_specs(config)),
            protocol_runner_id=validation_protocol_runner_id(config),
            metric_group_field=str(config.get("metric_group_field", "")).strip(),
            default_group=str(config.get("default_group", "")).strip(),
            notes_path=str(config.get("notes_path", "")).strip(),
            skip_neuron_mode=validation_skip_neuron_mode(config),
            defaults=validation_defaults(config),
            protocol_defaults=validation_protocol_defaults(config),
            rules=tuple(dict(rule) for rule in validation_rule_specs(config)),
            design_review_defaults=ValidationDesignReviewDefaultsSpec.from_config(config),
            skip_item=ReferenceValidationSkipItemSpec.from_config(config),
        )


def load_reference_validation_document(
    *,
    validation_id: str | None = None,
    path: Path | None = None,
) -> ReferenceValidationDocument:
    return ReferenceValidationDocument.from_config(
        load_reference_validation_config(validation_id=validation_id, path=path)
    )


__all__ = [
    "ReferenceValidationDocument",
    "ReferenceValidationSkipItemSpec",
    "ValidationDesignReviewDefaultsSpec",
    "load_reference_validation_document",
]
