"""Typed raw-document layer for declarative reference-validation configs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from olfactorybulb.audit.reference_validation_config import (
    load_reference_validation_config,
)
from olfactorybulb.audit.reference_validation_rule_records import (
    ValidationRuleRecord,
    coerce_validation_rule_records,
)
from olfactorybulb.neuronunit.frozen_payloads import FrozenMappingPayload, coerce_mapping_payload


class ReferenceValidationDefaultsMap(FrozenMappingPayload):
    """Typed frozen wrapper for top-level validation CLI defaults."""


class ReferenceValidationProtocolDefaultsMap(FrozenMappingPayload):
    """Typed frozen wrapper for top-level protocol config defaults."""


class ReferenceValidationSkipEvidenceMap(FrozenMappingPayload):
    """Typed frozen wrapper for declarative skip-item evidence payloads."""


def _optional_table(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Reference validation config '{key}' must be a table")
    return dict(value)


def _validation_title(config: dict[str, Any]) -> str:
    return str(config.get("title") or config.get("validation_id") or "Reference validation")


def _validation_protocol_runner_id(config: dict[str, Any]) -> str:
    return str(config.get("protocol_runner") or "").strip()


def _validation_rule_records(config: dict[str, Any]) -> tuple[ValidationRuleRecord, ...]:
    rules = config.get("checks", [])
    if not isinstance(rules, list):
        raise ValueError("Reference validation config 'checks' must be an array of tables")
    return coerce_validation_rule_records(dict(rule) for rule in rules)


def _validation_extension_specs(config: dict[str, Any]) -> tuple[str, ...]:
    raw = config.get("extensions", [])
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("Reference validation config 'extensions' must be an array of module specs")
    return tuple(str(spec).strip() for spec in raw if str(spec).strip())


def _validation_skip_neuron_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("skip_neuron_mode", "short_circuit") or "short_circuit").strip()
    if mode not in {"short_circuit", "protocol_handles_skip"}:
        raise ValueError(
            "Reference validation config 'skip_neuron_mode' must be "
            "'short_circuit' or 'protocol_handles_skip'"
        )
    return mode


@dataclass(frozen=True)
class ValidationDesignReviewDefaultsSpec:
    status: str = ""
    note: str = ""
    reviewer: str = ""
    required_expertise: str = ""
    focus: str = ""

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ValidationDesignReviewDefaultsSpec":
        defaults = _optional_table(config, "validation_design_review")
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
    evidence: ReferenceValidationSkipEvidenceMap | Mapping[str, object] = field(default_factory=dict)
    evidence_arg_keys: tuple[str, ...] = ()
    note: str = ""
    validation_design_review_status: str = ""
    validation_design_review_note: str = ""
    validation_design_review_reviewer: str = ""
    validation_design_review_required_expertise: str = ""
    validation_design_review_focus: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence",
            coerce_mapping_payload(self.evidence, payload_type=ReferenceValidationSkipEvidenceMap)
            or ReferenceValidationSkipEvidenceMap(entries=()),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ReferenceValidationSkipItemSpec | None":
        spec = config.get("skip_item")
        if spec is None:
            return None
        if not isinstance(spec, dict):
            raise ValueError("Reference validation config 'skip_item' must be a table")
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
    defaults: ReferenceValidationDefaultsMap | Mapping[str, object]
    protocol_defaults: ReferenceValidationProtocolDefaultsMap | Mapping[str, object]
    rule_records: tuple[ValidationRuleRecord, ...]
    design_review_defaults: ValidationDesignReviewDefaultsSpec
    skip_item: ReferenceValidationSkipItemSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "defaults",
            coerce_mapping_payload(self.defaults, payload_type=ReferenceValidationDefaultsMap)
            or ReferenceValidationDefaultsMap(entries=()),
        )
        object.__setattr__(
            self,
            "protocol_defaults",
            coerce_mapping_payload(self.protocol_defaults, payload_type=ReferenceValidationProtocolDefaultsMap)
            or ReferenceValidationProtocolDefaultsMap(entries=()),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ReferenceValidationDocument":
        return cls(
            validation_id=str(config.get("validation_id", "")).strip(),
            title=_validation_title(config),
            config_path=str(config.get("__path__", "")).strip(),
            extension_specs=_validation_extension_specs(config),
            protocol_runner_id=_validation_protocol_runner_id(config),
            metric_group_field=str(config.get("metric_group_field", "")).strip(),
            default_group=str(config.get("default_group", "")).strip(),
            notes_path=str(config.get("notes_path", "")).strip(),
            skip_neuron_mode=_validation_skip_neuron_mode(config),
            defaults=_optional_table(config, "defaults"),
            protocol_defaults=_optional_table(config, "protocol"),
            rule_records=_validation_rule_records(config),
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
    "ReferenceValidationDefaultsMap",
    "ReferenceValidationProtocolDefaultsMap",
    "ReferenceValidationSkipItemSpec",
    "ReferenceValidationSkipEvidenceMap",
    "ValidationRuleRecord",
    "ValidationDesignReviewDefaultsSpec",
    "load_reference_validation_document",
]
