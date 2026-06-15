"""Shared typed evidence payload helpers for maintained NeuronUnit bridges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from olfactorybulb.audit.core import rounded
from olfactorybulb.neuronunit.evidence_formatting import rounded_evidence_mapping
from olfactorybulb.neuronunit.frozen_payloads import FrozenMappingPayload, coerce_mapping_payload
from olfactorybulb.neuronunit.scalar_observations import (
    ScalarGroupPair,
    ScalarGroupValueSet,
    ScalarMetricValue,
    ScalarMetricValueMap,
    ScalarStatusMapPolicy,
)


class ScalarEvidenceValues(FrozenMappingPayload):
    """Frozen scalar evidence values for maintained suite bridges."""


@dataclass(frozen=True)
class ScalarRuleEvidencePayload:
    """Typed scalar/detail evidence for summary and comparison rule suites."""

    values: ScalarEvidenceValues | Mapping[str, object]
    metric_unit: str = ""
    metric_quantity_name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            coerce_mapping_payload(self.values, payload_type=ScalarEvidenceValues),
        )
        object.__setattr__(self, "metric_unit", str(self.metric_unit or "").strip())
        object.__setattr__(self, "metric_quantity_name", str(self.metric_quantity_name or "").strip())

    @classmethod
    def summary_rule(
        cls,
        *,
        observed: ScalarMetricValue,
        rule_kind: str,
        minimum: float | None = None,
        maximum: float | None = None,
        status_map_policy: ScalarStatusMapPolicy | None = None,
        extra_metrics: Mapping[str, object] | None = None,
    ) -> "ScalarRuleEvidencePayload":
        values: dict[str, object] = {
            "group": observed.group,
            "observed": observed.numeric,
        }
        if rule_kind == "summary_metric_min" and minimum is not None:
            values["minimum"] = float(minimum)
        elif rule_kind == "summary_metric_max" and maximum is not None:
            values["maximum"] = float(maximum)
        elif rule_kind == "summary_metric_range":
            if minimum is not None:
                values["minimum"] = float(minimum)
            if maximum is not None:
                values["maximum"] = float(maximum)
        elif rule_kind == "summary_metric_status_map" and status_map_policy is not None:
            values.update(status_map_policy.observation_payload())
        if extra_metrics:
            values.update(dict(extra_metrics))
        return cls(
            values=values,
            metric_unit=observed.unit_text,
            metric_quantity_name=observed.quantity_name,
        )

    @classmethod
    def all_finite(
        cls,
        *,
        prediction: ScalarMetricValueMap,
        failing_values: Mapping[str, object],
    ) -> "ScalarRuleEvidencePayload":
        return cls(
            values={
                "metric_key": prediction.metric_key,
                "cell_count": prediction.entity_count,
                "failing_values": dict(failing_values),
            },
            metric_unit=prediction.unit_text,
            metric_quantity_name=prediction.quantity_name,
        )

    @classmethod
    def all_exact(
        cls,
        *,
        prediction: ScalarMetricValueMap,
        expected: float,
        tolerance: float,
        failing_values: Mapping[str, object],
    ) -> "ScalarRuleEvidencePayload":
        return cls(
            values={
                "metric_key": prediction.metric_key,
                "expected": float(expected),
                "tolerance": float(tolerance),
                "failing_values": dict(failing_values),
            },
            metric_unit=prediction.unit_text,
            metric_quantity_name=prediction.quantity_name,
        )

    @classmethod
    def group_ordering(
        cls,
        *,
        prediction: ScalarGroupPair,
    ) -> "ScalarRuleEvidencePayload":
        return cls(
            values={
                f"{prediction.left_group}_mean": prediction.left_numeric,
                f"{prediction.right_group}_mean": prediction.right_numeric,
                f"{prediction.right_group}_minus_{prediction.left_group}": prediction.delta,
            },
            metric_unit=prediction.unit_text,
            metric_quantity_name=prediction.quantity_name,
        )

    @classmethod
    def group_abs_diff_max(
        cls,
        *,
        prediction: ScalarGroupPair,
        max_difference: float,
    ) -> "ScalarRuleEvidencePayload":
        return cls(
            values={
                f"{prediction.left_group}_mean": prediction.left_numeric,
                f"{prediction.right_group}_mean": prediction.right_numeric,
                "absolute_difference": prediction.absolute_difference,
                "max_difference": float(max_difference),
            },
            metric_unit=prediction.unit_text,
            metric_quantity_name=prediction.quantity_name,
        )

    @classmethod
    def group_positive(
        cls,
        *,
        prediction: ScalarGroupValueSet,
        failing_groups: Sequence[str],
    ) -> "ScalarRuleEvidencePayload":
        values: dict[str, object] = {
            f"{group}_mean": value
            for group, value in prediction.numeric_group_values().items()
        }
        if failing_groups:
            values["failing_groups"] = [str(group) for group in failing_groups]
        return cls(
            values=values,
            metric_unit=prediction.unit_text,
            metric_quantity_name=prediction.quantity_name,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = rounded_evidence_mapping(self.values.to_dict())
        if self.metric_unit:
            payload["metric_unit"] = self.metric_unit
        if self.metric_quantity_name:
            payload["metric_quantity_name"] = self.metric_quantity_name
        return payload


@dataclass(frozen=True)
class ReferenceBandEvidencePayload:
    """Typed detailed evidence payload for reference-band suite cases."""

    observed_key: str
    observed_value: float
    reference_mean: float
    reference_unit: str
    accepted_low: float
    accepted_high: float
    accepted_sigma_multiplier: float
    accepted_interval_mode: str
    accepted_interval_standard: str
    reference_annotation: str = ""
    accepted_lower_bound: float | None = None
    accepted_upper_bound: float | None = None
    unbounded_low: float | None = None
    unbounded_high: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_key", str(self.observed_key).strip())
        object.__setattr__(self, "observed_value", float(self.observed_value))
        object.__setattr__(self, "reference_mean", float(self.reference_mean))
        object.__setattr__(self, "reference_unit", str(self.reference_unit or "").strip())
        object.__setattr__(self, "accepted_low", float(self.accepted_low))
        object.__setattr__(self, "accepted_high", float(self.accepted_high))
        object.__setattr__(self, "accepted_sigma_multiplier", float(self.accepted_sigma_multiplier))
        object.__setattr__(self, "accepted_interval_mode", str(self.accepted_interval_mode).strip())
        object.__setattr__(self, "accepted_interval_standard", str(self.accepted_interval_standard).strip())
        object.__setattr__(self, "reference_annotation", str(self.reference_annotation or "").strip())
        if self.accepted_lower_bound is not None:
            object.__setattr__(self, "accepted_lower_bound", float(self.accepted_lower_bound))
        if self.accepted_upper_bound is not None:
            object.__setattr__(self, "accepted_upper_bound", float(self.accepted_upper_bound))
        if self.unbounded_low is not None:
            object.__setattr__(self, "unbounded_low", float(self.unbounded_low))
        if self.unbounded_high is not None:
            object.__setattr__(self, "unbounded_high", float(self.unbounded_high))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            self.observed_key: rounded(self.observed_value),
            "reference_mean": rounded(self.reference_mean),
            "reference_unit": self.reference_unit,
            "accepted_low": rounded(self.accepted_low),
            "accepted_high": rounded(self.accepted_high),
            "accepted_sigma_multiplier": self.accepted_sigma_multiplier,
            "accepted_interval_mode": self.accepted_interval_mode,
            "accepted_interval_standard": self.accepted_interval_standard,
        }
        if self.reference_annotation:
            payload["__reference_annotations__"] = {
                self.observed_key: self.reference_annotation,
            }
        if self.accepted_lower_bound is not None:
            payload["accepted_lower_bound"] = rounded(self.accepted_lower_bound)
        if self.accepted_upper_bound is not None:
            payload["accepted_upper_bound"] = rounded(self.accepted_upper_bound)
        if self.unbounded_low is not None:
            payload["unbounded_low"] = rounded(self.unbounded_low)
        if self.unbounded_high is not None:
            payload["unbounded_high"] = rounded(self.unbounded_high)
        return payload


__all__ = [
    "ReferenceBandEvidencePayload",
    "ScalarEvidenceValues",
    "ScalarRuleEvidencePayload",
]
