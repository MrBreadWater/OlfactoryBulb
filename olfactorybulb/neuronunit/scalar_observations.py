"""Typed scalar observation and prediction helpers for maintained validation rules."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import quantities as pq

from olfactorybulb.audit.core import rounded
from olfactorybulb.neuronunit.metric_quantities import MetricQuantitySpec, resolve_metric_quantity
from olfactorybulb.neuronunit.reference_bands import measurement_with_unit, numeric_value


def is_finite_scalar(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        number = numeric_value(value) if isinstance(value, pq.Quantity) else float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


@dataclass(frozen=True)
class ScalarMetricValue:
    metric_quantity: MetricQuantitySpec
    group: str = ""
    value: float | pq.Quantity = 0.0
    reducer: str = "mean"

    def __post_init__(self) -> None:
        object.__setattr__(self, "group", str(self.group or "").strip())
        object.__setattr__(self, "reducer", str(self.reducer or "mean").strip() or "mean")

    @property
    def metric_key(self) -> str:
        return self.metric_quantity.metric_key

    @property
    def unit_text(self) -> str:
        return self.metric_quantity.unit_text

    @property
    def quantity_name(self) -> str:
        return self.metric_quantity.resolved_quantity_name

    @property
    def observed_symbol(self) -> str:
        return self.metric_quantity.resolved_observed_symbol

    @property
    def numeric(self) -> float:
        return numeric_value(self.value)

    @property
    def rounded_numeric(self) -> float:
        return rounded(self.numeric)

    def metadata(self) -> dict[str, Any]:
        payload = {
            "metric_key": self.metric_key,
            "metric_quantity_name": self.quantity_name,
            "metric_unit_text": self.unit_text,
            "metric_observed_symbol": self.observed_symbol,
        }
        if self.group:
            payload["group"] = self.group
        if self.reducer:
            payload["reducer"] = self.reducer
        return payload

    def observation_payload(self, *, key: str = "observed") -> dict[str, Any]:
        payload = self.metadata()
        payload[str(key)] = self.rounded_numeric
        return payload


@dataclass(frozen=True)
class ScalarMetricValueMap:
    metric_quantity: MetricQuantitySpec
    values: dict[str, Any]
    entity_key: str = "cell_name"

    def __post_init__(self) -> None:
        normalized = {str(entity): value for entity, value in dict(self.values).items()}
        object.__setattr__(self, "values", normalized)
        object.__setattr__(self, "entity_key", str(self.entity_key or "cell_name").strip() or "cell_name")

    @property
    def metric_key(self) -> str:
        return self.metric_quantity.metric_key

    @property
    def unit_text(self) -> str:
        return self.metric_quantity.unit_text

    @property
    def quantity_name(self) -> str:
        return self.metric_quantity.resolved_quantity_name

    @property
    def observed_symbol(self) -> str:
        return self.metric_quantity.resolved_observed_symbol

    @property
    def entity_count(self) -> int:
        return len(self.values)

    def failing_nonfinite(self) -> dict[str, Any]:
        return {
            entity: value
            for entity, value in self.values.items()
            if not is_finite_scalar(value)
        }

    def failing_not_equal(self, *, expected: float, tolerance: float) -> dict[str, Any]:
        return {
            entity: value
            for entity, value in self.values.items()
            if not (is_finite_scalar(value) and abs(numeric_value(value) - float(expected)) <= float(tolerance))
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "metric_key": self.metric_key,
            "metric_quantity_name": self.quantity_name,
            "metric_unit_text": self.unit_text,
            "metric_observed_symbol": self.observed_symbol,
            "entity_key": self.entity_key,
            "entity_count": self.entity_count,
        }


@dataclass(frozen=True)
class ScalarGroupPair:
    metric_quantity: MetricQuantitySpec
    left_group: str
    right_group: str
    left_value: float | pq.Quantity
    right_value: float | pq.Quantity
    reducer: str = "mean"

    def __post_init__(self) -> None:
        object.__setattr__(self, "left_group", str(self.left_group or "").strip())
        object.__setattr__(self, "right_group", str(self.right_group or "").strip())
        object.__setattr__(self, "reducer", str(self.reducer or "mean").strip() or "mean")

    @property
    def metric_key(self) -> str:
        return self.metric_quantity.metric_key

    @property
    def unit_text(self) -> str:
        return self.metric_quantity.unit_text

    @property
    def quantity_name(self) -> str:
        return self.metric_quantity.resolved_quantity_name

    @property
    def observed_symbol(self) -> str:
        return self.metric_quantity.resolved_observed_symbol

    @property
    def left_numeric(self) -> float:
        return numeric_value(self.left_value)

    @property
    def right_numeric(self) -> float:
        return numeric_value(self.right_value)

    @property
    def delta(self) -> float:
        return self.right_numeric - self.left_numeric

    @property
    def absolute_difference(self) -> float:
        return abs(self.delta)

    def metadata(self) -> dict[str, Any]:
        payload = {
            "metric_key": self.metric_key,
            "metric_quantity_name": self.quantity_name,
            "metric_unit_text": self.unit_text,
            "metric_observed_symbol": self.observed_symbol,
            "left_group": self.left_group,
            "right_group": self.right_group,
        }
        if self.reducer:
            payload["reducer"] = self.reducer
        return payload


@dataclass(frozen=True)
class ScalarGroupValueSet:
    metric_quantity: MetricQuantitySpec
    values_by_group: dict[str, Any]
    reducer: str = "mean"

    def __post_init__(self) -> None:
        normalized = {str(group): value for group, value in dict(self.values_by_group).items()}
        object.__setattr__(self, "values_by_group", normalized)
        object.__setattr__(self, "reducer", str(self.reducer or "mean").strip() or "mean")

    @property
    def metric_key(self) -> str:
        return self.metric_quantity.metric_key

    @property
    def unit_text(self) -> str:
        return self.metric_quantity.unit_text

    @property
    def quantity_name(self) -> str:
        return self.metric_quantity.resolved_quantity_name

    @property
    def observed_symbol(self) -> str:
        return self.metric_quantity.resolved_observed_symbol

    def numeric_group_values(self) -> dict[str, float]:
        return {
            group: numeric_value(value)
            for group, value in self.values_by_group.items()
        }

    def failing_positive_groups(self) -> list[str]:
        failing: list[str] = []
        for group, value in self.values_by_group.items():
            if not (is_finite_scalar(value) and float(numeric_value(value)) > 0.0):
                failing.append(group)
        return failing

    def metadata(self) -> dict[str, Any]:
        payload = {
            "metric_key": self.metric_key,
            "metric_quantity_name": self.quantity_name,
            "metric_unit_text": self.unit_text,
            "metric_observed_symbol": self.observed_symbol,
        }
        if self.reducer:
            payload["reducer"] = self.reducer
        return payload


def scalar_metric_value(
    metric_key: str,
    value: float | pq.Quantity,
    *,
    unit_text: str = "",
    quantity_name: str = "",
    observed_symbol: str = "",
    group: str = "",
    reducer: str = "mean",
) -> ScalarMetricValue:
    return ScalarMetricValue(
        metric_quantity=resolve_metric_quantity(
            metric_key,
            unit_text=unit_text,
            quantity_name=quantity_name,
            observed_symbol=observed_symbol,
        ),
        group=group,
        value=value,
        reducer=reducer,
    )


def scalar_metric_map(
    metric_key: str,
    values: dict[str, Any],
    *,
    unit_text: str = "",
    quantity_name: str = "",
    observed_symbol: str = "",
    entity_key: str = "cell_name",
) -> ScalarMetricValueMap:
    return ScalarMetricValueMap(
        metric_quantity=resolve_metric_quantity(
            metric_key,
            unit_text=unit_text,
            quantity_name=quantity_name,
            observed_symbol=observed_symbol,
        ),
        values=values,
        entity_key=entity_key,
    )


__all__ = [
    "ScalarGroupPair",
    "ScalarGroupValueSet",
    "ScalarMetricValue",
    "ScalarMetricValueMap",
    "is_finite_scalar",
    "scalar_metric_map",
    "scalar_metric_value",
]
