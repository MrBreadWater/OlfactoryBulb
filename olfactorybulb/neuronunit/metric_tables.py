"""Typed metric-row and grouped-summary contracts for maintained validations."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from olfactorybulb.neuronunit.frozen_payloads import FrozenMappingPayload, coerce_mapping_payload


def _is_numeric_metric_value(value: Any) -> bool:
    if isinstance(value, (bool, str)) or value is None:
        return False
    if isinstance(value, (list, tuple, dict, set)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


@dataclass(frozen=True)
class MetricRowRecord(Mapping[str, Any]):
    """One typed protocol-metric row with mapping compatibility."""

    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            {str(key): copy.deepcopy(value) for key, value in dict(self.values).items()},
        )

    def __getitem__(self, key: str) -> Any:
        return self.values[str(key)]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(str(key), default)

    def items(self):
        return self.values.items()

    def keys(self):
        return self.values.keys()

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.values))


class MetricValueMapPayload(FrozenMappingPayload):
    """Frozen per-entity metric-value map used by the runtime/model seam."""


@dataclass(frozen=True)
class MetricSummaryRecord(Mapping[str, float]):
    """Grouped numeric summary values with mapping compatibility."""

    values: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            {str(key): float(value) for key, value in dict(self.values).items()},
        )

    def __getitem__(self, key: str) -> float:
        return self.values[str(key)]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(str(key), default)

    def items(self):
        return self.values.items()

    def keys(self):
        return self.values.keys()

    def to_dict(self) -> dict[str, float]:
        return dict(self.values)


@dataclass(frozen=True)
class MetricSummaryTable(Mapping[str, MetricSummaryRecord]):
    """Typed grouped summary table keyed by validation group."""

    groups: Mapping[str, MetricSummaryRecord | Mapping[str, float]]

    def __post_init__(self) -> None:
        normalized: dict[str, MetricSummaryRecord] = {}
        for group, values in dict(self.groups).items():
            normalized[str(group)] = (
                values
                if isinstance(values, MetricSummaryRecord)
                else MetricSummaryRecord(values)
            )
        object.__setattr__(self, "groups", normalized)

    def __getitem__(self, group: str) -> MetricSummaryRecord:
        return self.groups[str(group)]

    def __iter__(self) -> Iterator[str]:
        return iter(self.groups)

    def __len__(self) -> int:
        return len(self.groups)

    def get(self, group: str, default: Any = None) -> Any:
        return self.groups.get(str(group), default)

    def items(self):
        return self.groups.items()

    def keys(self):
        return self.groups.keys()

    def metric_value(self, group: str, metric_key: str, *, default: float = float("nan")) -> float:
        record = self.groups.get(str(group))
        if record is None:
            return float(default)
        value = record.get(metric_key, default)
        return float(value)

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            str(group): record.to_dict()
            for group, record in self.groups.items()
        }


@dataclass(frozen=True)
class MetricTable(Sequence[MetricRowRecord]):
    """Typed protocol-metric table with sequence compatibility."""

    rows: Sequence[MetricRowRecord | Mapping[str, Any]]
    group_field: str = "cell_type"

    def __post_init__(self) -> None:
        normalized = tuple(
            row if isinstance(row, MetricRowRecord) else MetricRowRecord(row)
            for row in tuple(self.rows)
        )
        object.__setattr__(self, "rows", normalized)
        object.__setattr__(self, "group_field", str(self.group_field or "cell_type").strip() or "cell_type")

    def __getitem__(self, index: int) -> MetricRowRecord:
        return self.rows[index]

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[MetricRowRecord]:
        return iter(self.rows)

    def to_rows(self) -> list[dict[str, Any]]:
        return [row.to_dict() for row in self.rows]

    def grouped_rows(self, *, group_field: str | None = None) -> dict[str, list[MetricRowRecord]]:
        resolved_group_field = str(group_field or self.group_field).strip() or self.group_field
        grouped: dict[str, list[MetricRowRecord]] = {}
        for row in self.rows:
            group = str(row.get(resolved_group_field, "")).strip() or "ungrouped"
            grouped.setdefault(group, []).append(row)
        return grouped

    def summarize(self, *, group_field: str | None = None) -> MetricSummaryTable:
        grouped = self.grouped_rows(group_field=group_field)
        summary: dict[str, dict[str, float]] = {}
        for group, rows in grouped.items():
            numeric_keys: set[str] = set()
            for row in rows:
                for key, value in row.items():
                    if _is_numeric_metric_value(value):
                        numeric_keys.add(str(key))
            summary[group] = {
                key: self._mean_metric(rows, key)
                for key in sorted(numeric_keys)
            }
        return MetricSummaryTable(summary)

    def metric_value_map(
        self,
        metric_key: str,
        *,
        entity_key: str = "cell_name",
    ) -> MetricValueMapPayload:
        values: dict[str, Any] = {}
        for index, row in enumerate(self.rows):
            entity = str(row.get(entity_key, f"row_{index}"))
            values[entity] = row.get(metric_key)
        return coerce_mapping_payload(values, payload_type=MetricValueMapPayload)

    @staticmethod
    def _mean_metric(rows: Iterable[MetricRowRecord], key: str) -> float:
        values = [
            float(value)
            for value in (row.get(key) for row in rows)
            if _is_numeric_metric_value(value) and np.isfinite(float(value))
        ]
        if not values:
            return float("nan")
        return float(np.mean(np.asarray(values, dtype=float)))


def coerce_metric_table(
    metrics: MetricTable | Sequence[MetricRowRecord | Mapping[str, Any]],
    *,
    group_field: str = "cell_type",
) -> MetricTable:
    if isinstance(metrics, MetricTable):
        if metrics.group_field == group_field:
            return metrics
        return MetricTable(metrics.rows, group_field=group_field)
    return MetricTable(metrics, group_field=group_field)


def coerce_metric_summary_table(
    summary: MetricSummaryTable | Mapping[str, MetricSummaryRecord | Mapping[str, float]],
) -> MetricSummaryTable:
    if isinstance(summary, MetricSummaryTable):
        return summary
    return MetricSummaryTable(summary)


__all__ = [
    "MetricValueMapPayload",
    "MetricRowRecord",
    "MetricSummaryRecord",
    "MetricSummaryTable",
    "MetricTable",
    "coerce_metric_summary_table",
    "coerce_metric_table",
]
