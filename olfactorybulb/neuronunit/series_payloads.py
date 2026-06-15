"""Typed row/context payloads for the NeuronUnit series-comparison core."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from olfactorybulb.neuronunit.frozen_payloads import FrozenMappingPayload, coerce_mapping_payload


class SeriesRowPayload(FrozenMappingPayload):
    """Frozen row payload used by the series-comparison core."""


class SeriesContextPayload(FrozenMappingPayload):
    """Frozen context payload used by the series-comparison core."""


@dataclass(frozen=True)
class SeriesRowTable(Sequence[SeriesRowPayload]):
    """Typed series row table with sequence compatibility."""

    rows: Sequence[SeriesRowPayload | Mapping[str, object]]

    def __post_init__(self) -> None:
        normalized = tuple(
            coerce_mapping_payload(row, payload_type=SeriesRowPayload)
            for row in tuple(self.rows)
        )
        object.__setattr__(self, "rows", normalized)

    def __getitem__(self, index: int) -> SeriesRowPayload:
        return self.rows[index]

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[SeriesRowPayload]:
        return iter(self.rows)

    def to_rows(self) -> list[dict[str, object]]:
        return [row.to_dict() for row in self.rows]


def coerce_series_row_table(
    rows: SeriesRowTable | Sequence[SeriesRowPayload | Mapping[str, object]],
) -> SeriesRowTable:
    if isinstance(rows, SeriesRowTable):
        return rows
    return SeriesRowTable(rows)


def coerce_series_context_payload(
    context: SeriesContextPayload | Mapping[str, object] | None,
) -> SeriesContextPayload | None:
    return coerce_mapping_payload(context, payload_type=SeriesContextPayload)


__all__ = [
    "SeriesContextPayload",
    "SeriesRowPayload",
    "SeriesRowTable",
    "coerce_series_context_payload",
    "coerce_series_row_table",
]
