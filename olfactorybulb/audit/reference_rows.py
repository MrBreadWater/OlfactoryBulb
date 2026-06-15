"""Typed reference-row contracts for maintained literature validation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence


@dataclass(frozen=True)
class ReferenceRowRecord(Mapping[str, Any]):
    """One typed literature/reference row with mapping compatibility."""

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


@dataclass(frozen=True)
class ReferenceRowTable(Sequence[ReferenceRowRecord]):
    """Typed literature/reference row table with sequence compatibility."""

    rows: Sequence[ReferenceRowRecord | Mapping[str, Any]]

    def __post_init__(self) -> None:
        normalized = tuple(
            row if isinstance(row, ReferenceRowRecord) else ReferenceRowRecord(row)
            for row in tuple(self.rows)
        )
        object.__setattr__(self, "rows", normalized)

    def __getitem__(self, index: int) -> ReferenceRowRecord:
        return self.rows[index]

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[ReferenceRowRecord]:
        return iter(self.rows)

    def to_rows(self) -> list[dict[str, Any]]:
        return [row.to_dict() for row in self.rows]


def coerce_reference_row_table(
    rows: ReferenceRowTable | Sequence[ReferenceRowRecord | Mapping[str, Any]],
) -> ReferenceRowTable:
    if isinstance(rows, ReferenceRowTable):
        return rows
    return ReferenceRowTable(rows)


__all__ = [
    "ReferenceRowRecord",
    "ReferenceRowTable",
    "coerce_reference_row_table",
]
