"""Shared frozen payload helpers for typed NeuronUnit-side public seams."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from collections.abc import Iterator, Mapping
from typing import TypeVar


def freeze_payload_value(value: object) -> object:
    if isinstance(value, FrozenMappingPayload):
        return value
    if isinstance(value, Mapping):
        return FrozenMappingPayload.from_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(freeze_payload_value(item) for item in value)
    return copy.deepcopy(value)


def thaw_payload_value(value: object) -> object:
    if isinstance(value, FrozenMappingPayload):
        return value.to_dict()
    if isinstance(value, tuple):
        return [thaw_payload_value(item) for item in value]
    return copy.deepcopy(value)


PayloadMapT = TypeVar("PayloadMapT", bound="FrozenMappingPayload")


@dataclass(frozen=True)
class FrozenMappingPayload(Mapping[str, object]):
    entries: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        normalized_entries: list[tuple[str, object]] = []
        for key, value in self.entries:
            normalized_key = str(key).strip()
            if not normalized_key:
                continue
            normalized_entries.append((normalized_key, freeze_payload_value(value)))
        object.__setattr__(self, "entries", tuple(normalized_entries))

    @classmethod
    def from_mapping(cls: type[PayloadMapT], mapping: Mapping[str, object]) -> PayloadMapT:
        return cls(entries=tuple((str(key), value) for key, value in mapping.items()))

    def __getitem__(self, key: str) -> object:
        for entry_key, value in self.entries:
            if entry_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        for key, _ in self.entries:
            yield key

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, key: str, default: object = None) -> object:
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict[str, object]:
        return {
            key: thaw_payload_value(value)
            for key, value in self.entries
        }


def coerce_mapping_payload(
    value: PayloadMapT | Mapping[str, object] | None,
    *,
    payload_type: type[PayloadMapT],
) -> PayloadMapT | None:
    if value is None:
        return None
    if isinstance(value, payload_type):
        return value
    if isinstance(value, FrozenMappingPayload):
        return payload_type(entries=value.entries)
    if isinstance(value, Mapping):
        return payload_type.from_mapping(value)
    raise TypeError(f"{payload_type.__name__} values must be mappings or {payload_type.__name__} instances")


__all__ = [
    "FrozenMappingPayload",
    "coerce_mapping_payload",
    "freeze_payload_value",
    "thaw_payload_value",
]
