"""Shared provenance primitives for the NeuronUnit overhaul."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from olfactorybulb.audit.core import rounded


def _freeze_sequence(values: list[Any]) -> tuple[Any, ...]:
    frozen: list[Any] = []
    for value in values:
        if isinstance(value, list):
            frozen.append(_freeze_sequence(value))
        else:
            frozen.append(value)
    return tuple(frozen)


def _thaw_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return value


def _is_scalar_metadata(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def compact_metadata_value(value: Any) -> Any:
    if _is_scalar_metadata(value):
        if isinstance(value, float) and math.isfinite(value):
            return rounded(float(value))
        return value
    if isinstance(value, list) and all(_is_scalar_metadata(item) for item in value):
        compact_items: list[Any] = []
        for item in value:
            if isinstance(item, float) and math.isfinite(item):
                compact_items.append(rounded(float(item)))
            else:
                compact_items.append(item)
        return compact_items
    return None


def _sorted_unique_text_values(rows: list[dict[str, Any]], *keys: str) -> tuple[str, ...]:
    values: set[str] = set()
    for row in rows:
        for key in keys:
            text = str(row.get(key, "")).strip()
            if text:
                values.add(text)
    return tuple(sorted(values))


def _parsed_note_ids(raw_text: str) -> tuple[str, ...]:
    note_ids: set[str] = set()
    for token in str(raw_text or "").replace(",", ";").split(";"):
        normalized = token.strip()
        if normalized:
            note_ids.add(normalized)
    return tuple(sorted(note_ids))


def note_id_values(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    note_ids: set[str] = set()
    for row in rows:
        note_ids.update(_parsed_note_ids(row.get("note_ids", "")))
    return tuple(sorted(note_ids))


def _series_id_values(rows: list[dict[str, Any]], *, series_id_key: str) -> tuple[str, ...]:
    if not series_id_key:
        return ()
    return _sorted_unique_text_values(rows, series_id_key)


@dataclass(frozen=True)
class ProvenanceRecord:
    source: str = ""
    source_file: str = ""
    source_location: str = ""
    source_url: str = ""
    extraction_method: str = ""
    note_ids: tuple[str, ...] = ()
    reported_value_raw: str = ""

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "ProvenanceRecord":
        return cls(
            source=str(row.get("Source", "") or row.get("source", "")).strip(),
            source_file=str(row.get("source_file", "")).strip(),
            source_location=str(row.get("source_location", "")).strip(),
            source_url=str(row.get("source_url", "")).strip(),
            extraction_method=str(row.get("extraction_method", "")).strip(),
            note_ids=_parsed_note_ids(row.get("note_ids", "")),
            reported_value_raw=str(row.get("reported_value_raw", "")).strip(),
        )


@dataclass(frozen=True)
class ValidationReview:
    status: str = ""
    note: str = ""
    reviewer: str = ""
    required_expertise: str = ""
    focus: str = ""


@dataclass(frozen=True)
class ProtocolContextSummary:
    entries: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_context(
        cls,
        context: dict[str, Any] | None,
        *,
        exclude_keys: set[str] | None = None,
    ) -> "ProtocolContextSummary":
        if not context:
            return cls()
        blocked_keys = set(exclude_keys or set())
        entries: list[tuple[str, Any]] = []
        for key in sorted(context):
            if key in blocked_keys:
                continue
            compact_value = compact_metadata_value(context[key])
            if compact_value is None:
                continue
            if isinstance(compact_value, list):
                entries.append((key, _freeze_sequence(compact_value)))
            else:
                entries.append((key, compact_value))
        return cls(entries=tuple(entries))

    def to_dict(self) -> dict[str, Any]:
        return {key: _thaw_value(value) for key, value in self.entries}


@dataclass(frozen=True)
class SeriesProvenanceSummary:
    row_count: int = 0
    series_count: int = 0
    series_ids: tuple[str, ...] = ()
    x_key: str = ""
    y_key: str = ""
    x_unit_text: str = ""
    y_unit_text: str = ""
    sources: tuple[str, ...] = ()
    source_files: tuple[str, ...] = ()
    source_locations: tuple[str, ...] = ()
    source_urls: tuple[str, ...] = ()
    protocol_ids: tuple[str, ...] = ()
    extraction_methods: tuple[str, ...] = ()
    sample_scopes: tuple[str, ...] = ()
    rate_definitions: tuple[str, ...] = ()
    note_ids: tuple[str, ...] = ()
    protocol_context: ProtocolContextSummary = field(default_factory=ProtocolContextSummary)

    @classmethod
    def from_rows(
        cls,
        rows: list[dict[str, Any]],
        *,
        series_id_key: str,
        x_key: str,
        y_key: str,
        x_unit_text: str,
        y_unit_text: str,
        context: dict[str, Any] | None = None,
        exclude_context_keys: set[str] | None = None,
    ) -> "SeriesProvenanceSummary":
        series_ids = _series_id_values(rows, series_id_key=series_id_key)
        return cls(
            row_count=len(rows),
            series_count=len(series_ids),
            series_ids=series_ids,
            x_key=str(x_key),
            y_key=str(y_key),
            x_unit_text=str(x_unit_text),
            y_unit_text=str(y_unit_text),
            sources=_sorted_unique_text_values(rows, "source"),
            source_files=_sorted_unique_text_values(rows, "source_file"),
            source_locations=_sorted_unique_text_values(rows, "source_location"),
            source_urls=_sorted_unique_text_values(rows, "source_url"),
            protocol_ids=_sorted_unique_text_values(rows, "protocol_id"),
            extraction_methods=_sorted_unique_text_values(rows, "extraction_method"),
            sample_scopes=_sorted_unique_text_values(rows, "sample_scope"),
            rate_definitions=_sorted_unique_text_values(rows, "rate_definition"),
            note_ids=note_id_values(rows),
            protocol_context=ProtocolContextSummary.from_context(
                context,
                exclude_keys=set(exclude_context_keys or set()),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "row_count": int(self.row_count),
            "series_count": int(self.series_count),
            "series_ids": list(self.series_ids),
            "x_key": self.x_key,
            "y_key": self.y_key,
            "x_unit_text": self.x_unit_text,
            "y_unit_text": self.y_unit_text,
            "sources": list(self.sources),
            "source_files": list(self.source_files),
            "source_locations": list(self.source_locations),
            "source_urls": list(self.source_urls),
            "protocol_ids": list(self.protocol_ids),
            "extraction_methods": list(self.extraction_methods),
            "sample_scopes": list(self.sample_scopes),
            "rate_definitions": list(self.rate_definitions),
            "note_ids": list(self.note_ids),
        }
        protocol_context = self.protocol_context.to_dict()
        if protocol_context:
            payload["protocol_context"] = protocol_context
        return payload


@dataclass(frozen=True)
class SeriesObservationProvenance:
    reference: SeriesProvenanceSummary = field(default_factory=SeriesProvenanceSummary)
    model: SeriesProvenanceSummary = field(default_factory=SeriesProvenanceSummary)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference.to_dict(),
            "model": self.model.to_dict(),
        }


__all__ = [
    "ProvenanceRecord",
    "ProtocolContextSummary",
    "SeriesObservationProvenance",
    "SeriesProvenanceSummary",
    "ValidationReview",
    "compact_metadata_value",
    "note_id_values",
]
