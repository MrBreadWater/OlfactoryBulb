"""Validation-note helpers for protocol and provenance caveats."""

from __future__ import annotations

import csv
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .reference_data import REFERENCE_DATA_DIR, VALIDATION_NOTES_FILENAME


@dataclass(frozen=True)
class ValidationNote:
    note_id: str
    severity: str
    scope: str
    target_type: str
    target: str
    message: str
    display_order: int
    source: str = ""
    source_location: str = ""

    @property
    def target_values(self) -> set[str]:
        return {value.strip() for value in str(self.target).split(";") if value.strip()}


def load_notes(path: Path | None = None) -> list[ValidationNote]:
    csv_path = path or (REFERENCE_DATA_DIR / VALIDATION_NOTES_FILENAME)
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    notes: list[ValidationNote] = []
    for row in rows:
        notes.append(
            ValidationNote(
                note_id=str(row.get("note_id", "") or "").strip(),
                severity=str(row.get("severity", "") or "").strip(),
                scope=str(row.get("scope", "") or "").strip(),
                target_type=str(row.get("target_type", "") or "").strip(),
                target=str(row.get("target", "") or "").strip(),
                message=str(row.get("message", "") or "").strip(),
                display_order=int(float(row.get("display_order", 0) or 0)),
                source=str(row.get("source", "") or "").strip(),
                source_location=str(row.get("source_location", "") or "").strip(),
            )
        )
    return sorted(notes, key=lambda note: (note.display_order, note.note_id))


def _as_set(value: str | Iterable[str] | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value} if value else set()
    return {str(item) for item in value if str(item)}


def notes_for(
    notes: Iterable[ValidationNote] | None = None,
    *,
    scope: str | None = None,
    protocol_id: str | Iterable[str] | None = None,
    metric: str | Iterable[str] | None = None,
    source: str | Iterable[str] | None = None,
    note_ids: str | Iterable[str] | None = None,
) -> list[ValidationNote]:
    loaded_notes = list(load_notes() if notes is None else notes)
    note_id_filter = _as_set(note_ids)
    protocol_values = _as_set(protocol_id)
    metric_values = _as_set(metric)
    source_values = _as_set(source)

    matched: list[ValidationNote] = []
    for note in loaded_notes:
        if scope is not None and note.scope != scope:
            continue
        if note_id_filter and note.note_id not in note_id_filter:
            continue
        if note.target_type == "protocol" and note.target_values:
            if not note.target_values.issubset(protocol_values):
                continue
        elif note.target_type == "metric" and note.target_values:
            if not note.target_values.intersection(metric_values):
                continue
        elif note.target_type == "source" and note.target_values:
            if not note.target_values.intersection(source_values):
                continue
        matched.append(note)

    deduped = {note.note_id: note for note in matched}
    return sorted(deduped.values(), key=lambda note: (note.display_order, note.note_id))


def notes_for_rows(
    rows: Iterable[dict[str, object]],
    *,
    scope: str | None = None,
    notes: Iterable[ValidationNote] | None = None,
) -> list[ValidationNote]:
    row_list = [dict(row) for row in rows]
    protocols = {str(row.get("protocol_id", "")).strip() for row in row_list if str(row.get("protocol_id", "")).strip()}
    properties = {str(row.get("Property", "")).strip() for row in row_list if str(row.get("Property", "")).strip()}
    sources = {
        str(row.get("Source", row.get("source", ""))).strip()
        for row in row_list
        if str(row.get("Source", row.get("source", ""))).strip()
    }
    note_ids: set[str] = set()
    for row in row_list:
        for note_id in str(row.get("note_ids", "")).split(";"):
            if note_id.strip():
                note_ids.add(note_id.strip())
    return notes_for(
        notes=notes,
        scope=scope,
        protocol_id=protocols,
        metric=properties,
        source=sources,
        note_ids=note_ids or None,
    )


def render_notes(notes: Iterable[ValidationNote], format: str = "plain") -> str:
    notes_list = list(notes)
    if not notes_list:
        return ""

    if format == "plain":
        lines = ["Notes / protocol caveats"]
        lines.append("------------------------")
        for note in notes_list:
            lines.append(f"- [{note.severity.upper()}] {note.message}")
        return "\n".join(lines)

    if format == "markdown":
        lines = ["## Notes / protocol caveats"]
        for note in notes_list:
            lines.append(f"- **{note.severity.upper()}** {note.message}")
        return "\n".join(lines)

    if format == "html":
        items = "".join(
            f"<li><strong>{html.escape(note.severity.upper())}</strong> {html.escape(note.message)}</li>"
            for note in notes_list
        )
        return f"<h2>Notes / protocol caveats</h2><ul>{items}</ul>"

    raise ValueError(f"Unsupported note-render format: {format!r}")

