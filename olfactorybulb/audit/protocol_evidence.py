"""Typed contracts for protocol-evidence payloads."""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from olfactorybulb.audit.core import series_visual_spec
from olfactorybulb.neuronunit.frozen_payloads import FrozenMappingPayload, coerce_mapping_payload


class ProtocolEvidenceStyleMap(FrozenMappingPayload):
    """Typed frozen mapping wrapper for protocol-evidence visual style tables."""


class ProtocolEvidenceValueMap(FrozenMappingPayload):
    """Typed frozen mapping wrapper for protocol-evidence payload tables."""


class ProtocolEvidenceRowPayload(FrozenMappingPayload):
    """Frozen row payload used by typed protocol-evidence row tables."""


@dataclass(frozen=True)
class ProtocolEvidenceRowTable(Sequence[ProtocolEvidenceRowPayload]):
    """Typed protocol-evidence row collection with sequence compatibility."""

    rows: Sequence[ProtocolEvidenceRowPayload | Mapping[str, object]]

    def __post_init__(self) -> None:
        normalized = tuple(
            coerce_mapping_payload(row, payload_type=ProtocolEvidenceRowPayload)
            for row in tuple(self.rows)
        )
        object.__setattr__(self, "rows", normalized)

    def __getitem__(self, index: int) -> ProtocolEvidenceRowPayload:
        return self.rows[index]

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[ProtocolEvidenceRowPayload]:
        return iter(self.rows)

    def to_rows(self) -> list[dict[str, object]]:
        return [row.to_dict() for row in self.rows]


@dataclass(frozen=True)
class ProtocolEvidenceSeriesSpec:
    evidence_key: str
    x_key: str
    y_keys: tuple[str, ...]
    x_unit_text: str
    y_unit_text: str
    x_quantity_name: str
    y_quantity_name: str
    series_id_key: str = ""
    kind: str = "fi_curve"
    backend: str = "matplotlib"
    title: str = ""
    style: ProtocolEvidenceStyleMap | Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "style",
            coerce_mapping_payload(self.style, payload_type=ProtocolEvidenceStyleMap) or ProtocolEvidenceStyleMap(entries=()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_key": self.evidence_key,
            "x_key": self.x_key,
            "y_keys": list(self.y_keys),
            "x_unit_text": self.x_unit_text,
            "y_unit_text": self.y_unit_text,
            "x_quantity_name": self.x_quantity_name,
            "y_quantity_name": self.y_quantity_name,
            "series_id_key": self.series_id_key,
            "kind": self.kind,
            "backend": self.backend,
            "title": self.title,
            "style": self.style.to_dict(),
        }

    def axis_label(self, *, axis: str) -> str:
        axis_name = str(axis or "").strip().lower()
        if axis_name == "x":
            quantity_name = self.x_quantity_name
            unit_text = self.x_unit_text
        elif axis_name == "y":
            quantity_name = self.y_quantity_name
            unit_text = self.y_unit_text
        else:
            raise ValueError(f"Unsupported axis {axis!r}")
        if quantity_name and unit_text:
            return f"{quantity_name} ({unit_text})"
        if quantity_name:
            return quantity_name
        return unit_text

    def to_visual_spec(self) -> dict[str, Any]:
        style = self.style.to_dict()
        style.setdefault("x_label", self.axis_label(axis="x"))
        style.setdefault("y_label", self.axis_label(axis="y"))
        if self.title:
            style.setdefault("title", self.title)
        return series_visual_spec(
            kind=self.kind,
            backend=self.backend,
            title=self.title or None,
            keys=[self.evidence_key],
            row_source_key=self.evidence_key,
            x_key=self.x_key,
            y_keys=list(self.y_keys),
            series_id_key=self.series_id_key or None,
            style=style,
            x_unit_text=self.x_unit_text,
            y_unit_text=self.y_unit_text,
            x_quantity_name=self.x_quantity_name,
            y_quantity_name=self.y_quantity_name,
        )


@dataclass(frozen=True)
class ProtocolEvidenceBundle:
    values: ProtocolEvidenceValueMap | Mapping[str, object] = field(default_factory=dict)
    series_specs: tuple[ProtocolEvidenceSeriesSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            coerce_mapping_payload(self.values, payload_type=ProtocolEvidenceValueMap) or ProtocolEvidenceValueMap(entries=()),
        )
        object.__setattr__(self, "series_specs", tuple(self.series_specs))

    def to_dict(self) -> dict[str, Any]:
        return self.values.to_dict()

    def with_value(self, key: str, value: Any) -> "ProtocolEvidenceBundle":
        updated = self.to_dict()
        updated[str(key)] = copy.deepcopy(value)
        return ProtocolEvidenceBundle(values=updated, series_specs=self.series_specs)

    def row_table(self, evidence_key: str) -> ProtocolEvidenceRowTable:
        rows = self.values.get(evidence_key, [])
        if not isinstance(rows, list | tuple):
            return ProtocolEvidenceRowTable(())
        return ProtocolEvidenceRowTable(
            row
            for row in rows
            if isinstance(row, FrozenMappingPayload | Mapping)
        )

    def rows(self, evidence_key: str) -> list[dict[str, Any]]:
        return self.row_table(evidence_key).to_rows()

    def series_spec_map(self) -> dict[str, ProtocolEvidenceSeriesSpec]:
        return protocol_series_spec_map(self.series_specs)


@runtime_checkable
class SupportsProtocolEvidence(Protocol):
    protocol_evidence: ProtocolEvidenceBundle | dict[str, Any] | None


def protocol_series_spec_map(
    specs: list[ProtocolEvidenceSeriesSpec] | tuple[ProtocolEvidenceSeriesSpec, ...] | None,
) -> dict[str, ProtocolEvidenceSeriesSpec]:
    result: dict[str, ProtocolEvidenceSeriesSpec] = {}
    for spec in specs or ():
        result[str(spec.evidence_key).strip()] = spec
    return result


def coerce_protocol_evidence_row_table(
    rows: ProtocolEvidenceRowTable | Sequence[ProtocolEvidenceRowPayload | Mapping[str, object]],
) -> ProtocolEvidenceRowTable:
    if isinstance(rows, ProtocolEvidenceRowTable):
        return rows
    return ProtocolEvidenceRowTable(rows)


def coerce_protocol_evidence_bundle(
    values: "ProtocolEvidenceBundle | dict[str, Any] | None" = None,
    *,
    series_specs: list[ProtocolEvidenceSeriesSpec] | tuple[ProtocolEvidenceSeriesSpec, ...] | None = None,
) -> ProtocolEvidenceBundle:
    if isinstance(values, ProtocolEvidenceBundle):
        if series_specs:
            merged_specs = tuple(values.series_specs) + tuple(series_specs)
            deduped: dict[str, ProtocolEvidenceSeriesSpec] = {}
            for spec in merged_specs:
                deduped[str(spec.evidence_key).strip()] = spec
            return ProtocolEvidenceBundle(values=values.values, series_specs=tuple(deduped.values()))
        return values
    return ProtocolEvidenceBundle(values=dict(values or {}), series_specs=tuple(series_specs or ()))


def protocol_evidence_bundle_from_resultish(result: SupportsProtocolEvidence | None) -> ProtocolEvidenceBundle:
    if result is None:
        return ProtocolEvidenceBundle()
    return coerce_protocol_evidence_bundle(
        getattr(result, "protocol_evidence", None),
        series_specs=getattr(result, "evidence_series_specs", ()) or (),
    )


def intrinsic_fi_curve_series_spec(
    *,
    evidence_key: str = "fi_curve_rows",
    series_id_key: str = "cell_name",
    title: str = "f-I curve",
    style: dict[str, Any] | None = None,
) -> ProtocolEvidenceSeriesSpec:
    resolved_style = {
        "line_width": 1.6,
        "marker_size": 2.8,
        "alpha": 0.78,
        "legend_loc": "lower center",
    }
    if style:
        resolved_style.update(style)
    return ProtocolEvidenceSeriesSpec(
        evidence_key=evidence_key,
        x_key="current_pA",
        y_keys=("firing_rate_Hz",),
        x_unit_text="pA",
        y_unit_text="Hz",
        x_quantity_name="Injected current",
        y_quantity_name="Firing rate",
        series_id_key=series_id_key,
        kind="fi_curve",
        backend="matplotlib",
        title=title,
        style=resolved_style,
    )


__all__ = [
    "ProtocolEvidenceBundle",
    "ProtocolEvidenceRowPayload",
    "ProtocolEvidenceRowTable",
    "ProtocolEvidenceSeriesSpec",
    "ProtocolEvidenceStyleMap",
    "ProtocolEvidenceValueMap",
    "coerce_protocol_evidence_bundle",
    "coerce_protocol_evidence_row_table",
    "intrinsic_fi_curve_series_spec",
    "protocol_evidence_bundle_from_resultish",
    "protocol_series_spec_map",
    "SupportsProtocolEvidence",
]
