"""Typed contracts for protocol-evidence payloads."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from olfactorybulb.audit.core import series_visual_spec


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
    style: dict[str, Any] = field(default_factory=dict)

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
            "style": dict(self.style),
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
        style = dict(self.style)
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
    values: dict[str, Any] = field(default_factory=dict)
    series_specs: tuple[ProtocolEvidenceSeriesSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", copy.deepcopy(dict(self.values)))
        object.__setattr__(self, "series_specs", tuple(self.series_specs))

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.values)

    def with_value(self, key: str, value: Any) -> "ProtocolEvidenceBundle":
        updated = self.to_dict()
        updated[str(key)] = copy.deepcopy(value)
        return ProtocolEvidenceBundle(values=updated, series_specs=self.series_specs)

    def rows(self, evidence_key: str) -> list[dict[str, Any]]:
        rows = self.values.get(evidence_key, [])
        if not isinstance(rows, list):
            return []
        return [dict(row) for row in rows if isinstance(row, dict)]

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
    "ProtocolEvidenceSeriesSpec",
    "coerce_protocol_evidence_bundle",
    "intrinsic_fi_curve_series_spec",
    "protocol_evidence_bundle_from_resultish",
    "protocol_series_spec_map",
    "SupportsProtocolEvidence",
]
