"""Typed metric-quantity helpers for maintained scalar validation rules."""

from __future__ import annotations

from dataclasses import dataclass


_UNIT_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_Hz_per_50pA", "Hz/50pA"),
    ("_Hz_per_nA", "Hz/nA"),
    ("_mV_per_ms", "mV/ms"),
    ("_MOhm", "MOhm"),
    ("_mV", "mV"),
    ("_ms", "ms"),
    ("_pA", "pA"),
    ("_pF", "pF"),
    ("_Hz", "Hz"),
    ("_hz", "Hz"),
    ("_um", "um"),
)

_TOKEN_ALIASES: dict[str, str] = {
    "AP": "AP",
    "AHP": "AHP",
    "CV": "CV",
    "EPLI": "EPLI",
    "FI": "FI",
    "FWHM": "FWHM",
    "GC": "GC",
    "GCS": "GCs",
    "Hz": "Hz",
    "ISI": "ISI",
    "MC": "MC",
    "MCS": "MCs",
    "TC": "TC",
    "TCS": "TCs",
}


def infer_metric_unit_text(metric_key: str) -> str:
    key = str(metric_key or "").strip()
    for suffix, unit_text in _UNIT_SUFFIXES:
        if key.endswith(suffix):
            return unit_text
    return ""


def _metric_stem(metric_key: str) -> str:
    key = str(metric_key or "").strip()
    for suffix, _unit_text in _UNIT_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def _humanize_token(token: str) -> str:
    text = str(token or "").strip()
    if not text:
        return ""
    alias = _TOKEN_ALIASES.get(text)
    if alias is not None:
        return alias
    alias = _TOKEN_ALIASES.get(text.upper())
    if alias is not None:
        return alias
    return text.replace("__", " ").replace("-", " ").title()


def infer_metric_quantity_name(metric_key: str) -> str:
    stem = _metric_stem(metric_key).replace("__", "_")
    parts = [_humanize_token(token) for token in stem.split("_") if str(token).strip()]
    return " ".join(part for part in parts if part)


@dataclass(frozen=True)
class MetricQuantitySpec:
    metric_key: str
    unit_text: str = ""
    quantity_name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric_key", str(self.metric_key).strip())
        object.__setattr__(self, "unit_text", str(self.unit_text or "").strip())
        object.__setattr__(self, "quantity_name", str(self.quantity_name or "").strip())

    @property
    def resolved_quantity_name(self) -> str:
        return self.quantity_name or infer_metric_quantity_name(self.metric_key) or self.metric_key

    @property
    def definition_label(self) -> str:
        label = self.resolved_quantity_name
        if self.unit_text:
            return f"{label} ({self.unit_text})"
        return label


def resolve_metric_quantity(
    metric_key: str,
    *,
    unit_text: str = "",
    quantity_name: str = "",
) -> MetricQuantitySpec:
    key = str(metric_key or "").strip()
    resolved_unit = str(unit_text or "").strip() or infer_metric_unit_text(key)
    resolved_name = str(quantity_name or "").strip() or infer_metric_quantity_name(key)
    return MetricQuantitySpec(
        metric_key=key,
        unit_text=resolved_unit,
        quantity_name=resolved_name,
    )


def metric_definition_text(
    metric_quantity: MetricQuantitySpec,
    *,
    group: str = "",
    reducer: str = "mean",
) -> str:
    label = metric_quantity.definition_label
    group_text = str(group or "").strip()
    reducer_text = str(reducer or "").strip()
    if group_text and reducer_text:
        return f"{group_text} {reducer_text} {label}"
    if group_text:
        return f"{group_text} {label}"
    if reducer_text:
        return f"{reducer_text} {label}"
    return label


__all__ = [
    "MetricQuantitySpec",
    "infer_metric_quantity_name",
    "infer_metric_unit_text",
    "metric_definition_text",
    "resolve_metric_quantity",
]
