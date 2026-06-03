"""Render audit reports as a maintained HTML dashboard."""

from __future__ import annotations

import argparse
import html
import math
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any
import time

from olfactorybulb.audit.cli import run_audit_by_id
from olfactorybulb.audit.core import AuditItem, AuditReport, _summary_chunks, _expand_terms, status_reason_text


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _status_class(status: str) -> str:
    return {
        "FAIL": "status-fail",
        "WARN": "status-warn",
        "PASS": "status-pass",
    }.get(str(status), "status-pass")


def _render_status_badge(status: str) -> str:
    status = str(status or "").upper()
    icon = {
        "PASS": "✓",
        "WARN": "!",
        "FAIL": "✕",
    }.get(status, "•")
    return (
        f"<span class='status-badge {_status_class(status)}'>"
        f"<span class='status-icon' aria-hidden='true'>{_esc(icon)}</span>"
        f"<span class='status-text'>{_esc(status)}</span>"
        "</span>"
    )


def _render_summary(summary: dict[str, int]) -> str:
    def _count_label(status: str, count: int) -> str:
        if status == "FAIL":
            suffix = "Failure" if count == 1 else "Failures"
        elif status == "WARN":
            suffix = "Warning" if count == 1 else "Warnings"
        else:
            suffix = "Pass" if count == 1 else "Passes"
        return f"{int(count)} {suffix}"

    return "".join(
        f"<span class='summary-chip {_status_class(status)}'>{_esc(_count_label(status, int(summary.get(status, 0))))}</span>"
        for status in ("FAIL", "WARN", "PASS")
    )


_INTERVAL_RESERVED_EVIDENCE_KEYS = {
    "reference_mean",
    "reference_unit",
    "accepted_low",
    "accepted_high",
    "accepted_sigma_multiplier",
    "accepted_interval_mode",
    "accepted_interval_standard",
    "accepted_lower_bound",
    "accepted_upper_bound",
    "unbounded_low",
    "unbounded_high",
    "__reference_annotations__",
}


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return None
    return numeric


def _format_numeric(value: float | None, *, unit: str = "") -> str:
    if value is None:
        return "--"
    magnitude = abs(value)
    if magnitude >= 1000:
        text = f"{value:,.2f}".rstrip("0").rstrip(".")
    elif magnitude >= 100:
        text = f"{value:.2f}".rstrip("0").rstrip(".")
    elif magnitude >= 10:
        text = f"{value:.3f}".rstrip("0").rstrip(".")
    else:
        text = f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{text} {unit}".strip()


def _format_evidence_value(value: Any) -> str:
    if isinstance(value, float):
        return _format_numeric(value)
    if isinstance(value, (list, tuple)):
        parts = [_format_evidence_value(entry) for entry in value]
        return ", ".join(part for part in parts if part) or "[]"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return "--"
    return str(value)


def _evidence_label(key: str) -> str:
    normalized = key.replace("__", " ").replace("_", " ")
    return _expand_terms(normalized, sentence_case=True)


def _safe_dom_id(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value)).strip("-")
    return slug or "item"


def _extract_interval_visual_data(item: AuditItem) -> dict[str, Any] | None:
    evidence = dict(item.evidence or {})
    accepted_low = _float_or_none(evidence.get("accepted_low"))
    accepted_high = _float_or_none(evidence.get("accepted_high"))
    if accepted_low is None or accepted_high is None:
        return None
    reference_unit = str(evidence.get("reference_unit", "")).strip()
    reference_mean = _float_or_none(evidence.get("reference_mean"))
    annotations = evidence.get("__reference_annotations__")
    observed_candidates = [
        key
        for key, value in evidence.items()
        if key not in _INTERVAL_RESERVED_EVIDENCE_KEYS and _float_or_none(value) is not None
    ]
    observed_key: str | None = None
    if isinstance(annotations, dict):
        for key in annotations:
            if key in observed_candidates:
                observed_key = key
                break
    if observed_key is None:
        mean_candidates = [key for key in observed_candidates if key.endswith("_mean")]
        if mean_candidates:
            observed_key = mean_candidates[0]
    if observed_key is None and observed_candidates:
        observed_key = observed_candidates[0]
    observed_value = _float_or_none(evidence.get(observed_key)) if observed_key else None
    if observed_value is None:
        return None
    domain_values = [accepted_low, accepted_high, observed_value]
    if reference_mean is not None:
        domain_values.append(reference_mean)
    domain_min = min(domain_values)
    domain_max = max(domain_values)
    span = domain_max - domain_min
    if span <= 0.0:
        pad = max(abs(domain_max) * 0.25, 1.0)
    else:
        pad = max(span * 0.12, abs(domain_max) * 0.03, 0.1)
    domain_low = domain_min - pad
    domain_high = domain_max + pad
    if min(domain_values) >= 0.0:
        domain_low = max(0.0, domain_low)

    def _position(value: float | None) -> float | None:
        if value is None:
            return None
        width = domain_high - domain_low
        if width <= 0.0:
            return 50.0
        return max(0.0, min(100.0, ((value - domain_low) / width) * 100.0))

    accepted_interval_standard = str(evidence.get("accepted_interval_standard", "")).strip()
    return {
        "observed_key": observed_key or "",
        "observed_value": observed_value,
        "reference_mean": reference_mean,
        "reference_unit": reference_unit,
        "accepted_low": accepted_low,
        "accepted_high": accepted_high,
        "accepted_interval_standard": accepted_interval_standard,
        "accepted_interval_mode": str(evidence.get("accepted_interval_mode", "")).strip(),
        "domain_low": domain_low,
        "domain_high": domain_high,
        "positions": {
            "accepted_low": _position(accepted_low),
            "accepted_high": _position(accepted_high),
            "reference_mean": _position(reference_mean),
            "observed_value": _position(observed_value),
        },
    }


def _render_interval_visual(item: AuditItem, interval: dict[str, Any]) -> str:
    unit = str(interval["reference_unit"])
    observed_text = _format_numeric(interval["observed_value"], unit=unit)
    reference_text = _format_numeric(interval["reference_mean"], unit=unit)
    low_text = _format_numeric(interval["accepted_low"], unit=unit)
    high_text = _format_numeric(interval["accepted_high"], unit=unit)
    interval_label = str(interval["accepted_interval_standard"] or "reference interval")
    domain_low_text = _format_numeric(interval["domain_low"], unit=unit)
    domain_high_text = _format_numeric(interval["domain_high"], unit=unit)
    positions = interval["positions"]
    band_left = min(float(positions["accepted_low"]), float(positions["accepted_high"]))
    band_width = max(0.0, abs(float(positions["accepted_high"]) - float(positions["accepted_low"])))
    status_text = "inside" if item.status == "PASS" else "outside"
    reference_tick_html = ""
    if positions["reference_mean"] is not None:
        reference_tick_html = (
            f"<div class='interval-tick interval-reference' "
            f"style='left:{float(positions['reference_mean']):.2f}%'></div>"
        )
    aria_label = (
        f"Observed value {observed_text}, {status_text} the accepted range from {low_text} to {high_text}. "
        f"Reference mean {reference_text}. Standard: {interval_label}."
    )
    return f"""
<div class='item-block interval-block'>
  <h4>Reference interval</h4>
  <div class='interval-metric-grid'>
    <div class='interval-metric'><span>Observed</span><strong>{_esc(observed_text)}</strong></div>
    <div class='interval-metric'><span>Reference mean</span><strong>{_esc(reference_text)}</strong></div>
    <div class='interval-metric'><span>Accepted range</span><strong>{_esc(low_text)} to {_esc(high_text)}</strong></div>
    <div class='interval-metric'><span>Standard</span><strong>{_esc(interval_label)}</strong></div>
  </div>
  <div class='interval-visual' data-interval-visual role='img' aria-label='{_esc(aria_label)}'>
    <div class='interval-range-labels'>
      <span>{_esc(domain_low_text)}</span>
      <span>{_esc(domain_high_text)}</span>
    </div>
    <div class='interval-track'>
      <div class='interval-band' style='left:{band_left:.2f}%; width:{band_width:.2f}%;'></div>
      {reference_tick_html}
      <div class='interval-marker {_status_class(item.status)}' style='left:{float(positions["observed_value"] or 0.0):.2f}%'></div>
      <span class='interval-marker-label observed-label' style='left:{float(positions["observed_value"] or 0.0):.2f}%'>Observed {observed_text}</span>
    </div>
    <div class='interval-legend'>
      <span><i class='legend-swatch accepted'></i>accepted range</span>
      <span><i class='legend-swatch reference'></i>reference mean</span>
      <span><i class='legend-swatch observed {_status_class(item.status)}'></i>observed</span>
    </div>
  </div>
</div>
"""


_SERIES_X_KEY_CANDIDATES = ("currents_pA", "step_currents_pA", "current_steps_pA", "current_pA")
_SERIES_Y_KEY_CANDIDATES = ("firing_rates_by_step_Hz", "reference_values_Hz", "model_values_Hz", "firing_rate_Hz")


def _float_list_or_none(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    values: list[float] = []
    for entry in value:
        numeric = _float_or_none(entry)
        if numeric is None:
            return None
        values.append(numeric)
    return values


def _series_domain(values: list[float]) -> tuple[float, float]:
    domain_low = min(values)
    domain_high = max(values)
    if all(value >= 0.0 for value in values):
        domain_low = 0.0
    elif all(value <= 0.0 for value in values):
        domain_high = 0.0
    span = domain_high - domain_low
    if span <= 0.0:
        pad = max(abs(domain_high) * 0.25, 1.0)
    else:
        pad = max(span * 0.12, abs(domain_high) * 0.03, 0.1)
    domain_low -= pad
    domain_high += pad
    if min(values) >= 0.0:
        domain_low = max(0.0, domain_low)
    if max(values) <= 0.0:
        domain_high = min(0.0, domain_high)
    if domain_low == domain_high:
        domain_high = domain_low + 1.0
    return domain_low, domain_high


def _series_nice_step(raw_step: float) -> float:
    if raw_step <= 0.0:
        return 1.0
    exponent = math.floor(math.log10(raw_step))
    fraction = raw_step / (10.0**exponent)
    candidates = (1.0, 2.0, 2.5, 5.0, 10.0)
    chosen = min(candidates, key=lambda candidate: abs(candidate - fraction))
    return chosen * (10.0**exponent)


def _series_tick_values(low: float, high: float, *, target_ticks: int = 5) -> list[float]:
    if low == high:
        return [low]
    raw_step = abs(high - low) / max(target_ticks - 1, 1)
    step = _series_nice_step(raw_step)
    start = math.floor(low / step) * step
    end = math.ceil(high / step) * step
    values: list[float] = []
    current = start
    guard = 0
    while current <= end + step * 0.5 and guard < 64:
        values.append(round(current, 12))
        current += step
        guard += 1
    return values


def _series_tick_label(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(int(round(value)))
    magnitude = abs(value)
    if magnitude >= 100:
        text = f"{value:.0f}"
    elif magnitude >= 10:
        text = f"{value:.1f}"
    elif magnitude >= 1:
        text = f"{value:.1f}"
    elif magnitude >= 0.1:
        text = f"{value:.2f}"
    else:
        text = f"{value:.3f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _series_color_for_key(key: str, index: int, *, status: str) -> dict[str, str]:
    lowered = key.lower()
    if "reference" in lowered:
        return {"stroke": "#64748b", "fill": "#64748b", "dash": "6 4"}
    if "model" in lowered:
        stroke = {"PASS": "#2563eb", "WARN": "#d97706", "FAIL": "#dc2626"}.get(status, "#2563eb")
        return {"stroke": stroke, "fill": stroke, "dash": ""}
    palette = ["#2563eb", "#0f766e", "#7c3aed", "#d97706"]
    color = palette[index % len(palette)]
    return {"stroke": color, "fill": color, "dash": ""}


def _visual_backend(spec: dict[str, Any] | None) -> str:
    if not isinstance(spec, dict):
        return "svg"
    backend = str(spec.get("backend") or "svg").strip().lower()
    return backend or "svg"


def _visual_kind(spec: dict[str, Any] | None) -> str:
    if not isinstance(spec, dict):
        return ""
    return str(spec.get("kind") or "").strip().lower()


def _figure_to_inline_svg(fig: Any) -> str:
    from io import StringIO

    buffer = StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    svg = buffer.getvalue()
    start = svg.find("<svg")
    return svg[start:] if start >= 0 else svg


def _series_graph_payload(
    evidence: dict[str, Any],
    *,
    exclude_keys: set[str] | None = None,
) -> tuple[dict[str, Any], set[str]] | tuple[None, set[str]]:
    exclude = set(exclude_keys or set())
    if not evidence:
        return None, set()

    series_kind = str(evidence.get("series_kind", "")).strip().lower()
    x_key = ""
    x_values: list[float] | None = None
    series_entries: list[dict[str, Any]] = []
    used_keys: set[str] = set()

    row_source = evidence.get("fi_curve_rows")
    if isinstance(row_source, list) and row_source:
        row_points: list[tuple[float, float]] = []
        for row in row_source:
            if not isinstance(row, dict):
                continue
            x_value = _float_or_none(row.get("current_pA"))
            y_value = _float_or_none(row.get("firing_rate_Hz"))
            if x_value is None or y_value is None:
                continue
            row_points.append((x_value, y_value))
        if row_points:
            row_points.sort(key=lambda pair: pair[0])
            x_values = [point[0] for point in row_points]
            x_key = "current_pA"
            series_entries.append(
                {
                    "key": "firing_rate_Hz",
                    "label": _evidence_label("firing_rate_Hz"),
                    "points": row_points,
                }
            )
            used_keys.update({"fi_curve_rows", "current_pA", "firing_rate_Hz"})
            if not series_kind:
                series_kind = "f-i curve"

    if x_values is None:
        for candidate in _SERIES_X_KEY_CANDIDATES:
            if candidate in exclude or candidate == "fi_curve_rows":
                continue
            candidate_values = _float_list_or_none(evidence.get(candidate))
            if candidate_values is not None and len(candidate_values) >= 2:
                x_key = candidate
                x_values = candidate_values
                used_keys.add(candidate)
                break

    if x_values is None:
        return None, set()

    x_len = len(x_values)
    for candidate in _SERIES_Y_KEY_CANDIDATES:
        if candidate in exclude or candidate == x_key:
            continue
        candidate_values = _float_list_or_none(evidence.get(candidate))
        if candidate_values is None or len(candidate_values) != x_len:
            continue
        ordered_pairs = sorted(zip(x_values, candidate_values), key=lambda pair: pair[0])
        series_entries.append(
            {
                "key": candidate,
                "label": _evidence_label(candidate),
                "points": ordered_pairs,
            }
        )
        used_keys.add(candidate)

    if not series_entries:
        return None, set()

    has_rate_series = any("rate" in entry["key"].lower() or entry["key"].lower().endswith("_hz") for entry in series_entries)
    is_fi_curve = series_kind == "f-i curve" or ("current" in x_key.lower() and has_rate_series)
    if not series_kind:
        series_kind = "f-i curve" if is_fi_curve else "series graph"

    x_axis_label = "Current (pA)" if "current" in x_key.lower() else _evidence_label(x_key)
    y_axis_label = "Firing rate (Hz)" if is_fi_curve or has_rate_series else "Value"
    return (
        {
            "series_kind": series_kind,
            "x_key": x_key,
            "x_values": x_values,
            "series_entries": series_entries,
            "x_axis_label": x_axis_label,
            "y_axis_label": y_axis_label,
        },
        used_keys,
    )


def _render_series_graph_matplotlib(
    item: AuditItem,
    evidence: dict[str, Any],
    *,
    spec: dict[str, Any] | None = None,
    exclude_keys: set[str] | None = None,
) -> tuple[str, set[str]]:
    kind = _visual_kind(spec) or "fi_curve"
    style = dict(spec.get("style") or {}) if isinstance(spec, dict) else {}

    if kind == "bar":
        values_key = str(spec.get("values_key") or spec.get("y_key") or "").strip()
        if not values_key and isinstance(spec.get("keys"), (list, tuple)) and spec["keys"]:
            values_key = str(spec["keys"][0]).strip()
        if not values_key:
            return "", set()
        values = _float_list_or_none(evidence.get(values_key))
        if values is None or not values:
            return "", set()
        labels: list[str] = []
        raw_labels = spec.get("labels")
        if raw_labels is None:
            labels_key = str(spec.get("labels_key") or "").strip()
            if labels_key:
                raw_labels = evidence.get(labels_key)
        if isinstance(raw_labels, str):
            labels = [part.strip() for part in raw_labels.split(",") if part.strip()]
        elif isinstance(raw_labels, (list, tuple)):
            labels = [str(part).strip() for part in raw_labels if str(part).strip()]
        if len(labels) != len(values):
            labels = [str(index + 1) for index in range(len(values))]

        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["svg.fonttype"] = "none"
        import matplotlib.pyplot as plt

        figsize = tuple(style.get("figsize") or (6.0, 2.8))
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#f8fafc")
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", color="#cbd5e1", linewidth=0.6, alpha=0.35)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#94a3b8")
        ax.spines["bottom"].set_color("#94a3b8")
        ax.tick_params(axis="both", labelsize=8, colors="#334155")
        positions = list(range(len(values)))
        ax.bar(positions, values, color=str(style.get("bar_color") or "#2563eb"), alpha=float(style.get("alpha") or 0.9))
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=float(style.get("x_label_rotation") or 25.0), ha=str(style.get("x_label_ha") or "right"))
        ax.set_ylabel(str(style.get("y_label") or _evidence_label(values_key)))
        if style.get("title"):
            ax.set_title(str(style["title"]), fontsize=9, pad=8)
        fig.tight_layout()
        svg = _figure_to_inline_svg(fig)
        plt.close(fig)
        svg = svg.replace(
            "<svg ",
            f"<svg class='series-graph-svg' data-series-graph role='img' aria-label='{_esc(str(style.get('title') or 'Bar chart'))}' ",
            1,
        )
        return (
            (
                f"<div class='item-block series-graph-block' data-visual-backend='matplotlib' "
                f"data-visual-kind='bar'>"
                f"<h4>{_esc(str(style.get('title') or 'Bar chart'))}</h4>"
                f"<div class='series-graph-meta'><span>{_esc(str(style.get('x_label') or 'Category'))}</span>"
                f"<span>{_esc(str(style.get('y_label') or _evidence_label(values_key)))}</span></div>"
                f"<div class='series-graph-shell'>{svg}</div>"
                f"</div>"
            ),
            {values_key},
        )

    if kind in {"scatter", "regression"}:
        x_key = str(spec.get("x_key") or "").strip()
        y_key = str(spec.get("y_key") or "").strip()
        raw_keys = spec.get("keys")
        if (not x_key or not y_key) and isinstance(raw_keys, (list, tuple)) and len(raw_keys) >= 2:
            x_key = x_key or str(raw_keys[0]).strip()
            y_key = y_key or str(raw_keys[1]).strip()
        if not x_key or not y_key:
            return "", set()
        x_values = _float_list_or_none(evidence.get(x_key))
        y_values = _float_list_or_none(evidence.get(y_key))
        if x_values is None or y_values is None:
            return "", set()
        point_pairs = [(x_value, y_value) for x_value, y_value in zip(x_values, y_values) if _float_or_none(x_value) is not None and _float_or_none(y_value) is not None]
        if len(point_pairs) < 2:
            return "", set()
        x_points = [pair[0] for pair in point_pairs]
        y_points = [pair[1] for pair in point_pairs]

        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["svg.fonttype"] = "none"
        import matplotlib.pyplot as plt

        figsize = tuple(style.get("figsize") or (6.0, 2.8))
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#f8fafc")
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", color="#cbd5e1", linewidth=0.6, alpha=0.35)
        ax.grid(True, axis="x", color="#e2e8f0", linewidth=0.4, alpha=0.25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#94a3b8")
        ax.spines["bottom"].set_color("#94a3b8")
        ax.tick_params(axis="both", labelsize=8, colors="#334155")
        color = str(style.get("color") or "#2563eb")
        ax.scatter(
            x_points,
            y_points,
            s=float(style.get("marker_size") or 3.4) ** 2 * 6.0,
            color=color,
            edgecolors=str(style.get("edge_color") or color),
            linewidths=0.8,
            alpha=float(style.get("alpha") or 0.9),
        )
        if kind == "regression":
            point_count = len(x_points)
            x_mean = sum(x_points) / point_count
            y_mean = sum(y_points) / point_count
            denom = sum((value - x_mean) ** 2 for value in x_points)
            if denom > 0.0:
                slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_points, y_points)) / denom
                intercept = y_mean - slope * x_mean
                fit_x = [min(x_points), max(x_points)]
                fit_y = [slope * x_value + intercept for x_value in fit_x]
                ax.plot(
                    fit_x,
                    fit_y,
                    color=str(style.get("fit_color") or "#0f172a"),
                    linewidth=float(style.get("line_width") or 1.6),
                    linestyle="--",
                    alpha=0.75,
                )
        ax.set_xlabel(str(style.get("x_label") or _evidence_label(x_key)))
        ax.set_ylabel(str(style.get("y_label") or _evidence_label(y_key)))
        if style.get("title"):
            ax.set_title(str(style["title"]), fontsize=9, pad=8)
        fig.tight_layout()
        svg = _figure_to_inline_svg(fig)
        plt.close(fig)
        svg = svg.replace(
            "<svg ",
            f"<svg class='series-graph-svg' data-series-graph role='img' aria-label='{_esc(str(style.get('title') or ('Regression' if kind == 'regression' else 'Scatter plot')))}' ",
            1,
        )
        return (
            (
                f"<div class='item-block series-graph-block' data-visual-backend='matplotlib' "
                f"data-visual-kind='{_esc(kind)}'>"
                f"<h4>{_esc(str(style.get('title') or ('Regression' if kind == 'regression' else 'Scatter plot')))}</h4>"
                f"<div class='series-graph-meta'><span>{_esc(str(style.get('x_label') or _evidence_label(x_key)))}</span>"
                f"<span>{_esc(str(style.get('y_label') or _evidence_label(y_key)))}</span></div>"
                f"<div class='series-graph-shell'>{svg}</div>"
                f"</div>"
            ),
            {x_key, y_key},
        )

    payload, used_keys = _series_graph_payload(evidence, exclude_keys=exclude_keys)
    if payload is None:
        return "", set()

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    figsize = tuple(style.get("figsize") or (6.0, 2.8))
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="#cbd5e1", linewidth=0.6, alpha=0.35)
    ax.grid(True, axis="x", color="#e2e8f0", linewidth=0.4, alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(axis="both", labelsize=8, colors="#334155")

    x_axis_label = str(style.get("x_label") or payload["x_axis_label"])
    y_axis_label = str(style.get("y_label") or payload["y_axis_label"])
    series_entries = list(payload["series_entries"])
    x_domain = _series_domain([point[0] for series in series_entries for point in series["points"]])
    y_domain = _series_domain([point[1] for series in series_entries for point in series["points"]])
    x_low, x_high = x_domain
    y_low, y_high = y_domain
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    x_ticks = _series_tick_values(x_low, x_high, target_ticks=5)
    y_ticks = _series_tick_values(y_low, y_high, target_ticks=5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([_series_tick_label(tick) for tick in x_ticks])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([_series_tick_label(tick) for tick in y_ticks])
    if x_low <= 0.0 <= x_high:
        ax.axvline(0.0, color="#94a3b8", linewidth=0.8, alpha=0.35, zorder=0)
    if y_low <= 0.0 <= y_high:
        ax.axhline(0.0, color="#94a3b8", linewidth=0.8, alpha=0.35, zorder=0)

    legend_items: list[str] = []
    line_width = float(style.get("line_width") or 1.8)
    marker_size = float(style.get("marker_size") or 3.4)
    for index, series in enumerate(series_entries):
        color = _series_color_for_key(series["key"], index, status=str(item.status))
        x_points = [point[0] for point in series["points"]]
        y_points = [point[1] for point in series["points"]]
        dash_style = (0, (6, 4)) if color["dash"] else None
        if kind == "scatter":
            ax.scatter(
                x_points,
                y_points,
                s=(marker_size**2) * 6.0,
                color=color["fill"],
                edgecolors=color["stroke"],
                linewidths=0.8,
                alpha=float(style.get("alpha") or 0.9),
                label=series["label"],
            )
        elif kind == "regression":
            ax.scatter(
                x_points,
                y_points,
                s=(marker_size**2) * 6.0,
                color=color["fill"],
                edgecolors=color["stroke"],
                linewidths=0.8,
                alpha=float(style.get("alpha") or 0.9),
                label=series["label"],
            )
            point_count = len(x_points)
            if point_count >= 2:
                x_mean = sum(x_points) / point_count
                y_mean = sum(y_points) / point_count
                denom = sum((value - x_mean) ** 2 for value in x_points)
                if denom > 0.0:
                    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_points, y_points)) / denom
                    intercept = y_mean - slope * x_mean
                    fit_x = [x_low, x_high]
                    fit_y = [slope * x_value + intercept for x_value in fit_x]
                    ax.plot(
                        fit_x,
                        fit_y,
                        color="#0f172a",
                        linewidth=max(1.2, line_width - 0.2),
                        linestyle="--",
                        alpha=0.75,
                    )
        else:
            ax.plot(
                x_points,
                y_points,
                color=color["stroke"],
                linewidth=line_width,
                linestyle="--" if dash_style else "-",
                marker="o",
                markersize=marker_size,
                markerfacecolor=color["fill"],
                markeredgecolor=color["stroke"],
                label=series["label"],
            )
        legend_items.append(
            f"<span class='series-legend-item'><i class='series-legend-swatch' style='background:{color['fill']}; border-color:{color['stroke']};"
            f"{' border-style:dashed;' if color['dash'] else ''}'></i>{_esc(series['label'])}</span>"
        )

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    if style.get("title"):
        ax.set_title(str(style["title"]), fontsize=9, pad=8)
    fig.tight_layout()
    svg = _figure_to_inline_svg(fig)
    plt.close(fig)
    svg = svg.replace(
        "<svg ",
        f"<svg class='series-graph-svg' data-series-graph role='img' aria-label='{_esc(str(style.get('title') or ('f-I curve' if payload['series_kind'] == 'f-i curve' else 'Series graph')))}' ",
        1,
    )

    visual_kind = kind or str(payload["series_kind"])
    return (
        (
            f"<div class='item-block series-graph-block' data-visual-backend='matplotlib' "
            f"data-visual-kind='{_esc(visual_kind)}'>"
            f"<h4>{_esc('f-I curve' if payload['series_kind'] == 'f-i curve' else 'Series graph')}</h4>"
            f"<div class='series-graph-meta'><span>{_esc(x_axis_label)}</span><span>{_esc(y_axis_label)}</span></div>"
            f"<div class='series-graph-shell'>"
            f"{svg}"
            f"</div>"
            f"<div class='series-legend'>{''.join(legend_items)}</div>"
            f"</div>"
        ),
        used_keys,
    )


def _render_series_graph(
    item: AuditItem,
    evidence: dict[str, Any],
    *,
    exclude_keys: set[str] | None = None,
    spec: dict[str, Any] | None = None,
) -> tuple[str, set[str]]:
    if _visual_backend(spec) == "matplotlib":
        block_html, used_keys = _render_series_graph_matplotlib(
            item,
            evidence,
            spec=spec,
            exclude_keys=exclude_keys,
        )
        if block_html:
            return block_html, used_keys
    exclude = set(exclude_keys or set())
    if not evidence:
        return "", set()

    series_kind = str(evidence.get("series_kind", "")).strip()
    x_key = ""
    x_values: list[float] | None = None
    series_entries: list[dict[str, Any]] = []
    used_keys: set[str] = set()

    row_source = evidence.get("fi_curve_rows")
    if isinstance(row_source, list) and row_source:
        row_points: list[tuple[float, float]] = []
        for row in row_source:
            if not isinstance(row, dict):
                continue
            x_value = _float_or_none(row.get("current_pA"))
            y_value = _float_or_none(row.get("firing_rate_Hz"))
            if x_value is None or y_value is None:
                continue
            row_points.append((x_value, y_value))
        if row_points:
            row_points.sort(key=lambda pair: pair[0])
            x_values = [point[0] for point in row_points]
            x_key = "current_pA"
            series_entries.append(
                {
                    "key": "firing_rate_Hz",
                    "label": _evidence_label("firing_rate_Hz"),
                    "points": row_points,
                }
            )
            used_keys.add("fi_curve_rows")
            used_keys.add("current_pA")
            used_keys.add("firing_rate_Hz")
            if not series_kind:
                series_kind = "f-i curve"

    if x_values is None:
        for candidate in _SERIES_X_KEY_CANDIDATES:
            if candidate in exclude or candidate == "fi_curve_rows":
                continue
            candidate_values = _float_list_or_none(evidence.get(candidate))
            if candidate_values is not None and len(candidate_values) >= 2:
                x_key = candidate
                x_values = candidate_values
                used_keys.add(candidate)
                break

    if x_values is None:
        return "", set()

    x_len = len(x_values)
    for candidate in _SERIES_Y_KEY_CANDIDATES:
        if candidate in exclude or candidate == x_key:
            continue
        candidate_values = _float_list_or_none(evidence.get(candidate))
        if candidate_values is None or len(candidate_values) != x_len:
            continue
        ordered_pairs = sorted(zip(x_values, candidate_values), key=lambda pair: pair[0])
        series_entries.append(
            {
                "key": candidate,
                "label": _evidence_label(candidate),
                "points": ordered_pairs,
            }
        )
        used_keys.add(candidate)

    if not series_entries:
        return "", set()

    x_domain = _series_domain([point[0] for series in series_entries for point in series["points"]])
    y_domain = _series_domain([point[1] for series in series_entries for point in series["points"]])
    x_low, x_high = x_domain
    y_low, y_high = y_domain
    plot_width = 520.0
    plot_height = 160.0
    margin_left = 62.0
    margin_right = 18.0
    margin_top = 18.0
    margin_bottom = 40.0
    svg_width = margin_left + plot_width + margin_right
    svg_height = margin_top + plot_height + margin_bottom

    def _x_pos(value: float) -> float:
        width = x_high - x_low
        if width <= 0.0:
            return margin_left + plot_width / 2.0
        return margin_left + ((value - x_low) / width) * plot_width

    def _y_pos(value: float) -> float:
        height = y_high - y_low
        if height <= 0.0:
            return margin_top + plot_height / 2.0
        return margin_top + plot_height - ((value - y_low) / height) * plot_height

    def _path_d(points: list[tuple[float, float]]) -> str:
        commands = [f"M {_x_pos(points[0][0]):.2f} {_y_pos(points[0][1]):.2f}"]
        commands.extend(f"L {_x_pos(x):.2f} {_y_pos(y):.2f}" for x, y in points[1:])
        return " ".join(commands)

    has_rate_series = any("rate" in entry["key"].lower() or entry["key"].lower().endswith("_hz") for entry in series_entries)
    is_fi_curve = series_kind == "f-i curve" or ("current" in x_key.lower() and has_rate_series)
    if not series_kind:
        series_kind = "f-i curve" if is_fi_curve else "series graph"

    x_axis_label = "Current (pA)" if "current" in x_key.lower() else _evidence_label(x_key)
    y_axis_label = "Firing rate (Hz)" if is_fi_curve or has_rate_series else "Value"
    x_ticks = _series_tick_values(x_low, x_high, target_ticks=5)
    y_ticks = _series_tick_values(y_low, y_high, target_ticks=5)

    svg_lines = [
        f"<svg class='series-graph-svg' data-series-graph role='img' viewBox='0 0 {svg_width:.0f} {svg_height:.0f}' "
        f"aria-label='{_esc(series_kind.title())} with {len(series_entries)} series on a shared scale. "
        f"X axis {x_axis_label} from {_format_numeric(x_low)} to {_format_numeric(x_high)}. "
        f"Y axis {y_axis_label} from {_format_numeric(y_low)} to {_format_numeric(y_high)}.'>"
        f"<rect class='series-graph-bg' x='0' y='0' width='{svg_width:.0f}' height='{svg_height:.0f}' rx='10' ry='10'></rect>",
        f"<line class='series-axis' x1='{margin_left:.2f}' y1='{margin_top + plot_height:.2f}' x2='{margin_left + plot_width:.2f}' y2='{margin_top + plot_height:.2f}'></line>",
        f"<line class='series-axis' x1='{margin_left:.2f}' y1='{margin_top:.2f}' x2='{margin_left:.2f}' y2='{margin_top + plot_height:.2f}'></line>",
    ]

    x_axis_y = margin_top + plot_height
    x_tick_label_y = x_axis_y + 16.0
    y_tick_x = margin_left
    y_tick_label_x = margin_left - 8.0
    for tick in x_ticks:
        x_pos = _x_pos(tick)
        svg_lines.append(
            f"<line class='series-axis-tick series-axis-x' x1='{x_pos:.2f}' y1='{x_axis_y:.2f}' x2='{x_pos:.2f}' y2='{x_axis_y + 4.0:.2f}'></line>"
        )
        svg_lines.append(
            f"<text class='series-tick-label series-tick-label-x' x='{x_pos:.2f}' y='{x_tick_label_y:.2f}' text-anchor='middle' dominant-baseline='hanging'>"
            f"{_esc(_series_tick_label(tick))}</text>"
        )

    for tick in y_ticks:
        y_pos = _y_pos(tick)
        svg_lines.append(
            f"<line class='series-axis-tick series-axis-y' x1='{y_tick_x - 4.0:.2f}' y1='{y_pos:.2f}' x2='{y_tick_x:.2f}' y2='{y_pos:.2f}'></line>"
        )
        svg_lines.append(
            f"<text class='series-tick-label series-tick-label-y' x='{y_tick_label_x:.2f}' y='{y_pos:.2f}' text-anchor='end' dominant-baseline='middle'>"
            f"{_esc(_series_tick_label(tick))}</text>"
        )

    if x_low <= 0.0 <= x_high:
        svg_lines.append(
            f"<line class='series-zero' x1='{_x_pos(0.0):.2f}' y1='{margin_top:.2f}' x2='{_x_pos(0.0):.2f}' y2='{margin_top + plot_height:.2f}'></line>"
        )

    if y_low <= 0.0 <= y_high:
        svg_lines.append(
            f"<line class='series-zero' x1='{margin_left:.2f}' y1='{_y_pos(0.0):.2f}' x2='{margin_left + plot_width:.2f}' y2='{_y_pos(0.0):.2f}'></line>"
        )

    legend_items: list[str] = []
    visual_backend = _visual_backend(spec)
    visual_kind = _visual_kind(spec) or series_kind
    for index, series in enumerate(series_entries):
        color = _series_color_for_key(series["key"], index, status=str(item.status))
        dash_attr = f" stroke-dasharray='{color['dash']}'" if color["dash"] else ""
        svg_lines.append(
            f"<path class='series-line' d='{_esc(_path_d(series['points']))}' "
            f"style='stroke:{color['stroke']}; fill:none;' {dash_attr}></path>"
        )
        for x_value, y_value in series["points"]:
            svg_lines.append(
                f"<circle class='series-point' cx='{_x_pos(x_value):.2f}' cy='{_y_pos(y_value):.2f}' r='3.4' "
                f"style='stroke:{color['stroke']}; fill:{color['fill']};'></circle>"
            )
        legend_items.append(
            f"<span class='series-legend-item'><i class='series-legend-swatch' style='background:{color['fill']}; border-color:{color['stroke']};"
            f"{' border-style:dashed;' if color['dash'] else ''}'></i>{_esc(series['label'])}</span>"
        )

    return (
        f"<div class='item-block series-graph-block' data-visual-backend='{_esc(visual_backend)}' "
        f"data-visual-kind='{_esc(visual_kind)}'>"
        f"<h4>{_esc('f-I curve' if series_kind == 'f-i curve' else 'Series graph')}</h4>"
        f"<div class='series-graph-meta'><span>{_esc(x_axis_label)}</span><span>{_esc(y_axis_label)}</span></div>"
        f"<div class='series-graph-shell'>"
        f"{''.join(svg_lines)}</svg>"
        f"</div>"
        f"<div class='series-legend'>{''.join(legend_items)}</div>"
        f"</div>"
    ), used_keys


def _render_numeric_companions(
    item: AuditItem,
    evidence: dict[str, Any],
    *,
    visuals: list[dict[str, Any]] | None = None,
    exclude_keys: set[str] | None = None,
) -> tuple[str, set[str]]:
    exclude = set(exclude_keys or set())
    requested_visuals = [spec for spec in (visuals or []) if isinstance(spec, dict)]
    if not evidence or not requested_visuals:
        return "", set()

    used_keys: set[str] = set()
    blocks: list[str] = []

    def _requested_keys(spec: dict[str, Any]) -> list[str]:
        raw_keys = spec.get("keys")
        if raw_keys is None:
            raw_key = spec.get("key")
            if raw_key is None:
                return []
            raw_keys = [raw_key]
        elif isinstance(raw_keys, str):
            raw_keys = [raw_keys]
        return [str(key).strip() for key in raw_keys if str(key).strip()]

    def _render_numeric_strip(spec: dict[str, Any]) -> tuple[str, set[str]]:
        keys = _requested_keys(spec)
        if not keys:
            return "", set()

        scalar_entries: list[dict[str, Any]] = []
        local_used: set[str] = set()
        for key in keys:
            if key in exclude or key == "__reference_annotations__":
                continue
            if key in _INTERVAL_RESERVED_EVIDENCE_KEYS:
                continue
            if key in _SERIES_X_KEY_CANDIDATES or key in _SERIES_Y_KEY_CANDIDATES or key == "fi_curve_rows":
                continue
            numeric_value = _float_or_none(evidence.get(key))
            if numeric_value is None:
                continue
            scalar_entries.append(
                {
                    "key": key,
                    "label": _evidence_label(key),
                    "value": numeric_value,
                }
            )
            local_used.add(key)

        if not scalar_entries:
            return "", set()

        values = [float(entry["value"]) for entry in scalar_entries]
        domain_low, domain_high = _series_domain(values)
        plot_width = 520.0
        plot_height = 78.0
        margin_left = 54.0
        margin_right = 18.0
        margin_top = 18.0
        margin_bottom = 34.0
        svg_width = margin_left + plot_width + margin_right
        svg_height = margin_top + plot_height + margin_bottom

        def _x_pos(value: float) -> float:
            width = domain_high - domain_low
            if width <= 0.0:
                return margin_left + plot_width / 2.0
            return margin_left + ((value - domain_low) / width) * plot_width

        axis_y = margin_top + plot_height / 2.0
        x_ticks = _series_tick_values(domain_low, domain_high, target_ticks=5)
        dots: list[str] = []
        legend_items: list[str] = []
        for index, entry in enumerate(scalar_entries):
            color = _series_color_for_key(entry["key"], index, status=str(item.status))
            x_pos = _x_pos(float(entry["value"]))
            dots.append(
                f"<circle class='numeric-dot' cx='{x_pos:.2f}' cy='{axis_y:.2f}' r='4.2' "
                f"style='stroke:{color['stroke']}; fill:{color['fill']};'></circle>"
            )
            legend_items.append(
                f"<span class='numeric-legend-item'><i class='numeric-legend-swatch' style='background:{color['fill']}; border-color:{color['stroke']};"
                f"{' border-style:dashed;' if color['dash'] else ''}'></i>{_esc(entry['label'])} {_esc(_format_numeric(float(entry['value'])))}"
                f"</span>"
            )

        block_title = str(spec.get("title") or "Numeric summary")
        left_meta = str(spec.get("left_meta") or "shared scale")
        right_meta = str(spec.get("right_meta") or "scalar metrics")
        svg_lines = [
            f"<svg class='numeric-strip-svg' data-numeric-strip role='img' viewBox='0 0 {svg_width:.0f} {svg_height:.0f}' "
            f"aria-label='Numeric summary with {len(scalar_entries)} values on a shared scale from {_format_numeric(domain_low)} to {_format_numeric(domain_high)}.'>",
            f"<rect class='numeric-strip-bg' x='0' y='0' width='{svg_width:.0f}' height='{svg_height:.0f}' rx='10' ry='10'></rect>",
            f"<line class='numeric-strip-axis' x1='{margin_left:.2f}' y1='{axis_y:.2f}' x2='{margin_left + plot_width:.2f}' y2='{axis_y:.2f}'></line>",
        ]
        if domain_low <= 0.0 <= domain_high:
            svg_lines.append(
                f"<line class='numeric-strip-zero' x1='{_x_pos(0.0):.2f}' y1='{margin_top:.2f}' x2='{_x_pos(0.0):.2f}' y2='{margin_top + plot_height:.2f}'></line>"
            )
        for tick in x_ticks:
            x_pos = _x_pos(tick)
            svg_lines.append(
                f"<line class='numeric-strip-tick' x1='{x_pos:.2f}' y1='{axis_y - 4.0:.2f}' x2='{x_pos:.2f}' y2='{axis_y + 4.0:.2f}'></line>"
            )
            svg_lines.append(
                f"<text class='numeric-strip-label' x='{x_pos:.2f}' y='{axis_y + 12.0:.2f}' text-anchor='middle' dominant-baseline='hanging'>"
                f"{_esc(_series_tick_label(tick))}</text>"
            )
        svg_lines.extend(dots)
        svg_lines.append("</svg>")
        return (
            "".join(
                [
                    "<div class='item-block numeric-strip-block'>",
                    f"<h4>{_esc(block_title)}</h4>",
                    f"<div class='numeric-strip-meta'><span>{_esc(left_meta)}</span><span>{_esc(right_meta)}</span></div>",
                    "<div class='numeric-strip-shell'>",
                    "".join(svg_lines),
                    "</div>",
                    f"<div class='numeric-legend'>{''.join(legend_items)}</div>",
                    "</div>",
                ]
            ),
            local_used,
        )

    def _render_numeric_sparkline(spec: dict[str, Any]) -> tuple[str, set[str]]:
        keys = _requested_keys(spec)
        if not keys:
            return "", set()

        key = keys[0]
        if key in exclude or key == "__reference_annotations__":
            return "", set()
        if key in _INTERVAL_RESERVED_EVIDENCE_KEYS:
            return "", set()
        if key in _SERIES_X_KEY_CANDIDATES or key in _SERIES_Y_KEY_CANDIDATES or key == "fi_curve_rows":
            return "", set()

        values = _float_list_or_none(evidence.get(key))
        if values is None or len(values) < 2:
            return "", set()

        x_low = 0.5
        x_high = float(len(values)) + 0.5
        y_low, y_high = _series_domain(values)
        plot_width = 520.0
        plot_height = 128.0
        margin_left = 62.0
        margin_right = 18.0
        margin_top = 18.0
        margin_bottom = 38.0
        svg_width = margin_left + plot_width + margin_right
        svg_height = margin_top + plot_height + margin_bottom

        def _x_pos(value: float) -> float:
            width = x_high - x_low
            if width <= 0.0:
                return margin_left + plot_width / 2.0
            return margin_left + ((value - x_low) / width) * plot_width

        def _y_pos(value: float) -> float:
            height = y_high - y_low
            if height <= 0.0:
                return margin_top + plot_height / 2.0
            return margin_top + plot_height - ((value - y_low) / height) * plot_height

        x_ticks = _series_tick_values(1.0, float(len(values)), target_ticks=min(5, len(values)))
        y_ticks = _series_tick_values(y_low, y_high, target_ticks=5)
        x_label = str(spec.get("x_label") or "Index")
        y_label = str(spec.get("y_label") or "Value")
        sequence_label = str(spec.get("label") or _evidence_label(key))
        block_title = str(spec.get("title") or "Numeric sequence")
        svg_lines = [
            f"<svg class='numeric-sparkline-svg' data-numeric-sparkline role='img' viewBox='0 0 {svg_width:.0f} {svg_height:.0f}' "
            f"aria-label='Numeric sequence for {_esc(sequence_label)} with {len(values)} points. "
            f"X axis {x_label} from 1 to {len(values)}. Y axis {y_label} from {_format_numeric(y_low)} to {_format_numeric(y_high)}.'>",
            f"<rect class='numeric-sparkline-bg' x='0' y='0' width='{svg_width:.0f}' height='{svg_height:.0f}' rx='10' ry='10'></rect>",
            f"<line class='numeric-sparkline-axis' x1='{margin_left:.2f}' y1='{margin_top + plot_height:.2f}' x2='{margin_left + plot_width:.2f}' y2='{margin_top + plot_height:.2f}'></line>",
            f"<line class='numeric-sparkline-axis' x1='{margin_left:.2f}' y1='{margin_top:.2f}' x2='{margin_left:.2f}' y2='{margin_top + plot_height:.2f}'></line>",
        ]
        if y_low <= 0.0 <= y_high:
            svg_lines.append(
                f"<line class='numeric-sparkline-zero' x1='{margin_left:.2f}' y1='{_y_pos(0.0):.2f}' x2='{margin_left + plot_width:.2f}' y2='{_y_pos(0.0):.2f}'></line>"
            )
        for tick in x_ticks:
            x_pos = _x_pos(tick)
            svg_lines.append(
                f"<line class='numeric-sparkline-tick' x1='{x_pos:.2f}' y1='{margin_top + plot_height:.2f}' x2='{x_pos:.2f}' y2='{margin_top + plot_height + 4.0:.2f}'></line>"
            )
            svg_lines.append(
                f"<text class='numeric-sparkline-label' x='{x_pos:.2f}' y='{margin_top + plot_height + 16.0:.2f}' text-anchor='middle' dominant-baseline='hanging'>"
                f"{_esc(_series_tick_label(tick))}</text>"
            )
        for tick in y_ticks:
            y_pos = _y_pos(tick)
            svg_lines.append(
                f"<line class='numeric-sparkline-tick' x1='{margin_left - 4.0:.2f}' y1='{y_pos:.2f}' x2='{margin_left:.2f}' y2='{y_pos:.2f}'></line>"
            )
            svg_lines.append(
                f"<text class='numeric-sparkline-label' x='{margin_left - 8.0:.2f}' y='{y_pos:.2f}' text-anchor='end' dominant-baseline='middle'>"
                f"{_esc(_series_tick_label(tick))}</text>"
            )

        path_commands = [f"M {_x_pos(1.0):.2f} {_y_pos(values[0]):.2f}"]
        path_commands.extend(f"L {_x_pos(float(index + 1)):.2f} {_y_pos(value):.2f}" for index, value in enumerate(values[1:]))
        color = _series_color_for_key(str(key), 0, status=str(item.status))
        svg_lines.append(
            f"<path class='numeric-sparkline-line' d='{_esc(' '.join(path_commands))}' style='stroke:{color['stroke']}; fill:none;'></path>"
        )
        for index, value in enumerate(values, start=1):
            svg_lines.append(
                f"<circle class='numeric-sparkline-point' cx='{_x_pos(float(index)):.2f}' cy='{_y_pos(value):.2f}' r='3.4' "
                f"style='stroke:{color['stroke']}; fill:{color['fill']};'></circle>"
            )
        svg_lines.append("</svg>")
        return (
            "".join(
                [
                    "<div class='item-block numeric-sparkline-block'>",
                    f"<h4>{_esc(block_title)}</h4>",
                    f"<div class='numeric-sparkline-meta'><span>{_esc(sequence_label)}</span><span>{_esc(x_label)} / {_esc(y_label)}</span></div>",
                    "<div class='numeric-sparkline-shell'>",
                    "".join(svg_lines),
                    "</div>",
                    "</div>",
                ]
            ),
            {key},
        )

    for spec in requested_visuals:
        kind = str(spec.get("kind") or "").strip().lower()
        if kind in {"numeric_strip", "scalar_strip", "strip", "numeric_summary", "summary"}:
            block_html, local_used = _render_numeric_strip(spec)
        elif kind in {"numeric_sparkline", "sparkline", "sequence", "numeric_sequence"}:
            block_html, local_used = _render_numeric_sparkline(spec)
        else:
            continue
        if block_html:
            blocks.append(block_html)
            used_keys.update(local_used)

    return "".join(blocks), used_keys


def _visual_keys(spec: dict[str, Any]) -> list[str]:
    raw_keys = spec.get("keys")
    if raw_keys is None:
        raw_x_key = spec.get("x_key")
        raw_y_keys = spec.get("y_keys")
        if raw_x_key is None or raw_y_keys is None:
            return []
        if isinstance(raw_y_keys, str):
            raw_y_keys = [raw_y_keys]
        raw_keys = [raw_x_key, *raw_y_keys]
    elif isinstance(raw_keys, str):
        raw_keys = [raw_keys]
    return [str(key).strip() for key in raw_keys if str(key).strip()]


def _render_compact_interval_summary(item: AuditItem, interval: dict[str, Any]) -> str:
    unit = str(interval["reference_unit"])
    observed_text = _format_numeric(interval["observed_value"], unit=unit)
    reference_text = _format_numeric(interval["reference_mean"], unit=unit)
    low_text = _format_numeric(interval["accepted_low"], unit=unit)
    high_text = _format_numeric(interval["accepted_high"], unit=unit)
    interval_label = str(interval["accepted_interval_standard"] or "reference interval")
    positions = interval["positions"]
    band_left = min(float(positions["accepted_low"]), float(positions["accepted_high"]))
    band_width = max(0.0, abs(float(positions["accepted_high"]) - float(positions["accepted_low"])))
    reference_tick_html = ""
    if positions["reference_mean"] is not None:
        reference_tick_html = (
            f"<div class='interval-tick interval-reference' "
            f"style='left:{float(positions['reference_mean']):.2f}%'></div>"
        )
    aria_label = (
        f"Observed value {observed_text}, {'inside' if item.status == 'PASS' else 'outside'} the accepted range "
        f"from {low_text} to {high_text}. Reference mean {reference_text}. Standard: {interval_label}."
    )
    return f"""
<div class='item-compact-interval' data-compact-interval data-interval-visual role='img' aria-label='{_esc(aria_label)}'>
  <div class='interval-track'>
    <div class='interval-band' style='left:{band_left:.2f}%; width:{band_width:.2f}%;'></div>
    {reference_tick_html}
    <div class='interval-marker {_status_class(item.status)}' style='left:{float(positions["observed_value"] or 0.0):.2f}%'></div>
    <span class='interval-marker-label observed-label' style='left:{float(positions["observed_value"] or 0.0):.2f}%'>Observed {observed_text}</span>
  </div>
  <div class='interval-legend'>
    <span><i class='legend-swatch accepted'></i>accepted range</span>
    <span><i class='legend-swatch reference'></i>reference mean</span>
    <span><i class='legend-swatch observed {_status_class(item.status)}'></i>observed</span>
  </div>
</div>
"""


def _render_structured_evidence(evidence: dict[str, Any], *, exclude_keys: set[str] | None = None) -> str:
    if not evidence:
        return ""
    exclude = set(exclude_keys or set())
    rows: list[tuple[str, str]] = []
    for key, value in evidence.items():
        if key in exclude or key == "__reference_annotations__":
            continue
        rows.append((_evidence_label(key), _format_evidence_value(value)))
    if not rows:
        return ""
    rows_html = "".join(
        f"<div class='evidence-row'><dt>{_esc(label)}</dt><dd>{_esc(value)}</dd></div>"
        for label, value in rows
    )
    return (
        "<div class='item-block evidence-block'>"
        "<h4>Details</h4>"
        f"<dl class='evidence-grid'>{rows_html}</dl>"
        "</div>"
    )


def _render_evidence(
    item: AuditItem,
    *,
    exclude_keys: set[str] | None = None,
    include_interval_visual: bool = True,
) -> str:
    evidence = dict(item.evidence or {})
    if not evidence:
        return ""
    interval = _extract_interval_visual_data(item)
    rendered_sections: list[str] = []
    excluded_keys: set[str] = set(exclude_keys or set())
    if interval is not None:
        if include_interval_visual:
            rendered_sections.append(_render_interval_visual(item, interval))
        excluded_keys.update(_INTERVAL_RESERVED_EVIDENCE_KEYS)
        observed_key = str(interval.get("observed_key", "")).strip()
        if observed_key:
            excluded_keys.add(observed_key)
    rendered_sections.append(_render_structured_evidence(evidence, exclude_keys=excluded_keys))
    return "".join(section for section in rendered_sections if section)


def _collapsed_summary_messages(item: AuditItem) -> list[str]:
    return []


def _notes_html(item: AuditItem) -> list[str]:
    notes: list[str] = []
    note_text = str(item.note).strip() if item.note else ""
    if note_text:
        notes.append(note_text)
    notes_list = item.evidence.get("notes") if isinstance(item.evidence, dict) else None
    if isinstance(notes_list, list):
        for note_text in notes_list:
            rendered = str(note_text).strip()
            if rendered and rendered not in notes:
                notes.append(rendered)
    return [
        "<div class='item-body-notes'>",
        "<div class='item-notes-label'>Notes / caveats</div>",
        "<ul class='item-notes-list'>",
        "".join(f"<li class='item-note'>{_esc(_expand_terms(note_text, sentence_case=True))}</li>" for note_text in notes),
        "</ul>",
        "</div>",
    ] if notes else []


def _item_search_blob(item: AuditItem) -> str:
    fields = [
        item.check_id,
        item.title,
        item.criterion,
        item.description,
        item.acceptable,
        item.acceptable_basis,
        item.status_reason,
        item.note,
        json.dumps(item.evidence, sort_keys=True),
    ]
    return " ".join(_expand_terms(field, sentence_case=True) for field in fields if field).lower()


def _clean_item_title_for_card(item: AuditItem) -> str:
    title = str(item.title or "").strip()
    if not title:
        return title

    title = re.sub(r"^\s*audit\s*group\s*:\s*", "", title, count=1, flags=re.IGNORECASE)
    group_hint = str(item.group_title or item.group_id or "").strip()
    if group_hint:
        stripped_with_group = re.sub(rf"^\s*{re.escape(group_hint)}\s*:\s*", "", title, count=1)
        if stripped_with_group and stripped_with_group.strip():
            title = stripped_with_group
    return title.strip()


def _render_item_card(item_payload: dict[str, Any]) -> str:
    item = AuditItem(**item_payload)
    evidence = dict(item.evidence or {})
    interval = _extract_interval_visual_data(item)
    card_id = _safe_dom_id(item.check_id)
    item_detail_body_id = f"item-detail-body-{card_id}"
    compact_messages = _collapsed_summary_messages(item)
    compact_messages_html = "".join(
        f"<div class='item-summary-message'>{_esc(_expand_terms(message, sentence_case=True))}</div>"
        for message in compact_messages
    )
    interval_summary_html = _render_compact_interval_summary(item, interval) if interval is not None else ""
    series_graph_html = ""
    series_graph_keys: set[str] = set()
    if item.series_visuals:
        series_spec = next((spec for spec in item.series_visuals if isinstance(spec, dict)), None)
        if series_spec:
            series_keys = _visual_keys(series_spec)
            if series_keys:
                series_evidence = {key: evidence[key] for key in series_keys if key in evidence}
                if str(series_spec.get("kind") or "").strip().lower() == "fi_curve":
                    series_evidence["series_kind"] = "f-i curve"
                series_graph_html, series_graph_keys = _render_series_graph(item, series_evidence, spec=series_spec)
    numeric_companion_html = ""
    numeric_companion_keys: set[str] = set()
    if interval is None and not series_graph_html and item.companion_visuals:
        numeric_companion_html, numeric_companion_keys = _render_numeric_companions(
            item,
            evidence,
            visuals=item.companion_visuals,
        )
    warning_text = status_reason_text(item)
    notes_html = _notes_html(item)
    sections = [
        (
            f"<article class='item-card {_status_class(item.status)} item-collapsed' data-item-card data-status='{_esc(item.status)}' "
            f"data-detail-level='{_esc(item.detail_level or 'detail')}' data-search='{_esc(_item_search_blob(item))}'>"
        ),
        "<header class='item-header'>",
        (
            f"<button class='item-toggle' type='button' data-item-toggle aria-expanded='false' "
            f"aria-controls='{_esc(item_detail_body_id)}' aria-label='Expand {_esc(_expand_terms(item.title, sentence_case=True))}'>"
        ),
        "<span class='item-toggle-icon' aria-hidden='true'>&#9656;</span>",
        "<span class='item-header-main'>",
        (
            f"<span class='item-group-label'>{_esc(item.group_title or item.group_id)}</span>"
            if item.group_title or item.group_id else ""
        ),
        "<span class='item-header-row'>",
        f"<span class='item-title'>{_esc(_expand_terms(_clean_item_title_for_card(item), sentence_case=True))}</span>",
        _render_status_badge(item.status),
        "</span>",
        (f"<span class='item-summary-text'>{compact_messages_html}</span>" if compact_messages_html else ""),
        "</span>",
        "</button>",
        "</header>",
    ]
    if interval_summary_html:
        sections.extend(
            [
                "<div class='item-body-summary'>",
                interval_summary_html,
                "</div>",
            ]
        )
    if series_graph_html:
        sections.append(series_graph_html)
    if numeric_companion_html:
        sections.append(numeric_companion_html)
    if notes_html:
        sections.extend(notes_html)
    if warning_text:
        sections.extend(
            [
                "<div class='item-body-warning'>",
                "<div class='warning-summary-label'>Warning</div>",
                f"<div class='warning-summary-text'>{_esc(_expand_terms(warning_text, sentence_case=True))}</div>",
                "</div>",
            ]
        )
    sections.extend(
        [
            f"<div class='item-body item-detail-body' id='{_esc(item_detail_body_id)}' data-item-detail-body hidden>",
        f"<div class='item-block'><h4>Check id</h4><p class='check-id'>{_esc(item.check_id)}</p></div>",
        f"<div class='item-block'><h4>Criterion</h4><p>{_esc(_expand_terms(item.criterion, sentence_case=True))}</p></div>",
        f"<div class='item-block'><h4>Description</h4><p>{_esc(_expand_terms(item.description, sentence_case=True))}</p></div>",
        f"<div class='item-block'><h4>Acceptable result</h4><p>{_esc(_expand_terms(item.acceptable, sentence_case=True))}</p></div>",
        (
            "<div class='item-block'><h4>Decision basis</h4>"
            f"<p>{_esc(_expand_terms(item.acceptable_basis, sentence_case=True))}</p></div>"
        ),
        ]
    )
    sections.append(
        _render_evidence(
            item,
            exclude_keys=series_graph_keys | numeric_companion_keys,
            include_interval_visual=interval is None,
        )
    )
    sections.extend(["</div>", "</article>"])
    return "".join(section for section in sections if section)


def _render_group(group: dict[str, Any]) -> str:
    items_html = "\n".join(_render_item_card(item) for item in group["items"])
    return "".join(
        [
            f"<section id='group-{_esc(group['group_id'])}' class='group-section group-collapsed' data-group-section "
            f"data-group-id='{_esc(group['group_id'])}' data-worst-status='{_esc(group['worst_status'])}'>",
            "<header class='group-header'>",
            f"<div class='group-heading'><h2>{_esc(_expand_terms(group['title'], sentence_case=True))}</h2>",
            "</div>",
            "<div class='group-header-actions'>",
            f"<div class='summary-row'>{_render_summary(group['summary'])}</div>",
            "<button class='action-button group-toggle' type='button' data-group-toggle>Expand</button>",
            "</div>",
            "</header>",
            f"<div class='items-grid' data-group-items>{items_html}</div>",
            "</section>",
        ]
    )


def render_audit_dashboard_html(
    payload: dict[str, Any],
    *,
    refresh_endpoint: str | None = None,
) -> str:
    groups = list(payload.get("groups") or [])
    has_results = bool(groups)
    has_non_detail_items = any(
        str(item.get("detail_level") or "detail") != "detail"
        for group in groups
        for item in list(group.get("items") or [])
    )
    detail_toggle_html = ""
    if has_non_detail_items:
        detail_toggle_html = '<button class="toggle-button" type="button" id="show-detail-toggle" aria-pressed="true">Show detail items</button>'
    generated_at = datetime.now().isoformat(timespec="seconds")
    group_nav_items: list[str] = []
    for group in groups:
        status_class = _status_class(str(group.get("worst_status", "PASS")))
        group_nav_items.append(
            (
                f"<button type='button' class='group-link {status_class}' data-group-target='group-{_esc(group['group_id'])}'>"
                "<div class='group-link-main'>"
                f"<span class='group-link-label'>{_esc(_expand_terms(group['title'], sentence_case=True))}</span>"
                f"<span class='group-link-count'>{int(group.get('item_count', 0))} items</span>"
                "</div>"
                "<span class='group-link-meta'>"
                f"<span class='status-dot {_esc(status_class)}'></span>"
                f"{_render_status_badge(str(group['worst_status']))}"
                "</span>"
                "</button>"
            )
        )
    group_nav = "\n".join(group_nav_items)
    group_sections = "\n".join(_render_group(group) for group in groups)
    empty_message = (
        "No audit results yet. Use the audit runner in the Audits tab to start one."
        if not groups
        else "No audit items match the current filters."
    )
    control_strip_html = ""
    if has_results:
        control_strip_html = f"""
      <section class="control-strip">
        <div class="control-card">
          <h2>Display controls</h2>
          <div class="control-grid">
            <div class="control-field">
              <label for="audit-search">Search findings</label>
              <input id="audit-search" type="search" placeholder="Search titles, criteria, notes, and evidence">
            </div>
            <div class="control-field">
              <label>Filters</label>
              <div class="toggle-row">
                <button class="toggle-button" type="button" id="failures-only-toggle" aria-pressed="false">Failures and warnings only</button>
                <button class="toggle-button" type="button" id="hide-passed-groups-toggle" aria-pressed="false">Hide fully passing groups</button>
                {detail_toggle_html}
              </div>
            </div>
            <div class="control-field">
              <label>Group view</label>
              <div class="toggle-row">
                <button class="action-button" type="button" id="expand-all-groups">Expand all groups</button>
                <button class="action-button" type="button" id="collapse-all-groups">Collapse all groups</button>
              </div>
            </div>
          </div>
        </div>
      </section>
    """
    refresh_button = ""
    refresh_script = ""
    if refresh_endpoint and payload.get("items"):
        refresh_button = "<button id='refresh-audit-button' class='action-button' type='button'>Rerun current audit</button>"
        refresh_script = f"""
      const button = document.getElementById("refresh-audit-button");
      if (button) {{
        button.addEventListener("click", async () => {{
          button.disabled = true;
          button.textContent = "Running...";
          try {{
            const response = await fetch("{_esc(refresh_endpoint)}", {{ method: "POST", cache: "no-store" }});
            const payload = await response.json().catch(() => ({{}}));
            if (!response.ok || !payload.ok) {{
              throw new Error(String(payload.error || "Audit refresh failed"));
            }}
            window.location.reload();
          }} catch (error) {{
            console.warn("Audit refresh failed", error);
            button.disabled = false;
            button.textContent = "Rerun current audit";
          }}
        }});
      }}
"""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(payload['title'])}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --ink: #172033;
      --muted: #64748b;
      --line: #d8e1ef;
      --panel: #ffffff;
      --panel-alt: #f8fafd;
      --blue: #2563eb;
      --red: #dc2626;
      --amber: #d97706;
      --green: #16a34a;
      --shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
      --ui-font-stack: Verdana, sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 var(--ui-font-stack); overflow-anchor: none; }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(247, 248, 251, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 14px clamp(18px, 2vw, 30px) 12px;
      backdrop-filter: blur(8px);
    }}
    h1 {{ margin: 0 0 4px; font-size: 24px; font-weight: 700; }}
    main {{ width: 100%; max-width: none; margin: 0; padding: 24px clamp(18px, 2vw, 24px) 32px; }}
    .subtle {{ color: var(--muted); font-size: 13px; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 18px 0 20px;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      box-shadow: var(--shadow);
    }}
    .stat span {{ display: block; color: var(--muted); font-size: 12px; }}
    .stat strong {{ display: block; margin-top: 3px; font-size: 15px; overflow-wrap: anywhere; }}
    .summary-chip, .status-badge {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      flex: 0 0 auto;
      max-width: 100%;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 700;
      border: 1px solid transparent;
      white-space: nowrap;
      line-height: 1.2;
      font-variant-numeric: tabular-nums;
    }}
    .status-icon {{
      display: inline-block;
      line-height: 1;
      transform: translateY(-0.02em);
    }}
    .status-text {{
      line-height: 1;
    }}
    .summary-chip:focus-visible,
    .status-badge:focus-visible,
    .action-button:focus-visible,
    .toggle-button:focus-visible,
    .group-link:focus-visible,
    .item-toggle:focus-visible,
    .control-field input:focus-visible {{
      outline: 2px solid rgba(37, 99, 235, 0.45);
      outline-offset: 2px;
    }}
    .status-pass {{ color: var(--green); background: #ecfdf3; border-color: #a7f3d0; }}
    .status-warn {{ color: var(--amber); background: #fff7ed; border-color: #fed7aa; }}
    .status-fail {{ color: var(--red); background: #fef2f2; border-color: #fecaca; }}
    .actions {{
      display: flex;
      gap: 10px;
      align-items: center;
      margin-top: 10px;
      flex-wrap: wrap;
    }}
    .overall-status {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-left: 2px;
      padding: 4px 10px;
      border: 1px solid #dbe3ef;
      border-radius: 999px;
      background: #f8fbff;
    }}
    .overall-status-label {{
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }}
    .overall-status .status-badge {{
      padding: 4px 10px;
      font-size: 13px;
    }}
    .action-button {{
      appearance: none;
      border: 1px solid #dbe3ef;
      background: #ffffff;
      color: #334155;
      border-radius: 8px;
      height: 44px;
      padding: 0 16px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    .action-button:hover {{ background: #f8fbff; border-color: #b8c9df; color: #1d4ed8; }}
    .control-strip {{
      display: block;
      margin-bottom: 16px;
    }}
    .control-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 20px;
      overflow: visible;
    }}
    .control-card h2 {{
      margin: 0 0 10px;
      font-size: 16px;
    }}
    .control-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr) minmax(0, 0.9fr);
      gap: 16px;
      align-items: start;
    }}
    .control-field {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .control-field label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }}
    .control-field input {{
      appearance: none;
      width: 100%;
      border: 1px solid #dbe3ef;
      border-radius: 8px;
      height: 44px;
      padding: 0 12px;
      font: inherit;
      background: #fff;
      color: var(--ink);
    }}
    .toggle-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }}
    .toggle-button {{
      appearance: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      height: 36px;
      padding: 0 12px;
      border: 1px solid #d5deeb;
      border-radius: 999px;
      background: #f9fbff;
      font-size: 12px;
      font-weight: 600;
      color: #334155;
      cursor: pointer;
      transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
    }}
    .toggle-button:hover {{
      background: #eef4ff;
      border-color: #aabdda;
      color: #1e3a5f;
    }}
    .toggle-button[aria-pressed="true"] {{
      background: #eaf2ff;
      border-color: #87a7d8;
      color: #1d4f9b;
      box-shadow: inset 0 0 0 1px rgba(29, 79, 155, 0.08);
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(250px, 320px) minmax(0, 1fr);
      gap: 20px;
      position: relative;
      z-index: 1;
      align-items: start;
      overflow-anchor: none;
    }}
    .sidebar, .group-section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .sidebar {{
      position: sticky;
      top: 126px;
      z-index: 1;
      align-self: start;
      padding: 20px;
    }}
    .sidebar h2 {{ margin: 0 0 10px; font-size: 16px; }}
    .group-links {{
      display: flex;
      flex-direction: column;
      gap: 10px;
      max-height: calc(100vh - 160px);
      overflow-y: auto;
      overscroll-behavior: contain;
      padding-right: 4px;
    }}
    .group-link {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      padding: 10px 12px;
      border: 1px solid #e6eaf1;
      border-left-width: 4px;
      border-left-style: solid;
      border-radius: 8px;
      color: inherit;
      background: #fbfcfe;
      appearance: none;
      width: 100%;
      text-align: left;
      cursor: pointer;
    }}
    .group-link:hover {{
      border-color: #c7d4e6;
      background: #f7faff;
    }}
    .group-link.status-pass {{ border-left-color: rgba(22, 163, 74, 0.56); }}
    .group-link.status-warn {{ border-left-color: rgba(217, 119, 6, 0.68); }}
    .group-link.status-fail {{ border-left-color: rgba(220, 38, 38, 0.78); }}
    .group-link-main {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      min-width: 0;
    }}
    .group-link-label {{
      min-width: 0;
      overflow: hidden;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      font-weight: 700;
      line-height: 1.3;
    }}
    .group-link-count {{
      color: var(--muted);
      font-size: 11px;
      font-weight: 600;
      font-variant-numeric: tabular-nums;
    }}
    .group-link-meta {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .group-link small {{
      color: var(--muted);
      font-size: 11px;
      font-weight: 600;
      white-space: nowrap;
    }}
    .content {{ display: flex; flex-direction: column; gap: 16px; min-width: 0; }}
    .content {{
      overflow-anchor: none;
    }}
    .group-section {{
      scroll-margin-top: 24px;
    }}
    .group-section.status-pass {{ border-left: 4px solid rgba(22, 163, 74, 0.22); }}
    .group-section.status-warn {{ border-left: 4px solid rgba(217, 119, 6, 0.30); }}
    .group-section.status-fail {{ border-left: 4px solid rgba(220, 38, 38, 0.34); }}
    .group-header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
      scroll-margin-top: 126px;
    }}
    .group-heading {{
      min-width: 0;
      flex: 1 1 auto;
    }}
    .group-header h2 {{ margin: 0; font-size: 16px; font-weight: 700; }}
    .group-header-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      justify-content: flex-end;
      min-width: 0;
      flex: 0 1 auto;
    }}
    .summary-row {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }}
    .items-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      align-items: start;
      gap: 16px;
      padding: 16px;
    }}
    .group-section.group-collapsed .items-grid {{
      display: none;
    }}
    .group-section {{
      position: relative;
      z-index: 1;
      overflow: visible;
    }}
    .item-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      align-self: start;
      overflow: hidden;
      box-shadow: var(--shadow);
      border-left: 4px solid transparent;
    }}
    .item-card.status-fail {{ border-left-color: var(--red); }}
    .item-card.status-warn {{ border-left-color: var(--amber); }}
    .item-card.status-pass {{ border-left-color: rgba(22, 163, 74, 0.32); }}
    .item-header {{
      display: block;
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
    }}
    .item-toggle {{
      appearance: none;
      display: flex;
      align-items: flex-start;
      gap: 12px;
      width: 100%;
      padding: 16px 18px;
      border: 0;
      background: transparent;
      text-align: left;
      cursor: pointer;
    }}
    .item-toggle:hover {{
      background: #f8fbff;
    }}
    .item-toggle-icon {{
      flex: 0 0 auto;
      width: 18px;
      color: #475569;
      font-size: 13px;
      line-height: 1.4;
      transform-origin: 50% 45%;
      transition: transform 140ms ease;
    }}
    .item-card:not(.item-collapsed) .item-toggle-icon {{
      transform: rotate(90deg);
    }}
    .item-header-main {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      min-width: 0;
      flex: 1 1 auto;
    }}
    .item-group-label {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.02em;
      line-height: 1.2;
      opacity: 0.85;
    }}
    .item-header-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-width: 0;
    }}
    .item-title {{
      min-width: 0;
      font-size: 15px;
      font-weight: 700;
      line-height: 1.3;
      overflow-wrap: anywhere;
    }}
    .item-summary-text {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      color: #475569;
      font-size: 12px;
      line-height: 1.45;
    }}
    .item-summary-message {{
      overflow-wrap: anywhere;
    }}
    .item-body-summary {{
      padding: clamp(12px, 1vw, 16px) clamp(16px, 3vw, 60px);
      background: #ffffff;
    }}
    .item-body-notes {{
      margin: 12px 16px 12px;
      padding: 12px 14px;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      background: #f8fafc;
      color: #334155;
    }}
    .item-notes-label {{
      color: #475569;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.02em;
      margin-bottom: 6px;
    }}
    .item-notes-list {{
      margin: 0;
      padding-left: 18px;
    }}
    .item-note {{
      font-size: 12px;
      line-height: 1.45;
      padding-left: 2px;
      overflow-wrap: anywhere;
    }}
    .item-note + .item-note {{
      margin-top: 4px;
    }}
    .item-note::marker {{
      color: #64748b;
    }}
    .item-body-warning {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin: 12px 16px 12px;
      padding: 12px 14px;
      border: 1px solid #f3d8a2;
      border-left: 4px solid #f59e0b;
      border-radius: 8px;
      background: #fffbeb;
    }}
    .warning-summary-label {{
      color: #92400e;
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }}
    .warning-summary-text {{
      color: #334155;
      font-size: 12px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }}
    .check-id {{ margin: 0; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }}
    .item-body {{ padding: 18px 18px 20px; display: flex; flex-direction: column; gap: 12px; }}
    .item-detail-body[hidden] {{
      display: none !important;
    }}
    .item-block h4 {{ margin: 0 0 4px; font-size: 12px; text-transform: uppercase; color: var(--muted); }}
    .item-block p {{ margin: 0; }}
    .interval-block {{
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      padding: 14px 14px 12px;
      background: #f8fbff;
    }}
    .interval-metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .interval-metric {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      padding: 10px;
      border-radius: 8px;
      border: 1px solid #dde6f3;
      background: rgba(255, 255, 255, 0.92);
    }}
    .interval-metric span {{
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }}
    .interval-metric strong {{
      font-size: 13px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    .interval-visual {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .interval-range-labels {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 11px;
    }}
    .interval-track {{
      position: relative;
      height: 18px;
      border-radius: 999px;
      background: linear-gradient(180deg, #eef3fb, #d9e3f0);
      overflow: visible;
      border: 1px solid #d3ddeb;
    }}
    .interval-band {{
      position: absolute;
      top: 2px;
      bottom: 2px;
      border-radius: 999px;
      background: rgba(37, 99, 235, 0.22);
      border: 1px solid rgba(37, 99, 235, 0.35);
    }}
    .interval-tick {{
      position: absolute;
      top: -4px;
      width: 2px;
      height: 24px;
      transform: translateX(-50%);
      border-radius: 999px;
      background: #475569;
      box-shadow: 0 0 0 1px rgba(255,255,255,0.8);
    }}
    .interval-tick.interval-reference {{
      width: 3px;
      background: #334155;
    }}
    .interval-marker {{
      position: absolute;
      top: 50%;
      width: 14px;
      height: 14px;
      transform: translate(-50%, -50%);
      border-radius: 999px;
      border: 2px solid #ffffff;
      box-shadow: 0 0 0 2px #ffffff, 0 0 0 1px rgba(15, 23, 42, 0.12);
      background: var(--blue);
    }}
    .interval-marker-label {{
      position: absolute;
      top: -23px;
      transform: translateX(-50%);
      padding: 1px 6px;
      border-radius: 999px;
      border: 1px solid #dbe3ef;
      background: #ffffff;
      color: #475569;
      font-size: 10px;
      font-weight: 700;
      line-height: 1.2;
      white-space: nowrap;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
      pointer-events: none;
    }}
    .interval-marker-label.observed-label {{
      border-color: #c9d5e5;
      color: #1f2937;
    }}
    .interval-marker.status-pass {{ background: var(--green); }}
    .interval-marker.status-warn {{ background: var(--amber); }}
    .interval-marker.status-fail {{ background: var(--red); }}
    .interval-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px 16px;
      color: var(--muted);
      font-size: 11px;
    }}
    .interval-legend span {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .item-compact-interval {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin: 14px 18px 12px;
    }}
    .legend-swatch {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: #cbd5e1;
      border: 1px solid #94a3b8;
    }}
    .legend-swatch.accepted {{
      background: rgba(37, 99, 235, 0.18);
      border-color: rgba(37, 99, 235, 0.28);
    }}
    .legend-swatch.reference {{
      width: 2px;
      height: 12px;
      border-radius: 999px;
      background: #475569;
      border-color: #475569;
    }}
    .legend-swatch.observed.status-pass {{ background: var(--green); border-color: var(--green); }}
    .legend-swatch.observed.status-warn {{ background: var(--amber); border-color: var(--amber); }}
    .legend-swatch.observed.status-fail {{ background: var(--red); border-color: var(--red); }}
    .series-graph-block,
    .numeric-strip-block,
    .numeric-sparkline-block {{
      margin: 14px 18px 12px;
      border: 1px solid #dbe4f0;
      border-radius: 10px;
      padding: 12px;
      background: #fbfdff;
    }}
    .series-graph-meta,
    .numeric-strip-meta,
    .numeric-sparkline-meta {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
      margin-bottom: 8px;
    }}
    .series-graph-shell,
    .numeric-strip-shell,
    .numeric-sparkline-shell {{
      border: 1px solid #d9e2ee;
      border-radius: 10px;
      background: linear-gradient(180deg, #eef3fb 0%, #e6edf7 100%);
      overflow: hidden;
    }}
    .series-graph-svg,
    .numeric-strip-svg,
    .numeric-sparkline-svg {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .series-graph-bg,
    .numeric-strip-bg,
    .numeric-sparkline-bg {{
      fill: transparent;
    }}
    .series-axis,
    .numeric-strip-axis,
    .numeric-sparkline-axis {{
      stroke: #cbd5e1;
      stroke-width: 1;
    }}
    .series-axis-tick,
    .numeric-strip-tick,
    .numeric-sparkline-tick {{
      stroke: #cbd5e1;
      stroke-width: 1;
    }}
    .series-zero,
    .numeric-strip-zero,
    .numeric-sparkline-zero {{
      stroke: #94a3b8;
      stroke-width: 1.5;
      stroke-dasharray: 4 4;
    }}
    .series-line,
    .numeric-sparkline-line {{
      fill: none;
      stroke-width: 2.5;
    }}
    .series-point,
    .numeric-dot,
    .numeric-sparkline-point {{
      stroke-width: 1.5;
      fill: #ffffff;
    }}
    .series-tick-label,
    .numeric-strip-label,
    .numeric-sparkline-label {{
      fill: #64748b;
      font-size: 10px;
      font-weight: 600;
    }}
    .series-legend,
    .numeric-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 11px;
    }}
    .series-legend-item,
    .numeric-legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .series-legend-swatch,
    .numeric-legend-swatch {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: #cbd5e1;
      border: 1px solid #94a3b8;
    }}
    .evidence-grid {{
      margin: 0;
      display: grid;
      gap: 8px;
    }}
    .evidence-row {{
      display: grid;
      grid-template-columns: minmax(110px, 160px) minmax(0, 1fr);
      gap: 10px;
      padding: 8px 10px;
      border: 1px solid #e6eaf1;
      border-radius: 8px;
      background: #fbfcfe;
    }}
    .evidence-row dt {{
      margin: 0;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    .evidence-row dd {{
      margin: 0;
      color: #243247;
      overflow-wrap: anywhere;
    }}
    .empty-state {{
      display: block;
      padding: 24px;
      border: 1px dashed #dbe3ef;
      border-radius: 8px;
      background: #fbfcfe;
      color: var(--muted);
      text-align: center;
    }}
    .results-meta {{
      color: var(--muted);
      font-size: 12px;
    }}
    @media (max-width: 980px) {{
      header {{ padding: 14px 16px; }}
      main {{ padding: 16px; }}
      .control-grid {{
        grid-template-columns: 1fr;
      }}
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{ position: static; }}
      .group-header-actions {{
        width: 100%;
        justify-content: flex-start;
      }}
      .summary-row {{
        justify-content: flex-start;
      }}
      .group-links {{
        max-height: none;
        overflow: visible;
        padding-right: 0;
      }}
      .evidence-row {{
        grid-template-columns: 1fr;
      }}
      .items-grid {{
        grid-template-columns: 1fr;
      }}
      .item-header-row {{
        align-items: flex-start;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{_esc(payload['title'])}</h1>
    <div class="subtle">audit_id={_esc(payload['audit_id'])} | generated {generated_at}</div>
    <div class="actions">
      {_render_summary(payload['summary'])}
      <span class="overall-status"><span class="overall-status-label">Overall</span>{_render_status_badge(str(payload['worst_status']))}</span>
      {refresh_button}
      <span class="results-meta" id="results-meta">Loading visible-item summary...</span>
    </div>
  </header>
  <main>
    {control_strip_html}
    <div class="layout">
      <aside class="sidebar">
        <h2>Audit groups</h2>
        <div class="group-links">
          {group_nav}
        </div>
      </aside>
      <div class="content">
        {group_sections}
        <div class="empty-state" id="audit-empty-state">
          <p class="empty-state-title" id="audit-empty-state-title">{_esc("No audit results yet")}</p>
          <p class="empty-state-subtitle" id="audit-empty-state-subtitle">{_esc("Run an audit to populate findings, groups, filters, and evidence.")}</p>
          <button class="action-button" type="button" id="audit-empty-state-run">Run selected audit</button>
        </div>
      </div>
    </div>
  </main>
  <script>
    (() => {{
{refresh_script}
      const searchInput = document.getElementById("audit-search");
      const failuresOnlyToggle = document.getElementById("failures-only-toggle");
      const hidePassedGroupsToggle = document.getElementById("hide-passed-groups-toggle");
      const showDetailToggle = document.getElementById("show-detail-toggle");
      const resultsMeta = document.getElementById("results-meta");
      const emptyState = document.getElementById("audit-empty-state");
      const emptyStateTitle = document.getElementById("audit-empty-state-title");
      const emptyStateSubtitle = document.getElementById("audit-empty-state-subtitle");
      const emptyStateDefaultText = {json.dumps(empty_message)};
      const groupSections = Array.from(document.querySelectorAll("[data-group-section]"));
      const itemCards = Array.from(document.querySelectorAll("[data-item-card]"));
      const emptyStateRunButton = document.getElementById("audit-empty-state-run");

      function togglePressed(button) {{
        if (!button) return false;
        const next = String(button.getAttribute("aria-pressed") || "false") !== "true";
        button.setAttribute("aria-pressed", next ? "true" : "false");
        return next;
      }}

      function isPressed(button) {{
        return String(button?.getAttribute("aria-pressed") || "false") === "true";
      }}

      function toggleGroup(section, collapse) {{
        if (!section) return;
        section.classList.toggle("group-collapsed", Boolean(collapse));
        const button = section.querySelector("[data-group-toggle]");
        if (button) {{
          button.textContent = collapse ? "Expand" : "Collapse";
        }}
      }}

      function toggleItem(item, collapse) {{
        if (!item) return;
        item.classList.toggle("item-collapsed", Boolean(collapse));
        const button = item.querySelector("[data-item-toggle]");
        const body = item.querySelector("[data-item-detail-body]");
        if (button) {{
          button.setAttribute("aria-expanded", collapse ? "false" : "true");
          button.setAttribute("aria-label", collapse ? "Expand item" : "Collapse item");
        }}
        if (body) {{
          body.hidden = Boolean(collapse);
        }}
      }}

      function applyFilters() {{
        const query = String(searchInput?.value || "").trim().toLowerCase();
        const failuresOnly = isPressed(failuresOnlyToggle);
        const hidePassedGroups = isPressed(hidePassedGroupsToggle);
        const showDetail = !showDetailToggle || isPressed(showDetailToggle);
        let visibleItems = 0;
        let visibleGroups = 0;

        groupSections.forEach((section) => {{
          const groupWorstStatus = String(section.dataset.worstStatus || "PASS");
          const items = Array.from(section.querySelectorAll("[data-item-card]"));
          let groupVisibleCount = 0;
          items.forEach((item) => {{
            const status = String(item.dataset.status || "PASS");
            const detailLevel = String(item.dataset.detailLevel || "detail");
            const searchBlob = String(item.dataset.search || "");
            const matchesQuery = !query || searchBlob.includes(query);
            const matchesStatus = !failuresOnly || status !== "PASS";
            const matchesDetail = showDetail || detailLevel !== "detail";
            const visible = matchesQuery && matchesStatus && matchesDetail;
            item.hidden = !visible;
            if (visible) {{
              groupVisibleCount += 1;
              visibleItems += 1;
            }}
          }});
          const hideForPassedGroup = hidePassedGroups && groupWorstStatus === "PASS";
          const groupVisible = groupVisibleCount > 0 && !hideForPassedGroup;
          section.hidden = !groupVisible;
          if (groupVisible) {{
            visibleGroups += 1;
          }}
        }});

        if (emptyState) {{
          emptyState.style.display = visibleItems > 0 ? "none" : "block";
          if (visibleItems === 0) {{
            if (emptyStateTitle) {{
              emptyStateTitle.textContent = groupSections.length === 0 ? "No audit results yet" : "No audit items match the current filters.";
            }}
            if (emptyStateSubtitle) {{
              emptyStateSubtitle.textContent = groupSections.length === 0
                ? emptyStateDefaultText
                : "Adjust the filters, or rerun the selected audit if you need a fresh result set.";
            }}
          }}
        }}
        if (resultsMeta) {{
          if (groupSections.length === 0) {{
            resultsMeta.hidden = true;
          }} else {{
            resultsMeta.hidden = false;
            resultsMeta.textContent = `${{visibleItems}} visible items across ${{visibleGroups}} visible groups`;
          }}
        }}
      }}

      document.querySelectorAll("[data-group-toggle]").forEach((button) => {{
        button.addEventListener("click", () => {{
          const section = button.closest("[data-group-section]");
          toggleGroup(section, !section?.classList.contains("group-collapsed"));
        }});
      }});
      emptyStateRunButton?.addEventListener("click", () => {{
        window.parent?.postMessage({{
          type: "control-center-run-audit",
          source: "audit-dashboard-empty-state",
        }}, "*");
      }});
      document.querySelectorAll(".group-link").forEach((link) => {{
        link.addEventListener("click", () => {{
          const target = document.getElementById(String(link.dataset.groupTarget || ""));
          if (!target) {{
            return;
          }}
          if (typeof link.blur === "function") {{
            link.blur();
          }}
          if (target.classList.contains("group-collapsed")) {{
            toggleGroup(target, false);
          }}
          const alignGroupHeader = () => {{
            const header = target.querySelector(".group-header") || target;
            const shellHeader = document.querySelector("body > header");
            const shellHeaderBottom = shellHeader ? shellHeader.getBoundingClientRect().bottom : 0;
            const nextScrollTop = Math.max(0, window.scrollY + header.getBoundingClientRect().top - shellHeaderBottom);
            window.scrollTo({{ top: nextScrollTop, behavior: "auto" }});
          }};
          alignGroupHeader();
          window.setTimeout(alignGroupHeader, 120);
          window.setTimeout(alignGroupHeader, 500);
        }});
      }});
      document.querySelectorAll("[data-item-toggle]").forEach((button) => {{
        button.addEventListener("click", () => {{
          const item = button.closest("[data-item-card]");
          toggleItem(item, !item?.classList.contains("item-collapsed"));
        }});
      }});
      document.getElementById("expand-all-groups")?.addEventListener("click", () => {{
        groupSections.forEach((section) => toggleGroup(section, false));
        itemCards.forEach((item) => toggleItem(item, false));
      }});
      document.getElementById("collapse-all-groups")?.addEventListener("click", () => {{
        groupSections.forEach((section) => toggleGroup(section, true));
        itemCards.forEach((item) => toggleItem(item, true));
      }});
      [failuresOnlyToggle, hidePassedGroupsToggle, showDetailToggle].forEach((button) => {{
        button?.addEventListener("click", () => {{
          togglePressed(button);
          applyFilters();
        }});
      }});
      searchInput?.addEventListener("input", applyFilters);
      searchInput?.addEventListener("change", applyFilters);
      groupSections.forEach((section) => toggleGroup(section, true));
      itemCards.forEach((item) => toggleItem(item, true));
      applyFilters();
    }})();
  </script>
</body>
</html>
"""


def export_audit_dashboard(
    report: AuditReport,
    output_dir: str | Path,
    *,
    refresh_endpoint: str | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    report_path = output_path / "report.json"
    index_path = output_path / "index.html"
    manifest_path = output_path / "manifest.json"

    report_tmp = report_path.with_name(f".{report_path.name}.tmp")
    report_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(report_tmp, report_path)

    html_text = render_audit_dashboard_html(payload, refresh_endpoint=refresh_endpoint)
    index_tmp = index_path.with_name(f".{index_path.name}.tmp")
    index_tmp.write_text(html_text)
    os.replace(index_tmp, index_path)

    manifest = {
        "audit_id": report.audit_id,
        "title": report.title,
        "output_dir": str(output_path),
        "index_html": str(index_path),
        "report_json": str(report_path),
        "summary": report.summary,
        "worst_status": report.worst_status,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "manifest_revision": time.time_ns(),
        "group_count": len(payload.get("groups") or []),
        "item_count": len(payload.get("items") or []),
    }
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(manifest_tmp, manifest_path)
    return manifest


def build_report_dashboard(
    audit_id: str,
    *,
    audit_args: list[str],
    output_dir: str | Path,
    refresh_endpoint: str | None = None,
) -> dict[str, Any]:
    report = run_audit_by_id(audit_id, audit_args)
    return export_audit_dashboard(report, output_dir, refresh_endpoint=refresh_endpoint)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_id", nargs="?", default="new_sweep", help="Audit id to render.")
    parser.add_argument("--output-dir", required=True, help="Directory to receive index.html and report.json.")
    parser.add_argument(
        "--refresh-endpoint",
        default="",
        help="Optional POST endpoint that refreshes the rendered audit report.",
    )
    args, extra_args = parser.parse_known_args(argv)
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    build_report_dashboard(
        str(args.audit_id),
        audit_args=extra_args,
        output_dir=args.output_dir,
        refresh_endpoint=str(args.refresh_endpoint or "") or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
