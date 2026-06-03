"""Render audit reports as a maintained HTML dashboard."""

from __future__ import annotations

import argparse
import html
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
    return f"<span class='status-badge {_status_class(status)}'>{_esc(status)}</span>"


def _render_summary(summary: dict[str, int]) -> str:
    return "".join(
        f"<span class='summary-chip {_status_class(status)}'>{_esc(status)}={int(summary.get(status, 0))}</span>"
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
    </div>
    <div class='interval-legend'>
      <span><i class='legend-swatch accepted'></i>accepted range</span>
      <span><i class='legend-swatch reference'></i>reference mean</span>
      <span><i class='legend-swatch observed {_status_class(item.status)}'></i>observed</span>
    </div>
  </div>
</div>
"""


def _extract_numeric_profile_data(evidence: dict[str, Any], *, exclude_keys: set[str] | None = None) -> dict[str, Any] | None:
    exclude = set(exclude_keys or set())
    entries: list[tuple[str, float]] = []
    for key, value in evidence.items():
        if key in exclude or key == "__reference_annotations__":
            continue
        numeric_value = _float_or_none(value)
        if numeric_value is not None:
            entries.append((key, numeric_value))
    if not entries:
        return None

    values = [value for _, value in entries]
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

    def _position(value: float) -> float:
        width = domain_high - domain_low
        if width <= 0.0:
            return 50.0
        return max(0.0, min(100.0, ((value - domain_low) / width) * 100.0))

    zero_position = _position(0.0)
    rows: list[dict[str, Any]] = []
    for key, value in entries:
        value_position = _position(value)
        left = min(zero_position, value_position)
        width = max(0.0, abs(value_position - zero_position))
        rows.append(
            {
                "key": key,
                "label": _evidence_label(key),
                "value": value,
                "value_text": _format_numeric(value),
                "left": left,
                "width": width,
            }
        )

    return {
        "rows": rows,
        "domain_low": domain_low,
        "domain_high": domain_high,
        "domain_low_text": _format_numeric(domain_low),
        "domain_high_text": _format_numeric(domain_high),
        "zero_position": zero_position,
        "numeric_keys": [key for key, _ in entries],
    }


def _render_numeric_profile(item: AuditItem, evidence: dict[str, Any], *, exclude_keys: set[str] | None = None) -> tuple[str, set[str]]:
    profile = _extract_numeric_profile_data(evidence, exclude_keys=exclude_keys)
    if profile is None:
        return "", set()
    rows = profile["rows"]
    row_count = len(rows)
    status_class = _status_class(item.status)
    rows_html = "".join(
        f"<div class='numeric-profile-row'>"
        f"<div class='numeric-profile-label'>{_esc(row['label'])}</div>"
        f"<div class='numeric-profile-track'>"
        f"<div class='numeric-profile-zero' style='left:{profile['zero_position']:.2f}%;'></div>"
        f"<div class='numeric-profile-bar {status_class}' style='left:{row['left']:.2f}%; width:{row['width']:.2f}%;'></div>"
        "</div>"
        f"<div class='numeric-profile-value'>{_esc(row['value_text'])}</div>"
        "</div>"
        for row in rows
    )
    aria_label = (
        f"Numeric profile with {row_count} values on a shared scale from "
        f"{profile['domain_low_text']} to {profile['domain_high_text']}."
    )
    return (
        "<div class='item-block numeric-profile-block'>"
        "<h4>Numeric profile</h4>"
        f"<div class='numeric-profile-scale'><span>{_esc(profile['domain_low_text'])}</span><span>{_esc(profile['domain_high_text'])}</span></div>"
        f"<div class='numeric-profile-graph' data-evidence-graph role='img' aria-label='{_esc(aria_label)}'>"
        f"{rows_html}"
        "</div>"
        "</div>"
    ), set(profile["numeric_keys"])


def _render_compact_interval_summary(item: AuditItem, interval: dict[str, Any]) -> str:
    unit = str(interval["reference_unit"])
    observed_text = _format_numeric(interval["observed_value"], unit=unit)
    reference_text = _format_numeric(interval["reference_mean"], unit=unit)
    low_text = _format_numeric(interval["accepted_low"], unit=unit)
    high_text = _format_numeric(interval["accepted_high"], unit=unit)
    positions = interval["positions"]
    band_left = min(float(positions["accepted_low"]), float(positions["accepted_high"]))
    band_width = max(0.0, abs(float(positions["accepted_high"]) - float(positions["accepted_low"])))
    reference_tick_html = ""
    if positions["reference_mean"] is not None:
        reference_tick_html = (
            f"<div class='compact-interval-tick' style='left:{float(positions['reference_mean']):.2f}%'></div>"
        )
    return f"""
<div class='item-compact-interval' data-compact-interval>
  <div class='compact-interval-track'>
    <div class='compact-interval-band' style='left:{band_left:.2f}%; width:{band_width:.2f}%;'></div>
    {reference_tick_html}
    <div class='compact-interval-marker {_status_class(item.status)}' style='left:{float(positions["observed_value"] or 0.0):.2f}%'></div>
  </div>
  <div class='compact-interval-meta'>
    <span>observed {_esc(observed_text)}</span>
    <span>reference {_esc(reference_text)}</span>
    <span>range {_esc(low_text)} to {_esc(high_text)}</span>
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


def _render_evidence(item: AuditItem) -> str:
    evidence = dict(item.evidence or {})
    if not evidence:
        return ""
    interval = _extract_interval_visual_data(item)
    rendered_sections: list[str] = []
    exclude_keys: set[str] = set()
    if interval is not None:
        rendered_sections.append(_render_interval_visual(item, interval))
        exclude_keys.update(_INTERVAL_RESERVED_EVIDENCE_KEYS)
        observed_key = str(interval.get("observed_key", "")).strip()
        if observed_key:
            exclude_keys.add(observed_key)
    else:
        numeric_profile_html, numeric_keys = _render_numeric_profile(item, evidence, exclude_keys=exclude_keys)
        if numeric_profile_html:
            rendered_sections.append(numeric_profile_html)
            exclude_keys.update(numeric_keys)
    rendered_sections.append(_render_structured_evidence(evidence, exclude_keys=exclude_keys))
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


def _render_item_card(item_payload: dict[str, Any]) -> str:
    item = AuditItem(**item_payload)
    interval = _extract_interval_visual_data(item)
    card_id = _safe_dom_id(item.check_id)
    item_detail_body_id = f"item-detail-body-{card_id}"
    compact_messages = _collapsed_summary_messages(item)
    compact_messages_html = "".join(
        f"<div class='item-summary-message'>{_esc(_expand_terms(message, sentence_case=True))}</div>"
        for message in compact_messages
    )
    compact_interval_html = _render_compact_interval_summary(item, interval) if interval is not None else ""
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
        "<span class='item-header-row'>",
        f"<span class='item-title'>{_esc(_expand_terms(item.title, sentence_case=True))}</span>",
        _render_status_badge(item.status),
        "</span>",
        (f"<span class='item-summary-text'>{compact_messages_html}</span>" if compact_messages_html else ""),
        "</span>",
        "</button>",
        "</header>",
    ]
    if compact_interval_html:
        sections.extend(
            [
                "<div class='item-body-summary'>",
                compact_interval_html,
                "</div>",
            ]
        )
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
    sections.append(_render_evidence(item))
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
    has_non_detail_items = any(
        str(item.get("detail_level") or "detail") != "detail"
        for group in groups
        for item in list(group.get("items") or [])
    )
    generated_at = datetime.now().isoformat(timespec="seconds")
    group_nav = "\n".join(
        (
            f"<button type='button' class='group-link' data-group-target='group-{_esc(group['group_id'])}'>"
            f"<span class='group-link-label'>{_esc(_expand_terms(group['title'], sentence_case=True))}</span>"
            "<span class='group-link-meta'>"
            f"<small>{int(group.get('item_count', 0))} items</small>"
            f"{_render_status_badge(str(group['worst_status']))}"
            "</span>"
            "</button>"
        )
        for group in groups
    )
    group_sections = "\n".join(_render_group(group) for group in groups)
    empty_message = (
        "No audit results yet. Use the audit runner in the Audits tab to start one."
        if not groups
        else "No audit items match the current filters."
    )
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
    detail_toggle_html = ""
    if has_non_detail_items:
        detail_toggle_html = '<button class="toggle-button" type="button" id="show-detail-toggle" aria-pressed="true">Show detail items</button>'
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(payload['title'])}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fb;
      --ink: #17202a;
      --muted: #667085;
      --line: #d9dee8;
      --panel: #ffffff;
      --blue: #2563eb;
      --red: #dc2626;
      --amber: #d97706;
      --green: #15803d;
      --shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; overflow-anchor: none; }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(247, 248, 251, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 14px clamp(18px, 2vw, 30px) 12px;
      backdrop-filter: blur(8px);
    }}
    h1 {{ margin: 0 0 4px; font-size: 18px; }}
    main {{ width: 100%; max-width: none; margin: 0; padding: 18px clamp(18px, 2vw, 30px) 40px; }}
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
      flex: 0 0 auto;
      max-width: 100%;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 700;
      border: 1px solid transparent;
      white-space: nowrap;
      line-height: 1.2;
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
    .action-button {{
      appearance: none;
      border: 1px solid #dbe3ef;
      background: #ffffff;
      color: #334155;
      border-radius: 8px;
      padding: 8px 12px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    .action-button:hover {{ background: #eff6ff; border-color: #93c5fd; color: #1d4ed8; }}
    .control-strip {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .control-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 12px;
      overflow: visible;
    }}
    .control-card h2 {{
      margin: 0 0 10px;
      font-size: 14px;
    }}
    .control-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
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
      padding: 8px 10px;
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
      padding: 7px 12px;
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
      padding: 16px;
    }}
    .sidebar h2 {{ margin: 0 0 10px; font-size: 16px; }}
    .group-links {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: calc(100vh - 160px);
      overflow-y: auto;
      overscroll-behavior: contain;
      padding-right: 4px;
    }}
    .group-link {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      padding: 8px 10px;
      border: 1px solid #e6eaf1;
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
    .group-link-label {{
      min-width: 0;
      overflow-wrap: anywhere;
      font-weight: 700;
      line-height: 1.25;
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
    .group-header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      padding: 13px 16px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
      scroll-margin-top: 126px;
    }}
    .group-heading {{
      min-width: 0;
      flex: 1 1 auto;
    }}
    .group-header h2 {{ margin: 0; font-size: 16px; }}
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
    }}
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
      padding: 14px 16px;
      border: 0;
      background: transparent;
      text-align: left;
      cursor: pointer;
    }}
    .item-toggle:hover {{
      background: rgba(239, 246, 255, 0.55);
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
      padding: 10px 12px;
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
      padding: 10px 12px;
      border: 1px solid #f3d8a2;
      border-left: 4px solid #f59e0b;
      border-radius: 8px;
      background: #fffbeb;
    }}
    .warning-summary-label {{
      display: none;
    }}
    .warning-summary-text {{
      color: #334155;
      font-size: 12px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }}
    .check-id {{ margin: 0; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }}
    .item-body {{ padding: 16px; display: flex; flex-direction: column; gap: 12px; }}
    .item-detail-body[hidden] {{
      display: none !important;
    }}
    .item-block h4 {{ margin: 0 0 4px; font-size: 12px; text-transform: uppercase; color: var(--muted); }}
    .item-block p {{ margin: 0; }}
    .interval-block {{
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      padding: 12px;
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
      height: 16px;
      border-radius: 999px;
      background: linear-gradient(180deg, #edf2fa, #dbe5f4);
      overflow: visible;
      border: 1px solid #d3ddeb;
    }}
    .interval-band {{
      position: absolute;
      top: 2px;
      bottom: 2px;
      border-radius: 999px;
      background: rgba(37, 99, 235, 0.18);
      border: 1px solid rgba(37, 99, 235, 0.28);
    }}
    .interval-tick {{
      position: absolute;
      top: -3px;
      width: 2px;
      height: 22px;
      transform: translateX(-50%);
      border-radius: 999px;
      background: #475569;
      box-shadow: 0 0 0 1px rgba(255,255,255,0.8);
    }}
    .interval-marker {{
      position: absolute;
      top: 50%;
      width: 12px;
      height: 12px;
      transform: translate(-50%, -50%);
      border-radius: 999px;
      border: 2px solid #ffffff;
      box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.14);
      background: var(--blue);
    }}
    .interval-marker.status-pass {{ background: var(--green); }}
    .interval-marker.status-warn {{ background: var(--amber); }}
    .interval-marker.status-fail {{ background: var(--red); }}
    .interval-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
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
      gap: 6px;
    }}
    .compact-interval-track {{
      position: relative;
      height: 10px;
      border-radius: 999px;
      background: linear-gradient(180deg, #edf2fa, #dbe5f4);
      overflow: visible;
      border: 1px solid #d3ddeb;
    }}
    .compact-interval-band {{
      position: absolute;
      top: 1px;
      bottom: 1px;
      border-radius: 999px;
      background: rgba(37, 99, 235, 0.18);
      border: 1px solid rgba(37, 99, 235, 0.26);
    }}
    .compact-interval-tick {{
      position: absolute;
      top: -2px;
      width: 2px;
      height: 14px;
      transform: translateX(-50%);
      border-radius: 999px;
      background: #475569;
    }}
    .compact-interval-marker {{
      position: absolute;
      top: 50%;
      width: 10px;
      height: 10px;
      transform: translate(-50%, -50%);
      border-radius: 999px;
      border: 2px solid #ffffff;
      box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.14);
      background: var(--blue);
    }}
    .compact-interval-marker.status-pass {{ background: var(--green); }}
    .compact-interval-marker.status-warn {{ background: var(--amber); }}
    .compact-interval-marker.status-fail {{ background: var(--red); }}
    .compact-interval-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
    }}
    .compact-interval-meta span {{
      white-space: nowrap;
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
    .numeric-profile-block {{
      border: 1px solid #dbe4f0;
      border-radius: 10px;
      padding: 12px;
      background: #fbfdff;
    }}
    .numeric-profile-scale {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
      margin-bottom: 8px;
    }}
    .numeric-profile-graph {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .numeric-profile-row {{
      display: grid;
      grid-template-columns: minmax(120px, 180px) minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      min-width: 0;
    }}
    .numeric-profile-label {{
      color: #334155;
      font-size: 12px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    .numeric-profile-track {{
      position: relative;
      height: 14px;
      border-radius: 999px;
      overflow: hidden;
      background: linear-gradient(180deg, #eef3fb 0%, #e6edf7 100%);
      border: 1px solid #d9e2ee;
    }}
    .numeric-profile-zero {{
      position: absolute;
      top: -1px;
      bottom: -1px;
      width: 2px;
      background: #94a3b8;
      border-radius: 999px;
      opacity: 0.9;
    }}
    .numeric-profile-bar {{
      position: absolute;
      top: 2px;
      bottom: 2px;
      border-radius: 999px;
      border: 1px solid transparent;
      min-width: 2px;
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.28);
    }}
    .numeric-profile-bar.status-pass {{
      background: rgba(21, 128, 61, 0.18);
      border-color: rgba(21, 128, 61, 0.28);
    }}
    .numeric-profile-bar.status-warn {{
      background: rgba(217, 119, 6, 0.18);
      border-color: rgba(217, 119, 6, 0.28);
    }}
    .numeric-profile-bar.status-fail {{
      background: rgba(220, 38, 38, 0.18);
      border-color: rgba(220, 38, 38, 0.28);
    }}
    .numeric-profile-value {{
      color: #334155;
      font-size: 12px;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
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
      .compact-interval-meta span {{
        white-space: normal;
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
      {_render_status_badge(str(payload['worst_status']))}
      {refresh_button}
      <span class="results-meta" id="results-meta">Loading visible-item summary...</span>
    </div>
  </header>
  <main>
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
    <div class="layout">
      <aside class="sidebar">
        <h2>Audit groups</h2>
        <div class="group-links">
          {group_nav}
        </div>
      </aside>
      <div class="content">
        {group_sections}
        <div class="empty-state" id="audit-empty-state">{_esc(empty_message)}</div>
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
      const emptyStateDefaultText = {json.dumps(empty_message)};
      const groupSections = Array.from(document.querySelectorAll("[data-group-section]"));
      const itemCards = Array.from(document.querySelectorAll("[data-item-card]"));

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
            emptyState.textContent = groupSections.length === 0 ? emptyStateDefaultText : "No audit items match the current filters.";
          }}
        }}
        if (resultsMeta) {{
          resultsMeta.textContent = `${{visibleItems}} visible items across ${{visibleGroups}} visible groups`;
        }}
      }}

      document.querySelectorAll("[data-group-toggle]").forEach((button) => {{
        button.addEventListener("click", () => {{
          const section = button.closest("[data-group-section]");
          toggleGroup(section, !section?.classList.contains("group-collapsed"));
        }});
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
