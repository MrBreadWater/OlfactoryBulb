"""Render audit reports as a maintained HTML dashboard."""

from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
import time

from olfactorybulb.audit.cli import run_audit_by_id
from olfactorybulb.audit.core import AuditItem, AuditReport, _pretty_evidence_lines, _summary_chunks, _expand_terms


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


def _render_evidence(evidence: dict[str, Any]) -> str:
    if not evidence:
        return ""
    lines = "\n".join(_esc(line) for line in _pretty_evidence_lines(evidence))
    return (
        "<div class='item-block'>"
        "<h4>Evidence</h4>"
        f"<pre>{lines}</pre>"
        "</div>"
    )


def _render_human_review(item: dict[str, Any]) -> str:
    status = str(item.get("human_review_status") or "").strip()
    reviewer = str(item.get("human_review_reviewer") or "").strip()
    note = str(item.get("human_review_note") or "").strip()
    if not status and not reviewer and not note:
        return ""
    parts = []
    if status:
        parts.append(_expand_terms(status.replace("_", " "), sentence_case=True))
    if reviewer:
        parts.append(f"reviewer: {reviewer}")
    if note:
        parts.append(_expand_terms(note, sentence_case=True))
    return (
        "<div class='item-block'>"
        "<h4>Human review</h4>"
        f"<p>{_esc(' | '.join(parts))}</p>"
        "</div>"
    )


def _item_search_blob(item: AuditItem) -> str:
    fields = [
        item.check_id,
        item.title,
        item.criterion,
        item.description,
        item.acceptable,
        item.acceptable_basis,
        item.note,
        json.dumps(item.evidence, sort_keys=True),
    ]
    return " ".join(_expand_terms(field, sentence_case=True) for field in fields if field).lower()


def _render_item_card(item_payload: dict[str, Any]) -> str:
    item = AuditItem(**item_payload)
    sections = [
        (
            f"<article class='item-card {_status_class(item.status)}' data-item-card data-status='{_esc(item.status)}' "
            f"data-detail-level='{_esc(item.detail_level or 'detail')}' data-search='{_esc(_item_search_blob(item))}'>"
        ),
        "<header class='item-header'>",
        _render_status_badge(item.status),
        f"<div><h3>{_esc(_expand_terms(item.title, sentence_case=True))}</h3>",
        f"<p class='check-id'>{_esc(item.check_id)}</p></div>",
        "</header>",
        "<div class='item-body'>",
        f"<div class='item-block'><h4>Criterion</h4><p>{_esc(_expand_terms(item.criterion, sentence_case=True))}</p></div>",
        f"<div class='item-block'><h4>Description</h4><p>{_esc(_expand_terms(item.description, sentence_case=True))}</p></div>",
        f"<div class='item-block'><h4>Acceptable result</h4><p>{_esc(_expand_terms(item.acceptable, sentence_case=True))}</p></div>",
        (
            "<div class='item-block'><h4>How acceptable result was determined</h4>"
            f"<p>{_esc(_expand_terms(item.acceptable_basis, sentence_case=True))}</p></div>"
        ),
        _render_human_review(item_payload),
        _render_evidence(item.evidence),
    ]
    if item.note:
        sections.append(
            "<div class='item-block'><h4>Note</h4>"
            f"<p>{_esc(_expand_terms(item.note, sentence_case=True))}</p></div>"
        )
    sections.extend(["</div>", "</article>"])
    return "".join(section for section in sections if section)


def _render_group(group: dict[str, Any]) -> str:
    items_html = "\n".join(_render_item_card(item) for item in group["items"])
    return "".join(
        [
            f"<section id='group-{_esc(group['group_id'])}' class='group-section' data-group-section "
            f"data-group-id='{_esc(group['group_id'])}' data-worst-status='{_esc(group['worst_status'])}'>",
            "<header class='group-header'>",
            f"<div class='group-heading'><h2>{_esc(_expand_terms(group['title'], sentence_case=True))}</h2>",
            f"<p>{_esc(group['group_id'])}</p></div>",
            "<div class='group-header-actions'>",
            f"<div class='summary-row'>{_render_summary(group['summary'])}</div>",
            "<button class='action-button group-toggle' type='button' data-group-toggle>Collapse</button>",
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
    generated_at = datetime.now().isoformat(timespec="seconds")
    group_nav = "\n".join(
        (
            f"<a href='#group-{_esc(group['group_id'])}' class='group-link'>"
            f"<span class='group-link-label'>{_esc(group['group_id'])}</span>"
            f"<small>{int(group.get('item_count', 0))} items</small>"
            f"{_render_status_badge(str(group['worst_status']))}"
            "</a>"
        )
        for group in groups
    )
    group_sections = "\n".join(_render_group(group) for group in groups)
    refresh_button = ""
    refresh_script = ""
    if refresh_endpoint:
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
    body {{ margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(247, 248, 251, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 18px 28px 14px;
      backdrop-filter: blur(8px);
    }}
    h1 {{ margin: 0 0 4px; font-size: 22px; }}
    main {{ max-width: 1540px; margin: 0 auto; padding: 24px 28px 60px; }}
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
      position: sticky;
      top: 76px;
      z-index: 28;
      padding-bottom: 4px;
      background: linear-gradient(to bottom, rgba(247, 248, 251, 0.98), rgba(247, 248, 251, 0.92));
      backdrop-filter: blur(8px);
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
      padding: 7px 10px;
      border: 1px solid #dbe3ef;
      border-radius: 999px;
      background: #fff;
      font-size: 12px;
      font-weight: 600;
      color: #334155;
      cursor: pointer;
      box-shadow: none;
    }}
    .toggle-button:hover {{
      background: #eff6ff;
      border-color: #93c5fd;
      color: #1d4ed8;
    }}
    .toggle-button[aria-pressed="true"] {{
      background: #eff6ff;
      border-color: #93c5fd;
      color: #1d4ed8;
      box-shadow: inset 0 0 0 1px rgba(37, 99, 235, 0.10);
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(220px, 260px) minmax(0, 1fr);
      gap: 18px;
      position: relative;
      z-index: 1;
    }}
    .sidebar, .group-section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .sidebar {{
      position: sticky;
      top: 88px;
      z-index: 18;
      align-self: start;
      padding: 14px;
      overflow: visible;
    }}
    .sidebar h2 {{ margin: 0 0 10px; font-size: 16px; }}
    .group-links {{ display: flex; flex-direction: column; gap: 8px; }}
    .group-link {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto auto;
      align-items: start;
      gap: 8px;
      padding: 8px 10px;
      border: 1px solid #e6eaf1;
      border-radius: 8px;
      color: inherit;
      text-decoration: none;
      background: #fbfcfe;
    }}
    .group-link-label {{
      min-width: 0;
      overflow-wrap: anywhere;
      font-weight: 700;
      line-height: 1.25;
    }}
    .group-link small {{
      color: var(--muted);
      font-size: 11px;
      font-weight: 600;
      white-space: nowrap;
      align-self: center;
    }}
    .content {{ display: flex; flex-direction: column; gap: 16px; }}
    .group-header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      padding: 13px 16px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
    }}
    .group-heading {{
      min-width: 0;
      flex: 1 1 auto;
    }}
    .group-header h2 {{ margin: 0; font-size: 16px; }}
    .group-header p {{ margin: 4px 0 0; color: var(--muted); font-size: 12px; }}
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
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 14px;
      padding: 14px;
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
      overflow: hidden;
    }}
    .item-header {{
      display: flex;
      align-items: flex-start;
      gap: 10px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
    }}
    .item-header h3 {{ margin: 0; font-size: 15px; }}
    .check-id {{ margin: 4px 0 0; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }}
    .item-body {{ padding: 14px; display: flex; flex-direction: column; gap: 12px; }}
    .item-block h4 {{ margin: 0 0 4px; font-size: 12px; text-transform: uppercase; color: var(--muted); }}
    .item-block p, .item-block pre {{ margin: 0; }}
    .item-block pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      padding: 10px;
      border-radius: 8px;
      background: #f8fafc;
      border: 1px solid #e6eaf1;
      font: 12px/1.45 ui-monospace, "SFMono-Regular", Consolas, monospace;
    }}
    .empty-state {{
      display: none;
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
      .control-strip {{
        position: static;
        padding-bottom: 0;
        background: transparent;
        backdrop-filter: none;
      }}
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{ position: static; }}
      .group-link {{
        grid-template-columns: minmax(0, 1fr) auto;
      }}
      .group-link .status-badge {{
        grid-column: 1 / -1;
        justify-self: start;
      }}
      .group-header-actions {{
        width: 100%;
        justify-content: flex-start;
      }}
      .summary-row {{
        justify-content: flex-start;
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
              <button class="toggle-button" type="button" id="show-detail-toggle" aria-pressed="true">Show detail items</button>
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
        <div class="empty-state" id="audit-empty-state">No audit items match the current filters.</div>
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
      const groupSections = Array.from(document.querySelectorAll("[data-group-section]"));

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

      function applyFilters() {{
        const query = String(searchInput?.value || "").trim().toLowerCase();
        const failuresOnly = isPressed(failuresOnlyToggle);
        const hidePassedGroups = isPressed(hidePassedGroupsToggle);
        const showDetail = isPressed(showDetailToggle);
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
      document.getElementById("expand-all-groups")?.addEventListener("click", () => {{
        groupSections.forEach((section) => toggleGroup(section, false));
      }});
      document.getElementById("collapse-all-groups")?.addEventListener("click", () => {{
        groupSections.forEach((section) => toggleGroup(section, true));
      }});
      [failuresOnlyToggle, hidePassedGroupsToggle, showDetailToggle].forEach((button) => {{
        button?.addEventListener("click", () => {{
          togglePressed(button);
          applyFilters();
        }});
      }});
      searchInput?.addEventListener("input", applyFilters);
      searchInput?.addEventListener("change", applyFilters);
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
