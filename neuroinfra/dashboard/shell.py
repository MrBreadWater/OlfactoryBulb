"""Shared dashboard shell renderer for repo and neuroinfra web surfaces."""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ShellTabSpec:
    key: str
    label: str
    src: str
    description: str = ""
    badge: str = ""


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def render_dashboard_shell(
    *,
    title: str,
    subtitle: str,
    tabs: Iterable[ShellTabSpec],
    initial_tab: str | None = None,
) -> str:
    tab_specs = list(tabs)
    if not tab_specs:
        raise ValueError("Dashboard shell requires at least one tab")
    active_key = initial_tab or tab_specs[0].key
    nav_html = "\n".join(
        (
            f"<button class='tab-button' type='button' role='tab' data-tab-button "
            f"data-tab-target='tab-{_esc(tab.key)}' aria-controls='tab-{_esc(tab.key)}' "
            f"aria-selected='{'true' if tab.key == active_key else 'false'}'>"
            f"<span>{_esc(tab.label)}</span>"
            f"{f'<em>{_esc(tab.badge)}</em>' if tab.badge else ''}"
            "</button>"
        )
        for tab in tab_specs
    )
    panel_html = "\n".join(
        (
            f"<section id='tab-{_esc(tab.key)}' class='tab-panel' role='tabpanel'"
            f"{'' if tab.key == active_key else ' hidden'}>"
            f"<header class='panel-header'><div>"
            f"<h2>{_esc(tab.label)}</h2>"
            f"{f'<p>{_esc(tab.description)}</p>' if tab.description else ''}"
            "</div></header>"
            f"<iframe src='{_esc(tab.src)}' title='{_esc(tab.label)}'></iframe>"
            "</section>"
        )
        for tab in tab_specs
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(title)}</title>
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
      --surface-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 20;
      background: rgba(247, 248, 251, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 18px 28px 14px;
      backdrop-filter: blur(8px);
    }}
    h1 {{ margin: 0 0 4px; font-size: 22px; letter-spacing: 0; }}
    .subtle {{ color: var(--muted); font-size: 13px; }}
    main {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 20px 28px 28px;
    }}
    .tab-shell {{
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .tab-bar {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      position: sticky;
      top: 76px;
      z-index: 19;
      padding: 8px 0 2px;
      background: rgba(247, 248, 251, 0.96);
      backdrop-filter: blur(8px);
    }}
    .tab-button {{
      appearance: none;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid #dbe3ef;
      background: #ffffff;
      color: #334155;
      border-radius: 8px;
      padding: 8px 12px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    .tab-button em {{
      font-style: normal;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }}
    .tab-button[aria-selected="true"] {{
      background: #eff6ff;
      border-color: #93c5fd;
      color: #1d4ed8;
    }}
    .tab-panel[hidden] {{ display: none !important; }}
    .tab-panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--surface-shadow);
      overflow: hidden;
    }}
    .panel-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 13px 16px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
    }}
    .panel-header h2 {{
      margin: 0;
      font-size: 16px;
    }}
    .panel-header p {{
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 12px;
    }}
    iframe {{
      display: block;
      width: 100%;
      min-height: calc(100vh - 220px);
      border: 0;
      background: #ffffff;
    }}
    @media (max-width: 760px) {{
      header {{ padding: 14px 16px; }}
      main {{ padding: 16px; }}
      .tab-bar {{ top: 72px; }}
      iframe {{ min-height: calc(100vh - 210px); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{_esc(title)}</h1>
    <div class="subtle">{_esc(subtitle)}</div>
  </header>
  <main>
    <div class="tab-shell">
      <nav class="tab-bar" aria-label="Dashboard sections" role="tablist">
        {nav_html}
      </nav>
      {panel_html}
    </div>
  </main>
  <script>
    (() => {{
      function setActiveTab(tabId) {{
        const buttons = document.querySelectorAll("[data-tab-button]");
        const panels = document.querySelectorAll(".tab-panel");
        let resolved = tabId;
        if (!resolved || !document.getElementById(resolved)) {{
          resolved = "tab-{_esc(active_key)}";
        }}
        buttons.forEach((button) => {{
          const selected = button.dataset.tabTarget === resolved;
          button.setAttribute("aria-selected", selected ? "true" : "false");
        }});
        panels.forEach((panel) => {{
          panel.hidden = panel.id !== resolved;
        }});
      }}
      document.addEventListener("click", (event) => {{
        const button = event.target.closest("[data-tab-button]");
        if (!button) return;
        setActiveTab(button.dataset.tabTarget || "tab-{_esc(active_key)}");
      }});
      setActiveTab("tab-{_esc(active_key)}");
    }})();
  </script>
</body>
</html>
"""
