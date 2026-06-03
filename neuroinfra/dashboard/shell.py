"""Shared dashboard shell renderer for repo and neuroinfra web surfaces."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ShellTabSpec:
    key: str
    label: str
    src: str
    description: str = ""
    badge: str = ""
    badge_tone: str = "neutral"


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def render_dashboard_shell(
    *,
    title: str,
    subtitle: str,
    tabs: Iterable[ShellTabSpec],
    initial_tab: str | None = None,
    toolbar_html: str = "",
    panel_toolbar_html_by_key: dict[str, str] | None = None,
    shell_state: dict[str, object] | None = None,
    state_endpoint: str = "",
    state_poll_interval_ms: int = 2500,
) -> str:
    tab_specs = list(tabs)
    if not tab_specs:
        raise ValueError("Dashboard shell requires at least one tab")
    active_key = initial_tab or tab_specs[0].key
    toolbar_block = f"<section class='shell-toolbar'>{toolbar_html}</section>" if toolbar_html.strip() else ""
    panel_toolbar_html_by_key = dict(panel_toolbar_html_by_key or {})
    shell_state = dict(shell_state or {})
    initial_state_json = html.escape(
        json.dumps(shell_state, indent=2, sort_keys=True).replace("</", "<\\/"),
        quote=False,
    )
    tab_state_payload = shell_state.get("tabs") if isinstance(shell_state.get("tabs"), dict) else {}
    header_status_html = "\n".join(
        (
            f"<div class='shell-status-chip' data-shell-status-card data-tab-key='{_esc(tab.key)}'>"
            f"<span>{_esc(tab.label)}</span>"
            f"<strong class='tone-{_esc(str((tab_state_payload.get(tab.key) or {}).get('badge_tone') or tab.badge_tone or 'neutral'))}' data-shell-status-badge>{_esc(str((tab_state_payload.get(tab.key) or {}).get('badge') or tab.badge or 'idle'))}</strong>"
            f"<small data-shell-status-detail>{_esc(str((shell_state.get(tab.key) or {}).get('message') if isinstance(shell_state.get(tab.key), dict) else tab.description or ''))}</small>"
            "</div>"
        )
        for tab in tab_specs
    )
    progress_payload = shell_state.get("progress") if isinstance(shell_state.get("progress"), dict) else {}
    progress_active = bool(progress_payload.get("active"))
    progress_label = str(progress_payload.get("label") or "")
    progress_value = str(progress_payload.get("value_text") or "")
    progress_fraction = float(progress_payload.get("fraction") or 0.0) if progress_payload.get("fraction") is not None else 0.0
    progress_width = max(0.0, min(progress_fraction, 1.0)) * 100.0
    progress_indeterminate = bool(progress_payload.get("indeterminate"))
    nav_html = "\n".join(
        (
            f"<button class='tab-button' type='button' role='tab' data-tab-button "
            f"data-tab-key='{_esc(tab.key)}' data-tab-target='tab-{_esc(tab.key)}' aria-controls='tab-{_esc(tab.key)}' "
            f"aria-selected='{'true' if tab.key == active_key else 'false'}'>"
            f"<span class='tab-label'>{_esc(tab.label)}</span>"
            f"<em class='tab-badge tone-{_esc(tab.badge_tone)}' data-tab-badge {'hidden' if not tab.badge else ''}>{_esc(tab.badge)}</em>"
            "</button>"
        )
        for tab in tab_specs
    )
    def _panel_toolbar_html(tab_key: str) -> str:
        panel_toolbar = str(panel_toolbar_html_by_key.get(tab_key, "") or "").strip()
        if not panel_toolbar:
            return ""
        return f"<div class='panel-toolbar-shell' data-panel-toolbar data-tab-key='{_esc(tab_key)}'>{panel_toolbar}</div>"

    panel_html = "\n".join(
        (
            (
                f"<section id='tab-{_esc(tab.key)}' class='tab-panel' role='tabpanel'"
                f"{'' if tab.key == active_key else ' hidden'}>"
                f"{_panel_toolbar_html(tab.key)}"
                f"<iframe src='{_esc(tab.src)}' title='{_esc(tab.label)}' data-tab-frame data-tab-key='{_esc(tab.key)}' data-base-src='{_esc(tab.src)}'></iframe>"
                "</section>"
            )
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
      --bg: #f3f6fb;
      --ink: #162132;
      --muted: #5f6f86;
      --line: #d7dfeb;
      --panel: #ffffff;
      --panel-alt: #f8fbff;
      --blue: #2b5fb8;
      --blue-soft: #e8f0ff;
      --red: #c23d3d;
      --amber: #b87414;
      --green: #177245;
      --surface-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
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
      background: rgba(243, 246, 251, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 10px 20px 10px;
      backdrop-filter: blur(8px);
    }}
    .shell-header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
    }}
    .shell-heading {{
      min-width: 0;
      flex: 1 1 auto;
    }}
    h1 {{ margin: 0 0 2px; font-size: 18px; letter-spacing: 0; }}
    .subtle {{ color: var(--muted); font-size: 12px; }}
    .shell-status-strip {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 8px;
      flex: 1 1 620px;
      max-width: 820px;
    }}
    .shell-status-chip {{
      display: flex;
      flex-direction: column;
      gap: 2px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: rgba(255, 255, 255, 0.92);
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
      min-width: 0;
    }}
    .shell-status-chip span {{
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }}
    .shell-status-chip strong {{
      font-size: 13px;
      line-height: 1.3;
    }}
    .shell-status-chip strong.tone-neutral,
    .shell-status-chip strong.tone-pass,
    .shell-status-chip strong.tone-warn,
    .shell-status-chip strong.tone-fail,
    .shell-status-chip strong.tone-running,
    .shell-status-chip strong.tone-info {{
      padding: 0;
      border: 0;
      background: transparent;
      box-shadow: none;
    }}
    .shell-status-chip small {{
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }}
    .shell-progress {{
      margin-top: 8px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .shell-progress[hidden] {{
      display: none !important;
    }}
    .shell-progress-meta {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      font-size: 12px;
      color: var(--muted);
    }}
    .shell-progress-label {{
      font-weight: 700;
      color: var(--ink);
    }}
    .shell-progress-track {{
      position: relative;
      height: 8px;
      border-radius: 999px;
      overflow: hidden;
      background: #dbe5f4;
    }}
    .shell-progress-bar {{
      height: 100%;
      width: 0%;
      border-radius: 999px;
      background: linear-gradient(90deg, #2563eb, #60a5fa);
      transition: width 180ms ease;
    }}
    .shell-progress-bar.indeterminate {{
      position: absolute;
      width: 32%;
      animation: shell-progress-indeterminate 1.15s ease-in-out infinite;
    }}
    @keyframes shell-progress-indeterminate {{
      0% {{ left: -30%; }}
      100% {{ left: 100%; }}
    }}
    main {{
      width: 100%;
      max-width: max(95%, 1700px);
      margin: 0 auto;
      padding: 16px 20px 24px;
    }}
    .tab-shell {{
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .shell-toolbar {{
      display: grid;
      gap: 16px;
    }}
    .control-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .toolbar-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      box-shadow: var(--surface-shadow);
      padding: 16px;
    }}
    .toolbar-card-header h2 {{
      margin: 0 0 4px;
      font-size: 16px;
    }}
    .toolbar-card-header p {{
      margin: 0;
      color: var(--muted);
      font-size: 12px;
    }}
    .form-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .form-field {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .form-field-wide {{
      grid-column: 1 / -1;
    }}
    .form-field span {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }}
    .form-help {{
      color: var(--muted);
      font-size: 11px;
      line-height: 1.4;
    }}
    .form-field input,
    .form-field select {{
      appearance: none;
      width: 100%;
      border: 1px solid #d6deea;
      border-radius: 8px;
      padding: 9px 11px;
      font: inherit;
      background: #fff;
      color: var(--ink);
    }}
    .toolbar-actions {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      margin-top: 14px;
    }}
    .toolbar-button {{
      appearance: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      border: 1px solid #d6deea;
      background: #fff;
      color: #334155;
      border-radius: 8px;
      padding: 9px 12px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    .toolbar-button:disabled {{
      cursor: wait;
      opacity: 0.7;
    }}
    .toolbar-button-primary {{
      background: var(--blue);
      border-color: var(--blue);
      color: #fff;
    }}
    .toolbar-meta {{
      color: var(--muted);
      font-size: 12px;
    }}
    .status-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .status-card {{
      border: 1px solid #dde5f0;
      border-radius: 8px;
      background: var(--panel-alt);
      padding: 12px;
      min-height: 94px;
    }}
    .status-card span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }}
    .status-card strong {{
      display: block;
      margin-top: 4px;
      font-size: 16px;
    }}
    .status-card small {{
      display: block;
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }}
    .tab-bar {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      position: sticky;
      top: 72px;
      z-index: 19;
      padding: 8px 0 2px;
      background: rgba(243, 246, 251, 0.96);
      backdrop-filter: blur(8px);
    }}
    .tab-button {{
      appearance: none;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid #d6deea;
      background: #ffffff;
      color: #334155;
      border-radius: 8px;
      padding: 9px 12px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
    }}
    .tab-label {{
      min-width: 0;
    }}
    .tab-badge {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 22px;
      padding: 3px 8px;
      border-radius: 999px;
      font-style: normal;
      font-size: 12px;
      font-weight: 600;
      border: 1px solid transparent;
      background: #eef2f8;
      color: var(--muted);
    }}
    .tab-badge[hidden] {{
      display: none !important;
    }}
    .tone-neutral {{
      background: #eef2f8;
      color: var(--muted);
      border-color: #d7dfeb;
    }}
    .tone-pass {{
      background: #ebfaf1;
      color: var(--green);
      border-color: #b6ead0;
    }}
    .tone-warn {{
      background: #fff7e8;
      color: var(--amber);
      border-color: #f4d7a4;
    }}
    .tone-fail {{
      background: #fdeeee;
      color: var(--red);
      border-color: #f0b6b6;
    }}
    .tone-running, .tone-info {{
      background: var(--blue-soft);
      color: var(--blue);
      border-color: #bfd0f7;
    }}
    .tab-button[aria-selected="true"] {{
      background: var(--blue-soft);
      border-color: #bfd0f7;
      color: #204c98;
    }}
    .tab-panel[hidden] {{ display: none !important; }}
    .tab-panel {{
      display: flex;
      flex-direction: column;
      gap: 12px;
      min-height: calc(100vh - 155px);
    }}
    .panel-toolbar-shell {{
      display: grid;
      gap: 12px;
    }}
    iframe {{
      display: block;
      width: 100%;
      min-height: calc(100vh - 190px);
      flex: 1 1 auto;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #ffffff;
      box-shadow: var(--surface-shadow);
    }}
    @media (max-width: 760px) {{
      header {{ padding: 10px 14px; }}
      main {{ padding: 14px; }}
      .shell-header {{
        flex-direction: column;
      }}
      .shell-status-strip {{
        width: 100%;
        max-width: none;
      }}
      .tab-bar {{ top: 64px; }}
      iframe {{ min-height: calc(100vh - 210px); }}
    }}
  </style>
</head>
<body data-state-endpoint="{_esc(state_endpoint)}" data-state-poll-interval-ms="{_esc(state_poll_interval_ms)}">
  <header>
    <div class="shell-header">
      <div class="shell-heading">
        <h1>{_esc(title)}</h1>
        <div class="subtle">{_esc(subtitle)}</div>
      </div>
      <div class="shell-status-strip">
        {header_status_html}
      </div>
    </div>
    <div class="shell-progress" id="shell-progress" {'hidden' if not progress_active else ''}>
      <div class="shell-progress-meta">
        <span class="shell-progress-label" id="shell-progress-label">{_esc(progress_label)}</span>
        <span id="shell-progress-value">{_esc(progress_value)}</span>
      </div>
      <div class="shell-progress-track">
        <div class="shell-progress-bar{' indeterminate' if progress_indeterminate else ''}" id="shell-progress-bar" style="width: {progress_width:.1f}%"></div>
      </div>
    </div>
  </header>
  <main>
    <div class="tab-shell">
      {toolbar_block}
      <nav class="tab-bar" aria-label="Dashboard sections" role="tablist">
        {nav_html}
      </nav>
      {panel_html}
    </div>
  </main>
  <script id="dashboard-shell-state" type="application/json">{initial_state_json}</script>
  <script>
    (() => {{
      function toneClass(tone) {{
        const normalized = String(tone || "neutral").toLowerCase();
        if (["pass", "warn", "fail", "info", "running", "neutral"].includes(normalized)) {{
          return "tone-" + normalized;
        }}
        return "tone-neutral";
      }}
      function withRevision(src, revision) {{
        const base = String(src || "");
        if (!base) return base;
        const url = new URL(base, window.location.href);
        if (revision) {{
          url.searchParams.set("__rev", String(revision));
        }} else {{
          url.searchParams.delete("__rev");
        }}
        if (url.origin === window.location.origin) {{
          return url.pathname + url.search + url.hash;
        }}
        return url.toString();
      }}
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
      function applyState(state) {{
        if (!state || typeof state !== "object") return;
        window.__dashboardShellState = state;
        const tabs = state.tabs || {{}};
        Object.entries(tabs).forEach(([key, tabState]) => {{
          const button = document.querySelector(`[data-tab-button][data-tab-key="${{key}}"]`);
          const frame = document.querySelector(`iframe[data-tab-frame][data-tab-key="${{key}}"]`);
          const statusCard = document.querySelector(`[data-shell-status-card][data-tab-key="${{key}}"]`);
          if (button) {{
            const badgeEl = button.querySelector("[data-tab-badge]");
            const badgeText = String(tabState.badge || "").trim();
            if (badgeEl) {{
              badgeEl.textContent = badgeText;
              badgeEl.hidden = !badgeText;
              badgeEl.className = `tab-badge ${{toneClass(tabState.badge_tone || tabState.status || "neutral")}}`;
            }}
          }}
          if (statusCard) {{
            const badgeEl = statusCard.querySelector("[data-shell-status-badge]");
            const detailEl = statusCard.querySelector("[data-shell-status-detail]");
            const statusValue = state[key] && typeof state[key] === "object" ? state[key] : {{}};
            if (badgeEl) {{
              badgeEl.textContent = String(tabState.badge || tabState.status || "idle");
              badgeEl.className = toneClass(tabState.badge_tone || tabState.status || "neutral");
            }}
            if (detailEl) {{
              detailEl.textContent = String(statusValue.message || "");
            }}
          }}
          if (frame) {{
            const desiredBaseSrc = String(tabState.src || frame.dataset.baseSrc || frame.getAttribute("src") || "").trim();
            const desiredSrc = withRevision(desiredBaseSrc, tabState.revision || "");
            if (desiredBaseSrc) {{
              frame.dataset.baseSrc = desiredBaseSrc;
            }}
            if (desiredSrc && frame.dataset.currentSrc !== desiredSrc) {{
              frame.dataset.currentSrc = desiredSrc;
              frame.src = desiredSrc;
            }}
          }}
        }});
        const progress = state.progress || {{}};
        const progressRoot = document.getElementById("shell-progress");
        const progressLabel = document.getElementById("shell-progress-label");
        const progressValue = document.getElementById("shell-progress-value");
        const progressBar = document.getElementById("shell-progress-bar");
        if (progressRoot && progressLabel && progressValue && progressBar) {{
          const active = Boolean(progress.active);
          progressRoot.hidden = !active;
          progressLabel.textContent = String(progress.label || "");
          progressValue.textContent = String(progress.value_text || "");
          const fraction = Number(progress.fraction || 0);
          const clamped = Number.isFinite(fraction) ? Math.max(0, Math.min(fraction, 1)) : 0;
          progressBar.style.width = `${{clamped * 100}}%`;
          progressBar.classList.toggle("indeterminate", Boolean(progress.indeterminate));
        }}
        window.dispatchEvent(new CustomEvent("dashboard-shell-state", {{ detail: state }}));
      }}
      async function pollState() {{
        const endpoint = String(document.body.dataset.stateEndpoint || "").trim();
        if (!endpoint || pollState.inFlight) return;
        pollState.inFlight = true;
        try {{
          const response = await fetch(endpoint + (endpoint.includes("?") ? "&" : "?") + "cache=" + Date.now(), {{
            cache: "no-store",
          }});
          if (!response.ok) {{
            throw new Error("State poll failed with status " + response.status);
          }}
          const payload = await response.json();
          applyState(payload);
        }} catch (_error) {{
        }} finally {{
          pollState.inFlight = false;
        }}
      }}
      document.addEventListener("click", (event) => {{
        const button = event.target.closest("[data-tab-button]");
        if (!button) return;
        setActiveTab(button.dataset.tabTarget || "tab-{_esc(active_key)}");
      }});
      const stateNode = document.getElementById("dashboard-shell-state");
      if (stateNode && stateNode.textContent) {{
        try {{
          applyState(JSON.parse(stateNode.textContent));
        }} catch (_error) {{
        }}
      }}
      setActiveTab("tab-{_esc(active_key)}");
      const pollIntervalMs = Number(document.body.dataset.statePollIntervalMs || "0");
      if (pollIntervalMs > 0 && String(document.body.dataset.stateEndpoint || "").trim()) {{
        window.setInterval(pollState, pollIntervalMs);
      }}
    }})();
  </script>
</body>
</html>
"""
