"""Unified dashboard shell for docs, audits, and optimization views."""

from __future__ import annotations

import argparse
import errno
import http.server
import importlib
import json
import os
import posixpath
import shlex
import threading
import time
import webbrowser
from datetime import datetime
from html import escape as html_escape
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from neuroinfra.dashboard import ShellTabSpec
from olfactorybulb.audit.cli import available_audit_entries, run_audit_by_id
from olfactorybulb.audit.core import AuditItem, AuditReport
from olfactorybulb.audit.dashboard import export_audit_dashboard
import tools.analysis.hfo_visual_dashboard as hfo_dashboard


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "dashboard" / "control_center"
DEFAULT_STATUS_JSON = (REPO_ROOT / hfo_dashboard.SUMMARY_STATUS_PATH).resolve()
DEFAULT_OPTIMIZATION_ROOT = REPO_ROOT / "results" / "notebook_runs" / "optimization"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 6006
DEFAULT_AUDIT_ID = "all"
DEFAULT_AUDIT_ARGS: list[str] = []
DEFAULT_CONTROL_CENTER_TOP_N = hfo_dashboard.DEFAULT_TOP_N
DEFAULT_CONTROL_CENTER_GENERATE_PACKETS_TOP_N = 0
DEFAULT_CONTROL_CENTER_GENERATE_PACKET_WORKERS = hfo_dashboard.DEFAULT_PACKET_GENERATION_WORKERS
DEFAULT_CONTROL_CENTER_CLEANUP_STALE_PACKETS = False
DEFAULT_STATE_POLL_INTERVAL_MS = 2000
DEFAULT_DEV_RELOAD_POLL_INTERVAL_MS = 1000


def _progress(message: str) -> None:
    print(f"[control_center] {message}", flush=True)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _json_script_payload(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True).replace("</", "<\\/")


def _source_revision(paths: list[Path]) -> str:
    parts: list[str] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            parts.append(f"{path.name}:missing")
            continue
        parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts)


def _dev_reload_source_paths() -> list[Path]:
    shell_module = importlib.import_module("neuroinfra.dashboard.shell")
    audit_dashboard_module = importlib.import_module("olfactorybulb.audit.dashboard")
    return [
        Path(str(shell_module.__file__)).resolve(),
        Path(str(audit_dashboard_module.__file__)).resolve(),
    ]


def _inject_dev_reload_script(html_text: str, *, endpoint: str = "/__control_center_dev_state__", poll_interval_ms: int = DEFAULT_DEV_RELOAD_POLL_INTERVAL_MS) -> str:
    script = f"""
  <script>
    (() => {{
      const endpoint = {json.dumps(endpoint)};
      const pollMs = {int(poll_interval_ms)};
      let seenRevision = "";
      let inFlight = false;
      async function pollDevRevision() {{
        if (inFlight) return;
        inFlight = true;
        try {{
          const response = await fetch(endpoint + "?cache=" + Date.now(), {{ cache: "no-store" }});
          if (!response.ok) return;
          const payload = await response.json();
          const revision = String(payload.revision || "");
          if (!seenRevision) {{
            seenRevision = revision;
          }} else if (revision && revision !== seenRevision) {{
            window.location.reload();
          }}
        }} catch (_error) {{
        }} finally {{
          inFlight = false;
        }}
      }}
      window.setInterval(pollDevRevision, pollMs);
      pollDevRevision();
    }})();
  </script>
"""
    if "</body>" in html_text:
        return html_text.replace("</body>", script + "</body>", 1)
    return html_text + script


def _default_audit_args_for(audit_id: str) -> list[str]:
    normalized = str(audit_id or "").strip() or DEFAULT_AUDIT_ID
    for entry in available_audit_entries():
        if str(entry.get("audit_id") or "") == normalized:
            return list(entry.get("default_args") or [])
    if normalized == DEFAULT_AUDIT_ID:
        return list(DEFAULT_AUDIT_ARGS)
    return []


def _badge_tone(value: str) -> str:
    normalized = str(value or "").strip().upper()
    if normalized == "PASS":
        return "pass"
    if normalized == "WARN":
        return "warn"
    if normalized == "FAIL":
        return "fail"
    if normalized in {"RUNNING", "STARTING", "LOADING"}:
        return "running"
    if normalized in {"READY", "UNAVAILABLE", "ERROR"}:
        return "info" if normalized == "READY" else ("warn" if normalized == "UNAVAILABLE" else "fail")
    return "neutral"


def _render_frame_message_html(*, title: str, message: str, auto_refresh_s: float | None = None) -> str:
    refresh_meta = ""
    if auto_refresh_s is not None and auto_refresh_s > 0:
        refresh_meta = f"<meta http-equiv='refresh' content='{float(auto_refresh_s):g}'>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {refresh_meta}
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fb;
      --ink: #17202a;
      --muted: #667085;
      --line: #d9dee8;
      --panel: #ffffff;
      --shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 920px;
      margin: 0 auto;
      padding: 40px 20px 64px;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 20px;
    }}
    h1 {{ margin: 0 0 10px; font-size: 22px; }}
    p {{ margin: 0; color: var(--muted); }}
  </style>
</head>
<body>
  <main>
    <section>
      <h1>{title}</h1>
      <p>{message}</p>
    </section>
  </main>
</body>
</html>
"""


def _write_loading_frame(output_dir: Path, *, title: str, message: str, auto_refresh_s: float = 3.0) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(
        output_dir / "index.html",
        _render_frame_message_html(title=title, message=message, auto_refresh_s=auto_refresh_s),
    )


def _write_error_frame(output_dir: Path, *, title: str, message: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(
        output_dir / "index.html",
        _render_frame_message_html(title=title, message=message),
    )


def _module_tabs() -> tuple[ShellTabSpec, ...]:
    return (
        ShellTabSpec(
            key="audits",
            label="Audits",
            src="/audits/index.html",
            description="Structured audit results, grouped summaries, and detailed findings.",
            badge="starting",
            badge_tone="running",
        ),
        ShellTabSpec(
            key="optimization",
            label="Optimization",
            src="/optimization/index.html",
            description="Campaign-level HFO packet review and candidate ranking.",
            badge="loading",
            badge_tone="running",
        ),
        ShellTabSpec(
            key="docs",
            label="Docs",
            src="/docs/index.html",
            description="Rendered maintained docs and current operational guidance.",
            badge="ready",
            badge_tone="info",
        ),
    )


def _available_audit_entries() -> list[dict[str, str]]:
    return [
        {
            "audit_id": str(entry.get("audit_id") or ""),
            "title": str(entry.get("title") or ""),
            "description": str(entry.get("description") or ""),
            "default_args": list(entry.get("default_args") or []),
        }
        for entry in available_audit_entries()
    ]


def _display_audit_args(audit_id: str, audit_args: list[str]) -> str:
    normalized_id = str(audit_id or "").strip()
    normalized_args = [str(arg) for arg in (audit_args or []) if str(arg).strip()]
    default_args = [str(arg) for arg in _default_audit_args_for(normalized_id) if str(arg).strip()]
    if normalized_args == default_args:
        return ""
    return " ".join(normalized_args)


def _render_audit_runner_panel(*, audit_id: str, audit_args: list[str]) -> str:
    options_html = "\n".join(
        (
            f"<option value='{html_escape(entry['audit_id'], quote=True)}'"
            f"{' selected' if entry['audit_id'] == audit_id else ''}>"
            f"{html_escape(entry['audit_id'])}"
            "</option>"
        )
        for entry in _available_audit_entries()
    )
    audits_payload = html_escape(_json_script_payload(_available_audit_entries()), quote=False)
    audit_args_text = html_escape(_display_audit_args(audit_id, audit_args))
    return f"""
<section class="toolbar-card audit-runner-card" id="control-center-audit-runner" data-audit-runner-mode="expanded">
  <div class="toolbar-card-header">
    <div>
      <h2>Audit runner</h2>
      <p id="audit-selection-description">Run a new sweep across every registered audit.</p>
    </div>
    <button class="toolbar-button toolbar-button-compact audit-runner-edit" type="button" id="control-center-edit-audit" hidden>Edit selection</button>
  </div>
  <div class="audit-runner-summary" id="audit-runner-summary" aria-live="polite">
    <div class="audit-runner-summary-grid">
      <div class="audit-runner-summary-item">
        <span>Audit</span>
        <strong id="audit-runner-summary-audit">—</strong>
      </div>
      <div class="audit-runner-summary-item">
        <span>Args</span>
        <strong id="audit-runner-summary-args">none</strong>
      </div>
      <div class="audit-runner-summary-item">
        <span>Last run</span>
        <strong id="audit-runner-summary-last-run">No audit has been run yet.</strong>
      </div>
    </div>
    <div class="audit-runner-summary-meta" id="audit-runner-summary-meta">No audit has been run in this session.</div>
  </div>
  <div class="audit-runner-form" id="control-center-audit-form">
  <div class="audit-runner-fields">
    <label class="form-field">
      <span>Audit id</span>
      <small id="control-center-audit-id-help" class="form-help">Use <code>all</code> to run every registered audit.</small>
      <select id="control-center-audit-id" title="Choose a registered audit to run from this page" aria-label="Audit id" aria-describedby="control-center-audit-id-help audit-selection-description">
        {options_html}
      </select>
    </label>
    <label class="form-field">
      <span>Audit arguments</span>
      <small id="control-center-audit-args-help" class="form-help">Optional command-line flags for the selected audit.</small>
      <input id="control-center-audit-args" type="text" value="{audit_args_text}" title="Optional extra command-line flags for the selected audit" aria-label="Audit arguments" aria-describedby="control-center-audit-args-help audit-selection-description" placeholder="--suite maintained_core --details">
    </label>
  </div>
  <div class="toolbar-actions audit-runner-actions">
    <button class="toolbar-button toolbar-button-primary" type="button" id="control-center-run-audit">Run selected audit</button>
    <button class="toolbar-button" type="button" id="control-center-reset-audit">Reset defaults</button>
  </div>
  </div>
</section>
<script id="control-center-audit-options" type="application/json">{audits_payload}</script>
<script>
(() => {{
  const optionsNode = document.getElementById("control-center-audit-options");
  const availableAudits = optionsNode && optionsNode.textContent ? JSON.parse(optionsNode.textContent) : [];
  const auditSelect = document.getElementById("control-center-audit-id");
  const auditArgsInput = document.getElementById("control-center-audit-args");
  const runButton = document.getElementById("control-center-run-audit");
  const resetButton = document.getElementById("control-center-reset-audit");
  const editButton = document.getElementById("control-center-edit-audit");
  const auditRunnerCard = document.getElementById("control-center-audit-runner");
  const auditRunnerForm = document.getElementById("control-center-audit-form");
  const runnerSummaryAudit = document.getElementById("audit-runner-summary-audit");
  const runnerSummaryArgs = document.getElementById("audit-runner-summary-args");
  const runnerSummaryLastRun = document.getElementById("audit-runner-summary-last-run");
  const runnerSummaryMeta = document.getElementById("audit-runner-summary-meta");
  const selectionDescription = document.getElementById("audit-selection-description");
  const defaultAuditId = {json.dumps(audit_id)};
  const defaultAuditArgs = {json.dumps(audit_args)};
  let formDirty = false;
  let suppressFormEvents = false;
  let manualRunnerExpanded = true;
  let runnerStateInitialized = false;

  function selectedAuditEntry() {{
    const selectedId = String(auditSelect?.value || "");
    return availableAudits.find((entry) => entry.audit_id === selectedId) || null;
  }}

  function auditEntryForId(auditId) {{
    return availableAudits.find((entry) => entry.audit_id === String(auditId || "")) || null;
  }}

  function selectedAuditDefaultArgs() {{
    const entry = selectedAuditEntry();
    return Array.isArray(entry?.default_args) ? entry.default_args : [];
  }}

  function defaultArgsForAuditId(auditId) {{
    const entry = auditEntryForId(auditId);
    return Array.isArray(entry?.default_args) ? entry.default_args : [];
  }}

  function arraysEqual(left, right) {{
    if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return false;
    return left.every((value, index) => String(value) === String(right[index]));
  }}

  function displayAuditArgs(auditId, auditArgs) {{
    const normalizedArgs = Array.isArray(auditArgs)
      ? auditArgs.map((value) => String(value || "").trim()).filter(Boolean)
      : [];
    const defaultArgs = defaultArgsForAuditId(auditId)
      .map((value) => String(value || "").trim())
      .filter(Boolean);
    if (arraysEqual(normalizedArgs, defaultArgs)) {{
      return "";
    }}
    return normalizedArgs.join(" ");
  }}

  function formatAuditTimestamp(value) {{
    const text = String(value || "").trim();
    if (!text) {{
      return "No audit has been run yet.";
    }}
    return text.replace("T", " ").slice(0, 16);
  }}

  function setRunnerMode(mode, {{ manual = false }} = {{}}) {{
    const expanded = String(mode || "") !== "compact";
    if (manual) {{
      manualRunnerExpanded = expanded;
    }}
    if (auditRunnerCard) {{
      auditRunnerCard.dataset.auditRunnerMode = expanded ? "expanded" : "compact";
    }}
    if (auditRunnerForm) {{
      auditRunnerForm.hidden = !expanded;
    }}
    if (editButton) {{
      editButton.hidden = expanded;
    }}
  }}

  function updateRunnerSummary(auditState) {{
    const auditIdText = String(auditState?.audit_id || defaultAuditId || "—");
    const argsArray = Array.isArray(auditState?.audit_args) ? auditState.audit_args : defaultAuditArgs;
    const auditArgsText = displayAuditArgs(auditIdText, argsArray) || "none";
    const lastRunText = formatAuditTimestamp(auditState?.generated_at || "");
    if (runnerSummaryAudit) {{
      runnerSummaryAudit.textContent = auditIdText || "—";
    }}
    if (runnerSummaryArgs) {{
      runnerSummaryArgs.textContent = auditArgsText;
    }}
    if (runnerSummaryLastRun) {{
      runnerSummaryLastRun.textContent = lastRunText;
    }}
    if (runnerSummaryMeta) {{
      runnerSummaryMeta.textContent = String(auditState?.message || "No audit has been run in this session.");
    }}
    if (runButton) {{
      const hasRun = Boolean(String(auditState?.generated_at || "").trim());
      runButton.textContent = String(auditState?.status || "") === "running"
        ? "Running..."
        : (hasRun ? "Run again" : "Run selected audit");
    }}
  }}

  function setFormValues(auditId, auditArgs, {{ preserveDirty = false }} = {{}}) {{
    suppressFormEvents = true;
    if (auditSelect && auditId) {{
      auditSelect.value = String(auditId);
    }}
    if (auditArgsInput) {{
      auditArgsInput.value = displayAuditArgs(auditId, auditArgs);
    }}
    suppressFormEvents = false;
    if (!preserveDirty) {{
      formDirty = false;
    }}
    updateSelectionDescription();
  }}

  function updateSelectionDescription() {{
    const entry = selectedAuditEntry();
    if (!selectionDescription) return;
    selectionDescription.textContent = entry ? entry.description : "Choose any registered audit, pass explicit arguments, and rerun it without restarting the shell.";
  }}

  async function runSelectedAudit() {{
    if (!runButton) return;
    const payload = {{
      audit_id: String(auditSelect?.value || defaultAuditId),
      audit_args_text: String(auditArgsInput?.value || "").trim(),
    }};
    runButton.disabled = true;
    if (runnerSummaryMeta) {{
      runnerSummaryMeta.textContent = `Starting ${{payload.audit_id}}...`;
    }}
    try {{
      const response = await fetch("/__audit_run__", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(payload),
        cache: "no-store",
      }});
      const result = await response.json().catch(() => ({{}}));
      if (!response.ok || !result.ok) {{
        throw new Error(String(result.error || "Audit run failed"));
      }}
      const auditTabButton = document.querySelector('[data-tab-button][data-tab-key="audits"]');
      auditTabButton?.click();
      formDirty = false;
      manualRunnerExpanded = false;
      setRunnerMode("compact");
      if (runButton) {{
        runButton.textContent = "Run again";
      }}
    }} catch (error) {{
      if (runnerSummaryMeta) {{
        runnerSummaryMeta.textContent = String(error && error.message ? error.message : error);
      }}
    }} finally {{
      runButton.disabled = false;
    }}
  }}

  auditSelect?.addEventListener("change", () => {{
    if (suppressFormEvents) return;
    formDirty = true;
    if (auditArgsInput) {{
      auditArgsInput.value = "";
    }}
    updateSelectionDescription();
  }});
  resetButton?.addEventListener("click", () => {{
    setFormValues(defaultAuditId, defaultAuditArgs);
    manualRunnerExpanded = true;
    setRunnerMode("expanded", {{ manual: true }});
  }});
  runButton?.addEventListener("click", runSelectedAudit);
  editButton?.addEventListener("click", () => {{
    manualRunnerExpanded = true;
    setRunnerMode("expanded", {{ manual: true }});
  }});
  window.addEventListener("message", (event) => {{
    const data = event && typeof event.data === "object" ? event.data : null;
    if (!data || data.type !== "control-center-run-audit") {{
      return;
    }}
    runSelectedAudit();
  }});
  auditArgsInput?.addEventListener("input", () => {{
    if (suppressFormEvents) return;
    formDirty = true;
    manualRunnerExpanded = true;
    setRunnerMode("expanded", {{ manual: true }});
  }});
  auditArgsInput?.addEventListener("keydown", (event) => {{
    if (event.key === "Enter") {{
      event.preventDefault();
      runSelectedAudit();
    }}
  }});

  window.addEventListener("dashboard-shell-state", (event) => {{
    const state = event.detail || {{}};
    const auditState = state.audit || {{}};
    if (!formDirty && auditState.audit_id) {{
      setFormValues(String(auditState.audit_id), Array.isArray(auditState.audit_args) ? auditState.audit_args : []);
    }}
    updateRunnerSummary(auditState);
    if (runButton) {{
      runButton.disabled = String(auditState.status || "") === "running";
    }}
    const hasPriorRun = Boolean(String(auditState.generated_at || "").trim()) || String(auditState.status || "") === "running" || String(auditState.status || "") === "ready";
    if (!runnerStateInitialized && hasPriorRun) {{
      manualRunnerExpanded = false;
      setRunnerMode("compact");
      runnerStateInitialized = true;
    }} else if (String(auditState.status || "") === "idle" && !String(auditState.generated_at || "").trim()) {{
      setRunnerMode("expanded");
    }} else if (!manualRunnerExpanded) {{
      setRunnerMode("compact");
    }}
    runnerStateInitialized = true;
    updateSelectionDescription();
  }});

  setFormValues(defaultAuditId, defaultAuditArgs);
  updateRunnerSummary({{ audit_id: defaultAuditId, audit_args: defaultAuditArgs, message: "No audit has been run in this session." }});
  setRunnerMode("expanded", {{ manual: true }});
}})();
</script>
"""


def _render_optimization_context_panel(*, campaign_label: str, campaign_path_text: str | None = None) -> str:
    normalized = str(campaign_path_text or campaign_label or "").strip() or "no active optimization campaign detected"
    details = (
        "The optimization tab shows the active campaign path that was auto-detected for this session."
        if normalized != "no active optimization campaign detected"
        else "No active optimization campaign could be auto-detected from the maintained status file or optimization results directory."
    )
    campaign_text = html_escape(normalized)
    return f"""
<section class="toolbar-card optimization-context-card">
  <div class="toolbar-card-header">
    <div>
      <h2>Optimization campaign</h2>
      <p>{html_escape(details)}</p>
    </div>
  </div>
  <div class="status-grid">
    <div class="status-card">
      <span>Campaign</span>
      <strong>{campaign_text}</strong>
      <small>{html_escape(details)}</small>
    </div>
  </div>
</section>
"""


def _initial_shell_state(*, audit_id: str, audit_args: list[str], campaign_label: str) -> dict[str, Any]:
    return {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "campaign_label": campaign_label,
        "audit": {
            "status": "idle",
            "badge": "idle",
            "badge_tone": "info",
            "audit_id": audit_id,
            "audit_args": list(audit_args),
            "message": "No audit has been run in this session yet.",
            "progress_active": False,
            "progress_label": "",
            "progress_current": 0,
            "progress_total": 0,
            "progress_value_text": "",
            "progress_indeterminate": False,
        },
        "optimization": {
            "status": "loading",
            "badge": "loading",
            "badge_tone": "running",
            "message": (
                "Preparing optimization view"
                if campaign_label and campaign_label != "no active optimization campaign detected"
                else "No active optimization campaign detected yet."
            ),
        },
        "docs": {
            "status": "ready",
            "badge": "ready",
            "badge_tone": "info",
            "message": "Rendered maintained docs portal.",
        },
        "tabs": {
            "audits": {
                "badge": "starting",
                "badge_tone": "running",
                "src": "/audits/index.html",
                "revision": "starting",
            },
            "optimization": {
                "badge": "loading",
                "badge_tone": "running",
                "src": "/optimization/index.html",
                "revision": "loading",
            },
            "docs": {
                "badge": "ready",
                "badge_tone": "info",
                "src": "/docs/index.html",
                "revision": "maintained-docs",
            },
        },
        "progress": {
            "active": False,
            "label": "",
            "value_text": "",
            "fraction": 0.0,
            "indeterminate": False,
        },
        "available_audits": _available_audit_entries(),
    }


def _write_control_center_shell(
    root_dir: Path,
    *,
    campaign_label: str,
    campaign_path_text: str | None,
    audit_id: str,
    audit_args: list[str],
    shell_state: dict[str, Any],
    dev_reload: bool = False,
) -> None:
    shell_module = importlib.import_module("neuroinfra.dashboard.shell")
    shell_html = shell_module.render_dashboard_shell(
        title="OlfactoryBulb Control Center",
        subtitle="docs, audits, and optimization in one maintained shell",
        tabs=_module_tabs(),
        initial_tab="audits",
        panel_toolbar_html_by_key={
            "audits": _render_audit_runner_panel(
                audit_id=audit_id,
                audit_args=audit_args,
            ),
            "optimization": _render_optimization_context_panel(
                campaign_label=campaign_label,
                campaign_path_text=campaign_path_text,
            ),
        },
        shell_state=shell_state,
        state_endpoint="/__control_center_state__",
        state_poll_interval_ms=DEFAULT_STATE_POLL_INTERVAL_MS,
    )
    if dev_reload:
        shell_html = _inject_dev_reload_script(shell_html)
    _write_text_atomic(root_dir / "index.html", shell_html)


def _read_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_audit_args_text(text: str) -> list[str]:
    return shlex.split(str(text or "").strip())


def _audit_history_path(audits_dir: Path) -> Path:
    return audits_dir / "history.json"


def _audit_history_entry_key(audit_id: str, audit_args: list[str]) -> str:
    return json.dumps({"audit_id": str(audit_id), "audit_args": list(audit_args)}, sort_keys=True)


def _audit_history_group_id(entry_index: int, audit_id: str, audit_args: list[str]) -> str:
    slug = str(audit_id or "audit").replace("/", "_").replace(" ", "_")
    return f"{slug}-{entry_index:03d}"


def _audit_registry_title(audit_id: str) -> str:
    normalized_id = str(audit_id or "").strip()
    if not normalized_id:
        return ""
    for entry in _available_audit_entries():
        if str(entry.get("audit_id") or "").strip() == normalized_id:
            return str(entry.get("title") or "").strip()
    return ""


def _audit_history_group_title(audit_id: str, audit_args: list[str]) -> str:
    title = _audit_registry_title(audit_id) or str(audit_id or "").strip()
    args_text = " ".join(str(arg) for arg in audit_args if str(arg).strip()).strip()
    return f"{title} ({args_text})".strip() if args_text else title


def _format_summary_brief(summary: Any) -> str:
    if not isinstance(summary, dict):
        return ""
    ordered_keys = ("FAIL", "WARN", "PASS", "SKIP", "INFO")
    parts: list[str] = []
    seen: set[str] = set()
    for key in ordered_keys:
        value = summary.get(key)
        seen.add(key)
        if value in (None, "", 0):
            continue
        parts.append(f"{key.lower()} {value}")
    for key, value in summary.items():
        normalized_key = str(key)
        if normalized_key in seen or value in (None, "", 0):
            continue
        parts.append(f"{normalized_key.lower()} {value}")
    return ", ".join(parts)


def _load_audit_history(audits_dir: Path) -> list[dict[str, Any]]:
    payload = _read_json_dict(_audit_history_path(audits_dir))
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return []
    return [dict(entry) for entry in entries if isinstance(entry, dict)]


def _write_audit_history(audits_dir: Path, entries: list[dict[str, Any]]) -> None:
    _write_json_atomic(
        _audit_history_path(audits_dir),
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "entries": entries,
        },
    )


def _combine_audit_history(entries: list[dict[str, Any]]) -> AuditReport:
    combined_items: list[AuditItem] = []
    for index, entry in enumerate(entries, start=1):
        report_payload = entry.get("report") if isinstance(entry.get("report"), dict) else {}
        item_payloads = report_payload.get("items") if isinstance(report_payload, dict) else None
        if not isinstance(item_payloads, list):
            continue
        entry_audit_args = list(entry.get("audit_args") or [])
        entry_group_id = str(entry.get("group_id") or _audit_history_group_id(index, str(entry.get("audit_id") or "audit"), list(entry.get("audit_args") or [])))
        entry_group_title = str(entry.get("group_title") or _audit_history_group_title(str(entry.get("audit_id") or "audit"), entry_audit_args))
        nested_group_count = len(report_payload.get("groups") or []) if isinstance(report_payload.get("groups"), list) else 0
        for item_payload in item_payloads:
            if not isinstance(item_payload, dict):
                continue
            item = AuditItem(**item_payload)
            nested_group_id = str(item.group_id or report_payload.get("audit_id") or "items")
            nested_group_title = str(item.group_title or item.title or nested_group_id)
            if nested_group_count > 1:
                display_title = nested_group_title
            else:
                report_title = str(report_payload.get("title") or "").strip()
                display_title = entry_group_title if entry_audit_args else (report_title or entry_group_title or nested_group_title)
            combined_group_id = f"{entry_group_id}.{nested_group_id}"
            combined_items.append(
                AuditItem(
                    check_id=f"{combined_group_id}.{item.check_id}",
                    status=item.status,
                    title=item.title,
                    criterion=item.criterion,
                    description=item.description,
                    acceptable=item.acceptable,
                    acceptable_basis=item.acceptable_basis,
                    evidence=item.evidence,
                    series_visuals=item.series_visuals,
                    companion_visuals=item.companion_visuals,
                    note=item.note,
                    human_review_status=item.human_review_status,
                    human_review_note=item.human_review_note,
                    human_review_reviewer=item.human_review_reviewer,
                    group_id=combined_group_id,
                    group_title=display_title,
                    detail_level=item.detail_level,
                )
            )
    if not combined_items:
        return AuditReport(audit_id="control_center_audits", title="Control center audits", items=[])
    return AuditReport(audit_id="control_center_audits", title="Control center audits", items=combined_items)


def _append_or_replace_audit_history_entry(
    audits_dir: Path,
    *,
    audit_id: str,
    audit_args: list[str],
    report: AuditReport,
) -> list[dict[str, Any]]:
    entries = _load_audit_history(audits_dir)
    key = _audit_history_entry_key(audit_id, audit_args)
    entry = {
        "entry_key": key,
        "audit_id": str(audit_id),
        "audit_args": list(audit_args),
        "group_title": _audit_history_group_title(audit_id, audit_args),
        "report": report.to_dict(),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    replaced = False
    for index, existing in enumerate(entries):
        if str(existing.get("entry_key") or "") == key:
            entries[index] = {**existing, **entry}
            replaced = True
            break
    if not replaced:
        entries.append(entry)
    for index, existing in enumerate(entries, start=1):
        existing["group_id"] = _audit_history_group_id(index, str(existing.get("audit_id") or "audit"), list(existing.get("audit_args") or []))
        existing["group_title"] = _audit_history_group_title(str(existing.get("audit_id") or "audit"), list(existing.get("audit_args") or []))
    _write_audit_history(audits_dir, entries)
    return entries


def _compose_control_center_state(
    *,
    base_state: dict[str, Any],
    audits_dir: Path,
    optimization_dir: Path,
) -> dict[str, Any]:
    state = json.loads(json.dumps(base_state))
    audit_manifest = _read_json_dict(audits_dir / "manifest.json")
    audit_history = _load_audit_history(audits_dir)
    optimization_manifest = _read_json_dict(optimization_dir / "manifest.json")

    audit_state = state.setdefault("audit", {})
    if audit_state.get("status") not in {"running", "starting", "error", "idle"}:
        if audit_manifest:
            latest_history = audit_history[-1] if audit_history else {}
            latest_label = str(latest_history.get("group_title") or audit_state.get("audit_id") or audit_manifest.get("title") or "")
            summary_brief = _format_summary_brief(audit_manifest.get("summary"))
            message = f"{latest_label}: {summary_brief}".strip(": ") if summary_brief else latest_label
            audit_state.update(
                {
                    "status": "ready",
                    "badge": str(audit_manifest.get("worst_status") or "ready"),
                    "badge_tone": _badge_tone(str(audit_manifest.get("worst_status") or "READY")),
                    "message": message,
                    "summary": audit_manifest.get("summary") or {},
                    "worst_status": audit_manifest.get("worst_status") or "PASS",
                    "generated_at": audit_manifest.get("generated_at") or "",
                    "revision": str(audit_manifest.get("manifest_revision") or audit_manifest.get("generated_at") or ""),
                }
            )
        elif audit_state.get("status") != "error":
            audit_state.update(
                {
                    "status": "idle",
                    "badge": "idle",
                    "badge_tone": "info",
                }
            )

    optimization_state = state.setdefault("optimization", {})
    if optimization_state.get("status") not in {"loading", "error"} and optimization_manifest:
        placeholder = bool(optimization_manifest.get("placeholder"))
        if placeholder:
            optimization_state.update(
                {
                    "status": "unavailable",
                    "badge": "unavailable",
                    "badge_tone": "warn",
                    "message": str(optimization_manifest.get("reason") or "Optimization dashboard unavailable."),
                    "revision": str(optimization_manifest.get("manifest_revision") or optimization_manifest.get("generated_at") or ""),
                }
            )
        else:
            packet_count = int(optimization_manifest.get("packet_count") or 0)
            optimization_state.update(
                {
                    "status": "ready",
                    "badge": f"{packet_count} packets",
                    "badge_tone": "info",
                    "message": (
                        f"{packet_count} packets from {optimization_manifest.get('candidate_rows', 0)} candidate rows"
                    ),
                    "packet_count": packet_count,
                    "generated_at": optimization_manifest.get("generated_at") or "",
                    "revision": str(optimization_manifest.get("manifest_revision") or optimization_manifest.get("generated_at") or ""),
                }
            )

    state["tabs"] = {
        "audits": {
            "badge": str(audit_state.get("badge") or audit_state.get("status") or ""),
            "badge_tone": str(audit_state.get("badge_tone") or _badge_tone(str(audit_state.get("badge") or audit_state.get("status") or ""))),
            "src": "/audits/index.html",
            "revision": str(audit_state.get("revision") or audit_state.get("generated_at") or audit_state.get("status") or ""),
        },
        "optimization": {
            "badge": str(optimization_state.get("badge") or optimization_state.get("status") or ""),
            "badge_tone": str(
                optimization_state.get("badge_tone")
                or _badge_tone(str(optimization_state.get("badge") or optimization_state.get("status") or ""))
            ),
            "src": "/optimization/index.html",
            "revision": str(
                optimization_state.get("revision") or optimization_state.get("generated_at") or optimization_state.get("status") or ""
            ),
        },
        "docs": {
            "badge": str(state.get("docs", {}).get("badge") or "ready"),
            "badge_tone": str(state.get("docs", {}).get("badge_tone") or "info"),
            "src": "/docs/index.html",
            "revision": "maintained-docs",
        },
    }
    audit_status = str(audit_state.get("status") or "")
    progress_active = bool(audit_state.get("progress_active")) or audit_status in {"running", "starting"}
    progress_current = int(audit_state.get("progress_current") or 0)
    progress_total = int(audit_state.get("progress_total") or 0)
    progress_fraction = (progress_current / progress_total) if progress_total > 0 else 0.0
    state["progress"] = {
        "active": progress_active,
        "label": str(audit_state.get("progress_label") or audit_state.get("message") or ""),
        "value_text": str(audit_state.get("progress_value_text") or (f"{progress_current}/{progress_total}" if progress_total > 0 else "")),
        "fraction": progress_fraction,
        "indeterminate": bool(audit_state.get("progress_indeterminate")) or progress_total <= 0,
    }
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    return state


def resolve_control_center_campaign(
    campaign_dir: str | Path | None = None,
    *,
    status_json: str | Path | None = None,
) -> Path | None:
    if campaign_dir is not None:
        path = Path(campaign_dir).expanduser().resolve()
        return path if path.exists() else None
    status_path = Path(status_json).expanduser().resolve() if status_json is not None else DEFAULT_STATUS_JSON
    status_payload = _read_json_dict(status_path) if status_path.exists() else {}
    status_campaign = str(status_payload.get("campaign_dir") or "").strip()
    if status_campaign:
        candidate = Path(status_campaign).expanduser().resolve()
        if candidate.exists():
            return candidate
    if DEFAULT_OPTIMIZATION_ROOT.exists():
        candidates = [
            path
            for path in DEFAULT_OPTIMIZATION_ROOT.iterdir()
            if path.is_dir()
            and (path / "candidate_archive.jsonl").exists()
            and path.name != "codex_big_hfo_logs"
        ]
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)
    return None


def _write_optimization_placeholder(output_dir: Path, *, reason: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Optimization dashboard unavailable</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fb;
      --ink: #17202a;
      --muted: #667085;
      --line: #d9dee8;
      --panel: #ffffff;
      --blue: #2563eb;
      --shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 900px;
      margin: 0 auto;
      padding: 48px 24px 64px;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 20px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    p {{ margin: 0 0 14px; color: var(--muted); }}
    code {{
      padding: 2px 6px;
      border-radius: 6px;
      background: #f8fafc;
      border: 1px solid #e6eaf1;
      color: var(--ink);
    }}
  </style>
</head>
<body>
  <main>
    <section>
      <h1>Optimization dashboard unavailable</h1>
      <p>{reason}</p>
      <p>Launch the control center with an explicit campaign path if needed:</p>
      <p><code>python -m olfactorybulb.dashboard.control_center /path/to/campaign</code></p>
    </section>
  </main>
</body>
</html>
"""
    _write_text_atomic(output_dir / "index.html", html)
    manifest = {
        "campaign_dir": None,
        "output_dir": str(output_dir),
        "index_html": str(output_dir / "index.html"),
        "entrypoint_html": str(output_dir / "index.html"),
        "packet_count": 0,
        "candidate_rows": 0,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "manifest_revision": time.time_ns(),
        "placeholder": True,
        "reason": reason,
    }
    _write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def export_control_center(
    campaign_dir: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    audit_id: str = DEFAULT_AUDIT_ID,
    audit_args: list[str] | None = None,
    run_audit_on_start: bool = False,
    top_n: int = DEFAULT_CONTROL_CENTER_TOP_N,
    refresh_s: float = hfo_dashboard.DEFAULT_REFRESH_S,
    generate_packets_top_n: int = DEFAULT_CONTROL_CENTER_GENERATE_PACKETS_TOP_N,
    generate_packet_workers: int = DEFAULT_CONTROL_CENTER_GENERATE_PACKET_WORKERS,
    cleanup_stale_packets_before_render: bool = DEFAULT_CONTROL_CENTER_CLEANUP_STALE_PACKETS,
    status_json: str | Path | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    log = _progress if progress else (lambda _message: None)
    resolved_audit_args = list(_default_audit_args_for(audit_id) if audit_args is None else audit_args)
    campaign_path = resolve_control_center_campaign(campaign_dir, status_json=status_json)
    root_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else DEFAULT_OUTPUT_DIR.resolve()
    root_dir.mkdir(parents=True, exist_ok=True)
    optimization_dir = root_dir / "optimization"
    audits_dir = root_dir / "audits"
    optimization_dir.mkdir(parents=True, exist_ok=True)
    audits_dir.mkdir(parents=True, exist_ok=True)

    if campaign_path is None:
        log("no active optimization campaign detected; using placeholder optimization tab")
        optimization_manifest = _write_optimization_placeholder(
            optimization_dir,
            reason="No active optimization campaign could be auto-detected from the maintained status file or optimization results directory.",
        )
        campaign_label = "no active optimization campaign detected"
        campaign_path_text = None
    else:
        log(f"rendering optimization dashboard from {campaign_path}")
        optimization_manifest = hfo_dashboard.export_visual_dashboard(
            campaign_path,
            output_dir=optimization_dir,
            top_n=top_n,
            refresh_s=refresh_s,
            generate_packets_top_n=generate_packets_top_n,
            generate_packet_workers=generate_packet_workers,
            cleanup_stale_packets_before_render=cleanup_stale_packets_before_render,
            status_json=status_json,
            asset_url_prefix="/repo",
        )
        campaign_label = campaign_path.name
        campaign_path_text = str(campaign_path)
    _write_audit_history(audits_dir, [])
    if run_audit_on_start:
        log(f"running audit {audit_id} {' '.join(resolved_audit_args)}".rstrip())
        audit_report = run_audit_by_id(audit_id, resolved_audit_args)
        history_entries = _append_or_replace_audit_history_entry(
            audits_dir,
            audit_id=audit_id,
            audit_args=resolved_audit_args,
            report=audit_report,
        )
        combined_report = _combine_audit_history(history_entries)
    else:
        combined_report = AuditReport(
            audit_id="control_center_audits",
            title="Control center audits",
            items=[],
        )
    audit_manifest = export_audit_dashboard(
        combined_report,
        audits_dir,
        refresh_endpoint="/__audit_refresh__",
    )
    log("writing unified dashboard shell")
    shell_state = _compose_control_center_state(
        base_state=_initial_shell_state(
            audit_id=audit_id,
            audit_args=resolved_audit_args,
            campaign_label=campaign_label,
        ),
        audits_dir=audits_dir,
        optimization_dir=optimization_dir,
    )
    if not run_audit_on_start:
        shell_state["audit"].update(
            {
                "status": "idle",
                "badge": "idle",
                "badge_tone": "info",
                "message": "No audit has been run in this session yet.",
                "summary": {},
                "worst_status": "",
                "generated_at": "",
                "progress_active": False,
                "progress_label": "",
                "progress_current": 0,
                "progress_total": 0,
                "progress_value_text": "",
                "progress_indeterminate": False,
            }
        )
        shell_state = _compose_control_center_state(
            base_state=shell_state,
            audits_dir=audits_dir,
            optimization_dir=optimization_dir,
        )

    _write_control_center_shell(
        root_dir,
        campaign_label=campaign_label,
        campaign_path_text=campaign_path_text,
        audit_id=audit_id,
        audit_args=resolved_audit_args,
        shell_state=shell_state,
    )
    manifest = {
        "campaign_dir": str(campaign_path) if campaign_path is not None else None,
        "output_dir": str(root_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "audit_id": audit_id,
        "audit_args": resolved_audit_args,
        "audits": audit_manifest,
        "optimization": optimization_manifest,
        "docs_root": str(DOCS_ROOT),
        "state": shell_state,
    }
    _write_json_atomic(root_dir / "manifest.json", manifest)
    _write_json_atomic(root_dir / "state.json", shell_state)
    return manifest


def _safe_static_path(base_dir: Path, relative_path: str) -> Path:
    base = base_dir.resolve()
    relative_clean = relative_path.lstrip("/")
    candidate = (base / relative_clean).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise FileNotFoundError(relative_path) from exc
    return candidate


def _bind_control_center_server(
    host: str,
    port: int,
    handler_cls: type[http.server.BaseHTTPRequestHandler],
    *,
    fallback_attempts: int = 20,
) -> tuple[http.server.ThreadingHTTPServer, int]:
    requested_port = int(port)
    if requested_port == 0:
        server = http.server.ThreadingHTTPServer((host, 0), handler_cls)
        return server, int(server.server_port)

    last_error: OSError | None = None
    for offset in range(max(int(fallback_attempts), 1)):
        candidate_port = requested_port + offset
        try:
            server = http.server.ThreadingHTTPServer((host, candidate_port), handler_cls)
            return server, int(server.server_port)
        except OSError as exc:
            last_error = exc
            if exc.errno != errno.EADDRINUSE:
                raise
    assert last_error is not None
    raise last_error


def serve_control_center(
    campaign_dir: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    audit_id: str = DEFAULT_AUDIT_ID,
    audit_args: list[str] | None = None,
    run_audit_on_start: bool = False,
    top_n: int = DEFAULT_CONTROL_CENTER_TOP_N,
    refresh_s: float = hfo_dashboard.DEFAULT_REFRESH_S,
    generate_packets_top_n: int = DEFAULT_CONTROL_CENTER_GENERATE_PACKETS_TOP_N,
    generate_packet_workers: int = DEFAULT_CONTROL_CENTER_GENERATE_PACKET_WORKERS,
    cleanup_stale_packets_before_render: bool = DEFAULT_CONTROL_CENTER_CLEANUP_STALE_PACKETS,
    status_json: str | Path | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    watch_optimization: bool = True,
    open_browser: bool = False,
    dev_reload: bool = True,
) -> None:
    resolved_audit_args = list(_default_audit_args_for(audit_id) if audit_args is None else audit_args)
    campaign_path = resolve_control_center_campaign(campaign_dir, status_json=status_json)
    root_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else DEFAULT_OUTPUT_DIR.resolve()
    requested_port = int(port)
    requested_url = f"http://{host}:{requested_port}/" if requested_port else f"http://{host}:<auto>/"
    _progress(f"preparing control center at {requested_url}")
    root_dir.mkdir(parents=True, exist_ok=True)
    optimization_dir = root_dir / "optimization"
    audits_dir = root_dir / "audits"
    state_path = root_dir / "state.json"
    optimization_dir.mkdir(parents=True, exist_ok=True)
    audits_dir.mkdir(parents=True, exist_ok=True)
    campaign_label = campaign_path.name if campaign_path is not None else "no active optimization campaign detected"
    campaign_path_text = str(campaign_path) if campaign_path is not None else None
    base_state_lock = threading.RLock()
    base_state = _initial_shell_state(
        audit_id=audit_id,
        audit_args=resolved_audit_args,
        campaign_label=campaign_label,
    )
    audit_thread_holder: dict[str, threading.Thread | None] = {"thread": None}
    audit_thread_lock = threading.Lock()
    dev_source_paths = _dev_reload_source_paths() if dev_reload else []
    dev_reload_state = {"revision": _source_revision(dev_source_paths) if dev_reload else ""}

    def _refresh_state_file() -> dict[str, Any]:
        with base_state_lock:
            composed = _compose_control_center_state(
                base_state=base_state,
                audits_dir=audits_dir,
                optimization_dir=optimization_dir,
            )
        _write_json_atomic(state_path, composed)
        return composed

    def _update_base_state(*, audit_patch: dict[str, Any] | None = None, optimization_patch: dict[str, Any] | None = None, docs_patch: dict[str, Any] | None = None) -> dict[str, Any]:
        with base_state_lock:
            if audit_patch:
                base_state.setdefault("audit", {}).update(audit_patch)
            if optimization_patch:
                base_state.setdefault("optimization", {}).update(optimization_patch)
            if docs_patch:
                base_state.setdefault("docs", {}).update(docs_patch)
            return _refresh_state_file()

    export_audit_dashboard(
        AuditReport(audit_id="control_center_audits", title="Control center audits", items=[]),
        audits_dir,
        refresh_endpoint="/__audit_refresh__",
    )
    _write_audit_history(audits_dir, [])
    _write_loading_frame(
        optimization_dir,
        title="Optimization dashboard starting",
        message=(
            f"Preparing optimization view for {campaign_label}"
            if campaign_path is not None
            else "No active optimization campaign was detected yet."
        ),
    )
    _write_control_center_shell(
        root_dir,
        campaign_label=campaign_label,
        campaign_path_text=campaign_path_text,
        audit_id=audit_id,
        audit_args=resolved_audit_args,
        shell_state=base_state,
        dev_reload=dev_reload,
    )
    _refresh_state_file()
    stop_event = threading.Event()
    watcher_holder: dict[str, threading.Thread | None] = {"thread": None}

    root_index = (root_dir / "index.html").resolve()

    def _run_audit_render(audit_id_to_run: str, audit_args_to_run: list[str]) -> None:
        run_token = time.time_ns()

        def _handle_audit_progress(update: dict[str, Any]) -> None:
            total = int(update.get("total") or 0)
            current = int(update.get("current") or 0)
            message = str(update.get("message") or f"Running {audit_id_to_run}").strip()
            _update_base_state(
                audit_patch={
                    "status": "running",
                    "badge": "running",
                    "badge_tone": "running",
                    "audit_id": audit_id_to_run,
                    "audit_args": list(audit_args_to_run),
                    "message": message,
                    "progress_active": True,
                    "progress_label": message,
                    "progress_current": current,
                    "progress_total": total,
                    "progress_value_text": f"{current}/{total}" if total > 0 else "",
                    "progress_indeterminate": total <= 0,
                    "revision": str(time.time_ns()),
                }
            )

        _update_base_state(
            audit_patch={
                "status": "running",
                "badge": "running",
                "badge_tone": "running",
                "audit_id": audit_id_to_run,
                "audit_args": list(audit_args_to_run),
                "message": f"Running {audit_id_to_run} {' '.join(audit_args_to_run)}".strip(),
                "summary": {},
                "worst_status": "",
                "generated_at": "",
                "progress_active": True,
                "progress_label": f"Running {audit_id_to_run}",
                "progress_current": 0,
                "progress_total": 0,
                "progress_value_text": "",
                "progress_indeterminate": True,
                "revision": str(run_token),
            }
        )
        try:
            report = run_audit_by_id(
                audit_id_to_run,
                audit_args_to_run,
                progress_callback=_handle_audit_progress,
            )
            history_entries = _append_or_replace_audit_history_entry(
                audits_dir,
                audit_id=audit_id_to_run,
                audit_args=audit_args_to_run,
                report=report,
            )
            combined_report = _combine_audit_history(history_entries)
            audit_manifest = export_audit_dashboard(
                combined_report,
                audits_dir,
                refresh_endpoint="/__audit_refresh__",
            )
            _update_base_state(
                audit_patch={
                    "status": "ready",
                    "badge": str(report.worst_status),
                    "badge_tone": _badge_tone(str(report.worst_status)),
                    "audit_id": audit_id_to_run,
                    "audit_args": list(audit_args_to_run),
                    "message": f"Completed {audit_id_to_run} with {report.worst_status}",
                    "summary": report.summary,
                    "worst_status": report.worst_status,
                    "generated_at": audit_manifest.get("generated_at") or "",
                    "progress_active": False,
                    "progress_label": "",
                    "progress_current": 0,
                    "progress_total": 0,
                    "progress_value_text": "",
                    "progress_indeterminate": False,
                    "revision": str(audit_manifest.get("manifest_revision") or audit_manifest.get("generated_at") or run_token),
                }
            )
        except BaseException as exc:
            error_text = str(exc) or exc.__class__.__name__
            _progress(f"audit render failed: {error_text}")
            _write_error_frame(
                audits_dir,
                title="Audit run failed",
                message=error_text,
            )
            _update_base_state(
                audit_patch={
                    "status": "error",
                    "badge": "error",
                    "badge_tone": "fail",
                    "audit_id": audit_id_to_run,
                    "audit_args": list(audit_args_to_run),
                    "message": error_text,
                    "progress_active": False,
                    "progress_label": "",
                    "progress_current": 0,
                    "progress_total": 0,
                    "progress_value_text": "",
                    "progress_indeterminate": False,
                    "revision": str(time.time_ns()),
                }
            )

    def _queue_audit_render(audit_id_to_run: str, audit_args_to_run: list[str]) -> tuple[bool, str]:
        with audit_thread_lock:
            active_thread = audit_thread_holder["thread"]
            if active_thread is not None and active_thread.is_alive():
                return False, "An audit run is already in progress."
            _update_base_state(
                audit_patch={
                    "status": "running",
                    "badge": "running",
                    "badge_tone": "running",
                    "audit_id": audit_id_to_run,
                    "audit_args": list(audit_args_to_run),
                    "message": f"Running {audit_id_to_run} {' '.join(audit_args_to_run)}".strip(),
                    "summary": {},
                    "worst_status": "",
                    "generated_at": "",
                    "progress_active": True,
                    "progress_label": f"Running {audit_id_to_run}",
                    "progress_current": 0,
                    "progress_total": 0,
                    "progress_value_text": "",
                    "progress_indeterminate": True,
                    "revision": str(time.time_ns()),
                }
            )
            thread = threading.Thread(
                target=_run_audit_render,
                args=(audit_id_to_run, list(audit_args_to_run)),
                name=f"control-center-audit-{audit_id_to_run}",
                daemon=True,
            )
            audit_thread_holder["thread"] = thread
            thread.start()
        return True, f"Running {audit_id_to_run}"

    def _maybe_refresh_dev_outputs() -> str:
        if not dev_reload:
            return ""
        current_revision = _source_revision(dev_source_paths)
        if current_revision == dev_reload_state.get("revision"):
            return current_revision
        importlib.invalidate_caches()
        shell_module = importlib.import_module("neuroinfra.dashboard.shell")
        importlib.reload(shell_module)
        audit_dashboard_module = importlib.import_module("olfactorybulb.audit.dashboard")
        reloaded_audit_dashboard = importlib.reload(audit_dashboard_module)
        globals()["export_audit_dashboard"] = reloaded_audit_dashboard.export_audit_dashboard
        current_state = _refresh_state_file()
        _write_control_center_shell(
            root_dir,
            campaign_label=campaign_label,
            campaign_path_text=campaign_path_text,
            audit_id=str(current_state.get("audit", {}).get("audit_id") or audit_id),
            audit_args=list(current_state.get("audit", {}).get("audit_args") or resolved_audit_args),
            shell_state=current_state,
            dev_reload=dev_reload,
        )
        history_entries = _load_audit_history(audits_dir)
        combined_report = _combine_audit_history(history_entries)
        export_audit_dashboard(
            combined_report,
            audits_dir,
            refresh_endpoint="/__audit_refresh__",
        )
        refreshed_revision = _source_revision(dev_source_paths)
        dev_reload_state["revision"] = refreshed_revision
        _progress("dev reload refreshed dashboard shell")
        return refreshed_revision

    class ControlCenterRequestHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("directory", str(REPO_ROOT))
            super().__init__(*args, **kwargs)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 - standard handler signature
            return

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 - HTTP handler API
            request_path = urlparse(self.path).path
            if request_path == "/__control_center_state__":
                self._send_json(200, _refresh_state_file())
                return
            if request_path == "/__control_center_dev_state__":
                self._send_json(200, {"ok": True, "revision": _maybe_refresh_dev_outputs()})
                return
            _maybe_refresh_dev_outputs()
            super().do_GET()

        def translate_path(self, path: str) -> str:
            parsed_path = posixpath.normpath(unquote(urlparse(path).path))
            if parsed_path in {"/", "/index.html"}:
                return str(root_index)
            if parsed_path.startswith("/audits/"):
                relative = parsed_path[len("/audits/") :]
                return str(_safe_static_path(audits_dir, relative))
            if parsed_path.startswith("/optimization/"):
                relative = parsed_path[len("/optimization/") :]
                return str(_safe_static_path(optimization_dir, relative))
            if parsed_path.startswith("/docs/"):
                relative = parsed_path[len("/docs/") :]
                return str(_safe_static_path(DOCS_ROOT, relative))
            if parsed_path.startswith("/repo/"):
                relative = parsed_path[len("/repo/") :]
                return str(_safe_static_path(REPO_ROOT, relative))
            return str(_safe_static_path(REPO_ROOT, parsed_path.lstrip("/")))

        def do_POST(self) -> None:  # noqa: N802 - HTTP handler API
            request_path = urlparse(self.path).path
            try:
                content_length = int(self.headers.get("Content-Length") or "0")
            except (TypeError, ValueError):
                content_length = 0
            raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                payload = json.loads(raw_body.decode("utf-8") or "{}")
            except json.JSONDecodeError as exc:
                self._send_json(400, {"ok": False, "error": f"Invalid JSON body: {exc}"})
                return
            if request_path == hfo_dashboard.GENERATE_PACKET_ENDPOINT:
                candidate_id = str((payload or {}).get("candidate_id") or "").strip()
                if not candidate_id:
                    self._send_json(400, {"ok": False, "error": "Missing candidate_id"})
                    return
                if campaign_path is None:
                    self._send_json(400, {"ok": False, "error": "No optimization campaign is available for packet generation"})
                    return
                try:
                    result = hfo_dashboard._queue_dashboard_packet_generation(  # type: ignore[attr-defined]
                        campaign_path,
                        candidate_id,
                        packet_output_dir=campaign_path / "figures" / f"packet_{candidate_id}",
                        output_dir=optimization_dir,
                        top_n=top_n,
                        refresh_s=refresh_s,
                        generate_packets_top_n=generate_packets_top_n,
                        generate_packet_workers=generate_packet_workers,
                        cleanup_stale_packets_before_render=cleanup_stale_packets_before_render,
                        status_json=status_json,
                        asset_url_prefix="/repo",
                    )
                except Exception as exc:  # pragma: no cover - exercised through integration, not unit tests
                    self._send_json(500, {"ok": False, "candidate_id": candidate_id, "error": str(exc)})
                    return
                self._send_json(202, result)
                return
            if request_path == "/__audit_run__":
                requested_audit_id = str((payload or {}).get("audit_id") or "").strip() or DEFAULT_AUDIT_ID
                audit_args_text = str((payload or {}).get("audit_args_text") or "")
                requested_args = _parse_audit_args_text(audit_args_text)
                ok, message = _queue_audit_render(requested_audit_id, requested_args)
                if not ok:
                    self._send_json(409, {"ok": False, "error": message, "state": _refresh_state_file()})
                    return
                self._send_json(
                    202,
                    {
                        "ok": True,
                        "message": message,
                        "audit_id": requested_audit_id,
                        "audit_args": requested_args,
                        "state": _refresh_state_file(),
                    },
                )
                return
            if request_path == "/__audit_refresh__":
                current_state = _refresh_state_file()
                current_audit_state = current_state.get("audit", {})
                requested_audit_id = str(current_audit_state.get("audit_id") or audit_id)
                requested_args = list(current_audit_state.get("audit_args") or resolved_audit_args)
                ok, message = _queue_audit_render(requested_audit_id, requested_args)
                if not ok:
                    self._send_json(409, {"ok": False, "error": message, "state": current_state})
                    return
                self._send_json(
                    202,
                    {
                        "ok": True,
                        "message": message,
                        "audit_id": requested_audit_id,
                        "audit_args": requested_args,
                        "state": _refresh_state_file(),
                    },
                )
                return
            self.send_error(404, "Not found")

    server, bound_port = _bind_control_center_server(host, requested_port, ControlCenterRequestHandler)
    if bound_port != requested_port and requested_port != 0:
        _progress(f"port {requested_port} is busy; using http://{host}:{bound_port}/ instead")
    url = f"http://{host}:{bound_port}/"
    _progress(f"ready at {url}")
    if open_browser:
        try:
            webbrowser.open(url, new=2, autoraise=True)
        except Exception:
            pass

    def _render_initial_content() -> None:
        try:
            if campaign_path is None:
                _progress("no active optimization campaign detected; using placeholder optimization tab")
                optimization_manifest = _write_optimization_placeholder(
                    optimization_dir,
                    reason="No active optimization campaign could be auto-detected from the maintained status file or optimization results directory.",
                )
                _update_base_state(
                    optimization_patch={
                        "status": "unavailable",
                        "badge": "unavailable",
                        "badge_tone": "warn",
                        "message": str(optimization_manifest.get("reason") or ""),
                        "revision": str(optimization_manifest.get("manifest_revision") or optimization_manifest.get("generated_at") or ""),
                    }
                )
            else:
                _progress(f"rendering optimization dashboard from {campaign_path}")
                optimization_manifest = hfo_dashboard.export_visual_dashboard(
                    campaign_path,
                    output_dir=optimization_dir,
                    top_n=top_n,
                    refresh_s=refresh_s,
                    generate_packets_top_n=generate_packets_top_n,
                    generate_packet_workers=generate_packet_workers,
                    cleanup_stale_packets_before_render=cleanup_stale_packets_before_render,
                    status_json=status_json,
                    asset_url_prefix="/repo",
                )
                packet_count = int(optimization_manifest.get("packet_count") or 0)
                _update_base_state(
                    optimization_patch={
                        "status": "ready",
                        "badge": f"{packet_count} packets",
                        "badge_tone": "info",
                        "message": f"{packet_count} packets from {optimization_manifest.get('candidate_rows', 0)} candidate rows",
                        "generated_at": optimization_manifest.get("generated_at") or "",
                        "revision": str(optimization_manifest.get("manifest_revision") or optimization_manifest.get("generated_at") or ""),
                    }
                )
            if run_audit_on_start:
                ok, _message = _queue_audit_render(audit_id, resolved_audit_args)
                if not ok:
                    raise RuntimeError(_message)
            if watch_optimization and campaign_path is not None and not bool(optimization_manifest.get("placeholder")):
                watcher_thread = threading.Thread(
                    target=hfo_dashboard.watch_visual_dashboard,
                    kwargs={
                        "campaign_dir": campaign_path,
                        "output_dir": optimization_dir,
                        "top_n": top_n,
                        "refresh_s": refresh_s,
                        "generate_packets_top_n": generate_packets_top_n,
                        "generate_packet_workers": generate_packet_workers,
                        "cleanup_stale_packets_before_render": cleanup_stale_packets_before_render,
                        "status_json": status_json,
                        "asset_url_prefix": "/repo",
                        "stop_event": stop_event,
                    },
                    name="control-center-optimization-watch",
                    daemon=True,
                )
                watcher_holder["thread"] = watcher_thread
                watcher_thread.start()
        except Exception as exc:
            _progress(f"startup render failed: {exc}")
            _write_error_frame(
                audits_dir,
                title="Audit startup failed",
                message=str(exc),
            )
            _write_error_frame(
                optimization_dir,
                title="Optimization startup failed",
                message=str(exc),
            )
            _update_base_state(
                audit_patch={
                    "status": "error",
                    "badge": "error",
                    "badge_tone": "fail",
                    "message": str(exc),
                    "revision": str(time.time_ns()),
                },
                optimization_patch={
                    "status": "error",
                    "badge": "error",
                    "badge_tone": "fail",
                    "message": str(exc),
                    "revision": str(time.time_ns()),
                },
            )

    render_thread = threading.Thread(
        target=_render_initial_content,
        name="control-center-initial-render",
        daemon=True,
    )
    render_thread.start()
    try:
        server.serve_forever()
    finally:
        stop_event.set()
        if render_thread.is_alive():
            render_thread.join(timeout=max(float(refresh_s), 1.0) + 2.0)
        audit_thread = audit_thread_holder["thread"]
        if audit_thread is not None:
            audit_thread.join(timeout=max(float(refresh_s), 1.0) + 2.0)
        watcher_thread = watcher_holder["thread"]
        if watcher_thread is not None:
            watcher_thread.join(timeout=max(float(refresh_s), 1.0) + 2.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        default="serve",
        choices=("export", "serve"),
        help="Export the unified dashboard shell or export and serve it locally.",
    )
    parser.add_argument(
        "campaign_dir",
        nargs="?",
        default="",
        help="Campaign directory to use for the optimization dashboard. Omit to auto-detect the active maintained campaign.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Control-center output directory.")
    parser.add_argument("--audit-id", default=DEFAULT_AUDIT_ID, help="Audit id to render inside the audit dashboard.")
    parser.add_argument("--top-n", type=int, default=DEFAULT_CONTROL_CENTER_TOP_N)
    parser.add_argument("--refresh-s", type=float, default=hfo_dashboard.DEFAULT_REFRESH_S)
    parser.add_argument("--generate-packets-top-n", type=int, default=DEFAULT_CONTROL_CENTER_GENERATE_PACKETS_TOP_N)
    parser.add_argument("--generate-packet-workers", type=int, default=DEFAULT_CONTROL_CENTER_GENERATE_PACKET_WORKERS)
    parser.add_argument("--no-cleanup-stale-packets", action="store_true")
    parser.add_argument("--status-json", default="")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-watch-optimization", action="store_true")
    parser.add_argument("--run-audit-on-start", action="store_true", help="Run the selected audit immediately instead of leaving the audit tab idle on startup.")
    parser.add_argument("--open-browser", action="store_true", help="Open the control center in a local browser after startup.")
    parser.add_argument("--no-dev-reload", action="store_true", help="Disable local source polling and browser reload for dashboard UI development.")
    args, extra_args = parser.parse_known_args(argv)
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    campaign_arg = str(args.campaign_dir or "").strip() or None
    common_kwargs = {
        "output_dir": args.output_dir,
        "audit_id": str(args.audit_id),
        "audit_args": extra_args if extra_args else None,
        "run_audit_on_start": bool(args.run_audit_on_start),
        "top_n": int(args.top_n),
        "refresh_s": float(args.refresh_s),
        "generate_packets_top_n": int(args.generate_packets_top_n),
        "generate_packet_workers": int(args.generate_packet_workers),
        "cleanup_stale_packets_before_render": not bool(args.no_cleanup_stale_packets),
        "status_json": str(args.status_json or "") or None,
    }
    if args.mode == "export":
        export_control_center(campaign_arg, progress=True, **common_kwargs)
        return 0
    serve_control_center(
        campaign_arg,
        host=str(args.host),
        port=int(args.port),
        watch_optimization=not bool(args.no_watch_optimization),
        open_browser=bool(args.open_browser),
        dev_reload=not bool(args.no_dev_reload),
        **common_kwargs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
