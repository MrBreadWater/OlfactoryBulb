"""Unified dashboard shell for docs, audits, and optimization views."""

from __future__ import annotations

import argparse
import errno
import http.server
import json
import os
import posixpath
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from neuroinfra.dashboard import ShellTabSpec, render_dashboard_shell
from olfactorybulb.audit.cli import run_audit_by_id
from olfactorybulb.audit.dashboard import export_audit_dashboard
import tools.analysis.hfo_visual_dashboard as hfo_dashboard


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "dashboard" / "control_center"
DEFAULT_STATUS_JSON = (REPO_ROOT / hfo_dashboard.SUMMARY_STATUS_PATH).resolve()
DEFAULT_OPTIMIZATION_ROOT = REPO_ROOT / "results" / "notebook_runs" / "optimization"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 6006
DEFAULT_AUDIT_ID = "repo_health"
DEFAULT_AUDIT_ARGS = ["--profile", "maintained"]
DEFAULT_CONTROL_CENTER_TOP_N = hfo_dashboard.DEFAULT_TOP_N
DEFAULT_CONTROL_CENTER_GENERATE_PACKETS_TOP_N = 0
DEFAULT_CONTROL_CENTER_GENERATE_PACKET_WORKERS = hfo_dashboard.DEFAULT_PACKET_GENERATION_WORKERS
DEFAULT_CONTROL_CENTER_CLEANUP_STALE_PACKETS = False


def _progress(message: str) -> None:
    print(f"[control_center] {message}", flush=True)


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


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


def _module_tabs(*, audit_badge: str, optimization_badge: str) -> tuple[ShellTabSpec, ...]:
    return (
        ShellTabSpec(
            key="audits",
            label="Audits",
            src="/audits/index.html",
            description="Structured audit results, grouped summaries, and detailed findings.",
            badge=audit_badge,
        ),
        ShellTabSpec(
            key="optimization",
            label="Optimization",
            src="/optimization/index.html",
            description="Campaign-level HFO packet review and candidate ranking.",
            badge=optimization_badge,
        ),
        ShellTabSpec(
            key="docs",
            label="Docs",
            src="/docs/index.html",
            description="Maintained markdown docs and current operational guidance.",
        ),
    )


def _write_control_center_shell(
    root_dir: Path,
    *,
    campaign_label: str,
    audit_badge: str,
    optimization_badge: str,
) -> None:
    shell_html = render_dashboard_shell(
        title="OlfactoryBulb Control Center",
        subtitle=f"{campaign_label} | docs, audits, and optimization in one maintained shell",
        tabs=_module_tabs(
            audit_badge=audit_badge,
            optimization_badge=optimization_badge,
        ),
        initial_tab="audits",
    )
    _write_text_atomic(root_dir / "index.html", shell_html)


def _read_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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
    top_n: int = DEFAULT_CONTROL_CENTER_TOP_N,
    refresh_s: float = hfo_dashboard.DEFAULT_REFRESH_S,
    generate_packets_top_n: int = DEFAULT_CONTROL_CENTER_GENERATE_PACKETS_TOP_N,
    generate_packet_workers: int = DEFAULT_CONTROL_CENTER_GENERATE_PACKET_WORKERS,
    cleanup_stale_packets_before_render: bool = DEFAULT_CONTROL_CENTER_CLEANUP_STALE_PACKETS,
    status_json: str | Path | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    log = _progress if progress else (lambda _message: None)
    resolved_audit_args = list(DEFAULT_AUDIT_ARGS if audit_args is None else audit_args)
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
        optimization_badge = "unavailable"
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
        )
        campaign_label = str(campaign_path)
        optimization_badge = f"{int(optimization_manifest.get('packet_count', 0))} packets"
    log(f"running audit {audit_id} {' '.join(resolved_audit_args)}".rstrip())
    audit_report = run_audit_by_id(audit_id, resolved_audit_args)
    audit_manifest = export_audit_dashboard(
        audit_report,
        audits_dir,
        refresh_endpoint="/__audit_refresh__",
    )
    log("writing unified dashboard shell")

    _write_control_center_shell(
        root_dir,
        campaign_label=campaign_label,
        audit_badge=str(audit_report.worst_status),
        optimization_badge=optimization_badge,
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
    }
    _write_json_atomic(root_dir / "manifest.json", manifest)
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
) -> None:
    resolved_audit_args = list(DEFAULT_AUDIT_ARGS if audit_args is None else audit_args)
    campaign_path = resolve_control_center_campaign(campaign_dir, status_json=status_json)
    root_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else DEFAULT_OUTPUT_DIR.resolve()
    requested_port = int(port)
    requested_url = f"http://{host}:{requested_port}/" if requested_port else f"http://{host}:<auto>/"
    _progress(f"preparing control center at {requested_url}")
    root_dir.mkdir(parents=True, exist_ok=True)
    optimization_dir = root_dir / "optimization"
    audits_dir = root_dir / "audits"
    optimization_dir.mkdir(parents=True, exist_ok=True)
    audits_dir.mkdir(parents=True, exist_ok=True)
    campaign_label = str(campaign_path) if campaign_path is not None else "no active optimization campaign detected"
    _write_loading_frame(
        audits_dir,
        title="Audit starting",
        message=f"Running {audit_id} {' '.join(resolved_audit_args)}".strip(),
    )
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
        audit_badge="starting",
        optimization_badge="loading",
    )
    stop_event = threading.Event()
    watcher_holder: dict[str, threading.Thread | None] = {"thread": None}
    manifest_holder: dict[str, Any] = {}

    root_index = (root_dir / "index.html").resolve()

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
            return str(_safe_static_path(REPO_ROOT, parsed_path.lstrip("/")))

        def do_POST(self) -> None:  # noqa: N802 - HTTP handler API
            request_path = urlparse(self.path).path
            if request_path == hfo_dashboard.GENERATE_PACKET_ENDPOINT:
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
                    )
                except Exception as exc:  # pragma: no cover - exercised through integration, not unit tests
                    self._send_json(500, {"ok": False, "candidate_id": candidate_id, "error": str(exc)})
                    return
                self._send_json(202, result)
                return
            if request_path == "/__audit_refresh__":
                try:
                    report = run_audit_by_id(audit_id, resolved_audit_args)
                    audit_manifest = export_audit_dashboard(
                        report,
                        audits_dir,
                        refresh_endpoint="/__audit_refresh__",
                    )
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "audit_id": report.audit_id,
                            "summary": report.summary,
                            "worst_status": report.worst_status,
                            "manifest": audit_manifest,
                        },
                    )
                except Exception as exc:  # pragma: no cover - exercised through integration, not unit tests
                    self._send_json(500, {"ok": False, "error": str(exc)})
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
            manifest = export_control_center(
                campaign_path,
                output_dir=root_dir,
                audit_id=audit_id,
                audit_args=resolved_audit_args,
                top_n=top_n,
                refresh_s=refresh_s,
                generate_packets_top_n=generate_packets_top_n,
                generate_packet_workers=generate_packet_workers,
                cleanup_stale_packets_before_render=cleanup_stale_packets_before_render,
                status_json=status_json,
                progress=True,
            )
            manifest_holder.update(manifest)
            optimization_manifest = manifest.get("optimization") or {}
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
    parser.add_argument("--open-browser", action="store_true", help="Open the control center in a local browser after startup.")
    args, extra_args = parser.parse_known_args(argv)
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    campaign_arg = str(args.campaign_dir or "").strip() or None
    common_kwargs = {
        "output_dir": args.output_dir,
        "audit_id": str(args.audit_id),
        "audit_args": extra_args if extra_args else None,
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
        **common_kwargs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
