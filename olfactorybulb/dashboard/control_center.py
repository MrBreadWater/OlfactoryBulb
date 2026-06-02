"""Unified dashboard shell for docs, audits, and optimization views."""

from __future__ import annotations

import argparse
import http.server
import json
import os
import posixpath
import threading
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
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 6010


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


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


def export_control_center(
    campaign_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    audit_id: str = "new_sweep",
    audit_args: list[str] | None = None,
    top_n: int = hfo_dashboard.DEFAULT_TOP_N,
    refresh_s: float = hfo_dashboard.DEFAULT_REFRESH_S,
    generate_packets_top_n: int = hfo_dashboard.DEFAULT_GENERATE_PACKETS_TOP_N,
    generate_packet_workers: int = hfo_dashboard.DEFAULT_PACKET_GENERATION_WORKERS,
    cleanup_stale_packets_before_render: bool = hfo_dashboard.DEFAULT_CLEANUP_STALE_PACKETS,
    status_json: str | Path | None = None,
) -> dict[str, Any]:
    campaign_path = Path(campaign_dir).expanduser().resolve()
    root_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else DEFAULT_OUTPUT_DIR.resolve()
    root_dir.mkdir(parents=True, exist_ok=True)
    optimization_dir = root_dir / "optimization"
    audits_dir = root_dir / "audits"
    optimization_dir.mkdir(parents=True, exist_ok=True)
    audits_dir.mkdir(parents=True, exist_ok=True)

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
    audit_report = run_audit_by_id(audit_id, list(audit_args or []))
    audit_manifest = export_audit_dashboard(
        audit_report,
        audits_dir,
        refresh_endpoint="/__audit_refresh__",
    )

    shell_html = render_dashboard_shell(
        title="OlfactoryBulb Control Center",
        subtitle=f"{campaign_path} | docs, audits, and optimization in one maintained shell",
        tabs=_module_tabs(
            audit_badge=str(audit_report.worst_status),
            optimization_badge=f"{int(optimization_manifest.get('packet_count', 0))} packets",
        ),
        initial_tab="audits",
    )
    _write_text_atomic(root_dir / "index.html", shell_html)
    manifest = {
        "campaign_dir": str(campaign_path),
        "output_dir": str(root_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "audit_id": audit_id,
        "audit_args": list(audit_args or []),
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


def serve_control_center(
    campaign_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    audit_id: str = "new_sweep",
    audit_args: list[str] | None = None,
    top_n: int = hfo_dashboard.DEFAULT_TOP_N,
    refresh_s: float = hfo_dashboard.DEFAULT_REFRESH_S,
    generate_packets_top_n: int = hfo_dashboard.DEFAULT_GENERATE_PACKETS_TOP_N,
    generate_packet_workers: int = hfo_dashboard.DEFAULT_PACKET_GENERATION_WORKERS,
    cleanup_stale_packets_before_render: bool = hfo_dashboard.DEFAULT_CLEANUP_STALE_PACKETS,
    status_json: str | Path | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    watch_optimization: bool = True,
) -> None:
    campaign_path = Path(campaign_dir).expanduser().resolve()
    root_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else DEFAULT_OUTPUT_DIR.resolve()
    manifest = export_control_center(
        campaign_path,
        output_dir=root_dir,
        audit_id=audit_id,
        audit_args=list(audit_args or []),
        top_n=top_n,
        refresh_s=refresh_s,
        generate_packets_top_n=generate_packets_top_n,
        generate_packet_workers=generate_packet_workers,
        cleanup_stale_packets_before_render=cleanup_stale_packets_before_render,
        status_json=status_json,
    )
    optimization_dir = Path(manifest["optimization"]["output_dir"])
    audits_dir = Path(manifest["audits"]["output_dir"])
    stop_event = threading.Event()
    watcher_thread: threading.Thread | None = None
    if watch_optimization:
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
        watcher_thread.start()

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
                    report = run_audit_by_id(audit_id, list(audit_args or []))
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

    server = http.server.ThreadingHTTPServer((host, int(port)), ControlCenterRequestHandler)
    print(f"Serving control center at http://{host}:{int(port)}/", flush=True)
    try:
        server.serve_forever()
    finally:
        stop_event.set()
        if watcher_thread is not None:
            watcher_thread.join(timeout=max(float(refresh_s), 1.0) + 2.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("export", "serve"),
        help="Export the unified dashboard shell or export and serve it locally.",
    )
    parser.add_argument("campaign_dir", help="Campaign directory to use for the optimization dashboard.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Control-center output directory.")
    parser.add_argument("--audit-id", default="new_sweep", help="Audit id to render inside the audit dashboard.")
    parser.add_argument("--top-n", type=int, default=hfo_dashboard.DEFAULT_TOP_N)
    parser.add_argument("--refresh-s", type=float, default=hfo_dashboard.DEFAULT_REFRESH_S)
    parser.add_argument("--generate-packets-top-n", type=int, default=hfo_dashboard.DEFAULT_GENERATE_PACKETS_TOP_N)
    parser.add_argument("--generate-packet-workers", type=int, default=hfo_dashboard.DEFAULT_PACKET_GENERATION_WORKERS)
    parser.add_argument("--no-cleanup-stale-packets", action="store_true")
    parser.add_argument("--status-json", default="")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-watch-optimization", action="store_true")
    args, extra_args = parser.parse_known_args(argv)
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    common_kwargs = {
        "output_dir": args.output_dir,
        "audit_id": str(args.audit_id),
        "audit_args": extra_args,
        "top_n": int(args.top_n),
        "refresh_s": float(args.refresh_s),
        "generate_packets_top_n": int(args.generate_packets_top_n),
        "generate_packet_workers": int(args.generate_packet_workers),
        "cleanup_stale_packets_before_render": not bool(args.no_cleanup_stale_packets),
        "status_json": str(args.status_json or "") or None,
    }
    if args.mode == "export":
        export_control_center(args.campaign_dir, **common_kwargs)
        return 0
    serve_control_center(
        args.campaign_dir,
        host=str(args.host),
        port=int(args.port),
        watch_optimization=not bool(args.no_watch_optimization),
        **common_kwargs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
