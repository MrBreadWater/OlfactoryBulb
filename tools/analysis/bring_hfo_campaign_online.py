#!/usr/bin/env python3
"""Refresh one HFO optimizer campaign and ensure its live dashboard runtime is online."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.analysis.hfo_visual_dashboard as hfo_vd


def _port_is_available(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((str(host), int(port)))
    except OSError:
        return False
    return True


def _find_available_port(host: str, requested_port: int) -> int:
    requested_port = int(requested_port)
    if requested_port > 0:
        return requested_port
    for candidate in range(6006, 6021):
        if _port_is_available(host, candidate):
            return candidate
    raise RuntimeError("No free dashboard port found in 6006-6020")


def _wait_for_dashboard(url: str, *, timeout_s: float) -> bool:
    deadline = time.monotonic() + max(float(timeout_s), 1.0)
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3.0) as response:
                if int(response.status) != 200:
                    time.sleep(0.5)
                    continue
                content = response.read(512).decode("utf-8", errors="ignore")
                if "HFO Campaign Visual Dashboard" in content or "<!doctype html>" in content.lower():
                    return True
        except (OSError, urllib.error.URLError):
            time.sleep(0.5)
            continue
    return False


def bring_campaign_online(
    campaign_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    top_n: int = hfo_vd.DEFAULT_TOP_N,
    refresh_s: float = hfo_vd.DEFAULT_REFRESH_S,
    generate_packets_top_n: int = hfo_vd.DEFAULT_TOP_N,
    generate_packet_workers: int = hfo_vd.DEFAULT_PACKET_GENERATION_WORKERS,
    cleanup_stale_packets_before_render: bool = hfo_vd.DEFAULT_CLEANUP_STALE_PACKETS,
    status_json: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
    supervise_s: float = hfo_vd.DEFAULT_WATCHDOG_SUPERVISE_S,
    stale_after_s: float = hfo_vd.DEFAULT_STALE_AFTER_S,
    restart_runtime: bool = True,
    wait_timeout_s: float = 20.0,
) -> dict[str, Any]:
    campaign_path = Path(campaign_dir).expanduser().resolve()
    output_path = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else campaign_path / hfo_vd.DEFAULT_OUTPUT_SUBDIR
    )

    manifest = hfo_vd.export_visual_dashboard(
        campaign_path,
        output_dir=output_path,
        top_n=int(top_n),
        refresh_s=float(refresh_s),
        generate_packets_top_n=int(generate_packets_top_n),
        generate_packet_workers=int(generate_packet_workers),
        cleanup_stale_packets_before_render=bool(cleanup_stale_packets_before_render),
        status_json=status_json,
    )

    stopped = None
    if restart_runtime:
        stopped = hfo_vd.stop_visual_dashboard_runtime(campaign_path, output_dir=output_path)

    chosen_port = _find_available_port(str(host), int(port))
    runtime = hfo_vd.ensure_visual_dashboard_runtime(
        campaign_path,
        output_dir=output_path,
        top_n=int(top_n),
        refresh_s=float(refresh_s),
        generate_packets_top_n=int(generate_packets_top_n),
        generate_packet_workers=int(generate_packet_workers),
        cleanup_stale_packets_before_render=bool(cleanup_stale_packets_before_render),
        status_json=status_json,
        host=str(host),
        port=int(chosen_port),
        supervise_s=float(supervise_s),
        stale_after_s=float(stale_after_s),
    )
    dashboard_url = f"http://{host}:{int(chosen_port)}/"
    dashboard_ready = _wait_for_dashboard(dashboard_url, timeout_s=float(wait_timeout_s))
    if not dashboard_ready:
        raise RuntimeError(f"Dashboard did not become reachable at {dashboard_url} within {float(wait_timeout_s):.1f}s")

    return {
        "campaign_dir": str(campaign_path),
        "output_dir": str(output_path),
        "dashboard_url": dashboard_url,
        "dashboard_ready": bool(dashboard_ready),
        "host": str(host),
        "port": int(chosen_port),
        "restart_runtime": bool(restart_runtime),
        "stopped_runtime": stopped,
        "manifest": manifest,
        "runtime": runtime,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=hfo_vd.DEFAULT_TOP_N)
    parser.add_argument("--refresh-s", type=float, default=hfo_vd.DEFAULT_REFRESH_S)
    parser.add_argument("--generate-packets-top-n", type=int, default=hfo_vd.DEFAULT_TOP_N)
    parser.add_argument("--generate-packet-workers", type=int, default=hfo_vd.DEFAULT_PACKET_GENERATION_WORKERS)
    parser.add_argument("--no-cleanup-stale-packets", action="store_true")
    parser.add_argument("--status-json", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to serve on. Use 0 to auto-pick a free port in 6006-6020.",
    )
    parser.add_argument("--supervise-s", type=float, default=hfo_vd.DEFAULT_WATCHDOG_SUPERVISE_S)
    parser.add_argument("--stale-after-s", type=float, default=hfo_vd.DEFAULT_STALE_AFTER_S)
    parser.add_argument("--no-restart-runtime", action="store_true")
    parser.add_argument("--wait-timeout-s", type=float, default=20.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    payload = bring_campaign_online(
        args.campaign_dir,
        output_dir=args.output_dir,
        top_n=args.top_n,
        refresh_s=args.refresh_s,
        generate_packets_top_n=args.generate_packets_top_n,
        generate_packet_workers=args.generate_packet_workers,
        cleanup_stale_packets_before_render=not args.no_cleanup_stale_packets,
        status_json=args.status_json,
        host=args.host,
        port=args.port,
        supervise_s=args.supervise_s,
        stale_after_s=args.stale_after_s,
        restart_runtime=not args.no_restart_runtime,
        wait_timeout_s=args.wait_timeout_s,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
