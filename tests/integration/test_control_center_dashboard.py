"""Integration tests for unified control-center dashboard export and serve flows."""

from __future__ import annotations

import http.server
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import socket
import subprocess
import threading
import time
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
from unittest.mock import patch

from olfactorybulb.audit.core import AuditItem, AuditReport
from olfactorybulb.dashboard.control_center import export_control_center, resolve_control_center_campaign, serve_control_center
from websocket import create_connection


def _sample_report(audit_id: str = "new_sweep", title: str = "New sweep") -> AuditReport:
    return AuditReport(
        audit_id=audit_id,
        title=title,
        items=[
            AuditItem(
                check_id=f"{audit_id}.alpha_pass",
                status="PASS",
                title="Alpha pass",
                criterion="Alpha should pass.",
                description="Description",
                acceptable="Acceptable",
                acceptable_basis="Configured",
                group_id=audit_id,
                group_title=title,
                detail_level="summary",
            )
        ],
    )


_export_call_kwargs: dict[str, object] = {}
_run_audit_calls: list[tuple[str, list[str]]] = []


def _fake_export_visual_dashboard(campaign_dir: Path, *, output_dir: Path, **kwargs):
    _export_call_kwargs.clear()
    _export_call_kwargs.update(kwargs)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text("<html><body>optimization</body></html>")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "packet_count": 7,
                "candidate_rows": 3,
                "generated_at": "now",
                "manifest_revision": 1,
                "output_dir": str(output_dir),
            }
        )
    )
    return {
        "campaign_dir": str(campaign_dir),
        "output_dir": str(output_dir),
        "index_html": str(output_dir / "index.html"),
        "entrypoint_html": str(output_dir / "index.html"),
        "packet_count": 7,
        "candidate_rows": 3,
        "generated_at": "now",
        "manifest_revision": 1,
    }


def _json_post(url: str, payload: dict[str, object]) -> tuple[int, dict[str, object]]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=3) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _capture_run_audit_by_id(audit_id: str, audit_args: list[str]):
    _run_audit_calls.append((audit_id, list(audit_args)))
    return _sample_report(audit_id=audit_id, title=f"{audit_id} report")


def _chromium_binary() -> str | None:
    return (
        shutil.which("chromium")
        or shutil.which("chromium-browser")
        or shutil.which("google-chrome")
        or shutil.which("google-chrome-stable")
    )


def _launch_chromium_cdp() -> tuple[subprocess.Popen[str], str, TemporaryDirectory]:
    binary = _chromium_binary()
    assert binary, "Chromium is required for control-center browser regression coverage"
    profile_dir = TemporaryDirectory()
    stderr_path = Path(profile_dir.name) / "chromium.stderr.log"
    stderr_handle = stderr_path.open("w+")
    proc = subprocess.Popen(
        [
            binary,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--no-first-run",
            "--no-default-browser-check",
            "--remote-allow-origins=*",
            "--remote-debugging-port=0",
            f"--user-data-dir={profile_dir.name}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=stderr_handle,
        text=True,
    )
    deadline = time.time() + 10.0
    while time.time() < deadline:
        stderr_handle.flush()
        stderr_text = stderr_path.read_text() if stderr_path.exists() else ""
        devtools_line = next((line for line in stderr_text.splitlines() if "DevTools listening on ws://" in line), "")
        if devtools_line:
            browser_ws = devtools_line.split("DevTools listening on ", 1)[-1].strip()
            port = urlparse(browser_ws).port
            targets = json.loads(urlopen(f"http://127.0.0.1:{port}/json/list", timeout=5).read().decode("utf-8"))
            page_target = next((target for target in targets if target.get("type") == "page"), None)
            if page_target and page_target.get("webSocketDebuggerUrl"):
                return proc, str(page_target["webSocketDebuggerUrl"]), profile_dir
        time.sleep(0.1)
    proc.terminate()
    stderr_handle.close()
    profile_dir.cleanup()
    raise RuntimeError("Chromium DevTools endpoint did not become ready")


class _CDPClient:
    def __init__(self, ws_url: str):
        self._ws = create_connection(ws_url, timeout=10)
        self._next_id = 0

    def close(self) -> None:
        self._ws.close()

    def call(self, method: str, params: dict[str, object] | None = None) -> dict[str, object]:
        self._next_id += 1
        message_id = self._next_id
        self._ws.send(json.dumps({"id": message_id, "method": method, "params": params or {}}))
        while True:
            payload = json.loads(self._ws.recv())
            if payload.get("id") == message_id:
                return payload

    def eval(self, expression: str, *, await_promise: bool = True) -> object:
        result = self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": await_promise,
                "returnByValue": True,
            },
        )
        return result.get("result", {}).get("result", {}).get("value")


with TemporaryDirectory() as tmp:
    root = Path(tmp)
    campaign_dir = root / "campaign"
    campaign_dir.mkdir()
    with (
        patch("olfactorybulb.dashboard.control_center.hfo_dashboard.export_visual_dashboard", side_effect=_fake_export_visual_dashboard),
        patch("olfactorybulb.dashboard.control_center.run_audit_by_id", side_effect=_capture_run_audit_by_id),
    ):
        manifest = export_control_center(campaign_dir, output_dir=root / "control_center")
    output_dir = Path(manifest["output_dir"])
    assert (output_dir / "index.html").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "state.json").exists()
    assert (output_dir / "audits" / "index.html").exists()
    assert (output_dir / "audits" / "report.json").exists()
    assert (output_dir / "optimization" / "index.html").exists()
    html = (output_dir / "index.html").read_text()
    audit_report = json.loads((output_dir / "audits" / "report.json").read_text())
    assert "OlfactoryBulb Control Center" in html
    assert "Run selected audit" in html
    assert "/__control_center_state__" in html
    assert ">default<" in html
    assert ">all<" in html
    assert 'id="control-center-audit-args" type="text" value=""' in html
    assert audit_report["audit_id"] == "control_center_audits"
    assert len(audit_report["groups"]) == 1
    assert audit_report["groups"][0]["title"] == "repo_health --profile maintained"
    assert _export_call_kwargs["generate_packets_top_n"] == 0
    assert _export_call_kwargs["cleanup_stale_packets_before_render"] is False
    assert _export_call_kwargs["asset_url_prefix"] == "/repo"
    assert _run_audit_calls[-1] == ("repo_health", ["--profile", "maintained"])

with TemporaryDirectory() as tmp:
    root = Path(tmp)
    campaign_dir = root / "campaign"
    campaign_dir.mkdir()
    _run_audit_calls.clear()
    with (
        patch("olfactorybulb.dashboard.control_center.hfo_dashboard.export_visual_dashboard", side_effect=_fake_export_visual_dashboard),
        patch("olfactorybulb.dashboard.control_center.run_audit_by_id", side_effect=_capture_run_audit_by_id),
    ):
        export_control_center(campaign_dir, output_dir=root / "control_center_custom", audit_id="human_review_status")
    assert _run_audit_calls[-1] == ("human_review_status", [])

with TemporaryDirectory() as tmp:
    root = Path(tmp)
    campaign_dir = root / "campaign"
    campaign_dir.mkdir()
    status_path = root / "status.json"
    status_path.write_text(json.dumps({"campaign_dir": str(campaign_dir)}))
    assert resolve_control_center_campaign(status_json=status_path) == campaign_dir.resolve()

with TemporaryDirectory() as tmp:
    root = Path(tmp)
    with patch("olfactorybulb.dashboard.control_center.DEFAULT_STATUS_JSON", root / "missing.json"), patch(
        "olfactorybulb.dashboard.control_center.DEFAULT_OPTIMIZATION_ROOT",
        root,
    ), patch("olfactorybulb.dashboard.control_center.run_audit_by_id", return_value=_sample_report()):
        manifest = export_control_center(None, output_dir=root / "control_center")
    output_dir = Path(manifest["output_dir"])
    assert manifest["campaign_dir"] is None
    assert manifest["audit_id"] == "repo_health"
    assert manifest["audit_args"] == ["--profile", "maintained"]
    placeholder = json.loads((output_dir / "optimization" / "manifest.json").read_text())
    assert placeholder["placeholder"] is True

print("control_center_dashboard_export: OK")

with TemporaryDirectory() as tmp:
    root = Path(tmp)
    campaign_dir = root / "campaign"
    campaign_dir.mkdir()
    stale_root = root / "control_center"
    (stale_root / "audits").mkdir(parents=True, exist_ok=True)
    (stale_root / "optimization").mkdir(parents=True, exist_ok=True)
    (stale_root / "audits" / "manifest.json").write_text(
        json.dumps(
            {
                "audit_id": "stale_audit",
                "title": "Stale audit",
                "summary": {"FAIL": 9, "WARN": 0, "PASS": 0},
                "worst_status": "FAIL",
                "generated_at": "stale",
                "manifest_revision": "stale-audit",
            }
        )
    )
    (stale_root / "optimization" / "manifest.json").write_text(
        json.dumps(
            {
                "packet_count": 999,
                "candidate_rows": 999,
                "generated_at": "stale",
                "manifest_revision": "stale-optimization",
                "placeholder": False,
            }
        )
    )
    server_holder: dict[str, http.server.ThreadingHTTPServer] = {}

    class CapturingServer(http.server.ThreadingHTTPServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            server_holder["server"] = self

    def _slow_export_visual_dashboard(campaign_dir_arg: Path, *, output_dir: Path, **_kwargs):
        time.sleep(0.5)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "index.html").write_text("<html><body>optimization ready</body></html>")
        manifest = {
            "campaign_dir": str(campaign_dir_arg),
            "output_dir": str(output_dir),
            "index_html": str(output_dir / "index.html"),
            "entrypoint_html": str(output_dir / "index.html"),
            "packet_count": 5,
            "candidate_rows": 9,
            "generated_at": "later",
            "manifest_revision": 22,
            "placeholder": False,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    def _slow_run_audit_by_id(audit_id: str, audit_args: list[str]):
        time.sleep(0.5)
        return _sample_report(audit_id=audit_id, title=f"{audit_id} report")

    thread = threading.Thread(
        target=serve_control_center,
        kwargs={
            "campaign_dir": campaign_dir,
            "output_dir": stale_root,
            "port": 0,
            "watch_optimization": False,
        },
        daemon=True,
    )
    with (
        patch("olfactorybulb.dashboard.control_center.http.server.ThreadingHTTPServer", CapturingServer),
        patch("olfactorybulb.dashboard.control_center.hfo_dashboard.export_visual_dashboard", side_effect=_slow_export_visual_dashboard),
        patch("olfactorybulb.dashboard.control_center.run_audit_by_id", side_effect=_slow_run_audit_by_id),
    ):
        thread.start()
        deadline = time.time() + 5.0
        while "server" not in server_holder and time.time() < deadline:
            time.sleep(0.05)
        assert "server" in server_holder
        server = server_holder["server"]
        base_url = f"http://127.0.0.1:{server.server_port}"

        root_html = urlopen(f"{base_url}/", timeout=3).read().decode("utf-8")
        initial_audit_html = urlopen(f"{base_url}/audits/index.html", timeout=3).read().decode("utf-8")
        docs_html = urlopen(f"{base_url}/docs/index.html", timeout=3).read().decode("utf-8")
        rendered_doc_html = urlopen(f"{base_url}/docs/maintained/readme.html", timeout=3).read().decode("utf-8")
        initial_state = json.loads(urlopen(f"{base_url}/__control_center_state__", timeout=3).read().decode("utf-8"))
        assert "Run selected audit" in root_html
        assert "__control_center_state__" in root_html
        assert "Audit running" in initial_audit_html or "Audit starting" in initial_audit_html
        assert "maintained/readme.html" in docs_html
        assert "View source markdown" in rendered_doc_html
        assert initial_state["audit"]["status"] in {"starting", "running"}
        assert initial_state["optimization"]["status"] in {"loading", "ready"}

        ready_deadline = time.time() + 6.0
        ready_state = initial_state
        while time.time() < ready_deadline:
            ready_state = json.loads(urlopen(f"{base_url}/__control_center_state__", timeout=3).read().decode("utf-8"))
            if ready_state["audit"]["status"] == "ready" and ready_state["optimization"]["status"] == "ready":
                break
            time.sleep(0.1)
        assert ready_state["audit"]["status"] == "ready"
        assert ready_state["optimization"]["status"] == "ready"
        assert ready_state["optimization"]["badge"] == "5 packets"

        audits_html = ""
        audits_deadline = time.time() + 4.0
        while time.time() < audits_deadline:
            audits_html = urlopen(f"{base_url}/audits/index.html", timeout=3).read().decode("utf-8")
            if "--profile maintained" in audits_html:
                break
            time.sleep(0.1)
        assert "Display controls" in audits_html
        assert "Collapse all groups" in audits_html
        assert "--profile maintained" in audits_html

        run_status, run_payload = _json_post(
            f"{base_url}/__audit_run__",
            {"audit_id": "test_suite_status", "audit_args_text": "--suite maintained_core --details"},
        )
        assert run_status == 202
        assert run_payload["ok"] is True
        rerun_deadline = time.time() + 6.0
        rerun_state = ready_state
        while time.time() < rerun_deadline:
            rerun_state = json.loads(urlopen(f"{base_url}/__control_center_state__", timeout=3).read().decode("utf-8"))
            if rerun_state["audit"]["status"] == "ready" and rerun_state["audit"]["audit_id"] == "test_suite_status":
                break
            time.sleep(0.1)
        assert rerun_state["audit"]["audit_id"] == "test_suite_status"
        assert rerun_state["audit"]["audit_args"] == ["--suite", "maintained_core", "--details"]
        rerun_audits_html = ""
        rerun_html_deadline = time.time() + 4.0
        while time.time() < rerun_html_deadline:
            rerun_audits_html = urlopen(f"{base_url}/audits/index.html", timeout=3).read().decode("utf-8")
            if "--suite maintained_core --details" in rerun_audits_html:
                break
            time.sleep(0.1)
        assert "--profile maintained" in rerun_audits_html
        assert "--suite maintained_core --details" in rerun_audits_html
        combined_report = json.loads(urlopen(f"{base_url}/audits/report.json", timeout=3).read().decode("utf-8"))
        assert combined_report["audit_id"] == "control_center_audits"
        assert len(combined_report["groups"]) == 2

        refresh_status, refresh_payload = _json_post(f"{base_url}/__audit_refresh__", {})
        assert refresh_status == 202
        assert refresh_payload["ok"] is True

        server.shutdown()
        thread.join(timeout=6.0)
        assert not thread.is_alive()

print("control_center_dashboard_serve: OK")

with TemporaryDirectory() as tmp:
    root = Path(tmp)
    campaign_dir = root / "campaign"
    campaign_dir.mkdir()
    server_holder: dict[str, http.server.ThreadingHTTPServer] = {}

    class CapturingServer(http.server.ThreadingHTTPServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            server_holder["server"] = self

    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    blocked_port = blocker.getsockname()[1]

    thread = threading.Thread(
        target=serve_control_center,
        kwargs={
            "campaign_dir": campaign_dir,
            "output_dir": root / "control_center_fallback",
            "port": blocked_port,
            "watch_optimization": False,
        },
        daemon=True,
    )
    with (
        patch("olfactorybulb.dashboard.control_center.http.server.ThreadingHTTPServer", CapturingServer),
        patch("olfactorybulb.dashboard.control_center.hfo_dashboard.export_visual_dashboard", side_effect=_fake_export_visual_dashboard),
        patch("olfactorybulb.dashboard.control_center.run_audit_by_id", return_value=_sample_report("repo_health", "repo_health report")),
    ):
        thread.start()
        deadline = time.time() + 5.0
        while "server" not in server_holder and time.time() < deadline:
            time.sleep(0.05)
        assert "server" in server_holder
        server = server_holder["server"]
        assert server.server_port != blocked_port
        server.shutdown()
        thread.join(timeout=5.0)
        assert not thread.is_alive()
    blocker.close()

print("control_center_dashboard_port_fallback: OK")

with TemporaryDirectory() as tmp:
    root = Path(tmp)
    campaign_dir = root / "campaign"
    campaign_dir.mkdir()
    server_holder: dict[str, http.server.ThreadingHTTPServer] = {}

    class CapturingServer(http.server.ThreadingHTTPServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            server_holder["server"] = self

    thread = threading.Thread(
        target=serve_control_center,
        kwargs={
            "campaign_dir": campaign_dir,
            "output_dir": root / "control_center_browser",
            "port": 0,
            "watch_optimization": False,
        },
        daemon=True,
    )
    with (
        patch("olfactorybulb.dashboard.control_center.http.server.ThreadingHTTPServer", CapturingServer),
        patch("olfactorybulb.dashboard.control_center.hfo_dashboard.export_visual_dashboard", side_effect=_fake_export_visual_dashboard),
        patch("olfactorybulb.dashboard.control_center.run_audit_by_id", side_effect=_capture_run_audit_by_id),
    ):
        thread.start()
        deadline = time.time() + 5.0
        while "server" not in server_holder and time.time() < deadline:
            time.sleep(0.05)
        assert "server" in server_holder
        server = server_holder["server"]
        base_url = f"http://127.0.0.1:{server.server_port}/"

        proc, ws_url, profile_dir = _launch_chromium_cdp()
        client = _CDPClient(ws_url)
        try:
            client.call("Page.enable")
            client.call("Runtime.enable")
            client.call("Page.navigate", {"url": base_url})
            ready = client.eval(
                "new Promise((resolve) => {"
                "  const deadline = Date.now() + 8000;"
                "  const tick = () => {"
                "    const select = document.getElementById('control-center-audit-id');"
                "    const input = document.getElementById('control-center-audit-args');"
                "    if (select && input) { resolve(true); return; }"
                "    if (Date.now() > deadline) { resolve(false); return; }"
                "    setTimeout(tick, 100);"
                "  };"
                "  tick();"
                "})"
            )
            assert ready is True

            blanked = client.eval(
                "(() => {"
                "  const select = document.getElementById('control-center-audit-id');"
                "  const input = document.getElementById('control-center-audit-args');"
                "  input.value = '--profile maintained';"
                "  input.dispatchEvent(new Event('input', { bubbles: true }));"
                "  select.value = 'human_review_status';"
                "  select.dispatchEvent(new Event('change', { bubbles: true }));"
                "  return { auditId: select.value, auditArgs: input.value };"
                "})()"
            )
            assert blanked == {"auditId": "human_review_status", "auditArgs": ""}

            preserved = client.eval(
                "new Promise((resolve) => {"
                "  const select = document.getElementById('control-center-audit-id');"
                "  const input = document.getElementById('control-center-audit-args');"
                "  input.value = '--custom-check';"
                "  input.dispatchEvent(new Event('input', { bubbles: true }));"
                "  setTimeout(() => resolve({ auditId: select.value, auditArgs: input.value }), 2600);"
                "})"
            )
            assert preserved == {"auditId": "human_review_status", "auditArgs": "--custom-check"}
        finally:
            client.close()
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            profile_dir.cleanup()
            server.shutdown()
            thread.join(timeout=6.0)
            assert not thread.is_alive()

print("control_center_dashboard_browser: OK")
