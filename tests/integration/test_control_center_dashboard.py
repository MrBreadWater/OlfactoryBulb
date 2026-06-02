"""Smoke tests for unified control-center dashboard export."""

from __future__ import annotations

import http.server
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
from urllib.request import urlopen
from unittest.mock import patch

from olfactorybulb.audit.core import AuditItem, AuditReport
from olfactorybulb.dashboard.control_center import export_control_center, resolve_control_center_campaign, serve_control_center


def _sample_report() -> AuditReport:
    return AuditReport(
        audit_id="new_sweep",
        title="New sweep",
        items=[
            AuditItem(
                check_id="audit_alpha.alpha_pass",
                status="PASS",
                title="Alpha pass",
                criterion="Alpha should pass.",
                description="Description",
                acceptable="Acceptable",
                acceptable_basis="Configured",
                group_id="audit_alpha",
                group_title="Audit alpha",
            )
        ],
    )


def _fake_export_visual_dashboard(campaign_dir: Path, *, output_dir: Path, **_kwargs):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text("<html><body>optimization</body></html>")
    (output_dir / "manifest.json").write_text(json.dumps({"packet_count": 7, "output_dir": str(output_dir)}))
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


with TemporaryDirectory() as tmp:
    root = Path(tmp)
    campaign_dir = root / "campaign"
    campaign_dir.mkdir()
    with (
        patch("olfactorybulb.dashboard.control_center.hfo_dashboard.export_visual_dashboard", side_effect=_fake_export_visual_dashboard),
        patch("olfactorybulb.dashboard.control_center.run_audit_by_id", return_value=_sample_report()),
    ):
        manifest = export_control_center(campaign_dir, output_dir=root / "control_center")
    output_dir = Path(manifest["output_dir"])
    assert (output_dir / "index.html").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "audits" / "index.html").exists()
    assert (output_dir / "audits" / "report.json").exists()
    assert (output_dir / "optimization" / "index.html").exists()
    html = (output_dir / "index.html").read_text()
    assert "OlfactoryBulb Control Center" in html
    assert "/audits/index.html" in html
    assert "/optimization/index.html" in html
    assert "/docs/index.html" in html

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

print("control_center_dashboard: OK")

with TemporaryDirectory() as tmp:
    root = Path(tmp)
    campaign_dir = root / "campaign"
    campaign_dir.mkdir()
    server_holder: dict[str, http.server.ThreadingHTTPServer] = {}

    class CapturingServer(http.server.ThreadingHTTPServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            server_holder["server"] = self

    def _fake_delayed_export(campaign_dir_arg, *, output_dir: Path, **_kwargs):
        time.sleep(1.0)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "index.html").write_text("<html><body>ready</body></html>")
        audits_dir = output_dir / "audits"
        optimization_dir = output_dir / "optimization"
        audits_dir.mkdir(parents=True, exist_ok=True)
        optimization_dir.mkdir(parents=True, exist_ok=True)
        (audits_dir / "index.html").write_text("<html><body>audit ready</body></html>")
        (optimization_dir / "index.html").write_text("<html><body>optimization ready</body></html>")
        return {
            "campaign_dir": str(campaign_dir_arg),
            "output_dir": str(output_dir),
            "generated_at": "now",
            "audit_id": "repo_health",
            "audit_args": ["--profile", "maintained"],
            "audits": {"output_dir": str(audits_dir)},
            "optimization": {"output_dir": str(optimization_dir), "placeholder": False},
            "docs_root": str(root),
        }

    thread = threading.Thread(
        target=serve_control_center,
        kwargs={
            "campaign_dir": campaign_dir,
            "output_dir": root / "control_center",
            "port": 0,
            "watch_optimization": False,
        },
        daemon=True,
    )
    with patch("olfactorybulb.dashboard.control_center.http.server.ThreadingHTTPServer", CapturingServer), patch(
        "olfactorybulb.dashboard.control_center.export_control_center",
        side_effect=_fake_delayed_export,
    ):
        thread.start()
        deadline = time.time() + 5.0
        while "server" not in server_holder and time.time() < deadline:
            time.sleep(0.05)
        assert "server" in server_holder
        server = server_holder["server"]
        base_url = f"http://127.0.0.1:{server.server_port}"
        root_html = urlopen(f"{base_url}/", timeout=2).read().decode("utf-8")
        audits_html = urlopen(f"{base_url}/audits/index.html", timeout=2).read().decode("utf-8")
        assert "OlfactoryBulb Control Center" in root_html
        assert "Audit starting" in audits_html
        server.shutdown()
        thread.join(timeout=5.0)
        assert not thread.is_alive()

print("control_center_dashboard_serve: OK")
