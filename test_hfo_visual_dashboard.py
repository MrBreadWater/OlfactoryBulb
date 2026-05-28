"""Regression checks for the HFO visual dashboard layout."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from tools.analysis.hfo_visual_dashboard import (
    PacketInfo,
    _dashboard_server_root_and_url,
    _primary_psd_image,
    _render_packet_card,
    _write_dashboard_entrypoint,
)


with TemporaryDirectory() as tmp:
    root = Path(tmp)
    packet_dir = root / "packet_C00042"
    packet_dir.mkdir()
    psd_overlay = packet_dir / "03_psd_overlay.png"
    psd_control = packet_dir / "01_psd_control.png"
    raster = packet_dir / "07_raster_control.png"
    kde = packet_dir / "13_spike_frequency_kde_2d_MT_full.png"
    contact = packet_dir / "contact_sheet.png"
    for path in (psd_overlay, psd_control, raster, kde, contact):
        path.write_bytes(b"placeholder")

    assert _primary_psd_image((raster, psd_control, psd_overlay)) == psd_overlay

    packet = PacketInfo(
        candidate_id="C00042",
        packet_dir=packet_dir,
        contact_sheet=contact,
        images=(psd_overlay, psd_control, raster, kde),
        manifest={"candidate_id": "C00042"},
        mtime=1.0,
    )
    row = {
        "candidate_id": "C00042",
        "batch_name": "batch_0002",
        "pair_score": 5.0,
        "target_delta": 0.1,
        "parameters": {"kar_mt_gmax": 0.02, "gaba_gmax": 1.5},
        "control_metrics": {
            "peak_hz": 110.0,
            "relative_band_power": {"target_hfo": 0.02, "high_gamma": 0.11},
            "mean_firing_rate_by_type": {"EPLI": 2.0, "TC": 8.0},
        },
        "ketamine_metrics": {
            "peak_hz": 195.0,
            "relative_band_power": {"target_hfo": 0.22, "high_gamma": 0.08},
            "mean_firing_rate_by_type": {"EPLI": 6.0, "TC": 16.0},
        },
    }

    html = _render_packet_card(row, packet, output_dir=root, rank=1)
    assert "Live PSD overlay with target PSD" in html
    assert "03_psd_overlay.png" in html
    assert "2D KDEs" in html
    assert "Contact sheet" in html
    assert html.index("Live PSD overlay with target PSD") < html.index("2D KDEs")
    assert html.index("Live PSD overlay with target PSD") < html.index("Contact sheet")

    campaign = root / "campaign"
    dashboard = campaign / "visual_dashboard"
    server_root, url_path = _dashboard_server_root_and_url(dashboard, campaign)
    assert server_root == campaign
    assert url_path == "/visual_dashboard/"
    entrypoint = _write_dashboard_entrypoint(server_root, url_path)
    entrypoint_html = entrypoint.read_text()
    assert 'src="/visual_dashboard/"' in entrypoint_html
    assert "Directory listing" not in entrypoint_html
