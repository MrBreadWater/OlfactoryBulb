"""Regression checks for the HFO visual dashboard layout."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import olfactorybulb.hfo_optimizer as hfo
from tools.analysis.generate_hfo_candidate_packet import VISUAL_STYLE_VERSION
from tools.analysis.hfo_visual_dashboard import (
    PacketInfo,
    _dashboard_server_root_and_url,
    _effective_packet_generation_workers,
    _load_ranked_rows,
    _packet_needs_refresh,
    _primary_psd_image,
    _render_html,
    _render_packet_card,
    _write_dashboard_entrypoint,
    find_candidate_packets,
)
from tools.analysis.regenerate_hfo_packet_psd import PSD_PACKET_RENDER_VERSION

assert _effective_packet_generation_workers(0, 0) == 1
assert _effective_packet_generation_workers(0, 1) == 1
assert _effective_packet_generation_workers(1, 8) == 1
assert _effective_packet_generation_workers(2, 8) == 2
assert _effective_packet_generation_workers(999, 3) == 3


with TemporaryDirectory() as tmp:
    root = Path(tmp)
    campaign = root / "campaign"
    figures_dir = root / "figures"
    packet_dir = figures_dir / "packet_C00042"
    packet_dir.mkdir(parents=True)
    psd_overlay = packet_dir / "03_psd_overlay.png"
    psd_control = packet_dir / "01_psd_control.png"
    raster = packet_dir / "07_raster_control.png"
    raster_k = packet_dir / "08_raster_ketamine.png"
    spec_c = packet_dir / "04_spectrogram_control.png"
    spec_k = packet_dir / "05_spectrogram_ketamine.png"
    kde1d_mt_c = packet_dir / "13_spike_frequency_kde_1d_control_MT.png"
    kde1d_mt_k = packet_dir / "13_spike_frequency_kde_1d_ketamine_MT.png"
    kde2d_mt_c = packet_dir / "13_spike_frequency_kde_2d_control_MT.png"
    kde2d_mt_k = packet_dir / "13_spike_frequency_kde_2d_ketamine_MT.png"
    kde1d_epli_c = packet_dir / "13_spike_frequency_kde_1d_control_EPLI.png"
    kde1d_epli_k = packet_dir / "13_spike_frequency_kde_1d_ketamine_EPLI.png"
    kde2d_epli_c = packet_dir / "13_spike_frequency_kde_2d_control_EPLI.png"
    kde2d_epli_k = packet_dir / "13_spike_frequency_kde_2d_ketamine_EPLI.png"
    legacy_kde = packet_dir / "kde_control_MC.png"
    contact = packet_dir / "contact_sheet.png"
    for path in (
        psd_overlay,
        psd_control,
        raster,
        raster_k,
        spec_c,
        spec_k,
        kde1d_mt_c,
        kde1d_mt_k,
        kde2d_mt_c,
        kde2d_mt_k,
        kde1d_epli_c,
        kde1d_epli_k,
        kde2d_epli_c,
        kde2d_epli_k,
        legacy_kde,
        contact,
    ):
        path.write_bytes(b"placeholder")
    stale_population_rates = packet_dir / "09_population_rates.png"
    stale_population_rates.write_bytes(b"placeholder")
    row_score_version = int(hfo.PAIR_SCORE_VERSION)
    packet_overlay = {
        "render_version": PSD_PACKET_RENDER_VERSION,
        "target_hfo_hz": list(hfo.DEFAULT_SCORE_BANDS["target_hfo"]),
        "high_gamma_hz": list(hfo.DEFAULT_SCORE_BANDS["high_gamma"]),
    }
    (packet_dir / "manifest.json").write_text(
        json.dumps(
            {
                "candidate_id": "C00042",
                "visual_style_version": VISUAL_STYLE_VERSION,
                "pair_score_version": row_score_version,
                "psd_target_overlay": packet_overlay,
            }
        )
    )

    assert _primary_psd_image((raster, psd_control, psd_overlay)) == psd_overlay
    discovered = find_candidate_packets(root)
    assert legacy_kde not in discovered["C00042"].images
    assert stale_population_rates not in discovered["C00042"].images

    packet = PacketInfo(
        candidate_id="C00042",
        packet_dir=packet_dir,
        contact_sheet=contact,
        images=(
            psd_overlay,
            psd_control,
            raster,
            raster_k,
            spec_c,
            spec_k,
            kde1d_mt_c,
            kde1d_mt_k,
            kde2d_mt_c,
            kde2d_mt_k,
            kde1d_epli_c,
            kde1d_epli_k,
            kde2d_epli_c,
            kde2d_epli_k,
        ),
        manifest={
            "candidate_id": "C00042",
            "visual_style_version": VISUAL_STYLE_VERSION,
            "pair_score_version": row_score_version,
            "psd_target_overlay": packet_overlay,
        },
        mtime=1.0,
    )
    row = {
        "candidate_id": "C00042",
        "batch_name": "batch_0002",
        "pair_score": 5.0,
        "pair_score_version": row_score_version,
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
    assert "Live PSD overlay with scoring template" in html
    assert "03_psd_overlay.png" in html
    assert "LFP spectrogram" in html
    assert "Control" in html
    assert "Ketamine" in html
    assert "Soma spike frequency 1D KDE" in html
    assert "13_spike_frequency_kde_1d_control_MT.png" in html
    assert "13_spike_frequency_kde_1d_control_EPLI.png" in html
    assert "Soma spike frequency 1D KDE: MT" in html
    assert "Soma spike frequency 1D KDE: EPLI" in html
    assert "09_population_rates.png" not in html
    assert "Contact sheet" in html
    assert html.index("Live PSD overlay with scoring template") < html.index("LFP spectrogram")
    assert html.index("Live PSD overlay with scoring template") < html.index("Contact sheet")
    assert _packet_needs_refresh(packet, row) is False

    stale_packet_version = PacketInfo(
        candidate_id="C00042",
        packet_dir=packet_dir,
        contact_sheet=contact,
        images=packet.images,
        manifest={**packet.manifest, "pair_score_version": row_score_version - 1},
        mtime=1.0,
    )
    assert _packet_needs_refresh(stale_packet_version, row) is True

    stale_packet_overlay = PacketInfo(
        candidate_id="C00042",
        packet_dir=packet_dir,
        contact_sheet=contact,
        images=packet.images,
        manifest={
            **packet.manifest,
            "psd_target_overlay": {
                **packet_overlay,
                "render_version": PSD_PACKET_RENDER_VERSION - 1,
            },
        },
        mtime=1.0,
    )
    assert _packet_needs_refresh(stale_packet_overlay, row) is True

    dashboard_html = _render_html(
        campaign_dir=campaign,
        output_dir=root,
        rows=[row],
        packets={"C00042": packet},
        top_n=1,
        refresh_s=60.0,
        generated_packets=[],
        status_payload={},
        generated_at="2026-05-28T01:23:45",
    )
    assert "http-equiv='refresh'" not in dashboard_html
    assert "fetch(\"manifest.json?cache=\" + Date.now()" in dashboard_html
    assert "dashboard-main" in dashboard_html
    assert "scrollTo(state.scrollX" in dashboard_html

    dashboard = campaign / "visual_dashboard"
    server_root, url_path = _dashboard_server_root_and_url(dashboard, campaign)
    assert server_root == campaign
    assert url_path == "/visual_dashboard/"
    entrypoint = _write_dashboard_entrypoint(server_root, url_path)
    entrypoint_html = entrypoint.read_text()
    assert 'src="/visual_dashboard/"' in entrypoint_html
    assert "Directory listing" not in entrypoint_html

with TemporaryDirectory() as tmp:
    campaign = Path(tmp)
    archive_path = campaign / "candidate_archive.jsonl"
    rows = [
        {
            "batch_name": "batch_0117",
            "candidate_id": "C02815",
            "pair_score": 8.7,
            "pair_score_version": int(hfo.PAIR_SCORE_VERSION),
            "parameters": {},
        },
        {
            "batch_name": "batch_0103",
            "candidate_id": "C01743",
            "pair_score": 25.2,
            "pair_score_version": int(hfo.PAIR_SCORE_VERSION) - 1,
            "parameters": {},
        },
    ]
    with archive_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    ranked_rows = _load_ranked_rows(campaign)
    assert [row["candidate_id"] for row in ranked_rows] == ["C02815"]
