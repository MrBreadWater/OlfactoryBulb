"""Regression checks for the HFO visual dashboard layout."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import types
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import obgpu_experiment_helpers as hlp
from olfactorybulb.hfo_features import parameter_contract_snapshot
import olfactorybulb.hfo_optimizer as hfo
from olfactorybulb.hfo_visuals import visual_contract_snapshot
import tools.analysis.hfo_visual_dashboard as hfo_vd
from tools.analysis.generate_hfo_candidate_packet import (
    VISUAL_STYLE_VERSION,
    SPECTROGRAM_FILE_CONTROL,
    SPECTROGRAM_FILE_KETAMINE,
    SPECTROGRAM_PIPELINE,
    _save_spectrogram,
    _spectrogram_window_geometry,
)
from tools.analysis.hfo_visual_dashboard import (
    PacketInfo,
    _dashboard_server_root_and_url,
    _ensure_visual_dashboard_sidecars,
    _effective_packet_generation_workers,
    _generate_missing_packets,
    _load_ranked_rows,
    _packet_needs_refresh,
    _primary_psd_image,
    _recent_rows,
    _render_html,
    _render_packet_card,
    _write_dashboard_entrypoint,
    ensure_visual_dashboard_runtime,
    find_candidate_packets,
)
from tools.analysis.regenerate_hfo_packet_psd import PSD_PACKET_RENDER_VERSION

assert _effective_packet_generation_workers(0, 0) == 1
assert _effective_packet_generation_workers(0, 1) == 1
assert _effective_packet_generation_workers(1, 8) == 1
assert _effective_packet_generation_workers(2, 8) == 2
assert _effective_packet_generation_workers(999, 3) == 3

recent_fixture_rows = [
    {"batch_name": "batch_0007", "candidate_id": "C00007", "pair_score": 1.0},
    {"batch_name": "batch_0009", "candidate_id": "C00009", "pair_score": 2.0},
    {"batch_name": "batch_0009", "candidate_id": "C00008", "pair_score": 3.0},
    {"batch_name": "batch_0008", "candidate_id": "C00004", "pair_score": 9.0},
]
assert [row["candidate_id"] for row in _recent_rows(recent_fixture_rows, limit=3)] == ["C00008", "C00009"]

recent_archive_fixture_rows = [
    {"batch_name": "batch_0199", "candidate_id": "C08000", "pair_score": 5.0, "_archive_seq": 100},
    {"batch_name": "batch_0108", "candidate_id": "C01000", "pair_score": 1.0, "_archive_seq": 101},
    {"batch_name": "batch_0108", "candidate_id": "C02000", "pair_score": 2.0, "_archive_seq": 102},
    {"batch_name": "batch_0088", "candidate_id": "C03000", "pair_score": 9.0, "_archive_seq": 99},
]
assert [row["candidate_id"] for row in _recent_rows(recent_archive_fixture_rows, limit=3)] == [
    "C02000",
    "C01000",
]

window_t = np.arange(0.0, 1000.0, 0.1, dtype=float)
windowed = {
    "lfp_t": window_t,
    "lfp": np.sin(2.0 * np.pi * 180.0 * window_t / 1000.0),
}
nperseg, noverlap = _spectrogram_window_geometry(windowed)
assert nperseg >= 128
assert noverlap >= int(0.85 * nperseg)
assert 1 + max(0, (window_t.size - nperseg) // max(1, nperseg - noverlap)) >= 100

fake_psd_module = types.SimpleNamespace(PSD_PACKET_RENDER_VERSION=123)
fake_visuals_module = types.SimpleNamespace(VISUAL_STYLE_VERSION=456)
fake_packet_module = types.SimpleNamespace(VISUAL_STYLE_VERSION=456)
with (
    patch.object(hfo_vd, "VISUAL_STYLE_VERSION", 4),
    patch.object(hfo_vd, "PSD_PACKET_RENDER_VERSION", 5),
    patch.object(hfo_vd, "_STYLE_SOURCE_SIGNATURE", (1, 2, 3)),
    patch.object(hfo_vd, "hfo_visuals", object()),
    patch.object(hfo_vd, "packet_generator_module", object()),
    patch.object(hfo_vd, "psd_packet_module", object()),
    patch.object(hfo_vd.importlib, "reload", side_effect=[fake_visuals_module, fake_psd_module, fake_packet_module]) as reload_mock,
):
    changed = hfo_vd._reload_visual_packet_modules_if_needed(source_signature=(3, 4, 5))
    assert changed is True
    assert hfo_vd.VISUAL_STYLE_VERSION == 456
    assert hfo_vd.PSD_PACKET_RENDER_VERSION == 123
    assert reload_mock.call_count == 3


with TemporaryDirectory() as tmp:
    root = Path(tmp)
    campaign = root / "campaign"
    figures_dir = root / "figures"
    packet_dir = figures_dir / "packet_C00042"
    packet_dir.mkdir(parents=True)
    campaign.mkdir(parents=True)
    helper_spec = packet_dir / "helper_spectrogram.png"
    psd_overlay = packet_dir / "03_psd_overlay.png"
    psd_control = packet_dir / "01_psd_control.png"
    raster = packet_dir / "07_raster_control.png"
    raster_k = packet_dir / "08_raster_ketamine.png"
    spec_c = packet_dir / SPECTROGRAM_FILE_CONTROL
    spec_k = packet_dir / SPECTROGRAM_FILE_KETAMINE
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
        helper_spec,
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
    _save_spectrogram(windowed, "control", helper_spec, nperseg=nperseg, noverlap=noverlap)
    assert helper_spec.exists()
    fig, ax = plt.subplots()
    try:
        hlp.plot_spectrogram(
            {"lfp_t": window_t, "lfp": windowed["lfp"]},
            dt_ms=0.1,
            max_freq_hz=250.0,
            nperseg=nperseg,
            noverlap=noverlap,
            modulus=None,
            ax=ax,
        )
        assert ax.get_xlabel() == "Time (ms)"
    finally:
        plt.close(fig)
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
                "visual_contract": visual_contract_snapshot(),
                "parameter_contract": parameter_contract_snapshot(campaign_dir=campaign),
                "pair_score_version": row_score_version,
                "psd_target_overlay": packet_overlay,
                "spectrogram_geometry": {
                    "control": {"nperseg": 256, "noverlap": 192},
                    "ketamine": {"nperseg": 256, "noverlap": 192},
                    "dt_ms": 0.1,
                    "max_freq_hz": float(list(hfo.DEFAULT_SCORE_BANDS["target_hfo"])[1]),
                },
                "spectrogram_window_ms": 1000.0,
                "spectrogram_switch_time_ms": 1000.0,
                "spectrogram_window_ms_by_condition": {
                    "control": [0.0, 1000.0],
                    "ketamine": [1000.0, 2000.0],
                },
                "spectrogram_generation": {
                    "pipeline": SPECTROGRAM_PIPELINE,
                    "control_file": SPECTROGRAM_FILE_CONTROL,
                    "ketamine_file": SPECTROGRAM_FILE_KETAMINE,
                },
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
            "visual_contract": visual_contract_snapshot(),
            "parameter_contract": parameter_contract_snapshot(campaign_dir=campaign),
            "pair_score_version": row_score_version,
            "psd_target_overlay": packet_overlay,
            "spectrogram_geometry": {
                "control": {"nperseg": 256, "noverlap": 192},
                "ketamine": {"nperseg": 256, "noverlap": 192},
                "dt_ms": 0.1,
                "max_freq_hz": float(list(hfo.DEFAULT_SCORE_BANDS["target_hfo"])[1]),
            },
            "spectrogram_window_ms": 1000.0,
            "spectrogram_switch_time_ms": 1000.0,
            "spectrogram_window_ms_by_condition": {
                "control": [0.0, 1000.0],
                "ketamine": [1000.0, 2000.0],
            },
            "spectrogram_generation": {
                "pipeline": SPECTROGRAM_PIPELINE,
                "control_file": SPECTROGRAM_FILE_CONTROL,
                "ketamine_file": SPECTROGRAM_FILE_KETAMINE,
            },
        },
        mtime=1.0,
    )
    row = {
        "candidate_id": "C00042",
        "batch_name": "batch_0002",
        "pair_score": 5.0,
        "pair_score_version": row_score_version,
        "target_delta": 0.1,
        "parameters": {
            "kar_mt_gmax": 0.02,
            "gaba_gmax": 1.5,
            "gaba_tau2_ms": 103.5,
            "input_syn_tau1_ms": 6.2,
            "kar_tau2_ms": 84.1,
        },
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
    assert "gaba_tau2_ms" in html
    assert "input_syn_tau1_ms" in html
    assert "kar_tau2_ms" in html
    assert html.index("Live PSD overlay with scoring template") < html.index("LFP spectrogram")
    assert html.index("Live PSD overlay with scoring template") < html.index("Contact sheet")
    assert _packet_needs_refresh(packet, row) is False

    missing_html = _render_packet_card(row, None, output_dir=root, rank=1, dom_prefix="recent")
    assert "No PSD packet has been generated for this candidate yet." in missing_html
    assert "data-generate-packet" in missing_html
    assert "data-candidate-id='C00042'" in missing_html
    assert "Generate packet" in missing_html

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

    stale_packet_spectrogram_window = PacketInfo(
        candidate_id="C00042",
        packet_dir=packet_dir,
        contact_sheet=contact,
        images=packet.images,
        manifest={
            **packet.manifest,
            "spectrogram_window_ms": 300.0,
            "spectrogram_switch_time_ms": 300.0,
            "spectrogram_window_ms_by_condition": {
                "control": [0.0, 300.0],
                "ketamine": [300.0, 1300.0],
            },
        },
        mtime=1.0,
    )
    assert _packet_needs_refresh(stale_packet_spectrogram_window, row) is True

    missing_spectrogram_packet = PacketInfo(
        candidate_id="C00042",
        packet_dir=packet_dir,
        contact_sheet=contact,
        images=packet.images[:2],  # omit spectrogram/other visuals intentionally
        manifest=packet.manifest,
        mtime=1.0,
    )
    assert _packet_needs_refresh(missing_spectrogram_packet, row) is True

    malformed_spectrogram_packet = PacketInfo(
        candidate_id="C00042",
        packet_dir=packet_dir,
        contact_sheet=contact,
        images=packet.images,
        manifest={
            **packet.manifest,
            "spectrogram_geometry": {
                "control": {"nperseg": 1, "noverlap": 0},
                "ketamine": {"nperseg": 2, "noverlap": 2},
                "dt_ms": 0.1,
                "max_freq_hz": 250.0,
            },
        },
        mtime=1.0,
    )
    assert _packet_needs_refresh(malformed_spectrogram_packet, row) is True

    wrong_pipeline_packet = PacketInfo(
        candidate_id="C00042",
        packet_dir=packet_dir,
        contact_sheet=contact,
        images=packet.images,
        manifest={
            **packet.manifest,
            "spectrogram_generation": {
                "pipeline": {"generator": "wrong.generator.path"},
                "control_file": SPECTROGRAM_FILE_CONTROL,
                "ketamine_file": SPECTROGRAM_FILE_KETAMINE,
            },
        },
        mtime=1.0,
    )
    assert _packet_needs_refresh(wrong_pipeline_packet, row) is True

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
    assert "data-tab-target='tab-best'" in dashboard_html
    assert "data-tab-target='tab-recent'" in dashboard_html
    assert "Most Recent Candidates" in dashboard_html
    assert "Recent Visual Packets" in dashboard_html
    assert "Best Visual Packets" in dashboard_html
    assert f'fetch("{hfo_vd.GENERATE_PACKET_ENDPOINT}"' in dashboard_html
    assert "setActiveTab(" in dashboard_html

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
    packet_dir = campaign / "figures" / "packet_C00042"
    with (
        patch.object(hfo_vd.packet_generator_module, "generate_packet", return_value=packet_dir) as generate_mock,
        patch.object(
            hfo_vd,
            "export_visual_dashboard",
            return_value={
                "campaign_dir": str(campaign),
                "output_dir": str(campaign / "visual_dashboard"),
                "index_html": str(campaign / "visual_dashboard" / "index.html"),
                "entrypoint_html": str(campaign / "index.html"),
                "entrypoint_url_path": "/visual_dashboard/",
                "generated_at": "2026-05-28T12:00:00",
                "candidate_rows": 1,
                "packet_count": 1,
                "generated_packets": [str(packet_dir)],
                "generate_packet_workers": 1,
                "cleanup_stale_packets_before_render": True,
                "top_candidate_id": "C00042",
                "top_score": 1.0,
            },
        ) as export_mock,
    ):
        payload = hfo_vd._generate_dashboard_packet(
            campaign,
            "C00042",
            output_dir=campaign / "visual_dashboard",
            top_n=1,
            refresh_s=60.0,
            generate_packets_top_n=1,
            generate_packet_workers=1,
            cleanup_stale_packets_before_render=True,
            status_json=None,
            reload_modules=False,
        )

    assert payload["ok"] is True
    assert payload["candidate_id"] == "C00042"
    assert payload["packet_dir"] == str(packet_dir)
    generate_mock.assert_called_once()
    export_mock.assert_called_once()

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

with TemporaryDirectory() as tmp:
    campaign = Path(tmp)
    rows = [
        {"batch_name": "batch_0001", "candidate_id": "C00001", "pair_score": 10.0},
        {"batch_name": "batch_0002", "candidate_id": "C00002", "pair_score": 9.0},
        {"batch_name": "batch_0003", "candidate_id": "C00003", "pair_score": 1.0},
    ]
    captured: list[str] = []

    def fake_generate(task):
        _, candidate_id = task
        captured.append(candidate_id)
        path = campaign / "figures" / f"packet_{candidate_id}"
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    with (
        patch.object(hfo_vd, "find_candidate_packets", return_value={}),
        patch.object(hfo_vd, "_packet_needs_refresh", return_value=True),
        patch.object(hfo_vd, "_generate_one_packet", side_effect=fake_generate),
    ):
        generated = _generate_missing_packets(campaign, rows, top_n=2, workers=1)

    assert sorted(captured) == ["C00001", "C00002", "C00003"]
    assert len(generated) == 3

with TemporaryDirectory() as tmp:
    campaign = Path(tmp)
    (campaign / "figures").mkdir(parents=True)
    (campaign / "candidate_archive.jsonl").write_text("{}\n")
    status_json = campaign / "status.json"
    status_json.write_text("{}\n")
    stop_event = threading.Event()
    calls: list[str] = []

    def flaky_export(*args, **kwargs):
        calls.append("export")
        if len(calls) == 1:
            raise RuntimeError("transient export failure")
        stop_event.set()
        return {
            "campaign_dir": str(campaign),
            "output_dir": str(campaign / "visual_dashboard"),
            "index_html": str(campaign / "visual_dashboard" / "index.html"),
            "entrypoint_html": str(campaign / "index.html"),
            "entrypoint_url_path": "/visual_dashboard/",
            "generated_at": "2026-05-28T16:20:00",
            "candidate_rows": 1,
            "packet_count": 0,
            "generated_packets": [],
            "generate_packet_workers": 1,
            "cleanup_stale_packets_before_render": True,
            "top_candidate_id": None,
            "top_score": None,
        }

    with patch.object(hfo_vd, "export_visual_dashboard", side_effect=flaky_export):
        hfo_vd.watch_visual_dashboard(
            campaign,
            output_dir=campaign / "visual_dashboard",
            refresh_s=0.01,
            generate_packets_top_n=0,
            status_json=status_json,
            stop_event=stop_event,
        )

    assert len(calls) == 2

with TemporaryDirectory() as tmp:
    campaign = Path(tmp)
    output_dir = campaign / "visual_dashboard"
    status_json = campaign / "status.json"
    output_dir.mkdir(parents=True)
    status_json.write_text("{}\n")
    spawned_kinds: list[str] = []

    def fake_spawn(command, *, cwd, stdout_path, stderr_path, meta_path, meta):
        kind = str(meta["kind"])
        spawned_kinds.append(kind)
        return hfo_vd.RuntimeProcessInfo(
            kind=kind,
            pid=1000 + len(spawned_kinds),
            pid_path=meta_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            meta={**meta, "pid": 1000 + len(spawned_kinds), "command": list(command)},
        )

    with (
        patch.object(hfo_vd, "export_visual_dashboard", return_value={"output_dir": str(output_dir)}),
        patch.object(hfo_vd, "_read_runtime_process_info", return_value=None),
        patch.object(hfo_vd, "_spawn_detached_process", side_effect=fake_spawn),
        patch.object(hfo_vd, "_port_in_use", return_value=False),
    ):
        payload = _ensure_visual_dashboard_sidecars(
            campaign,
            output_dir=output_dir,
            generate_packets_top_n=0,
            status_json=status_json,
        )

    assert spawned_kinds == ["watcher", "server"]
    assert payload["watcher"]["alive"] is True
    assert payload["server"]["alive"] is True

with TemporaryDirectory() as tmp:
    campaign = Path(tmp)
    output_dir = campaign / "visual_dashboard"
    status_file = output_dir / hfo_vd.RUNTIME_SUBDIR / "watchdog.status.json"
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(json.dumps({"watcher": {"alive": True}}))
    spawned_kinds: list[str] = []

    def fake_spawn(command, *, cwd, stdout_path, stderr_path, meta_path, meta):
        kind = str(meta["kind"])
        spawned_kinds.append(kind)
        pid = 4242 + len(spawned_kinds)
        return hfo_vd.RuntimeProcessInfo(
            kind=kind,
            pid=pid,
            pid_path=meta_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            meta={**meta, "pid": pid, "command": list(command)},
        )

    with (
        patch.object(hfo_vd, "_read_runtime_process_info", return_value=None),
        patch.object(hfo_vd, "_spawn_detached_process", side_effect=fake_spawn) as spawn_mock,
        patch.object(hfo_vd, "_port_in_use", return_value=False),
    ):
        payload = ensure_visual_dashboard_runtime(
            campaign,
            output_dir=output_dir,
            generate_packets_top_n=0,
            status_json=campaign / "status.json",
            port=6006,
        )

    assert spawned_kinds == ["watchdog", "watcher", "server"]
    assert spawn_mock.call_count == 3
    assert payload["watchdog"]["alive"] is True
    assert payload["watchdog"]["pid"] == 4243
    assert payload["sidecars"]["watcher"]["alive"] is True
    assert payload["sidecars"]["server"]["alive"] is True
