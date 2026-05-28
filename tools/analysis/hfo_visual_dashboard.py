#!/usr/bin/env python3
"""Build a static visual dashboard for HFO optimizer campaigns.

This is intentionally simpler than TensorBoard: it reads the campaign archive,
finds the diagnostic PNG packets already generated for candidates, and writes
an auto-refreshing HTML page with the same visual artifacts used in notebook
review packets.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import olfactorybulb.hfo_optimizer as hfo


DEFAULT_OUTPUT_SUBDIR = "visual_dashboard"
DEFAULT_REFRESH_S = 60.0
DEFAULT_TOP_N = 20
SUMMARY_STATUS_PATH = Path("results/notebook_runs/optimization/codex_big_hfo_logs/latest_big_hfo_optimizer_status.json")
PRIMARY_PSD_NAME_ORDER = (
    "03_psd_overlay.png",
    "03_power_spectrum_control_vs_ketamine.png",
    "01_lfp_psd_ketamine.png",
    "01_psd_ketamine.png",
    "01_lfp_psd_control.png",
    "01_psd_control.png",
)


@dataclass(frozen=True)
class PacketInfo:
    candidate_id: str
    packet_dir: Path
    contact_sheet: Path | None
    images: tuple[Path, ...]
    manifest: dict[str, Any]
    mtime: float


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _fmt(value: Any, digits: int = 3, *, missing: str = "-") -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return missing
    if abs(numeric) >= 1000.0:
        return f"{numeric:.1f}"
    if abs(numeric) >= 100.0:
        return f"{numeric:.2f}"
    return f"{numeric:.{digits}f}"


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _relpath(path: Path, *, from_dir: Path) -> str:
    return os.path.relpath(path.resolve(), from_dir.resolve()).replace(os.sep, "/")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _candidate_id_from_path(path: Path) -> str | None:
    match = re.search(r"(C\d+)", path.name)
    return match.group(1) if match else None


def _condition_metrics(row: dict[str, Any], condition: str) -> dict[str, Any]:
    payload = row.get(f"{condition}_metrics") or {}
    return payload if isinstance(payload, dict) else {}


def _relative_band(metrics: dict[str, Any], band_name: str) -> float | None:
    relative = metrics.get("relative_band_power") or {}
    if not isinstance(relative, dict):
        return None
    return _safe_float(relative.get(band_name))


def _rate(metrics: dict[str, Any], cell_type: str) -> float | None:
    rates = metrics.get("mean_firing_rate_by_type") or {}
    if not isinstance(rates, dict):
        return None
    return _safe_float(rates.get(cell_type))


def _load_ranked_rows(campaign_dir: Path) -> list[dict[str, Any]]:
    rows = hfo.load_candidate_archive_rows(campaign_dir)
    rows = [row for row in rows if _safe_float(row.get("pair_score")) is not None]
    rows.sort(key=lambda row: float(row.get("pair_score", float("-inf"))), reverse=True)
    return rows


def _packet_mtime(paths: list[Path]) -> float:
    mtimes = [path.stat().st_mtime for path in paths if path.exists()]
    return max(mtimes) if mtimes else 0.0


def find_candidate_packets(campaign_dir: str | Path) -> dict[str, PacketInfo]:
    """Return the newest diagnostic packet per candidate ID."""
    campaign_path = Path(campaign_dir)
    figures_dir = campaign_path / "figures"
    if not figures_dir.exists():
        return {}

    packets: dict[str, PacketInfo] = {}
    for packet_dir in sorted(path for path in figures_dir.iterdir() if path.is_dir()):
        manifest_path = packet_dir / "manifest.json"
        manifest = _read_json(manifest_path) if manifest_path.exists() else {}
        candidate_id = str(manifest.get("candidate_id") or _candidate_id_from_path(packet_dir) or "")
        if not candidate_id:
            continue
        contact_sheet = None
        for name in ("contact_sheet.png", "00_contact_sheet.png"):
            candidate = packet_dir / name
            if candidate.exists():
                contact_sheet = candidate
                break
        images = tuple(
            sorted(
                path
                for path in packet_dir.glob("*.png")
                if path.name not in {"contact_sheet.png", "00_contact_sheet.png"}
            )
        )
        mtime = _packet_mtime([manifest_path, contact_sheet or packet_dir, *images])
        packet = PacketInfo(
            candidate_id=candidate_id,
            packet_dir=packet_dir,
            contact_sheet=contact_sheet,
            images=images,
            manifest=manifest,
            mtime=mtime,
        )
        previous = packets.get(candidate_id)
        if previous is None or packet.mtime >= previous.mtime:
            packets[candidate_id] = packet
    return packets


def _generate_missing_packets(campaign_dir: Path, rows: list[dict[str, Any]], *, top_n: int) -> list[Path]:
    if top_n <= 0:
        return []
    from generate_hfo_candidate_packet import generate_packet

    generated: list[Path] = []
    packets = find_candidate_packets(campaign_dir)
    for row in rows[: int(top_n)]:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or candidate_id in packets:
            continue
        generated.append(generate_packet(campaign_dir, candidate_id))
        packets = find_candidate_packets(campaign_dir)
    return generated


def _metric_summary(row: dict[str, Any]) -> dict[str, Any]:
    control = _condition_metrics(row, "control")
    ketamine = _condition_metrics(row, "ketamine")
    params = row.get("parameters") or {}
    return {
        "candidate_id": row.get("candidate_id"),
        "batch_name": row.get("batch_name"),
        "score": row.get("pair_score"),
        "target_delta": row.get("target_delta"),
        "control_peak": control.get("peak_hz"),
        "ketamine_peak": ketamine.get("peak_hz"),
        "control_target": _relative_band(control, "target_hfo"),
        "ketamine_target": _relative_band(ketamine, "target_hfo"),
        "control_high_gamma": _relative_band(control, "high_gamma"),
        "ketamine_high_gamma": _relative_band(ketamine, "high_gamma"),
        "control_epli": _rate(control, "EPLI"),
        "ketamine_epli": _rate(ketamine, "EPLI"),
        "control_tc": _rate(control, "TC"),
        "ketamine_tc": _rate(ketamine, "TC"),
        "control_gc": _rate(control, "GC"),
        "ketamine_gc": _rate(ketamine, "GC"),
        "ketamine_low_support_penalty": row.get("ketamine_epli_low_support_penalty"),
        "ketamine_silence_penalty": row.get("ketamine_epli_silence_penalty"),
        "control_leak_penalty": row.get("control_leak_penalty"),
        "center_penalty": row.get("ketamine_center_penalty"),
        "params": params if isinstance(params, dict) else {},
    }


def _parameter_chips(parameters: dict[str, Any]) -> str:
    preferred = [
        "kar_mt_gmax",
        "kar_gc_gmax",
        "kar_osn_weight_scale",
        "kar_gc_weight_scale",
        "ampa_nmda_gmax",
        "gaba_gmax",
        "epli_ampa_weight_scale",
        "epli_gaba_weight_scale",
        "gc_ka_gbar_scale",
        "tc_input_weight",
        "mc_input_weight",
    ]
    chunks = []
    for name in preferred:
        if name in parameters:
            chunks.append(f"<span><b>{_esc(name)}</b> {_esc(_fmt(parameters[name], 4))}</span>")
    return "\n".join(chunks)


def _image_figure(image_path: Path, *, output_dir: Path, css_class: str = "", caption: str | None = None) -> str:
    href = _relpath(image_path, from_dir=output_dir)
    label = caption or image_path.stem.replace("_", " ")
    class_attr = f" class='{_esc(css_class)}'" if css_class else ""
    return (
        f"<figure{class_attr}><a href='{_esc(href)}' target='_blank'>"
        f"<img loading='lazy' src='{_esc(href)}' alt='{_esc(label)}'></a>"
        f"<figcaption>{_esc(label)}</figcaption></figure>"
    )


def _primary_psd_image(images: tuple[Path, ...]) -> Path | None:
    by_name = {image.name: image for image in images}
    for name in PRIMARY_PSD_NAME_ORDER:
        if name in by_name:
            return by_name[name]
    psd_images = [image for image in images if "psd" in image.name.lower() or "power_spectrum" in image.name.lower()]
    return sorted(psd_images)[0] if psd_images else None


def _psd_images(images: tuple[Path, ...]) -> list[Path]:
    selected = [
        image
        for image in images
        if "psd" in image.name.lower() or "power_spectrum" in image.name.lower()
    ]
    order = {name: index for index, name in enumerate(PRIMARY_PSD_NAME_ORDER)}
    return sorted(selected, key=lambda image: (order.get(image.name, 100), image.name))


def _kde_images(images: tuple[Path, ...]) -> list[Path]:
    return sorted([image for image in images if "kde" in image.name.lower()])


def _gallery_html(images: list[Path], *, output_dir: Path, css_class: str = "gallery") -> str:
    if not images:
        return ""
    return "<div class='{css_class}'>{items}</div>".format(
        css_class=_esc(css_class),
        items="\n".join(_image_figure(image, output_dir=output_dir) for image in images),
    )


def _details_gallery(
    title: str,
    images: list[Path],
    *,
    output_dir: Path,
    open_by_default: bool = False,
) -> str:
    if not images:
        return ""
    open_attr = " open" if open_by_default else ""
    return (
        f"<details class='figure-group'{open_attr}>"
        f"<summary>{_esc(title)} <span>{len(images)} plots</span></summary>"
        f"{_gallery_html(images, output_dir=output_dir)}"
        "</details>"
    )


def _status_payload(campaign_dir: Path, status_json: Path | None) -> dict[str, Any]:
    state = _read_json(campaign_dir / "state.json")
    objective_filter = hfo.load_objective_filter(campaign_dir)
    status = _read_json(status_json) if status_json and status_json.exists() else {}
    newest_batch = None
    batch_dir = campaign_dir / "batches"
    if batch_dir.exists():
        files = sorted(batch_dir.glob("batch_*_*json"), key=lambda path: path.stat().st_mtime, reverse=True)
        if files:
            newest_batch = {
                "name": files[0].name,
                "mtime": datetime.fromtimestamp(files[0].stat().st_mtime).isoformat(timespec="seconds"),
            }
    return {
        "state": state,
        "objective_filter": objective_filter,
        "status": status,
        "newest_batch": newest_batch,
    }


def _render_status(campaign_dir: Path, rows: list[dict[str, Any]], status_payload: dict[str, Any]) -> str:
    state = status_payload.get("state") or {}
    status = status_payload.get("status") or {}
    newest_batch = status_payload.get("newest_batch") or {}
    cells = [
        ("Campaign", campaign_dir.name),
        ("Candidates", len(rows)),
        ("Worker", status.get("status", "-")),
        ("Current batch", status.get("batch_name") or state.get("next_batch_index", "-")),
        ("Newest artifact", newest_batch.get("name", "-")),
        ("Updated", datetime.now().isoformat(timespec="seconds")),
    ]
    return "\n".join(
        f"<div class='stat'><span>{_esc(label)}</span><strong>{_esc(value)}</strong></div>"
        for label, value in cells
    )


def _render_top_table(rows: list[dict[str, Any]], *, top_n: int) -> str:
    headers = [
        "rank",
        "candidate",
        "score",
        "K target",
        "C target",
        "K peak",
        "C peak",
        "K high-gamma",
        "C high-gamma",
        "K EPLI",
        "C EPLI",
        "K TC",
        "C TC",
    ]
    body = []
    for index, row in enumerate(rows[: int(top_n)], start=1):
        s = _metric_summary(row)
        cells = [
            index,
            s["candidate_id"],
            _fmt(s["score"]),
            _fmt(s["ketamine_target"], 4),
            _fmt(s["control_target"], 4),
            _fmt(s["ketamine_peak"], 1),
            _fmt(s["control_peak"], 1),
            _fmt(s["ketamine_high_gamma"], 4),
            _fmt(s["control_high_gamma"], 4),
            _fmt(s["ketamine_epli"], 2),
            _fmt(s["control_epli"], 2),
            _fmt(s["ketamine_tc"], 2),
            _fmt(s["control_tc"], 2),
        ]
        body.append("<tr>" + "".join(f"<td>{_esc(cell)}</td>" for cell in cells) + "</tr>")
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{_esc(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "\n".join(body)
        + "</tbody></table>"
    )


def _render_packet_card(row: dict[str, Any], packet: PacketInfo | None, *, output_dir: Path, rank: int) -> str:
    s = _metric_summary(row)
    candidate_id = str(s["candidate_id"] or "")
    open_attr = " open" if rank <= 3 else ""
    packet_meta = ""
    primary_psd_html = "<div class='missing'>No PSD packet has been generated for this candidate yet.</div>"
    secondary_psd_html = ""
    kde_html = ""
    other_gallery_html = ""
    contact_html = ""
    if packet is not None:
        when = datetime.fromtimestamp(packet.mtime).isoformat(timespec="seconds") if packet.mtime else "-"
        packet_meta = f"<span>Packet: {_esc(packet.packet_dir.name)}</span><span>Updated: {_esc(when)}</span>"
        primary_psd = _primary_psd_image(packet.images)
        if primary_psd is not None:
            primary_psd_html = _image_figure(
                primary_psd,
                output_dir=output_dir,
                css_class="primary-psd",
                caption="Live PSD overlay with target PSD",
            )
        psd_images = _psd_images(packet.images)
        supporting_psd = [image for image in psd_images if image != primary_psd]
        secondary_psd_html = _details_gallery(
            "PSD details",
            supporting_psd,
            output_dir=output_dir,
            open_by_default=rank == 1,
        )
        kde_set = set(_kde_images(packet.images))
        kde_html = _details_gallery(
            "2D KDEs",
            sorted(kde_set),
            output_dir=output_dir,
            open_by_default=False,
        )
        excluded = set(psd_images) | kde_set
        other_images = [image for image in packet.images if image not in excluded]
        other_gallery_html = _details_gallery(
            "Other diagnostics",
            other_images,
            output_dir=output_dir,
            open_by_default=False,
        )
        if packet.contact_sheet is not None:
            contact_html = (
                "<details class='figure-group'><summary>Contact sheet <span>all plots</span></summary>"
                "<a class='contact' href='{href}' target='_blank'>"
                "<img loading='lazy' src='{href}' alt='{alt}'></a></details>"
            ).format(
                href=_esc(_relpath(packet.contact_sheet, from_dir=output_dir)),
                alt=_esc(f"{candidate_id} contact sheet"),
            )

    badges = [
        f"score {_fmt(s['score'])}",
        f"K target {_fmt(s['ketamine_target'], 4)}",
        f"C target {_fmt(s['control_target'], 4)}",
        f"K peak {_fmt(s['ketamine_peak'], 1)} Hz",
        f"EPLI {_fmt(s['ketamine_epli'], 2)} Hz",
    ]
    penalty_rows = [
        ("K low EPLI penalty", s["ketamine_low_support_penalty"]),
        ("K silence penalty", s["ketamine_silence_penalty"]),
        ("Control leak penalty", s["control_leak_penalty"]),
        ("Center penalty", s["center_penalty"]),
    ]
    penalty_html = "".join(f"<span><b>{_esc(name)}</b> {_esc(_fmt(value, 3))}</span>" for name, value in penalty_rows)
    return f"""
<details class="candidate"{open_attr}>
  <summary>
    <span class="rank">#{rank}</span>
    <span class="candidate-id">{_esc(candidate_id)}</span>
    <span class="batch">{_esc(s.get("batch_name") or "")}</span>
    <span class="badge-row">{''.join(f"<em>{_esc(badge)}</em>" for badge in badges)}</span>
  </summary>
  <div class="card-body">
    <div class="packet-meta">{packet_meta}</div>
    <div class="chips">{penalty_html}</div>
    <div class="chips params">{_parameter_chips(s["params"])}</div>
    {primary_psd_html}
    {secondary_psd_html}
    {kde_html}
    {other_gallery_html}
    {contact_html}
  </div>
</details>
"""


def _render_html(
    *,
    campaign_dir: Path,
    output_dir: Path,
    rows: list[dict[str, Any]],
    packets: dict[str, PacketInfo],
    top_n: int,
    refresh_s: float | None,
    generated_packets: list[Path],
    status_payload: dict[str, Any],
) -> str:
    refresh_meta = f"<meta http-equiv='refresh' content='{int(refresh_s)}'>" if refresh_s and refresh_s > 0 else ""
    packet_cards = "\n".join(
        _render_packet_card(row, packets.get(str(row.get("candidate_id"))), output_dir=output_dir, rank=index)
        for index, row in enumerate(rows[: int(top_n)], start=1)
    )
    generated_html = ""
    if generated_packets:
        generated_html = (
            "<p class='generated'>Generated packets this refresh: "
            + ", ".join(_esc(path.name) for path in generated_packets)
            + "</p>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {refresh_meta}
  <title>HFO Campaign Visual Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fb;
      --ink: #17202a;
      --muted: #667085;
      --line: #d9dee8;
      --panel: #ffffff;
      --blue: #2563eb;
      --red: #dc2626;
      --amber: #d97706;
      --green: #15803d;
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
      z-index: 10;
      background: rgba(247, 248, 251, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 18px 28px 14px;
      backdrop-filter: blur(8px);
    }}
    h1 {{ margin: 0 0 4px; font-size: 22px; letter-spacing: 0; }}
    .subtle {{ color: var(--muted); font-size: 13px; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 24px 28px 60px; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin: 18px 0 20px;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
    }}
    .stat span {{ display: block; color: var(--muted); font-size: 12px; }}
    .stat strong {{ display: block; margin-top: 3px; font-size: 15px; overflow-wrap: anywhere; }}
    section {{
      margin: 20px 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    section > h2 {{
      margin: 0;
      padding: 13px 16px;
      border-bottom: 1px solid var(--line);
      font-size: 16px;
    }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 980px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #edf0f5; text-align: right; white-space: nowrap; }}
    th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ color: var(--muted); font-size: 12px; font-weight: 700; background: #fbfcfe; }}
    details.candidate {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 14px 0;
      overflow: hidden;
    }}
    details.candidate > summary {{
      cursor: pointer;
      display: grid;
      grid-template-columns: 52px minmax(86px, 110px) minmax(110px, 180px) 1fr;
      align-items: center;
      gap: 10px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
    }}
    .rank {{ color: var(--muted); font-weight: 700; }}
    .candidate-id {{ color: var(--blue); font-size: 16px; font-weight: 800; }}
    .batch {{ color: var(--muted); overflow-wrap: anywhere; }}
    .badge-row {{ display: flex; gap: 7px; flex-wrap: wrap; }}
    .badge-row em {{
      font-style: normal;
      background: #f3f6fb;
      border: 1px solid #e3e8f2;
      border-radius: 999px;
      padding: 3px 8px;
      color: #1f2937;
      font-size: 12px;
    }}
    .card-body {{ padding: 14px; }}
    .packet-meta, .chips {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 12px;
    }}
    .chips span {{
      border: 1px solid #e6eaf1;
      border-radius: 7px;
      padding: 4px 7px;
      background: #fbfcfe;
    }}
    .chips.params span {{ color: #1f2937; }}
    figure.primary-psd {{
      margin: 10px 0 14px;
      border-color: #c7d2fe;
      box-shadow: 0 10px 28px rgba(37, 99, 235, 0.10);
    }}
    figure.primary-psd figcaption {{
      color: #1d4ed8;
      font-weight: 700;
      background: #f8fbff;
    }}
    .figure-group {{
      margin: 12px 0;
      border: 1px solid #e3e8f2;
      border-radius: 8px;
      overflow: hidden;
      background: #fbfcfe;
    }}
    .figure-group > summary {{
      cursor: pointer;
      padding: 10px 12px;
      font-weight: 700;
      color: #263244;
    }}
    .figure-group > summary span {{
      margin-left: 8px;
      color: var(--muted);
      font-weight: 500;
      font-size: 12px;
    }}
    .contact img {{
      display: block;
      width: min(100%, 1260px);
      height: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 14px;
      margin-top: 14px;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: white;
    }}
    .figure-group .gallery {{
      padding: 12px;
      border-top: 1px solid #e3e8f2;
      margin-top: 0;
    }}
    .figure-group .contact {{
      display: block;
      padding: 12px;
      border-top: 1px solid #e3e8f2;
    }}
    figure img {{ display: block; width: 100%; height: auto; }}
    figcaption {{
      padding: 7px 9px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
    }}
    .missing {{
      padding: 18px;
      border: 1px dashed #cbd5e1;
      border-radius: 8px;
      color: var(--muted);
      background: #fbfcfe;
    }}
    .generated {{ color: var(--green); }}
    @media (max-width: 760px) {{
      header {{ padding: 14px 16px; }}
      main {{ padding: 16px; }}
      details.candidate > summary {{ grid-template-columns: 40px 88px 1fr; }}
      .batch {{ display: none; }}
      .gallery {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>HFO Campaign Visual Dashboard</h1>
    <div class="subtle">{_esc(campaign_dir)}{f" | refreshes every {int(refresh_s)} s" if refresh_s and refresh_s > 0 else ""}</div>
  </header>
  <main>
    <div class="stats">{_render_status(campaign_dir, rows, status_payload)}</div>
    {generated_html}
    <section>
      <h2>Top Candidates</h2>
      <div class="table-wrap">{_render_top_table(rows, top_n=top_n)}</div>
    </section>
    <section>
      <h2>Visual Packets</h2>
      <div style="padding: 0 14px 14px;">{packet_cards}</div>
    </section>
  </main>
</body>
</html>
"""


def export_visual_dashboard(
    campaign_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    refresh_s: float | None = DEFAULT_REFRESH_S,
    generate_packets_top_n: int = 0,
    status_json: str | Path | None = None,
) -> dict[str, Any]:
    """Write ``index.html`` for one campaign and return a small manifest."""
    campaign_path = Path(campaign_dir).expanduser().resolve()
    output_path = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else campaign_path / DEFAULT_OUTPUT_SUBDIR
    )
    output_path.mkdir(parents=True, exist_ok=True)

    rows = _load_ranked_rows(campaign_path)
    generated_packets = _generate_missing_packets(campaign_path, rows, top_n=int(generate_packets_top_n))
    packets = find_candidate_packets(campaign_path)
    status_path = Path(status_json).expanduser().resolve() if status_json else (REPO_ROOT / SUMMARY_STATUS_PATH)
    payload = _status_payload(campaign_path, status_path)
    html_text = _render_html(
        campaign_dir=campaign_path,
        output_dir=output_path,
        rows=rows,
        packets=packets,
        top_n=int(top_n),
        refresh_s=refresh_s,
        generated_packets=generated_packets,
        status_payload=payload,
    )
    index_path = output_path / "index.html"
    index_path.write_text(html_text)
    server_root, url_path = _dashboard_server_root_and_url(output_path, campaign_path)
    entrypoint_path = _write_dashboard_entrypoint(server_root, url_path)
    manifest = {
        "campaign_dir": str(campaign_path),
        "output_dir": str(output_path),
        "index_html": str(index_path),
        "entrypoint_html": str(entrypoint_path),
        "entrypoint_url_path": url_path,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_rows": len(rows),
        "packet_count": len(packets),
        "generated_packets": [str(path) for path in generated_packets],
        "top_candidate_id": rows[0].get("candidate_id") if rows else None,
        "top_score": rows[0].get("pair_score") if rows else None,
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def watch_visual_dashboard(
    campaign_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    refresh_s: float = DEFAULT_REFRESH_S,
    generate_packets_top_n: int = 0,
    status_json: str | Path | None = None,
) -> None:
    campaign_path = Path(campaign_dir).expanduser().resolve()
    archive = campaign_path / "candidate_archive.jsonl"
    figures = campaign_path / "figures"
    last_signature: tuple[int, int, int] | None = None
    while True:
        archive_sig = int(archive.stat().st_mtime_ns if archive.exists() else 0) ^ int(archive.stat().st_size if archive.exists() else 0)
        figures_sig = int(figures.stat().st_mtime_ns if figures.exists() else 0)
        status_path = Path(status_json).expanduser().resolve() if status_json else (REPO_ROOT / SUMMARY_STATUS_PATH)
        status_sig = int(status_path.stat().st_mtime_ns if status_path.exists() else 0)
        signature = (archive_sig, figures_sig, status_sig)
        if signature != last_signature:
            manifest = export_visual_dashboard(
                campaign_path,
                output_dir=output_dir,
                top_n=top_n,
                refresh_s=refresh_s,
                generate_packets_top_n=generate_packets_top_n,
                status_json=status_path,
            )
            print(
                "Wrote visual dashboard for {candidate_rows} candidates "
                "({packet_count} packets) to {index_html}".format(**manifest),
                flush=True,
            )
            last_signature = signature
        time.sleep(max(float(refresh_s), 1.0))


def _dashboard_server_root_and_url(output_path: Path, campaign_path: Path) -> tuple[Path, str]:
    """Return the HTTP root and URL path that keep packet-relative image links valid."""
    output_path = output_path.expanduser().resolve()
    campaign_path = campaign_path.expanduser().resolve()
    try:
        root = Path(os.path.commonpath([str(output_path), str(campaign_path)]))
    except ValueError:
        root = output_path
    if root == output_path:
        return root, "/"
    relative = output_path.relative_to(root).as_posix()
    url_path = "/" + "/".join(quote(part) for part in relative.split("/") if part) + "/"
    return root, url_path


def _write_dashboard_entrypoint(server_root: Path, dashboard_url_path: str) -> Path:
    """Write a root index that opens the visual dashboard for static serving."""
    index_path = server_root / "index.html"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    href = dashboard_url_path if dashboard_url_path.startswith("/") else f"/{dashboard_url_path}"
    index_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HFO Campaign Visual Dashboard</title>
  <style>
    html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; }}
    iframe {{ display: block; width: 100%; height: 100%; border: 0; }}
    a {{ font: 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  </style>
</head>
<body>
  <iframe src="{_esc(href)}" title="HFO Campaign Visual Dashboard"></iframe>
  <noscript><a href="{_esc(href)}">Open HFO Campaign Visual Dashboard</a></noscript>
</body>
</html>
"""
    )
    return index_path


def serve_visual_dashboard(
    campaign_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    refresh_s: float = DEFAULT_REFRESH_S,
    generate_packets_top_n: int = 0,
    host: str = "127.0.0.1",
    port: int = 6006,
) -> None:
    manifest = export_visual_dashboard(
        campaign_dir,
        output_dir=output_dir,
        top_n=top_n,
        refresh_s=refresh_s,
        generate_packets_top_n=generate_packets_top_n,
    )
    output_path = Path(manifest["output_dir"])
    campaign_path = Path(manifest["campaign_dir"])
    server_root, url_path = _dashboard_server_root_and_url(output_path, campaign_path)
    entrypoint_path = Path(manifest.get("entrypoint_html") or _write_dashboard_entrypoint(server_root, url_path))
    command = [
        sys.executable,
        "-m",
        "http.server",
        str(int(port)),
        "--bind",
        str(host),
        "--directory",
        str(server_root),
    ]
    print(
        f"Serving {entrypoint_path} at http://{host}:{int(port)}/ "
        f"(dashboard: http://{host}:{int(port)}{url_path})",
        flush=True,
    )
    subprocess.run(command, check=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("campaign_dir", type=Path)
        subparser.add_argument("--output-dir", type=Path, default=None)
        subparser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
        subparser.add_argument("--refresh-s", type=float, default=DEFAULT_REFRESH_S)
        subparser.add_argument(
            "--generate-packets-top-n",
            type=int,
            default=0,
            help="Generate missing diagnostic packets for the current top N candidates before rendering.",
        )
        subparser.add_argument("--status-json", type=Path, default=None)

    export_parser = subparsers.add_parser("export", help="Write the dashboard once.")
    add_common(export_parser)

    watch_parser = subparsers.add_parser("watch", help="Rewrite the dashboard when campaign artifacts change.")
    add_common(watch_parser)

    serve_parser = subparsers.add_parser("serve", help="Write once and serve the dashboard over HTTP.")
    add_common(serve_parser)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=6006)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "export":
        manifest = export_visual_dashboard(
            args.campaign_dir,
            output_dir=args.output_dir,
            top_n=args.top_n,
            refresh_s=args.refresh_s,
            generate_packets_top_n=args.generate_packets_top_n,
            status_json=args.status_json,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.command == "watch":
        watch_visual_dashboard(
            args.campaign_dir,
            output_dir=args.output_dir,
            top_n=args.top_n,
            refresh_s=args.refresh_s,
            generate_packets_top_n=args.generate_packets_top_n,
            status_json=args.status_json,
        )
    elif args.command == "serve":
        serve_visual_dashboard(
            args.campaign_dir,
            output_dir=args.output_dir,
            top_n=args.top_n,
            refresh_s=args.refresh_s,
            generate_packets_top_n=args.generate_packets_top_n,
            host=args.host,
            port=args.port,
        )
    else:
        parser.error(f"Unsupported command {args.command!r}")


if __name__ == "__main__":
    main()
