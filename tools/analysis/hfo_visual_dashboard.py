#!/usr/bin/env python3
"""Build a static visual dashboard for HFO optimizer campaigns.

This is intentionally simpler than TensorBoard: it reads the campaign archive,
finds the diagnostic PNG packets already generated for candidates, and writes
an auto-refreshing HTML page with the same visual artifacts used in notebook
review packets.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fcntl
import http.server
import html
import importlib
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import olfactorybulb.hfo_optimizer as hfo
from olfactorybulb.hfo_features import PARAMETER_CONTRACT_VERSION, parameter_display_order
import olfactorybulb.hfo_visuals as hfo_visuals
import generate_hfo_candidate_packet as packet_generator_module
import regenerate_hfo_packet_psd as psd_packet_module
from generate_hfo_candidate_packet import VISUAL_STYLE_VERSION
from regenerate_hfo_packet_psd import PSD_PACKET_RENDER_VERSION


DEFAULT_OUTPUT_SUBDIR = "visual_dashboard"
DEFAULT_REFRESH_S = 60.0
DEFAULT_TOP_N = 20
DEFAULT_GENERATE_PACKETS_TOP_N = DEFAULT_TOP_N
DEFAULT_PACKET_GENERATION_WORKERS = 0
DEFAULT_CLEANUP_STALE_PACKETS = True
DEFAULT_RUNTIME_GENERATE_PACKETS_TOP_N = 5
DEFAULT_WATCHDOG_SUPERVISE_S = 20.0
DEFAULT_STALE_AFTER_S = 180.0
GENERATE_PACKET_ENDPOINT = "/__hfo_generate_packet__"
EXPECTED_SPECTROGRAM_FILES = dict(hfo_visuals.SPECTROGRAM_FILE_BY_CONDITION)
EXPECTED_SPECTROGRAM_PIPELINE = str(hfo_visuals.SPECTROGRAM_PIPELINE["generator"])
EXPECTED_SPECTROGRAM_WINDOW_MS = float(hfo_visuals.NOTEBOOK_SPECTROGRAM_VISUAL_WINDOW_MS)
SUMMARY_STATUS_PATH = Path("results/notebook_runs/optimization/codex_big_hfo_logs/latest_big_hfo_optimizer_status.json")
PRIMARY_PSD_NAME_ORDER = tuple(hfo_visuals.PRIMARY_PSD_NAME_ORDER)
_STYLE_SOURCE_SIGNATURE: tuple[int, int, int] = (0, 0, 0)
RUNTIME_SUBDIR = ".runtime"
EXPORT_LOCK_NAME = "export.lock"
_PACKET_GENERATION_JOBS: dict[tuple[str, str], threading.Thread] = {}
_PACKET_GENERATION_JOBS_LOCK = threading.Lock()


@dataclass(frozen=True)
class PacketInfo:
    candidate_id: str
    packet_dir: Path
    contact_sheet: Path | None
    images: tuple[Path, ...]
    manifest: dict[str, Any]
    mtime: float


@dataclass(frozen=True)
class RuntimeProcessInfo:
    kind: str
    pid: int
    pid_path: Path
    stdout_path: Path
    stderr_path: Path
    meta: dict[str, Any]


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


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _wait_with_stop(delay_s: float, stop_event: threading.Event | None) -> None:
    if stop_event is None:
        time.sleep(max(float(delay_s), 0.0))
        return
    stop_event.wait(timeout=max(float(delay_s), 0.0))


def _runtime_dir(output_path: Path) -> Path:
    return output_path / RUNTIME_SUBDIR


def _export_lock_path(output_path: Path) -> Path:
    return _runtime_dir(output_path) / EXPORT_LOCK_NAME


def _runtime_process_paths(output_path: Path, kind: str) -> dict[str, Path]:
    runtime_dir = _runtime_dir(output_path)
    return {
        "runtime_dir": runtime_dir,
        "pid": runtime_dir / f"{kind}.pid.json",
        "stdout": runtime_dir / f"{kind}.stdout.log",
        "stderr": runtime_dir / f"{kind}.stderr.log",
    }


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _process_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()


def _process_cmdargs(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except OSError:
        return []
    return [arg for arg in raw.decode("utf-8", errors="ignore").split("\x00") if arg]


def _process_matches_tokens(pid: int, expected_tokens: list[str]) -> bool:
    cmdline = _process_cmdline(pid)
    if not cmdline:
        return False
    return all(token in cmdline for token in expected_tokens)


def _process_matches_command(pid: int, expected_command: list[str]) -> bool:
    actual = _process_cmdargs(pid)
    if not actual:
        return False
    return actual == [str(arg) for arg in expected_command]


def _matching_pids(expected_tokens: list[str]) -> list[int]:
    matches: list[int] = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        if _process_matches_tokens(pid, expected_tokens):
            matches.append(pid)
    return sorted(set(matches))


def _read_runtime_process_info(output_path: Path, kind: str) -> RuntimeProcessInfo | None:
    paths = _runtime_process_paths(output_path, kind)
    payload = _read_json(paths["pid"])
    pid = int(payload.get("pid") or 0)
    if pid <= 0:
        return None
    return RuntimeProcessInfo(
        kind=kind,
        pid=pid,
        pid_path=paths["pid"],
        stdout_path=paths["stdout"],
        stderr_path=paths["stderr"],
        meta=payload,
    )


def _terminate_process(pid: int, *, grace_s: float = 5.0) -> None:
    if pid <= 0 or not _pid_is_alive(pid):
        return
    try:
        pgid = os.getpgid(int(pid))
    except OSError:
        pgid = None
    try:
        if pgid is not None and pgid > 0:
            os.killpg(pgid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.time() + max(float(grace_s), 0.0)
    while time.time() < deadline:
        if not _pid_is_alive(pid):
            return
        time.sleep(0.1)
    try:
        if pgid is not None and pgid > 0:
            os.killpg(pgid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGKILL)
    except OSError:
        return


def _spawn_detached_process(
    command: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    meta_path: Path,
    meta: dict[str, Any],
) -> RuntimeProcessInfo:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("ab") as stdout_handle, stderr_path.open("ab") as stderr_handle:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
    payload = dict(meta)
    payload.update(
        {
            "pid": int(proc.pid),
            "command": list(command),
            "cwd": str(cwd),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _write_json_atomic(meta_path, payload)
    return RuntimeProcessInfo(
        kind=str(meta.get("kind") or ""),
        pid=int(proc.pid),
        pid_path=meta_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        meta=payload,
    )


def _port_in_use(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, int(port)))
            except OSError:
                return True
    except PermissionError:
        return False
    return False


def _watch_sources_mtime(campaign_path: Path, status_path: Path) -> float:
    paths = [
        campaign_path / "candidate_archive.jsonl",
        campaign_path / "figures",
        status_path,
    ]
    return max((path.stat().st_mtime for path in paths if path.exists()), default=0.0)


def _dashboard_outputs_mtime(output_path: Path) -> float:
    paths = [
        output_path / "index.html",
        output_path / "manifest.json",
        output_path / "live_packet_manifest.json",
    ]
    return max((path.stat().st_mtime for path in paths if path.exists()), default=0.0)


def _candidate_id_from_path(path: Path) -> str | None:
    match = re.search(r"(C\d+)", path.name)
    return match.group(1) if match else None


def _is_legacy_ad_hoc_kde_image(path: Path) -> bool:
    return bool(re.match(r"kde_(control|ketamine)_[A-Za-z0-9]+\.png$", path.name))


def _is_hidden_packet_image(path: Path) -> bool:
    return path.name in {"contact_sheet.png", "00_contact_sheet.png", "09_population_rates.png"}


def _is_hidden_packet_dir(path: Path) -> bool:
    name = path.name
    return name.startswith(".")


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


def _is_current_visual_style_manifest(manifest: dict[str, Any], *, allow_legacy: bool = False) -> bool:
    if allow_legacy:
        return True
    style = manifest.get("visual_style_version")
    if style is None:
        return False
    try:
        return int(style) == int(VISUAL_STYLE_VERSION)
    except (TypeError, ValueError):
        return False


def _load_ranked_rows(campaign_dir: Path) -> list[dict[str, Any]]:
    rows = hfo.load_candidate_archive_rows(campaign_dir)
    rows = [row for row in rows if _safe_float(row.get("pair_score")) is not None]
    current_score_version = int(getattr(hfo, "PAIR_SCORE_VERSION", 0) or 0)
    current_version_rows = []
    for row in rows:
        try:
            version = int(row.get("pair_score_version", current_score_version))
        except (TypeError, ValueError):
            version = current_score_version
        if version == current_score_version:
            current_version_rows.append(row)
    if current_version_rows:
        rows = current_version_rows
    rows.sort(key=lambda row: float(row.get("pair_score", float("-inf"))), reverse=True)
    return rows


def _batch_index(row: dict[str, Any]) -> int:
    batch_name = str(row.get("batch_name") or "")
    try:
        return int(batch_name.rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return -1


def _archive_seq(row: dict[str, Any]) -> int:
    try:
        return int(row.get("_archive_seq"))
    except (TypeError, ValueError):
        return -1


def _latest_completed_batch_name(campaign_dir: Path) -> str | None:
    state = hfo.load_campaign_state(campaign_dir)
    completed = state.get("completed_batches") or []
    if not isinstance(completed, list):
        return None
    for batch_name in reversed(completed):
        text = str(batch_name or "").strip()
        if text:
            return text
    return None


def _recent_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    recent_batch_name: str | None = None,
) -> list[dict[str, Any]]:
    batch_name = str(recent_batch_name or "").strip()
    if batch_name:
        ranked = sorted(
            [row for row in rows if str(row.get("batch_name") or "") == batch_name],
            key=lambda row: (
                float(row.get("pair_score", float("-inf"))),
                _archive_seq(row),
                str(row.get("candidate_id") or ""),
            ),
            reverse=True,
        )
        if ranked:
            return ranked[: int(limit)]

    latest_batch_name = ""
    ranked_by_arrival = sorted(
        rows,
        key=lambda row: (
            _archive_seq(row),
            _batch_index(row),
            str(row.get("candidate_id") or ""),
        ),
        reverse=True,
    )
    for row in ranked_by_arrival:
        batch_name = str(row.get("batch_name") or "")
        if batch_name:
            latest_batch_name = batch_name
            break
    if latest_batch_name:
        ranked = sorted(
            [row for row in rows if str(row.get("batch_name") or "") == latest_batch_name],
            key=lambda row: (
                float(row.get("pair_score", float("-inf"))),
                _archive_seq(row),
                str(row.get("candidate_id") or ""),
            ),
            reverse=True,
        )
    else:
        ranked = sorted(
            rows,
            key=lambda row: (
                float(row.get("pair_score", float("-inf"))),
                _archive_seq(row),
                _batch_index(row),
                str(row.get("candidate_id") or ""),
            ),
            reverse=True,
        )
    return ranked[: int(limit)]


def _packet_mtime(paths: list[Path]) -> float:
    mtimes: list[float] = []
    for path in paths:
        try:
            if path.exists():
                mtimes.append(path.stat().st_mtime)
        except (FileNotFoundError, OSError):
            continue
    return max(mtimes) if mtimes else 0.0


def _path_mtime_ns(path: Path) -> int:
    try:
        return int(path.stat().st_mtime_ns)
    except OSError:
        return 0


def _style_source_signature() -> tuple[int, int, int]:
    return tuple(
        _path_mtime_ns(path)
        for path in (
            Path(hfo_visuals.__file__).resolve(),
            SCRIPT_DIR / "generate_hfo_candidate_packet.py",
            SCRIPT_DIR / "regenerate_hfo_packet_psd.py",
        )
    )  # type: ignore[return-value]


def _reload_visual_packet_modules_if_needed(
    source_signature: tuple[int, int, int] | None = None,
    *,
    force: bool = False,
) -> bool:
    """Reload packet-generation modules when their source changes on disk."""
    global VISUAL_STYLE_VERSION, PSD_PACKET_RENDER_VERSION, hfo_visuals, packet_generator_module, psd_packet_module, _STYLE_SOURCE_SIGNATURE
    signature = source_signature or _style_source_signature()
    if not force and signature == _STYLE_SOURCE_SIGNATURE:
        return False
    importlib.invalidate_caches()
    hfo_visuals = importlib.reload(hfo_visuals)
    psd_packet_module = importlib.reload(psd_packet_module)
    packet_generator_module = importlib.reload(packet_generator_module)
    VISUAL_STYLE_VERSION = int(hfo_visuals.VISUAL_STYLE_VERSION)
    PSD_PACKET_RENDER_VERSION = int(psd_packet_module.PSD_PACKET_RENDER_VERSION)
    _STYLE_SOURCE_SIGNATURE = signature
    return True


def find_candidate_packets(
    campaign_dir: str | Path,
    *,
    require_current_visual_style: bool = True,
) -> dict[str, PacketInfo]:
    """Return the newest diagnostic packet per candidate ID."""
    campaign_path = Path(campaign_dir)
    figures_dir = campaign_path / "figures"
    if not figures_dir.exists():
        return {}

    packets: dict[str, PacketInfo] = {}
    for packet_dir in sorted(path for path in figures_dir.iterdir() if path.is_dir()):
        if _is_hidden_packet_dir(packet_dir):
            continue
        manifest_path = packet_dir / "manifest.json"
        manifest = _read_json(manifest_path) if manifest_path.exists() else {}
        if not _is_current_visual_style_manifest(
            manifest, allow_legacy=(not require_current_visual_style)
        ):
            continue
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
                if not _is_hidden_packet_image(path) and not _is_legacy_ad_hoc_kde_image(path)
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


def cleanup_stale_packets(
    campaign_dir: str | Path,
    *,
    require_current_visual_style: bool = True,
    dry_run: bool = False,
) -> tuple[int, list[str]]:
    """Delete stale packet directories and return (removed_count, removed_names)."""
    campaign_path = Path(campaign_dir)
    figures_dir = campaign_path / "figures"
    if not figures_dir.exists():
        return 0, []

    removed: list[str] = []
    for packet_dir in sorted(path for path in figures_dir.iterdir() if path.is_dir()):
        if _is_hidden_packet_dir(packet_dir):
            continue
        manifest = _read_json(packet_dir / "manifest.json") if (packet_dir / "manifest.json").exists() else {}
        if _is_current_visual_style_manifest(manifest, allow_legacy=(not require_current_visual_style)):
            continue
        removed.append(packet_dir.name)
        if dry_run:
            continue
        try:
            shutil.rmtree(packet_dir)
        except FileNotFoundError:
            continue
    return len(removed), removed


@contextlib.contextmanager
def _dashboard_export_lock(output_path: Path) -> Any:
    lock_path = _export_lock_path(output_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _packet_needs_refresh(packet: PacketInfo | None, row: dict[str, Any]) -> bool:
    if packet is None:
        return True
    if packet.manifest.get("visual_style_version") != VISUAL_STYLE_VERSION:
        return True
    visual_contract = packet.manifest.get("visual_contract") or {}
    if not isinstance(visual_contract, dict):
        return True
    if int(visual_contract.get("style_version", 0) or 0) != int(VISUAL_STYLE_VERSION):
        return True
    parameter_contract = packet.manifest.get("parameter_contract") or {}
    if not isinstance(parameter_contract, dict):
        return True
    if int(parameter_contract.get("version", 0) or 0) != int(PARAMETER_CONTRACT_VERSION):
        return True
    if not list(parameter_contract.get("search_space_paths") or []):
        return True
    try:
        row_score_version = int(row.get("pair_score_version", getattr(hfo, "PAIR_SCORE_VERSION", 0)) or 0)
    except (TypeError, ValueError):
        row_score_version = int(getattr(hfo, "PAIR_SCORE_VERSION", 0) or 0)
    try:
        packet_score_version = int(packet.manifest.get("pair_score_version", row_score_version) or row_score_version)
    except (TypeError, ValueError):
        packet_score_version = row_score_version
    if packet_score_version != row_score_version:
        return True
    overlay = packet.manifest.get("psd_target_overlay") or {}
    if not isinstance(overlay, dict):
        return True
    if int(overlay.get("render_version", 0) or 0) != int(PSD_PACKET_RENDER_VERSION):
        return True
    if list(overlay.get("target_hfo_hz") or []) != list(hfo.DEFAULT_SCORE_BANDS["target_hfo"]):
        return True
    if list(overlay.get("high_gamma_hz") or []) != list(hfo.DEFAULT_SCORE_BANDS["high_gamma"]):
        return True

    required_spectrogram_files = (
        packet.packet_dir / EXPECTED_SPECTROGRAM_FILES["control"],
        packet.packet_dir / EXPECTED_SPECTROGRAM_FILES["ketamine"],
    )
    if any(not path.exists() for path in required_spectrogram_files):
        return True
    if any(not any(img == path for img in packet.images) for path in required_spectrogram_files):
        return True

    generation = packet.manifest.get("spectrogram_generation") or {}
    if not isinstance(generation, dict):
        return True
    pipeline = generation.get("pipeline") or {}
    if not isinstance(pipeline, dict):
        return True
    if str(pipeline.get("generator") or "") != EXPECTED_SPECTROGRAM_PIPELINE:
        return True
    if str(generation.get("control_file") or "") != EXPECTED_SPECTROGRAM_FILES["control"]:
        return True
    if str(generation.get("ketamine_file") or "") != EXPECTED_SPECTROGRAM_FILES["ketamine"]:
        return True
    try:
        spectrogram_window_ms = float(packet.manifest.get("spectrogram_window_ms", 0.0) or 0.0)
    except (TypeError, ValueError):
        return True
    if abs(spectrogram_window_ms - EXPECTED_SPECTROGRAM_WINDOW_MS) > 1e-9:
        return True
    try:
        spectrogram_switch_time_ms = float(packet.manifest.get("spectrogram_switch_time_ms", 0.0) or 0.0)
    except (TypeError, ValueError):
        return True
    if spectrogram_switch_time_ms <= 0.0 or not math.isfinite(spectrogram_switch_time_ms):
        return True
    windows_by_condition = packet.manifest.get("spectrogram_window_ms_by_condition") or {}
    if not isinstance(windows_by_condition, dict):
        return True
    for condition in ("control", "ketamine"):
        window = windows_by_condition.get(condition)
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            return True
        try:
            start_ms = float(window[0])
            stop_ms = float(window[1])
        except (TypeError, ValueError):
            return True
        if not math.isfinite(start_ms) or not math.isfinite(stop_ms) or stop_ms <= start_ms:
            return True
        if abs((stop_ms - start_ms) - EXPECTED_SPECTROGRAM_WINDOW_MS) > 1e-3:
            return True
    control_window = windows_by_condition.get("control")
    ketamine_window = windows_by_condition.get("ketamine")
    try:
        control_stop = float(control_window[1])
        ketamine_start = float(ketamine_window[0])
    except (TypeError, ValueError, IndexError):
        return True
    if abs(control_stop - spectrogram_switch_time_ms) > 1e-3:
        return True
    if abs(ketamine_start - spectrogram_switch_time_ms) > 1e-3:
        return True

    geometry = packet.manifest.get("spectrogram_geometry") or {}
    control = geometry.get("control") or {}
    ketamine = geometry.get("ketamine") or {}
    for name, state in (("control", control), ("ketamine", ketamine)):
        try:
            nperseg = int(state.get("nperseg", 0))
            noverlap = int(state.get("noverlap", 0))
        except (TypeError, ValueError):
            return True
        if nperseg < 2 or noverlap < 0 or noverlap >= nperseg:
            return True
    for key in ("dt_ms", "max_freq_hz"):
        if key not in geometry:
            return True

    all_pngs = tuple(packet.packet_dir.glob("*.png"))
    has_legacy_kde = any(_is_legacy_ad_hoc_kde_image(path) for path in all_pngs)
    has_pipeline_kde = any("spike_frequency_kde_2d" in path.name for path in all_pngs)
    has_notebook_kde_1d = any("spike_frequency_kde_1d" in path.name for path in all_pngs)
    return (has_legacy_kde and not has_pipeline_kde) or not has_notebook_kde_1d


def _effective_packet_generation_workers(requested_workers: int | None, candidate_count: int) -> int:
    if candidate_count <= 1:
        return 1
    if requested_workers is None:
        requested = 0
    else:
        requested = int(requested_workers)
    if requested <= 0:
        requested = int(os.cpu_count() or 1)
    return max(1, min(requested, int(candidate_count)))


def _packet_generation_job_key(campaign_dir: Path, candidate_id: str) -> tuple[str, str]:
    return (str(Path(campaign_dir).expanduser().resolve()), str(candidate_id))


def _packet_generation_lock_is_active(campaign_dir: Path, candidate_id: str) -> bool:
    lock_path = packet_generator_module.packet_build_lock_path(campaign_dir, candidate_id)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        else:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            return False


def _generate_one_packet(task: tuple[str, str]) -> str:
    campaign_dir_str, candidate_id = task
    from generate_hfo_candidate_packet import generate_packet

    packet_dir = generate_packet(Path(campaign_dir_str), candidate_id)
    return str(packet_dir)


def _generate_missing_packets(
    campaign_dir: Path,
    rows: list[dict[str, Any]],
    *,
    top_n: int,
    workers: int | None = None,
    recent_batch_name: str | None = None,
) -> list[Path]:
    if top_n <= 0:
        return []

    packets = find_candidate_packets(campaign_dir)
    selected_rows: list[dict[str, Any]] = []
    seen_candidate_ids: set[str] = set()
    for source_rows in (
        rows[: int(top_n)],
        _recent_rows(rows, limit=int(top_n), recent_batch_name=recent_batch_name),
    ):
        for row in source_rows:
            candidate_id = str(row.get("candidate_id") or "")
            if not candidate_id or candidate_id in seen_candidate_ids:
                continue
            seen_candidate_ids.add(candidate_id)
            selected_rows.append(row)
    missing_candidate_ids: list[str] = []
    for row in selected_rows:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            continue
        if _packet_generation_lock_is_active(campaign_dir, candidate_id):
            continue
        if not _packet_needs_refresh(packets.get(candidate_id), row):
            continue
        missing_candidate_ids.append(candidate_id)
    if not missing_candidate_ids:
        return []

    max_workers = _effective_packet_generation_workers(workers, len(missing_candidate_ids))
    if max_workers <= 1:
        return [Path(_generate_one_packet((str(campaign_dir), candidate_id))) for candidate_id in missing_candidate_ids]

    tasks = [(str(campaign_dir), candidate_id) for candidate_id in missing_candidate_ids]
    generated: list[Path] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        for packet_dir in pool.map(_generate_one_packet, tasks):
            generated.append(Path(packet_dir))
    return generated


def _generate_dashboard_packet(
    campaign_dir: Path,
    candidate_id: str,
    *,
    packet_output_dir: str | Path | None = None,
    output_dir: str | Path | None,
    top_n: int,
    refresh_s: float | None,
    generate_packets_top_n: int,
    generate_packet_workers: int,
    cleanup_stale_packets_before_render: bool,
    status_json: str | Path | None,
    export_generate_packets_top_n: int | None = None,
    export_cleanup_stale_packets_before_render: bool | None = None,
    reload_modules: bool = True,
) -> dict[str, Any]:
    if reload_modules:
        _reload_visual_packet_modules_if_needed()
    packet_dir = Path(packet_generator_module.generate_packet(campaign_dir, candidate_id, output_dir=packet_output_dir))
    export_top_n = (
        int(generate_packets_top_n)
        if export_generate_packets_top_n is None
        else int(export_generate_packets_top_n)
    )
    export_cleanup_stale_packets = (
        bool(cleanup_stale_packets_before_render)
        if export_cleanup_stale_packets_before_render is None
        else bool(export_cleanup_stale_packets_before_render)
    )
    manifest = export_visual_dashboard(
        campaign_dir,
        output_dir=output_dir,
        top_n=top_n,
        refresh_s=refresh_s,
        generate_packets_top_n=export_top_n,
        generate_packet_workers=generate_packet_workers,
        cleanup_stale_packets_before_render=export_cleanup_stale_packets,
        status_json=status_json,
    )
    return {
        "ok": True,
        "candidate_id": candidate_id,
        "packet_dir": str(packet_dir),
        "manifest": manifest,
    }


def _queue_dashboard_packet_generation(
    campaign_dir: Path,
    candidate_id: str,
    *,
    packet_output_dir: str | Path | None,
    output_dir: str | Path | None,
    top_n: int,
    refresh_s: float | None,
    generate_packets_top_n: int,
    generate_packet_workers: int,
    cleanup_stale_packets_before_render: bool,
    status_json: str | Path | None,
    reload_modules: bool = True,
) -> dict[str, Any]:
    campaign_path = Path(campaign_dir).expanduser().resolve()
    packet_output_path = (
        Path(packet_output_dir).expanduser().resolve()
        if packet_output_dir is not None
        else campaign_path / "figures" / f"packet_{candidate_id}"
    )
    job_key = _packet_generation_job_key(campaign_path, candidate_id)

    def _worker() -> None:
        try:
            _generate_dashboard_packet(
                campaign_path,
                candidate_id,
                packet_output_dir=packet_output_path,
                output_dir=output_dir,
                top_n=top_n,
                refresh_s=refresh_s,
                generate_packets_top_n=generate_packets_top_n,
                generate_packet_workers=generate_packet_workers,
                cleanup_stale_packets_before_render=cleanup_stale_packets_before_render,
                status_json=status_json,
                export_generate_packets_top_n=0,
                export_cleanup_stale_packets_before_render=False,
                reload_modules=reload_modules,
            )
        finally:
            with _PACKET_GENERATION_JOBS_LOCK:
                current = _PACKET_GENERATION_JOBS.get(job_key)
                if current is threading.current_thread():
                    _PACKET_GENERATION_JOBS.pop(job_key, None)

    with _PACKET_GENERATION_JOBS_LOCK:
        existing = _PACKET_GENERATION_JOBS.get(job_key)
        if existing is not None and existing.is_alive():
            return {
                "ok": True,
                "queued": False,
                "running": True,
                "candidate_id": candidate_id,
                "packet_output_dir": str(packet_output_path),
            }
        thread = threading.Thread(
            target=_worker,
            name=f"hfo-packet-{candidate_id}",
            daemon=True,
        )
        _PACKET_GENERATION_JOBS[job_key] = thread
        thread.start()
    return {
        "ok": True,
        "queued": True,
        "running": True,
        "candidate_id": candidate_id,
        "packet_output_dir": str(packet_output_path),
    }


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


def _parameter_chips(parameters: dict[str, Any], *, packet: PacketInfo | None = None) -> str:
    contract = packet.manifest.get("parameter_contract") if packet is not None else {}
    if not isinstance(contract, dict):
        contract = {}
    preferred = list(contract.get("search_space_paths") or [])
    if not preferred:
        campaign_dir = packet.manifest.get("campaign_dir") if packet is not None else None
        preferred = parameter_display_order({}, campaign_dir=campaign_dir)
        preferred = [key for key in preferred if key in parameters or not str(key).startswith("optimizer_")]
    ordered = parameter_display_order(parameters, search_space_paths=preferred)
    chunks = []
    for name in ordered:
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
    dom_id: str | None = None,
) -> str:
    if not images:
        return ""
    open_attr = " open" if open_by_default else ""
    id_attr = f" id='{_esc(dom_id)}'" if dom_id else ""
    return (
        f"<details class='figure-group'{id_attr}{open_attr}>"
        f"<summary>{_esc(title)} <span>{len(images)} plots</span></summary>"
        f"{_gallery_html(images, output_dir=output_dir)}"
        "</details>"
    )


def _image_by_name(images: tuple[Path, ...]) -> dict[str, Path]:
    return {image.name: image for image in images}


def _condition_pair_html(
    title: str,
    control_image: Path | None,
    ketamine_image: Path | None,
    *,
    output_dir: Path,
    dom_id: str,
    open_by_default: bool = False,
) -> str:
    if control_image is None and ketamine_image is None:
        return ""
    open_attr = " open" if open_by_default else ""

    def column(condition: str, image: Path | None) -> str:
        if image is None:
            body = "<div class='missing'>No plot generated.</div>"
        else:
            body = _image_figure(image, output_dir=output_dir, caption=condition)
        return f"<div class='condition-column {condition.lower()}'><h3>{_esc(condition)}</h3>{body}</div>"

    return (
        f"<details class='figure-group condition-group' id='{_esc(dom_id)}'{open_attr}>"
        f"<summary>{_esc(title)} <span>control vs ketamine</span></summary>"
        "<div class='condition-grid'>"
        f"{column('Control', control_image)}"
        f"{column('Ketamine', ketamine_image)}"
        "</div></details>"
    )


def _frequency_kde_pairs(images: tuple[Path, ...], *, kind: str) -> dict[tuple[str, str | None], dict[str, Path]]:
    pairs: dict[tuple[str, str | None], dict[str, Path]] = {}
    for image in images:
        parsed = hfo_visuals.parse_kde_filename(image.name, kind=kind)
        if parsed is None:
            continue
        condition, group, variant = parsed
        pairs.setdefault((group, variant), {})[condition] = image
    return pairs


def _condition_comparison_sections(
    packet: PacketInfo,
    *,
    output_dir: Path,
    rank: int,
) -> tuple[str, set[Path]]:
    by_name = _image_by_name(packet.images)
    used: set[Path] = set()
    sections: list[str] = []

    for pair_spec in hfo_visuals.fixed_condition_pair_specs():
        control_image = by_name.get(pair_spec.control_file)
        ketamine_image = by_name.get(pair_spec.ketamine_file)
        used.update(path for path in (control_image, ketamine_image) if path is not None)
        sections.append(
            _condition_pair_html(
                pair_spec.title,
                control_image,
                ketamine_image,
                output_dir=output_dir,
                dom_id=f"{packet.candidate_id}-{pair_spec.dom_id_suffix}",
                open_by_default=pair_spec.open_by_default and rank == 1,
            )
        )

    for kind, title in [("1d", "Soma spike frequency 1D KDE"), ("2d", "Soma spike time/frequency 2D KDE")]:
        for (group, variant), pair in sorted(
            _frequency_kde_pairs(packet.images, kind=kind).items(),
            key=lambda item: (str(item[0][0]), "" if item[0][1] is None else str(item[0][1])),
        ):
            control_image = pair.get("control")
            ketamine_image = pair.get("ketamine")
            used.update(path for path in (control_image, ketamine_image) if path is not None)
            suffix = " (mod 200 ms)" if variant == "mod200" else ""
            sections.append(
                _condition_pair_html(
                    f"{title}: {group}{suffix}",
                    control_image,
                    ketamine_image,
                    output_dir=output_dir,
                    dom_id=f"{packet.candidate_id}-kde-{kind}-{group}{'-' + variant if variant else ''}",
                    open_by_default=kind == "1d" and variant is None and rank == 1,
                )
            )

    return "\n".join(section for section in sections if section), used


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


def _render_recent_table(
    rows: list[dict[str, Any]],
    *,
    top_n: int,
    recent_batch_name: str | None = None,
) -> str:
    headers = [
        "order",
        "candidate",
        "batch",
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
    for index, row in enumerate(
        _recent_rows(rows, limit=int(top_n), recent_batch_name=recent_batch_name),
        start=1,
    ):
        s = _metric_summary(row)
        cells = [
            index,
            s["candidate_id"],
            s["batch_name"],
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


def _render_packet_card(
    row: dict[str, Any],
    packet: PacketInfo | None,
    *,
    output_dir: Path,
    rank: int,
    dom_prefix: str = "best",
) -> str:
    s = _metric_summary(row)
    candidate_id = str(s["candidate_id"] or "")
    open_attr = " open" if rank <= 3 else ""
    candidate_dom_id = (
        f"{dom_prefix}-candidate-{candidate_id}"
        if candidate_id
        else f"{dom_prefix}-candidate-rank-{rank}"
    )
    packet_meta = ""
    primary_psd_html = (
        "<div class='missing'>"
        "<div>No PSD packet has been generated for this candidate yet.</div>"
        + (
            f"<button type='button' class='generate-packet-button' data-generate-packet "
            f"data-candidate-id='{_esc(candidate_id)}'>Generate packet</button>"
            if candidate_id
            else ""
        )
        + "</div>"
    )
    secondary_psd_html = ""
    comparison_html = ""
    other_gallery_html = ""
    contact_html = ""
    if packet is not None:
        when = datetime.fromtimestamp(packet.mtime).isoformat(timespec="seconds") if packet.mtime else "-"
        style = packet.manifest.get("visual_style_version", "legacy")
        packet_meta = (
            f"<span>Packet: {_esc(packet.packet_dir.name)}</span>"
            f"<span>Updated: {_esc(when)}</span>"
            f"<span>Visual style: {_esc(style)}</span>"
        )
        primary_psd = _primary_psd_image(packet.images)
        if primary_psd is not None:
            primary_psd_html = _image_figure(
                primary_psd,
                output_dir=output_dir,
                css_class="primary-psd",
                caption="Live PSD overlay with scoring template",
            )
        psd_images = _psd_images(packet.images)
        supporting_psd = [image for image in psd_images if image != primary_psd]
        secondary_psd_html = _details_gallery(
            "PSD details",
            supporting_psd,
            output_dir=output_dir,
            open_by_default=rank == 1,
            dom_id=f"{candidate_dom_id}-psd-details",
        )
        excluded = set(psd_images)
        comparison_html, comparison_images = _condition_comparison_sections(packet, output_dir=output_dir, rank=rank)
        excluded.update(comparison_images)
        other_images = [image for image in packet.images if image not in excluded]
        other_gallery_html = _details_gallery(
            "Additional diagnostics",
            other_images,
            output_dir=output_dir,
            open_by_default=False,
            dom_id=f"{candidate_dom_id}-additional",
        )
        if packet.contact_sheet is not None:
            contact_html = (
                f"<details class='figure-group' id='{_esc(candidate_dom_id)}-contact'>"
                "<summary>Contact sheet <span>all plots</span></summary>"
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
<details class="candidate" id="{_esc(candidate_dom_id)}"{open_attr}>
  <summary>
    <span class="rank">#{rank}</span>
    <span class="candidate-id">{_esc(candidate_id)}</span>
    <span class="batch">{_esc(s.get("batch_name") or "")}</span>
    <span class="badge-row">{''.join(f"<em>{_esc(badge)}</em>" for badge in badges)}</span>
  </summary>
  <div class="card-body">
    <div class="packet-meta">{packet_meta}</div>
    <div class="chips">{penalty_html}</div>
    <div class="chips params">{_parameter_chips(s["params"], packet=packet)}</div>
    {primary_psd_html}
    {secondary_psd_html}
    {comparison_html}
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
    generated_at: str,
    recent_batch_name: str | None = None,
) -> str:
    tab_specs = hfo_visuals.dashboard_tabs()
    best_rows = rows[: int(top_n)]
    recent_tab_spec = next((spec for spec in tab_specs if spec.key == "recent"), None)
    recent_limit = int(top_n)
    if recent_tab_spec is not None and recent_tab_spec.display_limit is not None:
        recent_limit = min(int(top_n), int(recent_tab_spec.display_limit))
    recent_rows = _recent_rows(rows, limit=recent_limit, recent_batch_name=recent_batch_name)
    best_packet_cards = "\n".join(
        _render_packet_card(
            row,
            packets.get(str(row.get("candidate_id"))),
            output_dir=output_dir,
            rank=index,
            dom_prefix="best",
        )
        for index, row in enumerate(best_rows, start=1)
    )
    recent_packet_cards = "\n".join(
        _render_packet_card(
            row,
            packets.get(str(row.get("candidate_id"))),
            output_dir=output_dir,
            rank=index,
            dom_prefix="recent",
        )
        for index, row in enumerate(recent_rows, start=1)
    )
    tab_views = {
        "best": {
            "table_html": _render_top_table(best_rows, top_n=top_n),
            "packet_cards": best_packet_cards,
        },
        "recent": {
            "table_html": _render_recent_table(rows, top_n=recent_limit, recent_batch_name=recent_batch_name),
            "packet_cards": recent_packet_cards,
        },
    }
    tab_nav_html = "\n".join(
        (
            f"<button class='tab-button' type='button' role='tab' data-tab-button "
            f"data-tab-target='tab-{_esc(spec.key)}' aria-controls='tab-{_esc(spec.key)}' "
            f"aria-selected='{'true' if index == 0 else 'false'}'>{_esc(spec.label)}</button>"
        )
        for index, spec in enumerate(tab_specs)
    )
    tab_panels_html = "\n".join(
        (
            f"<div id='tab-{_esc(spec.key)}' class='tab-panel' role='tabpanel'"
            f"{'' if index == 0 else ' hidden'}>"
            f"<section><h2>{_esc(spec.table_heading)}</h2>"
            f"<div class='table-wrap'>{tab_views[spec.key]['table_html']}</div></section>"
            f"<section><h2>{_esc(spec.packet_heading)}</h2>"
            f"<div style='padding: 0 14px 14px;'>{tab_views[spec.key]['packet_cards']}</div></section>"
            "</div>"
        )
        for index, spec in enumerate(tab_specs)
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
    .tab-shell {{
      display: flex;
      flex-direction: column;
      gap: 14px;
      margin-top: 18px;
    }}
    .tab-bar {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      position: sticky;
      top: 76px;
      z-index: 9;
      padding: 8px 0 2px;
      background: rgba(247, 248, 251, 0.96);
      backdrop-filter: blur(8px);
    }}
    .tab-button {{
      appearance: none;
      border: 1px solid #dbe3ef;
      background: #ffffff;
      color: #334155;
      border-radius: 8px;
      padding: 8px 12px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    .tab-button[aria-selected="true"] {{
      background: #eff6ff;
      border-color: #93c5fd;
      color: #1d4ed8;
    }}
    .tab-panel[hidden] {{ display: none !important; }}
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
    .condition-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      padding: 12px;
      border-top: 1px solid #e3e8f2;
    }}
    .condition-column {{
      min-width: 0;
    }}
    .condition-column h3 {{
      margin: 0 0 8px;
      font-size: 13px;
      line-height: 1.2;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    .condition-column.control h3 {{ color: var(--blue); }}
    .condition-column.ketamine h3 {{ color: var(--red); }}
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
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      gap: 12px;
      padding: 18px;
      border: 1px dashed #cbd5e1;
      border-radius: 8px;
      color: var(--muted);
      background: #fbfcfe;
    }}
    .generate-packet-button {{
      appearance: none;
      border: 1px solid #c7d2fe;
      border-radius: 8px;
      padding: 8px 12px;
      background: #eff6ff;
      color: #1d4ed8;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    .generate-packet-button:hover {{
      background: #dbeafe;
    }}
    .generate-packet-button[data-pending="true"] {{
      background: #dbeafe;
      border-color: #93c5fd;
      color: #1d4ed8;
      cursor: progress;
    }}
    .generate-packet-button:disabled {{
      opacity: 0.65;
      cursor: progress;
    }}
    .generated {{ color: var(--green); }}
    .refresh-indicator {{
      position: fixed;
      right: 14px;
      bottom: 14px;
      z-index: 20;
      max-width: min(420px, calc(100vw - 28px));
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.96);
      color: var(--muted);
      font-size: 12px;
      box-shadow: 0 8px 22px rgba(15, 23, 42, 0.10);
      opacity: 0;
      transform: translateY(6px);
      transition: opacity 160ms ease, transform 160ms ease;
      pointer-events: none;
    }}
    .refresh-indicator.visible {{
      opacity: 1;
      transform: translateY(0);
    }}
    @media (max-width: 760px) {{
      header {{ padding: 14px 16px; }}
      main {{ padding: 16px; }}
      details.candidate > summary {{ grid-template-columns: 40px 88px 1fr; }}
      .batch {{ display: none; }}
      .gallery {{ grid-template-columns: 1fr; }}
      .condition-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body data-generated-at="{_esc(generated_at)}" data-refresh-s="{_esc(refresh_s or 0)}">
  <header>
    <h1>HFO Campaign Visual Dashboard</h1>
    <div class="subtle">{_esc(campaign_dir)}{f" | updates in place every {int(refresh_s)} s" if refresh_s and refresh_s > 0 else ""}</div>
  </header>
  <main id="dashboard-main" data-active-tab="tab-best">
    <div class="stats">{_render_status(campaign_dir, rows, status_payload)}</div>
    {generated_html}
    <div class="tab-shell">
      <nav class="tab-bar" aria-label="Dashboard sections" role="tablist">
        {tab_nav_html}
      </nav>
      {tab_panels_html}
    </div>
  </main>
  <div id="refresh-indicator" class="refresh-indicator" aria-live="polite"></div>
  <script>
    (() => {{
      const refreshSeconds = Number(document.body.dataset.refreshS || 0);
      const indicator = document.getElementById("refresh-indicator");
      const pendingPacketRequests = new Set();
      let refreshing = false;

      function showIndicator(text) {{
        if (!indicator) return;
        indicator.textContent = text;
        indicator.classList.add("visible");
        window.setTimeout(() => indicator.classList.remove("visible"), 1800);
      }}

      function sleep(ms) {{
        return new Promise((resolve) => window.setTimeout(resolve, ms));
      }}

      async function pollDashboardManifestUntilUpdated(candidateId, baselineGeneratedAt) {{
        const deadline = Date.now() + 10 * 60 * 1000;
        while (Date.now() < deadline) {{
          try {{
            const manifestResp = await fetch("manifest.json?cache=" + Date.now(), {{ cache: "no-store" }});
            if (manifestResp.ok) {{
              const manifest = await manifestResp.json();
              const generatedAt = String(manifest.generated_at || "");
              if (generatedAt && generatedAt !== baselineGeneratedAt) {{
                showIndicator(`Packet ready for ${{candidateId}}. Reloading dashboard...`);
                window.setTimeout(() => window.location.reload(), 350);
                return true;
              }}
            }}
          }} catch (exc) {{
            console.warn("Packet generation poll failed", exc);
          }}
          await sleep(2000);
        }}
        return false;
      }}

      function setActiveTab(tabId) {{
        const main = document.getElementById("dashboard-main");
        if (!main) return;
        const buttons = main.querySelectorAll("[data-tab-button]");
        const panels = main.querySelectorAll(".tab-panel");
        let resolved = tabId;
        if (!resolved || !document.getElementById(resolved)) {{
          resolved = "tab-best";
        }}
        main.dataset.activeTab = resolved;
        buttons.forEach((button) => {{
          const selected = button.dataset.tabTarget === resolved;
          button.setAttribute("aria-selected", selected ? "true" : "false");
        }});
        panels.forEach((panel) => {{
          panel.hidden = panel.id !== resolved;
        }});
      }}

      document.addEventListener("click", (event) => {{
        const button = event.target.closest("[data-tab-button]");
        if (!button) return;
        setActiveTab(button.dataset.tabTarget || "tab-best");
      }});

      document.addEventListener("click", async (event) => {{
        const button = event.target.closest("[data-generate-packet]");
        if (!button) return;
        event.preventDefault();
        const candidateId = String(button.dataset.candidateId || "").trim();
        if (!candidateId) return;
        if (pendingPacketRequests.has(candidateId)) return;
        pendingPacketRequests.add(candidateId);
        const originalText = button.textContent || "Generate packet";
        const baselineGeneratedAt = String(document.body.dataset.generatedAt || "");
        button.dataset.pending = "true";
        button.setAttribute("aria-busy", "true");
        button.textContent = "Queued...";
        showIndicator(`Queued packet generation for ${{candidateId}}...`);
        try {{
          const response = await fetch("{GENERATE_PACKET_ENDPOINT}", {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify({{ candidate_id: candidateId }}),
            cache: "no-store",
          }});
          const payload = await response.json().catch(() => ({{}}));
          if (!response.ok || !payload.ok) {{
            throw new Error(String(payload.error || `Packet generation failed with status ${{response.status}}`));
          }}
          showIndicator(
            payload.queued === false
              ? `Packet already running for ${{candidateId}}. Watching for completion...`
              : `Packet queued for ${{candidateId}}. Watching for completion...`
          );
          const updated = await pollDashboardManifestUntilUpdated(candidateId, baselineGeneratedAt);
          if (!updated) {{
            showIndicator(`Packet generation is still running for ${{candidateId}}.`);
            pendingPacketRequests.delete(candidateId);
            button.removeAttribute("aria-busy");
            button.dataset.pending = "";
            button.textContent = originalText;
          }}
        }} catch (exc) {{
          console.warn("Packet generation failed", exc);
          showIndicator(`Packet generation failed for ${{candidateId}}.`);
          pendingPacketRequests.delete(candidateId);
          button.removeAttribute("aria-busy");
          button.dataset.pending = "";
          button.textContent = originalText;
        }}
      }});

      function captureState() {{
        const openDetails = Array.from(document.querySelectorAll("details[id][open]")).map((node) => node.id);
        const active = document.activeElement && document.activeElement.id ? document.activeElement.id : null;
        const activeTab = document.getElementById("dashboard-main")?.dataset.activeTab || "tab-best";
        return {{ scrollX: window.scrollX, scrollY: window.scrollY, openDetails, active, activeTab }};
      }}

      function restoreState(state) {{
        setActiveTab(state.activeTab || "tab-best");
        for (const id of state.openDetails || []) {{
          const node = document.getElementById(id);
          if (node && node.tagName.toLowerCase() === "details") {{
            node.open = true;
          }}
        }}
        if (state.active) {{
          const active = document.getElementById(state.active);
          if (active && typeof active.focus === "function") active.focus({{ preventScroll: true }});
        }}
        requestAnimationFrame(() => window.scrollTo(state.scrollX || 0, state.scrollY || 0));
      }}

      setActiveTab(document.getElementById("dashboard-main")?.dataset.activeTab || "tab-best");

      if (Number.isFinite(refreshSeconds) && refreshSeconds > 0) {{
        async function refreshIfChanged() {{
          if (refreshing) return;
          refreshing = true;
          try {{
            const manifestResp = await fetch("manifest.json?cache=" + Date.now(), {{ cache: "no-store" }});
            if (!manifestResp.ok) return;
            const manifest = await manifestResp.json();
            const generatedAt = String(manifest.generated_at || "");
            if (!generatedAt || generatedAt === document.body.dataset.generatedAt) return;

            const state = captureState();
            const htmlResp = await fetch("index.html?cache=" + Date.now(), {{ cache: "no-store" }});
            if (!htmlResp.ok) return;
            const nextText = await htmlResp.text();
            const nextDoc = new DOMParser().parseFromString(nextText, "text/html");
            const nextMain = nextDoc.getElementById("dashboard-main");
            const currentMain = document.getElementById("dashboard-main");
            if (!nextMain || !currentMain) return;
            currentMain.replaceWith(document.importNode(nextMain, true));
            document.body.dataset.generatedAt = generatedAt;
            restoreState(state);
            showIndicator("Dashboard updated without changing scroll position.");
          }} catch (exc) {{
            console.warn("Dashboard refresh failed", exc);
          }} finally {{
            refreshing = false;
          }}
        }}

        window.setInterval(refreshIfChanged, Math.max(refreshSeconds, 5) * 1000);
      }}
    }})();
  </script>
</body>
</html>
"""


def export_visual_dashboard(
    campaign_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    refresh_s: float | None = DEFAULT_REFRESH_S,
    generate_packets_top_n: int = DEFAULT_GENERATE_PACKETS_TOP_N,
    generate_packet_workers: int = DEFAULT_PACKET_GENERATION_WORKERS,
    cleanup_stale_packets_before_render: bool = DEFAULT_CLEANUP_STALE_PACKETS,
    status_json: str | Path | None = None,
) -> dict[str, Any]:
    """Write ``index.html`` for one campaign and return a small manifest."""
    _reload_visual_packet_modules_if_needed()
    campaign_path = Path(campaign_dir).expanduser().resolve()
    output_path = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else campaign_path / DEFAULT_OUTPUT_SUBDIR
    )
    output_path.mkdir(parents=True, exist_ok=True)
    with _dashboard_export_lock(output_path):
        if cleanup_stale_packets_before_render:
            cleanup_stale_packets(campaign_path, dry_run=False)

        rows = _load_ranked_rows(campaign_path)
        recent_batch_name = _latest_completed_batch_name(campaign_path)
        packet_candidate_count = len(
            {
                str(row.get("candidate_id") or "")
                for row in [
                    *rows[: int(generate_packets_top_n)],
                    *_recent_rows(
                        rows,
                        limit=int(generate_packets_top_n),
                        recent_batch_name=recent_batch_name,
                    ),
                ]
                if str(row.get("candidate_id") or "")
            }
        )
        generated_packets = _generate_missing_packets(
            campaign_path,
            rows,
            top_n=int(generate_packets_top_n),
            workers=int(generate_packet_workers),
            recent_batch_name=recent_batch_name,
        )
        packets = find_candidate_packets(campaign_path)
        status_path = Path(status_json).expanduser().resolve() if status_json else (REPO_ROOT / SUMMARY_STATUS_PATH)
        payload = _status_payload(campaign_path, status_path)
        generated_at = datetime.now().isoformat(timespec="seconds")
        html_text = _render_html(
            campaign_dir=campaign_path,
            output_dir=output_path,
            rows=rows,
            packets=packets,
            top_n=int(top_n),
            refresh_s=refresh_s,
            generated_packets=generated_packets,
            status_payload=payload,
            generated_at=generated_at,
            recent_batch_name=recent_batch_name,
        )
        index_path = output_path / "index.html"
        index_tmp = index_path.with_name(f".{index_path.name}.tmp")
        index_tmp.write_text(html_text)
        os.replace(index_tmp, index_path)
        server_root, url_path = _dashboard_server_root_and_url(output_path, campaign_path)
        entrypoint_path = _write_dashboard_entrypoint(server_root, url_path)
        manifest = {
            "campaign_dir": str(campaign_path),
            "output_dir": str(output_path),
            "index_html": str(index_path),
            "entrypoint_html": str(entrypoint_path),
            "entrypoint_url_path": url_path,
            "generated_at": generated_at,
            "candidate_rows": len(rows),
            "packet_count": len(packets),
            "generated_packets": [str(path) for path in generated_packets],
            "generate_packet_workers": _effective_packet_generation_workers(
                int(generate_packet_workers),
                packet_candidate_count,
            )
            if int(generate_packets_top_n) > 0
            else 0,
            "cleanup_stale_packets_before_render": cleanup_stale_packets_before_render,
            "top_candidate_id": rows[0].get("candidate_id") if rows else None,
            "top_score": rows[0].get("pair_score") if rows else None,
        }
        manifest_path = output_path / "manifest.json"
        manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.tmp")
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(manifest_tmp, manifest_path)
        return manifest


def watch_visual_dashboard(
    campaign_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    refresh_s: float = DEFAULT_REFRESH_S,
    generate_packets_top_n: int = DEFAULT_GENERATE_PACKETS_TOP_N,
    generate_packet_workers: int = DEFAULT_PACKET_GENERATION_WORKERS,
    cleanup_stale_packets_before_render: bool = DEFAULT_CLEANUP_STALE_PACKETS,
    status_json: str | Path | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    campaign_path = Path(campaign_dir).expanduser().resolve()
    archive = campaign_path / "candidate_archive.jsonl"
    figures = campaign_path / "figures"
    last_signature: tuple[int, ...] | None = None
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        archive_sig = int(archive.stat().st_mtime_ns if archive.exists() else 0) ^ int(archive.stat().st_size if archive.exists() else 0)
        figures_sig = int(figures.stat().st_mtime_ns if figures.exists() else 0)
        status_path = Path(status_json).expanduser().resolve() if status_json else (REPO_ROOT / SUMMARY_STATUS_PATH)
        status_sig = int(status_path.stat().st_mtime_ns if status_path.exists() else 0)
        signature = (archive_sig, figures_sig, status_sig, *_style_source_signature())
        try:
            if signature != last_signature:
                manifest = export_visual_dashboard(
                    campaign_path,
                    output_dir=output_dir,
                    top_n=top_n,
                    refresh_s=refresh_s,
                    generate_packets_top_n=generate_packets_top_n,
                    generate_packet_workers=generate_packet_workers,
                    cleanup_stale_packets_before_render=cleanup_stale_packets_before_render,
                    status_json=status_path,
                )
                print(
                    "Wrote visual dashboard for {candidate_rows} candidates "
                    "({packet_count} packets) to {index_html}".format(**manifest),
                    flush=True,
                )
                last_signature = signature
        except Exception as exc:  # pragma: no cover - retry behavior exercised through regression script
            print(
                f"[HFO dashboard watch] export failed: {exc!r}. Will retry after {max(float(refresh_s), 1.0):.1f}s",
                flush=True,
            )
        _wait_with_stop(max(float(refresh_s), 1.0), stop_event)


def _dashboard_runtime_command(
    subcommand: str,
    campaign_path: Path,
    *,
    output_path: Path,
    top_n: int,
    refresh_s: float,
    generate_packets_top_n: int,
    generate_packet_workers: int,
    cleanup_stale_packets_before_render: bool,
    status_path: Path,
    host: str | None = None,
    port: int | None = None,
    supervise_s: float | None = None,
    stale_after_s: float | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(subcommand),
        str(campaign_path),
        "--output-dir",
        str(output_path),
        "--top-n",
        str(int(top_n)),
        "--refresh-s",
        str(float(refresh_s)),
        "--generate-packets-top-n",
        str(int(generate_packets_top_n)),
        "--generate-packet-workers",
        str(int(generate_packet_workers)),
        "--status-json",
        str(status_path),
    ]
    if not cleanup_stale_packets_before_render:
        command.append("--no-cleanup-stale-packets")
    if host is not None:
        command.extend(["--host", str(host)])
    if port is not None:
        command.extend(["--port", str(int(port))])
    if supervise_s is not None:
        command.extend(["--supervise-s", str(float(supervise_s))])
    if stale_after_s is not None:
        command.extend(["--stale-after-s", str(float(stale_after_s))])
    return command


def _watcher_is_stale(
    campaign_path: Path,
    output_path: Path,
    *,
    status_path: Path,
    refresh_s: float,
    stale_after_s: float,
) -> bool:
    sources_mtime = _watch_sources_mtime(campaign_path, status_path)
    outputs_mtime = _dashboard_outputs_mtime(output_path)
    if outputs_mtime <= 0.0:
        return True
    allowed_lag_s = max(float(stale_after_s), 2.5 * max(float(refresh_s), 1.0))
    return (sources_mtime - outputs_mtime) > allowed_lag_s


def _ensure_visual_dashboard_sidecars(
    campaign_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    refresh_s: float = DEFAULT_REFRESH_S,
    generate_packets_top_n: int = DEFAULT_RUNTIME_GENERATE_PACKETS_TOP_N,
    generate_packet_workers: int = DEFAULT_PACKET_GENERATION_WORKERS,
    cleanup_stale_packets_before_render: bool = DEFAULT_CLEANUP_STALE_PACKETS,
    status_json: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 6006,
    stale_after_s: float = DEFAULT_STALE_AFTER_S,
) -> dict[str, Any]:
    campaign_path = Path(campaign_dir).expanduser().resolve()
    output_path = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else campaign_path / DEFAULT_OUTPUT_SUBDIR
    )
    output_path.mkdir(parents=True, exist_ok=True)
    status_path = Path(status_json).expanduser().resolve() if status_json else (REPO_ROOT / SUMMARY_STATUS_PATH)
    runtime_dir = _runtime_dir(output_path)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "campaign_dir": str(campaign_path),
        "output_dir": str(output_path),
        "runtime_dir": str(runtime_dir),
        "status_json": str(status_path),
        "host": str(host),
        "port": int(port),
    }

    watcher_cmd = _dashboard_runtime_command(
        "watch",
        campaign_path,
        output_path=output_path,
        top_n=top_n,
        refresh_s=refresh_s,
        generate_packets_top_n=generate_packets_top_n,
        generate_packet_workers=generate_packet_workers,
        cleanup_stale_packets_before_render=cleanup_stale_packets_before_render,
        status_path=status_path,
    )
    watcher_info = _read_runtime_process_info(output_path, "watcher")
    watcher_alive = (
        watcher_info is not None
        and _pid_is_alive(watcher_info.pid)
        and _process_matches_command(watcher_info.pid, watcher_cmd)
    )
    watcher_stale = _watcher_is_stale(
        campaign_path,
        output_path,
        status_path=status_path,
        refresh_s=refresh_s,
        stale_after_s=stale_after_s,
    )
    if watcher_alive and watcher_stale and watcher_info is not None:
        _terminate_process(watcher_info.pid)
        watcher_alive = False
    if not watcher_alive:
        paths = _runtime_process_paths(output_path, "watcher")
        watcher_info = _spawn_detached_process(
            watcher_cmd,
            cwd=REPO_ROOT,
            stdout_path=paths["stdout"],
            stderr_path=paths["stderr"],
            meta_path=paths["pid"],
            meta={
                "kind": "watcher",
                "campaign_dir": str(campaign_path),
                "output_dir": str(output_path),
            },
        )
        watcher_alive = True
    result["watcher"] = {
        "pid": watcher_info.pid if watcher_info is not None else None,
        "alive": bool(watcher_alive),
        "stale": bool(watcher_stale),
        "stdout_log": str(_runtime_process_paths(output_path, "watcher")["stdout"]),
        "stderr_log": str(_runtime_process_paths(output_path, "watcher")["stderr"]),
    }

    server_cmd = _dashboard_runtime_command(
        "serve-static",
        campaign_path,
        output_path=output_path,
        top_n=top_n,
        refresh_s=refresh_s,
        generate_packets_top_n=generate_packets_top_n,
        generate_packet_workers=generate_packet_workers,
        cleanup_stale_packets_before_render=cleanup_stale_packets_before_render,
        status_path=status_path,
        host=host,
        port=port,
    )
    server_info = _read_runtime_process_info(output_path, "server")
    server_alive = (
        server_info is not None
        and _pid_is_alive(server_info.pid)
        and _process_matches_command(server_info.pid, server_cmd)
    )
    external_port_in_use = _port_in_use(host, port)
    if not server_alive:
        paths = _runtime_process_paths(output_path, "server")
        server_info = _spawn_detached_process(
            server_cmd,
            cwd=REPO_ROOT,
            stdout_path=paths["stdout"],
            stderr_path=paths["stderr"],
            meta_path=paths["pid"],
            meta={
                "kind": "server",
                "campaign_dir": str(campaign_path),
                "output_dir": str(output_path),
                "host": str(host),
                "port": int(port),
            },
        )
        server_alive = True
        external_port_in_use = True
    result["server"] = {
        "pid": server_info.pid if server_info is not None else None,
        "alive": bool(server_alive),
        "external_port_in_use": bool(external_port_in_use),
        "stdout_log": str(_runtime_process_paths(output_path, "server")["stdout"]),
        "stderr_log": str(_runtime_process_paths(output_path, "server")["stderr"]),
    }
    _write_json_atomic(runtime_dir / "sidecars.status.json", result)
    return result


def watch_visual_dashboard_runtime(
    campaign_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    refresh_s: float = DEFAULT_REFRESH_S,
    generate_packets_top_n: int = DEFAULT_RUNTIME_GENERATE_PACKETS_TOP_N,
    generate_packet_workers: int = DEFAULT_PACKET_GENERATION_WORKERS,
    cleanup_stale_packets_before_render: bool = DEFAULT_CLEANUP_STALE_PACKETS,
    status_json: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 6006,
    supervise_s: float = DEFAULT_WATCHDOG_SUPERVISE_S,
    stale_after_s: float = DEFAULT_STALE_AFTER_S,
) -> None:
    campaign_path = Path(campaign_dir).expanduser().resolve()
    output_path = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else campaign_path / DEFAULT_OUTPUT_SUBDIR
    )
    runtime_dir = _runtime_dir(output_path)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    pid_paths = _runtime_process_paths(output_path, "watchdog")
    _write_json_atomic(
        pid_paths["pid"],
        {
            "kind": "watchdog",
            "campaign_dir": str(campaign_path),
            "output_dir": str(output_path),
            "pid": int(os.getpid()),
            "command": list(sys.argv),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    status_file = runtime_dir / "watchdog.status.json"
    try:
        while True:
            status_payload: dict[str, Any]
            try:
                status_payload = _ensure_visual_dashboard_sidecars(
                    campaign_path,
                    output_dir=output_path,
                    top_n=top_n,
                    refresh_s=refresh_s,
                    generate_packets_top_n=generate_packets_top_n,
                    generate_packet_workers=generate_packet_workers,
                    cleanup_stale_packets_before_render=cleanup_stale_packets_before_render,
                    status_json=status_json,
                    host=host,
                    port=port,
                    stale_after_s=stale_after_s,
                )
                status_payload["watchdog"] = {
                    "pid": int(os.getpid()),
                    "ok": True,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
            except Exception as exc:  # pragma: no cover - exercised through live runtime, not unit tests
                status_payload = {
                    "campaign_dir": str(campaign_path),
                    "output_dir": str(output_path),
                    "watchdog": {
                        "pid": int(os.getpid()),
                        "ok": False,
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "error": repr(exc),
                    },
                }
                print(f"[HFO dashboard runtime] ensure failed: {exc!r}", flush=True)
            _write_json_atomic(status_file, status_payload)
            time.sleep(max(float(supervise_s), 1.0))
    finally:
        watcher_info = _read_runtime_process_info(output_path, "watcher")
        if watcher_info is not None:
            _terminate_process(watcher_info.pid)
        server_info = _read_runtime_process_info(output_path, "server")
        if server_info is not None:
            _terminate_process(server_info.pid)


def ensure_visual_dashboard_runtime(
    campaign_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    refresh_s: float = DEFAULT_REFRESH_S,
    generate_packets_top_n: int = DEFAULT_RUNTIME_GENERATE_PACKETS_TOP_N,
    generate_packet_workers: int = DEFAULT_PACKET_GENERATION_WORKERS,
    cleanup_stale_packets_before_render: bool = DEFAULT_CLEANUP_STALE_PACKETS,
    status_json: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 6006,
    supervise_s: float = DEFAULT_WATCHDOG_SUPERVISE_S,
    stale_after_s: float = DEFAULT_STALE_AFTER_S,
) -> dict[str, Any]:
    campaign_path = Path(campaign_dir).expanduser().resolve()
    output_path = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else campaign_path / DEFAULT_OUTPUT_SUBDIR
    )
    runtime_dir = _runtime_dir(output_path)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    watchdog_info = _read_runtime_process_info(output_path, "watchdog")
    watchdog_alive = (
        watchdog_info is not None
        and _pid_is_alive(watchdog_info.pid)
        and _process_matches_command(watchdog_info.pid, command)
    )
    if not watchdog_alive:
        paths = _runtime_process_paths(output_path, "watchdog")
        command = _dashboard_runtime_command(
            "watchdog",
            campaign_path,
            output_path=output_path,
            top_n=top_n,
            refresh_s=refresh_s,
            generate_packets_top_n=generate_packets_top_n,
            generate_packet_workers=generate_packet_workers,
            cleanup_stale_packets_before_render=cleanup_stale_packets_before_render,
            status_path=Path(status_json).expanduser().resolve() if status_json else (REPO_ROOT / SUMMARY_STATUS_PATH),
            host=host,
            port=port,
            supervise_s=supervise_s,
            stale_after_s=stale_after_s,
        )
        watchdog_info = _spawn_detached_process(
            command,
            cwd=REPO_ROOT,
            stdout_path=paths["stdout"],
            stderr_path=paths["stderr"],
            meta_path=paths["pid"],
            meta={
                "kind": "watchdog",
                "campaign_dir": str(campaign_path),
                "output_dir": str(output_path),
                "host": str(host),
                "port": int(port),
            },
        )
        watchdog_alive = True
    sidecars = _ensure_visual_dashboard_sidecars(
        campaign_path,
        output_dir=output_path,
        top_n=top_n,
        refresh_s=refresh_s,
        generate_packets_top_n=generate_packets_top_n,
        generate_packet_workers=generate_packet_workers,
        cleanup_stale_packets_before_render=cleanup_stale_packets_before_render,
        status_json=status_json,
        host=host,
        port=port,
        stale_after_s=stale_after_s,
    )
    status_file = runtime_dir / "watchdog.status.json"
    return {
        "campaign_dir": str(campaign_path),
        "output_dir": str(output_path),
        "runtime_dir": str(runtime_dir),
        "watchdog": {
            "pid": watchdog_info.pid if watchdog_info is not None else None,
            "alive": bool(watchdog_alive),
            "stdout_log": str(_runtime_process_paths(output_path, "watchdog")["stdout"]),
            "stderr_log": str(_runtime_process_paths(output_path, "watchdog")["stderr"]),
            "status_file": str(status_file),
        },
        "sidecars": sidecars,
    }


def stop_visual_dashboard_runtime(
    campaign_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    campaign_path = Path(campaign_dir).expanduser().resolve()
    output_path = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else campaign_path / DEFAULT_OUTPUT_SUBDIR
    )
    stopped: dict[str, Any] = {"campaign_dir": str(campaign_path), "output_dir": str(output_path)}
    token_map = {
        "watchdog": [str(Path(__file__).resolve()), " watchdog ", str(campaign_path)],
        "watcher": [str(Path(__file__).resolve()), " watch ", str(campaign_path)],
        "server": [str(Path(__file__).resolve()), " serve-static ", str(campaign_path)],
    }
    for kind in ("watchdog", "watcher", "server"):
        info = _read_runtime_process_info(output_path, kind)
        killed_pids: list[int] = []
        if info is None:
            pid = None
        else:
            pid = info.pid
            _terminate_process(info.pid)
            killed_pids.append(int(info.pid))
        for match_pid in _matching_pids(token_map[kind]):
            if match_pid == os.getpid():
                continue
            _terminate_process(match_pid)
            killed_pids.append(int(match_pid))
        stopped[kind] = {
            "pid": pid,
            "stopped": bool(killed_pids),
            "killed_pids": sorted(set(killed_pids)),
        }
    return stopped


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
    generate_packets_top_n: int = DEFAULT_GENERATE_PACKETS_TOP_N,
    generate_packet_workers: int = DEFAULT_PACKET_GENERATION_WORKERS,
    cleanup_stale_packets_before_render: bool = DEFAULT_CLEANUP_STALE_PACKETS,
    status_json: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 6006,
) -> None:
    manifest = export_visual_dashboard(
        campaign_dir,
        output_dir=output_dir,
        top_n=top_n,
        refresh_s=refresh_s,
        generate_packets_top_n=generate_packets_top_n,
        generate_packet_workers=generate_packet_workers,
        cleanup_stale_packets_before_render=cleanup_stale_packets_before_render,
        status_json=status_json,
    )
    output_path = Path(manifest["output_dir"])
    campaign_path = Path(manifest["campaign_dir"])
    _serve_dashboard_server(
        campaign_path,
        output_path,
        output_dir=output_dir,
        top_n=top_n,
        refresh_s=refresh_s,
        generate_packets_top_n=generate_packets_top_n,
        generate_packet_workers=generate_packet_workers,
        cleanup_stale_packets_before_render=cleanup_stale_packets_before_render,
        status_json=status_json,
        host=host,
        port=port,
        entrypoint_path=Path(manifest.get("entrypoint_html") or ""),
    )


def _serve_dashboard_server(
    campaign_path: Path,
    output_path: Path,
    *,
    output_dir: str | Path | None,
    top_n: int,
    refresh_s: float,
    generate_packets_top_n: int,
    generate_packet_workers: int,
    cleanup_stale_packets_before_render: bool,
    status_json: str | Path | None,
    host: str,
    port: int,
    entrypoint_path: Path | None = None,
) -> None:
    output_path = Path(output_path).expanduser().resolve()
    campaign_path = Path(campaign_path).expanduser().resolve()
    server_root, url_path = _dashboard_server_root_and_url(output_path, campaign_path)
    if entrypoint_path is None or not str(entrypoint_path):
        entrypoint_path = _write_dashboard_entrypoint(server_root, url_path)
    else:
        entrypoint_path = Path(entrypoint_path)
        if not entrypoint_path.exists():
            entrypoint_path = _write_dashboard_entrypoint(server_root, url_path)
    class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("directory", str(server_root))
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

        def do_POST(self) -> None:  # noqa: N802 - HTTP handler API
            if urlparse(self.path).path != GENERATE_PACKET_ENDPOINT:
                self.send_error(404, "Not found")
                return
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
                result = _queue_dashboard_packet_generation(
                    campaign_path,
                    candidate_id,
                    packet_output_dir=campaign_path / "figures" / f"packet_{candidate_id}",
                    output_dir=output_dir,
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

    server = http.server.ThreadingHTTPServer((host, int(port)), DashboardRequestHandler)
    print(
        f"Serving {entrypoint_path} at http://{host}:{int(port)}/ "
        f"(dashboard: http://{host}:{int(port)}{url_path})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def serve_existing_visual_dashboard(
    campaign_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    refresh_s: float = DEFAULT_REFRESH_S,
    generate_packets_top_n: int = DEFAULT_GENERATE_PACKETS_TOP_N,
    generate_packet_workers: int = DEFAULT_PACKET_GENERATION_WORKERS,
    cleanup_stale_packets_before_render: bool = DEFAULT_CLEANUP_STALE_PACKETS,
    status_json: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 6006,
) -> None:
    campaign_path = Path(campaign_dir).expanduser().resolve()
    output_path = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else campaign_path / DEFAULT_OUTPUT_SUBDIR
    )
    output_path.mkdir(parents=True, exist_ok=True)
    _serve_dashboard_server(
        campaign_path,
        output_path,
        output_dir=output_dir,
        top_n=top_n,
        refresh_s=refresh_s,
        generate_packets_top_n=generate_packets_top_n,
        generate_packet_workers=generate_packet_workers,
        cleanup_stale_packets_before_render=cleanup_stale_packets_before_render,
        status_json=status_json,
        host=host,
        port=port,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(
        subparser: argparse.ArgumentParser,
        *,
        generate_packets_top_n_default: int = DEFAULT_GENERATE_PACKETS_TOP_N,
    ) -> None:
        subparser.add_argument("campaign_dir", type=Path)
        subparser.add_argument("--output-dir", type=Path, default=None)
        subparser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
        subparser.add_argument("--refresh-s", type=float, default=DEFAULT_REFRESH_S)
        subparser.add_argument(
            "--generate-packets-top-n",
            type=int,
            default=generate_packets_top_n_default,
            help=(
                "Generate missing diagnostic packets for the current top N candidates before rendering."
                " Use 0 to disable and rely on precomputed packets."
            ),
        )
        subparser.add_argument(
            "--generate-packet-workers",
            type=int,
            default=DEFAULT_PACKET_GENERATION_WORKERS,
            help="Worker processes for packet generation. Use 0 to auto-scale to all local CPU cores.",
        )
        subparser.add_argument(
            "--no-cleanup-stale-packets",
            action="store_true",
            help="Keep packet directories from older visual styles instead of deleting them.",
        )
        subparser.add_argument("--status-json", type=Path, default=None)

    def add_runtime(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--host", default="127.0.0.1")
        subparser.add_argument("--port", type=int, default=6006)
        subparser.add_argument("--supervise-s", type=float, default=DEFAULT_WATCHDOG_SUPERVISE_S)
        subparser.add_argument("--stale-after-s", type=float, default=DEFAULT_STALE_AFTER_S)

    export_parser = subparsers.add_parser("export", help="Write the dashboard once.")
    add_common(export_parser)

    watch_parser = subparsers.add_parser("watch", help="Rewrite the dashboard when campaign artifacts change.")
    add_common(watch_parser)

    serve_parser = subparsers.add_parser("serve", help="Write once and serve the dashboard over HTTP.")
    add_common(serve_parser)
    add_runtime(serve_parser)

    serve_static_parser = subparsers.add_parser(
        "serve-static",
        help="Serve the existing dashboard directory over HTTP without forcing a fresh export.",
    )
    add_common(serve_static_parser, generate_packets_top_n_default=DEFAULT_RUNTIME_GENERATE_PACKETS_TOP_N)
    add_runtime(serve_static_parser)

    watchdog_parser = subparsers.add_parser(
        "watchdog",
        help="Supervise the dashboard watcher/server sidecars and restart them if they die or go stale.",
    )
    add_common(watchdog_parser, generate_packets_top_n_default=DEFAULT_RUNTIME_GENERATE_PACKETS_TOP_N)
    add_runtime(watchdog_parser)

    ensure_parser = subparsers.add_parser(
        "ensure-runtime",
        help="Ensure the dashboard watchdog is running for this campaign.",
    )
    add_common(ensure_parser, generate_packets_top_n_default=DEFAULT_RUNTIME_GENERATE_PACKETS_TOP_N)
    add_runtime(ensure_parser)

    stop_parser = subparsers.add_parser(
        "stop-runtime",
        help="Stop the dashboard watchdog and its sidecars for this campaign.",
    )
    stop_parser.add_argument("campaign_dir", type=Path)
    stop_parser.add_argument("--output-dir", type=Path, default=None)
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
            generate_packet_workers=args.generate_packet_workers,
            cleanup_stale_packets_before_render=not args.no_cleanup_stale_packets,
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
            generate_packet_workers=args.generate_packet_workers,
            cleanup_stale_packets_before_render=not args.no_cleanup_stale_packets,
            status_json=args.status_json,
        )
    elif args.command == "serve":
        serve_visual_dashboard(
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
        )
    elif args.command == "serve-static":
        serve_existing_visual_dashboard(
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
        )
    elif args.command == "watchdog":
        watch_visual_dashboard_runtime(
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
        )
    elif args.command == "ensure-runtime":
        payload = ensure_visual_dashboard_runtime(
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
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "stop-runtime":
        payload = stop_visual_dashboard_runtime(
            args.campaign_dir,
            output_dir=args.output_dir,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        parser.error(f"Unsupported command {args.command!r}")


if __name__ == "__main__":
    main()
