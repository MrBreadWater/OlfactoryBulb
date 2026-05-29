#!/usr/bin/env python3
"""Resume a live HFO optimization inside the authenticated notebook kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from jupyter_client import BlockingKernelClient

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from olfactorybulb.hfo_optimizer import resume_pending_batch_name

JSON_PREFIX = "__CODEX_JSON__"
DEFAULT_STATUS_JSON = (
    REPO_ROOT / "results" / "notebook_runs" / "optimization" / "codex_big_hfo_logs" / "latest_big_hfo_optimizer_status.json"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _candidate_kernel_files() -> list[Path]:
    runtime_dir = Path.home() / ".local" / "share" / "jupyter" / "runtime"
    return sorted(runtime_dir.glob("kernel-*.json"), key=lambda path: path.stat().st_mtime, reverse=True)


def _kernel_json(connection_file: Path, code: str, *, timeout_s: float = 20.0) -> dict:
    client = BlockingKernelClient()
    client.load_connection_file(str(connection_file))
    client.start_channels()
    try:
        msg_id = client.execute(code, allow_stdin=False, stop_on_error=True)
        deadline = time.time() + float(timeout_s)
        while True:
            remaining = max(deadline - time.time(), 0.1)
            msg = client.get_iopub_msg(timeout=remaining)
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            msg_type = msg["header"]["msg_type"]
            if msg_type == "stream":
                text = msg["content"].get("text", "")
                for line in text.splitlines():
                    if line.startswith(JSON_PREFIX):
                        return json.loads(line[len(JSON_PREFIX) :])
            if msg_type == "error":
                raise RuntimeError("\n".join(msg["content"].get("traceback") or []))
            if msg_type == "status" and msg["content"].get("execution_state") == "idle":
                break
    finally:
        client.stop_channels()
    raise RuntimeError(f"Kernel {connection_file} did not emit a {JSON_PREFIX} payload")


def _find_campaign_kernel(campaign_dir: Path) -> Path:
    campaign_real = campaign_dir.expanduser().resolve()
    campaign_str = str(campaign_dir)
    probe = """
import json, os
payload = {
    "pid": os.getpid(),
    "campaign_dir": str(globals().get("CAMPAIGN_DIR", "")),
}
print("__CODEX_JSON__" + json.dumps(payload))
"""
    for connection_file in _candidate_kernel_files():
        try:
            payload = _kernel_json(connection_file, probe, timeout_s=5.0)
        except Exception:
            continue
        payload_dir = str(payload.get("campaign_dir", "")).strip()
        if payload_dir and Path(payload_dir).expanduser().resolve() == campaign_real:
            return connection_file
    raise RuntimeError(f"Could not find a live kernel with CAMPAIGN_DIR={campaign_str}")


def _worker_code(payload: dict) -> str:
    payload_json = json.dumps(payload)
    return f"""
import importlib
import json
from pathlib import Path
import threading
import time
import traceback

import obgpu_experiment_helpers as _hlp
import tools.run_hfo_campaign as _runner
import olfactorybulb.hfo_optimizer as _hfo

importlib.reload(_hlp)
importlib.reload(_hfo)
importlib.reload(_runner)

_PAYLOAD = json.loads({payload_json!r})
campaign_dir = Path(_PAYLOAD["campaign_dir"])
status_path = Path(_PAYLOAD["status_json"])

def _state_tail():
    state = _hfo.load_campaign_state(campaign_dir)
    completed = list(state.get("completed_batches", []))
    return {{
        "next_batch_index": int(state.get("next_batch_index", 0) or 0),
        "next_candidate_index": int(state.get("next_candidate_index", 0) or 0),
        "completed_batches_tail": completed[-10:],
    }}

def _write_status(status: str, **extra):
    payload = {{
        "campaign_dir": str(campaign_dir),
        "status": status,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }}
    payload.update(_state_tail())
    payload.update(extra)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2) + "\\n")

def _is_retryable_transport_error(exc: BaseException) -> bool:
    message = str(exc)
    if exc.__class__.__name__ in {"SSHException", "NoValidConnectionsError", "TimeoutError", "EOFError", "OSError"}:
        return True
    needles = (
        "Error reading SSH protocol banner",
        "Connection reset by peer",
        "Unable to connect to port",
        "timed out",
        "No existing session",
        "EOFError",
    )
    return any(token in message for token in needles)

def _worker():
    transport_retry_count = 0
    try:
        remaining = int(_PAYLOAD["batches_to_run"])
        while remaining > 0:
            if bool(globals().get("_CODEX_HFO_OPT_STOP", False)):
                _write_status("stopped_by_flag")
                return
            state = _hfo.load_campaign_state(campaign_dir)
            pending = _hfo.resume_pending_batch_name(campaign_dir)
            next_batch_index = int(state.get("next_batch_index", 0) or 0)
            planned_batch = pending or f"batch_{{next_batch_index:04d}}"
            max_batches = next_batch_index if pending else next_batch_index + 1
            try:
                _write_status("running", batch_name=planned_batch)
                _runner.run_campaign(
                    allocation=_PAYLOAD["allocation"],
                    campaign_name=campaign_dir.name,
                    max_batches=max_batches,
                    total_tasks=int(_PAYLOAD["total_tasks"]),
                    nranks=int(_PAYLOAD["nranks"]),
                    tstop_ms=float(_PAYLOAD["tstop_ms"]),
                    cell_permute=int(_PAYLOAD["cell_permute"]),
                    early_stop_score=float("inf"),
                    min_ketamine_target=0.0,
                    max_control_target=1.0,
                    min_ketamine_peak_ratio=0.0,
                    min_target_contrast_log10=float("-inf"),
                    max_control_score=float("inf"),
                    require_criteria_for_early_stop=False,
                    require_live_paramiko_session=True,
                    verify_auth=False,
                )
                remaining -= 1
                transport_retry_count = 0
            except Exception as exc:
                if _is_retryable_transport_error(exc):
                    transport_retry_count += 1
                    max_transport_retries = int(_PAYLOAD.get("max_transport_retries", 12))
                    if transport_retry_count <= max_transport_retries:
                        retry_sleep_s = min(
                            float(_PAYLOAD.get("transport_retry_backoff_s", 10.0)) * transport_retry_count,
                            60.0,
                        )
                        _write_status(
                            "retrying_after_transport_error",
                            batch_name=planned_batch,
                            retry_attempt=transport_retry_count,
                            error=str(exc),
                            error_type=exc.__class__.__name__,
                        )
                        time.sleep(retry_sleep_s)
                        continue
                raise
        _write_status("idle")
    except Exception as exc:
        tb = traceback.format_exc()
        message = str(exc)
        blocked = (
            "cached Paramiko SSH session is no longer usable" in message
            or "blocked_no_active_paramiko_transport" in message
        )
        _write_status(
            "blocked_no_active_paramiko_transport" if blocked else "error",
            error=message,
            error_type=exc.__class__.__name__,
            traceback=tb,
        )
        raise

thread = globals().get("_CODEX_HFO_OPT_THREAD")
if thread is not None and getattr(thread, "is_alive", lambda: False)():
    print("__CODEX_JSON__" + json.dumps({{
        "status": "already_running",
        "thread_name": getattr(thread, "name", "unknown"),
        **_state_tail(),
    }}))
else:
    globals()["_CODEX_HFO_OPT_STOP"] = False
    thread = threading.Thread(target=_worker, name="codex_hfo_opt_resume", daemon=True)
    globals()["_CODEX_HFO_OPT_THREAD"] = thread
    thread.start()
    print("__CODEX_JSON__" + json.dumps({{
        "status": "started",
        "thread_name": thread.name,
        **_state_tail(),
    }}))
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_dir", help="Existing campaign directory")
    parser.add_argument("--connection-file", default=None, help="Kernel connection file; auto-detect by CAMPAIGN_DIR when omitted")
    parser.add_argument("--batches-to-run", default=1, type=int, help="Number of optimizer steps to run; a pending batch counts as one step")
    parser.add_argument("--status-json", default=str(DEFAULT_STATUS_JSON), help="Status JSON path for dashboard updates")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    campaign_dir = Path(args.campaign_dir).expanduser()
    campaign_config = _read_json(campaign_dir / "campaign_config.json")
    base_config = dict(campaign_config.get("base_config") or {})
    connection_file = Path(args.connection_file).expanduser() if args.connection_file else _find_campaign_kernel(campaign_dir)
    payload = {
        "campaign_dir": str(campaign_dir),
        "status_json": str(Path(args.status_json).expanduser()),
        "allocation": str(base_config.get("slurm_allocation_job_id") or ""),
        "total_tasks": int(base_config.get("optimizer_total_tasks") or 120),
        "nranks": int(base_config.get("nranks") or 5),
        "tstop_ms": float(base_config.get("tstop_ms") or 2000.0),
        "cell_permute": int(base_config.get("cell_permute") or 0),
        "batches_to_run": int(args.batches_to_run),
        "pending_batch": resume_pending_batch_name(campaign_dir),
        "max_transport_retries": 12,
        "transport_retry_backoff_s": 10.0,
    }
    result = _kernel_json(connection_file, _worker_code(payload), timeout_s=20.0)
    result["connection_file"] = str(connection_file)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
