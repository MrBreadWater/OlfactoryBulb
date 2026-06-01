"""Generic sidecar/runtime process helpers for dashboard-style tools."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from typing import Any


@dataclass(frozen=True)
class RuntimeProcessInfo:
    kind: str
    pid: int
    pid_path: Path
    stdout_path: Path
    stderr_path: Path
    meta: dict[str, Any]


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = json_path.with_name(f".{json_path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, json_path)
    return json_path


def runtime_dir(output_path: str | Path, *, runtime_subdir: str = ".runtime") -> Path:
    return Path(output_path) / str(runtime_subdir)


def runtime_process_paths(
    output_path: str | Path,
    kind: str,
    *,
    runtime_subdir: str = ".runtime",
) -> dict[str, Path]:
    current_runtime_dir = runtime_dir(output_path, runtime_subdir=runtime_subdir)
    return {
        "runtime_dir": current_runtime_dir,
        "pid": current_runtime_dir / f"{kind}.pid.json",
        "stdout": current_runtime_dir / f"{kind}.stdout.log",
        "stderr": current_runtime_dir / f"{kind}.stderr.log",
    }


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def process_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()


def process_cmdargs(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except OSError:
        return []
    return [arg for arg in raw.decode("utf-8", errors="ignore").split("\x00") if arg]


def process_matches_tokens(pid: int, expected_tokens: list[str]) -> bool:
    cmdline = process_cmdline(pid)
    if not cmdline:
        return False
    return all(token in cmdline for token in expected_tokens)


def process_matches_command(pid: int, expected_command: list[str]) -> bool:
    actual = process_cmdargs(pid)
    if not actual:
        return False
    return actual == [str(arg) for arg in expected_command]


def matching_pids(expected_tokens: list[str]) -> list[int]:
    matches: list[int] = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        if process_matches_tokens(pid, expected_tokens):
            matches.append(pid)
    return sorted(set(matches))


def _read_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_runtime_process_info(
    output_path: str | Path,
    kind: str,
    *,
    runtime_subdir: str = ".runtime",
) -> RuntimeProcessInfo | None:
    paths = runtime_process_paths(output_path, kind, runtime_subdir=runtime_subdir)
    payload = _read_json_dict(paths["pid"])
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


def terminate_process(pid: int, *, grace_s: float = 5.0) -> None:
    if pid <= 0 or not pid_is_alive(pid):
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
        if not pid_is_alive(pid):
            return
        time.sleep(0.1)
    try:
        if pgid is not None and pgid > 0:
            os.killpg(pgid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGKILL)
    except OSError:
        return


def spawn_detached_process(
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
    write_json_atomic(meta_path, payload)
    return RuntimeProcessInfo(
        kind=str(meta.get("kind") or ""),
        pid=int(proc.pid),
        pid_path=meta_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        meta=payload,
    )


def port_in_use(host: str, port: int) -> bool:
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


__all__ = [
    "RuntimeProcessInfo",
    "matching_pids",
    "pid_is_alive",
    "port_in_use",
    "process_cmdargs",
    "process_cmdline",
    "process_matches_command",
    "process_matches_tokens",
    "read_runtime_process_info",
    "runtime_dir",
    "runtime_process_paths",
    "spawn_detached_process",
    "terminate_process",
    "write_json_atomic",
]
