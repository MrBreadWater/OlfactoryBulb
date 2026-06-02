"""Regression tests for generic dashboard runtime process helpers."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from neuroinfra.dashboard.runtime import (
    RuntimeProcessInfo,
    matching_pids,
    pid_is_alive,
    port_in_use,
    process_matches_command,
    read_runtime_process_info,
    runtime_dir,
    runtime_process_paths,
    spawn_detached_process,
    terminate_process,
    write_json_atomic,
)


with TemporaryDirectory() as tmp:
    output_dir = Path(tmp) / "dashboard"
    current_runtime_dir = runtime_dir(output_dir)
    assert current_runtime_dir == output_dir / ".runtime"
    paths = runtime_process_paths(output_dir, "watcher")
    assert paths["runtime_dir"] == current_runtime_dir
    assert paths["pid"].name == "watcher.pid.json"
    assert paths["stdout"].name == "watcher.stdout.log"
    assert paths["stderr"].name == "watcher.stderr.log"

    payload_path = write_json_atomic(current_runtime_dir / "status.json", {"ok": True})
    assert payload_path.exists()
    assert payload_path.read_text().strip().startswith("{")

    info = spawn_detached_process(
        ["/bin/sh", "-c", "sleep 30"],
        cwd=output_dir,
        stdout_path=paths["stdout"],
        stderr_path=paths["stderr"],
        meta_path=paths["pid"],
        meta={"kind": "watcher", "campaign_dir": str(output_dir)},
    )
    assert isinstance(info, RuntimeProcessInfo)
    assert pid_is_alive(info.pid) is True
    assert process_matches_command(info.pid, ["/bin/sh", "-c", "sleep 30"]) is True
    assert info.pid in matching_pids(["sleep 30"])

    read_back = read_runtime_process_info(output_dir, "watcher")
    assert read_back is not None
    assert read_back.pid == info.pid
    assert read_back.kind == "watcher"

    terminate_process(info.pid, grace_s=1.0)
    deadline = time.time() + 2.0
    while time.time() < deadline:
        try:
            waited_pid, _status = os.waitpid(info.pid, os.WNOHANG)
        except ChildProcessError:
            break
        if waited_pid == info.pid:
            break
        time.sleep(0.05)
    assert pid_is_alive(info.pid) is False

assert port_in_use("127.0.0.1", 65534) in {True, False}
