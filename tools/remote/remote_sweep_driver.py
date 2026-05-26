"""Run one full OBGPU parameter sweep inside a single remote Slurm job."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from base64 import b64decode
from pathlib import Path
from typing import Any


def decode_items(payload_b64: str) -> list[dict[str, Any]]:
    """Decode the serialized sweep-item manifest."""
    items = json.loads(b64decode(payload_b64).decode("utf-8"))
    return normalize_items(items)


def load_items_json(items_json_path: str) -> list[dict[str, Any]]:
    """Load the serialized sweep-item manifest from a JSON file."""
    with open(items_json_path) as handle:
        items = json.load(handle)
    return normalize_items(items)


def normalize_items(items: Any) -> list[dict[str, Any]]:
    """Validate and normalize one decoded sweep-item manifest."""
    if not isinstance(items, list):
        raise ValueError("Sweep item payload must decode to a list")
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each sweep item must be a dict")
        command = item.get("command")
        if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
            raise ValueError("Each sweep item command must be a list[str]")
        normalized.append(
            {
                "index": int(item["index"]),
                "label": str(item["label"]),
                "value": item.get("value"),
                "result_dir": str(item["result_dir"]),
                "command": [str(part) for part in command],
                "overrides_file": None if item.get("overrides_file") in (None, "") else str(item["overrides_file"]),
                "overrides": item.get("overrides"),
            }
        )
    return normalized


def shell_join(parts: list[str]) -> str:
    """Portable equivalent of shlex.join."""
    return " ".join(shlex.quote(str(part)) for part in parts)


def path_is_within(path_value: str, root_value: str) -> bool:
    """Return whether one string path is equal to or nested under another."""
    root_text = str(root_value).rstrip("/")
    path_text = str(path_value)
    if not root_text:
        return False
    return path_text == root_text or path_text.startswith(root_text + "/")


def relocate_repo_paths(command: list[str], *, shared_repo_root: str, repo_root: str) -> list[str]:
    """Rewrite shared-repo paths to the active job repo root."""
    relocated = []
    for part in command:
        if path_is_within(part, shared_repo_root):
            relocated.append(repo_root.rstrip("/") + part[len(shared_repo_root.rstrip("/")) :])
        else:
            relocated.append(part)
    return relocated


def requested_mpi_rank_count(command: list[str]) -> int | None:
    """Return the requested MPI rank count from one command list, if present."""
    options_with_values = {"-n", "-np", "--np", "--ntasks", "--ntasks-per-job"}
    for index, part in enumerate(command):
        if part in options_with_values and index + 1 < len(command):
            try:
                return int(command[index + 1])
            except ValueError:
                continue
        for prefix in ("-n", "-np"):
            suffix = part[len(prefix) :]
            if part.startswith(prefix) and suffix:
                try:
                    return int(suffix)
                except ValueError:
                    pass
        for prefix in ("--ntasks=", "--ntasks-per-job="):
            if part.startswith(prefix):
                try:
                    return int(part.split("=", 1)[1])
                except ValueError:
                    pass
    return None


def add_srun_exclusive(command: list[str]) -> list[str]:
    """Inject ``--exclusive`` into srun commands when not already present."""
    if not command:
        return list(command)
    base = os.path.basename(command[0])
    if base != "srun":
        return list(command)
    if any(part == "--exclusive" or part.startswith("--exclusive=") for part in command[1:]):
        return list(command)
    return [command[0], "--exclusive", *command[1:]]


def inject_neuron_dll_args(command: list[str], dll_path: Path | None) -> list[str]:
    """Inject ``-dll <libnrnmech.so>`` immediately after ``nrniv`` when needed."""
    if dll_path is None:
        return list(command)
    result: list[str] = []
    inserted = False
    for part in command:
        result.append(part)
        if not inserted and os.path.basename(part) == "nrniv":
            result.extend(["-dll", str(dll_path)])
            inserted = True
    return result


def build_neuron_mpi_preflight(command: list[str]) -> list[str] | None:
    """Build a cheap NEURON MPI preflight command from one benchmark command."""
    try:
        nrniv_index = command.index("nrniv")
    except ValueError:
        for idx, part in enumerate(command):
            if os.path.basename(part) == "nrniv":
                nrniv_index = idx
                break
        else:
            return None
    code = (
        "import os\n"
        "from neuron import h\n"
        "pc = h.ParallelContext()\n"
        "rank = int(pc.id())\n"
        "nhost = int(pc.nhost())\n"
        "expected = int(os.environ.get('OBGPU_EXPECTED_NRANKS') or '0')\n"
        "if rank == 0:\n"
        "    print('OBGPU MPI preflight: ParallelContext.nhost()=%d expected=%d' % (nhost, expected), flush=True)\n"
        "if expected > 1 and nhost != expected:\n"
        "    raise RuntimeError('NEURON MPI preflight saw %d ranks, expected %d' % (nhost, expected))\n"
        "pc.barrier()\n"
    )
    prefix = command[:nrniv_index]
    nrniv = command[nrniv_index]
    return prefix + [nrniv, "-mpi", "-python", "-c", code]


def dll_path_from_env() -> Path | None:
    """Return the active mechanism DLL path when one exists."""
    mechanism_root = os.environ.get("OBGPU_MECHANISM_ROOT")
    if not mechanism_root:
        return None
    dll_path = Path(mechanism_root) / os.uname().machine / "libnrnmech.so"
    return dll_path if dll_path.exists() else None


def launch_cwd(repo_root: Path, sweep_root: Path, dll_path: Path | None) -> Path:
    """Return the cwd that child NEURON commands should use."""
    if dll_path is None:
        return repo_root
    neutral = sweep_root / ".neuron-launch"
    neutral.mkdir(parents=True, exist_ok=True)
    return neutral


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON file atomically enough for notebook polling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def resolve_completed_result_dir(requested_result_dir: Path, requested_label: str) -> Path:
    """Resolve the actual benchmark payload directory for one completed sweep item."""
    if (requested_result_dir / "summary.json").exists():
        return requested_result_dir

    candidates: list[tuple[str, float, Path]] = []
    for candidate in requested_result_dir.parent.glob(f"{requested_label}_*"):
        if not candidate.is_dir():
            continue
        summary_path = candidate / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            continue
        if str(summary.get("requested_label") or "") != requested_label:
            continue
        candidates.append(
            (
                str(summary.get("timestamp") or ""),
                float(candidate.stat().st_mtime),
                candidate,
            )
        )

    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1], item[2].name))
        return candidates[-1][2]
    return requested_result_dir


def progress_payload(
    *,
    sweep_label: str,
    total_items: int,
    pending_items: list[dict[str, Any]],
    running_items: list[dict[str, Any]],
    finished_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the aggregate sweep progress payload."""
    completed_ok = [item for item in finished_items if bool(item.get("ok", False))]
    failed_items = [item for item in finished_items if not bool(item.get("ok", False))]
    done_count = len(finished_items)
    total_float = float(total_items) if total_items > 0 else 1.0
    return {
        "kind": "remote_sweep",
        "sweep_label": sweep_label,
        "current_ms": done_count,
        "total_ms": total_items,
        "percent": (100.0 * done_count / total_float) if total_items > 0 else 100.0,
        "pending_labels": [str(item["label"]) for item in pending_items],
        "running_items": running_items,
        "finished_items": finished_items,
        "completed_labels": [str(item["label"]) for item in completed_ok],
        "failed_labels": [str(item["label"]) for item in failed_items],
    }


def terminate_process_tree(processes: dict[int, dict[str, Any]]) -> None:
    """Best-effort termination of all active child processes."""
    for payload in processes.values():
        proc = payload.get("process")
        if proc is None:
            continue
        try:
            proc.terminate()
        except Exception:
            pass
    time.sleep(1.0)
    for payload in processes.values():
        proc = payload.get("process")
        if proc is None:
            continue
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass


def main() -> None:
    """Run all sweep items serially or concurrently inside the current Slurm job."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--items-b64")
    parser.add_argument("--items-json")
    parser.add_argument("--max-concurrent", type=int, default=1)
    args = parser.parse_args()
    if bool(args.items_b64) == bool(args.items_json):
        raise ValueError("Pass exactly one of --items-b64 or --items-json")

    repo_root = Path(args.repo_root).expanduser().resolve()
    sweep_root = Path(args.sweep_root).expanduser().resolve()
    sweep_root.mkdir(parents=True, exist_ok=True)
    (sweep_root / "runs").mkdir(parents=True, exist_ok=True)

    items = decode_items(args.items_b64) if args.items_b64 else load_items_json(args.items_json)
    sweep_label = sweep_root.name
    shared_repo_root = str(Path(os.environ.get("OBGPU_SHARED_REPO_ROOT", str(repo_root))).resolve())
    dll_path = dll_path_from_env()
    child_cwd = launch_cwd(repo_root, sweep_root, dll_path)
    max_concurrent = max(int(args.max_concurrent or 1), 1)

    pending = sorted(items, key=lambda item: int(item["index"]))
    running: dict[int, dict[str, Any]] = {}
    finished: list[dict[str, Any]] = []

    manifest_path = sweep_root / "sweep_manifest.json"
    write_json(
        manifest_path,
        {
            "sweep_label": sweep_label,
            "repo_root": str(repo_root),
            "shared_repo_root": shared_repo_root,
            "max_concurrent": max_concurrent,
            "items": [
                {
                    "index": int(item["index"]),
                    "label": str(item["label"]),
                    "value": item.get("value"),
                    "result_dir": str(item["result_dir"]),
                }
                for item in pending
            ],
        },
    )

    def update_progress() -> None:
        payload = progress_payload(
            sweep_label=sweep_label,
            total_items=len(items),
            pending_items=pending,
            running_items=[
                {
                    "index": int(payload["item"]["index"]),
                    "label": str(payload["item"]["label"]),
                    "result_dir": str(payload["item"]["result_dir"]),
                    "pid": int(payload["process"].pid),
                    "started_at": float(payload["started_at"]),
                }
                for payload in running.values()
            ],
            finished_items=finished,
        )
        write_json(sweep_root / "sim_progress.json", payload)

    def prepare_command(item: dict[str, Any]) -> list[str]:
        overrides_file = item.get("overrides_file")
        overrides = item.get("overrides")
        if overrides_file not in (None, "") and overrides is not None:
            write_json(Path(str(overrides_file)).expanduser(), overrides)
        command = relocate_repo_paths(
            list(item["command"]),
            shared_repo_root=shared_repo_root,
            repo_root=str(repo_root),
        )
        if max_concurrent > 1:
            command = add_srun_exclusive(command)
        command = inject_neuron_dll_args(command, dll_path)
        return command

    def launch_item(item: dict[str, Any]) -> dict[str, Any]:
        result_dir = Path(item["result_dir"]).expanduser().resolve()
        result_dir.mkdir(parents=True, exist_ok=True)
        command = prepare_command(item)
        (result_dir / "command.txt").write_text(shell_join(command) + "\n")
        stdout_handle = open(result_dir / "stdout.txt", "w")
        stderr_handle = open(result_dir / "stderr.txt", "w")
        proc = subprocess.Popen(
            command,
            cwd=str(child_cwd),
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=os.environ.copy(),
        )
        print(
            f"[OBGPU sweep] started item {int(item['index']) + 1}/{len(items)} "
            f"{item['label']} pid={proc.pid}",
            flush=True,
        )
        return {
            "item": item,
            "command": command,
            "process": proc,
            "stdout_handle": stdout_handle,
            "stderr_handle": stderr_handle,
            "started_at": time.time(),
        }

    terminating = False

    def on_signal(signum: int, _frame: Any) -> None:
        nonlocal terminating
        terminating = True
        print(f"[OBGPU sweep] received signal {signum}; terminating child steps", flush=True)
        terminate_process_tree(running)
        sys.exit(128 + int(signum))

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    first_command = prepare_command(pending[0]) if pending else None
    preflight_command = build_neuron_mpi_preflight(first_command) if first_command else None
    if preflight_command is not None:
        print("[OBGPU sweep] running one NEURON MPI preflight before the sweep", flush=True)
        preflight_completed = subprocess.run(
            preflight_command,
            cwd=str(child_cwd),
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            check=False,
        )
        preflight_log = sweep_root / "mpi_preflight.log"
        preflight_log.write_text(preflight_completed.stdout or "")
        if preflight_completed.returncode != 0:
            raise RuntimeError(
                "NEURON MPI preflight failed before the sweep started.\n"
                f"See {preflight_log}"
            )

    update_progress()

    try:
        while pending or running:
            while pending and len(running) < max_concurrent:
                item = pending.pop(0)
                payload = launch_item(item)
                running[int(item["index"])] = payload
                update_progress()

            time.sleep(0.2)
            finished_indices = []
            for index, payload in list(running.items()):
                proc = payload["process"]
                returncode = proc.poll()
                if returncode is None:
                    continue
                payload["stdout_handle"].close()
                payload["stderr_handle"].close()
                item = payload["item"]
                requested_result_dir = Path(item["result_dir"]).expanduser().resolve()
                result_dir = resolve_completed_result_dir(requested_result_dir, str(item["label"]))
                status = {
                    "index": int(item["index"]),
                    "label": str(item["label"]),
                    "value": item.get("value"),
                    "result_dir": str(result_dir),
                    "requested_result_dir": str(requested_result_dir),
                    "returncode": int(returncode),
                    "ok": int(returncode) == 0 and (result_dir / "summary.json").exists(),
                    "started_at": float(payload["started_at"]),
                    "finished_at": time.time(),
                }
                if not status["ok"]:
                    print(
                        f"[OBGPU sweep] item {item['label']} finished with returncode {returncode}",
                        flush=True,
                    )
                else:
                    print(f"[OBGPU sweep] item {item['label']} completed", flush=True)
                finished.append(status)
                finished_indices.append(index)
            for index in finished_indices:
                running.pop(index, None)
            if finished_indices:
                update_progress()
    finally:
        if not terminating:
            terminate_process_tree(running)

    payload = progress_payload(
        sweep_label=sweep_label,
        total_items=len(items),
        pending_items=[],
        running_items=[],
        finished_items=finished,
    )
    write_json(sweep_root / "sim_progress.json", payload)

    summary = {
        "kind": "remote_sweep",
        "sweep_label": sweep_label,
        "repo_root": str(repo_root),
        "shared_repo_root": shared_repo_root,
        "max_concurrent": max_concurrent,
        "total_items": len(items),
        "completed_items": [item for item in finished if bool(item.get("ok", False))],
        "failed_items": [item for item in finished if not bool(item.get("ok", False))],
        "items": finished,
    }
    write_json(sweep_root / "summary.json", summary)
    print(
        json.dumps(
            {
                "sweep_label": sweep_label,
                "total_items": len(items),
                "completed_ok": len(summary["completed_items"]),
                "failed": len(summary["failed_items"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
