"""Reusable notebook-managed remote allocation orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import PurePosixPath
import subprocess
from typing import Any, Callable, MutableMapping

from .allocation_cache import (
    allocation_record,
    disabled_allocation_record,
    manual_allocation_record,
)
from .slurm_state import REMOTE_SLURM_TERMINAL_FAIL, REMOTE_SLURM_TERMINAL_OK


@dataclass
class RemoteAllocationRuntimeContext:
    """State and callbacks for notebook-managed reusable Slurm allocations."""

    config: dict[str, Any]
    live_slurm_allocations: MutableMapping[str, Any]
    live_remote_stale_cleanups: MutableMapping[str, Any]
    progress_write: Callable[[str], None]
    connection_key_fn: Callable[[dict[str, Any]], str]
    remote_results_root_fn: Callable[[dict[str, Any]], PurePosixPath]
    poll_command_timeout_s_fn: Callable[[dict[str, Any]], float | None]
    heartbeat_timeout_s_fn: Callable[[dict[str, Any]], int]
    allocation_cache_key_fn: Callable[[dict[str, Any]], str]
    allocation_runtime_config_fn: Callable[[dict[str, Any]], dict[str, Any]]
    build_remote_touch_command_fn: Callable[[str | PurePosixPath], str]
    build_remote_cleanup_allocations_command_fn: Callable[[dict[str, Any], PurePosixPath | None], str]
    build_remote_allocation_discovery_command_fn: Callable[[dict[str, Any], PurePosixPath | None], tuple[str, PurePosixPath, str]]
    build_remote_allocation_submit_command_fn: Callable[[dict[str, Any], PurePosixPath | None], tuple[str, PurePosixPath, str]]
    build_remote_cancel_command_fn: Callable[[str], str]
    run_ssh_shell_fn: Callable[[dict[str, Any], str, bool, float | None], subprocess.CompletedProcess[str]]
    query_remote_slurm_job_state_fn: Callable[[dict[str, Any], str], dict[str, str]]
    time_fn: Callable[[], float] = time.time
    sleep_fn: Callable[[float], None] = time.sleep

    def refresh_heartbeat(
        self,
        heartbeat_path: str | PurePosixPath | None,
        *,
        warn: bool = False,
    ) -> bool:
        """Best-effort refresh of a remote notebook heartbeat file."""
        if heartbeat_path in (None, ""):
            return False
        try:
            completed = self.run_ssh_shell_fn(
                self.config,
                self.build_remote_touch_command_fn(str(heartbeat_path)),
                False,
                self.poll_command_timeout_s_fn(self.config),
            )
        except Exception as exc:
            if warn:
                self.progress_write(f"[Sol remote] Heartbeat refresh failed: {exc}")
            return False
        if completed.returncode != 0:
            if warn:
                stderr = (completed.stderr or "").strip()
                self.progress_write(f"[Sol remote] Heartbeat refresh failed: {stderr or 'unknown error'}")
            return False
        return True

    def cleanup_stale_allocations(
        self,
        *,
        remote_helper_dir: PurePosixPath | None = None,
    ) -> list[dict[str, Any]]:
        """Cancel stale remote notebook-managed reusable allocations before a new run."""
        completed = self.run_ssh_shell_fn(
            self.config,
            self.build_remote_cleanup_allocations_command_fn(self.config, remote_helper_dir),
            False,
            None,
        )
        if completed.returncode != 0:
            self.progress_write(f"[Sol remote] Stale allocation cleanup failed: {(completed.stderr or '').strip()}")
            return []
        try:
            actions = json.loads((completed.stdout or "").strip() or "[]")
        except json.JSONDecodeError:
            self.progress_write("[Sol remote] Stale allocation cleanup returned invalid JSON.")
            return []
        if not isinstance(actions, list):
            return []
        cancelled = [action for action in actions if isinstance(action, dict) and action.get("action") == "cancel_requested"]
        for action in cancelled:
            job_id = action.get("job_id")
            reason = action.get("reason")
            self.progress_write(f"[Sol remote] Cancelled stale reusable allocation {job_id} ({reason}).")
        return [action for action in actions if isinstance(action, dict)]

    def maybe_cleanup_stale_allocations(
        self,
        *,
        remote_helper_dir: PurePosixPath | None = None,
    ) -> list[dict[str, Any]]:
        """Throttle stale-allocation cleanup so warm sessions do not repeat the same scan."""
        if not bool(self.config.get("remote_cleanup_stale_allocations", True)):
            return []
        if not bool(self.config.get("slurm_reuse_allocation", False)):
            return []
        if self.config.get("slurm_allocation_job_id") not in (None, ""):
            return []
        cache_key = f"{self.connection_key_fn(self.config)}::{self.remote_results_root_fn(self.config).as_posix()}"
        cached = self.live_remote_stale_cleanups.get(cache_key)
        if cached is not None:
            return list(cached.get("actions", []))
        actions = self.cleanup_stale_allocations(remote_helper_dir=remote_helper_dir)
        self.live_remote_stale_cleanups[cache_key] = {
            "timestamp": self.time_fn(),
            "actions": list(actions),
        }
        return actions

    def ensure_cached_allocation(
        self,
        *,
        remote_helper_dir: PurePosixPath | None = None,
    ) -> dict[str, Any]:
        """Acquire or reuse one notebook-cached remote Slurm allocation."""
        manual_job_id = self.config.get("slurm_allocation_job_id")
        if manual_job_id not in (None, ""):
            return manual_allocation_record(str(manual_job_id))
        if not bool(self.config.get("slurm_reuse_allocation", False)):
            return disabled_allocation_record()

        cache_key = self.allocation_cache_key_fn(self.config)
        allocation = self.live_slurm_allocations.get(cache_key)
        created_now = False
        runtime_config = self.allocation_runtime_config_fn(self.config)

        if allocation is not None:
            if allocation.get("heartbeat_path") in (None, ""):
                self.live_slurm_allocations.pop(cache_key, None)
                allocation = None
            else:
                self.refresh_heartbeat(str(allocation["heartbeat_path"]), warn=True)

        if allocation is not None:
            status = self.query_remote_slurm_job_state_fn(self.config, str(allocation["job_id"]))
            state = status.get("state", "UNKNOWN")
            if state in REMOTE_SLURM_TERMINAL_OK or state in REMOTE_SLURM_TERMINAL_FAIL or state == "UNKNOWN":
                self.live_slurm_allocations.pop(cache_key, None)
                allocation = None
            else:
                print(f"[Sol remote] Reusing cached allocation {allocation['job_id']}.", flush=True)

        if allocation is None:
            discover_command, allocation_root, allocation_name = self.build_remote_allocation_discovery_command_fn(
                self.config,
                remote_helper_dir,
            )
            discover_completed = self.run_ssh_shell_fn(self.config, discover_command, False, None)
            if discover_completed.returncode != 0:
                raise RuntimeError(
                    "Remote Slurm allocation discovery failed.\n"
                    f"Stdout:\n{discover_completed.stdout}\n\nStderr:\n{discover_completed.stderr}"
                )
            discovered_text = (discover_completed.stdout or "").strip()
            if discovered_text:
                try:
                    discovered = json.loads(discovered_text)
                except json.JSONDecodeError:
                    discovered = None
                if isinstance(discovered, dict) and discovered.get("job_id") not in (None, ""):
                    discovered_job_id = str(discovered["job_id"])
                    status = self.query_remote_slurm_job_state_fn(self.config, discovered_job_id)
                    state = status.get("state", "UNKNOWN")
                    if state not in REMOTE_SLURM_TERMINAL_OK and state not in REMOTE_SLURM_TERMINAL_FAIL and state != "UNKNOWN":
                        heartbeat_path = str(discovered.get("heartbeat_path") or "")
                        if not heartbeat_path:
                            print(
                                f"[Sol remote] Cancelling legacy reusable allocation {discovered_job_id} without heartbeat lease.",
                                flush=True,
                            )
                            self.run_ssh_shell_fn(
                                self.config,
                                self.build_remote_cancel_command_fn(discovered_job_id),
                                False,
                                None,
                            )
                        else:
                            self.refresh_heartbeat(heartbeat_path, warn=True)
                            allocation = allocation_record(
                                job_id=discovered_job_id,
                                cache_key=cache_key,
                                allocation_root=str(discovered.get("allocation_root") or allocation_root.as_posix()),
                                batch_script=str(discovered.get("batch_script") or ""),
                                heartbeat_path=heartbeat_path,
                                heartbeat_timeout_s=discovered.get("heartbeat_timeout_s"),
                                slurm_log_pattern=str(discovered.get("slurm_log_pattern") or ""),
                                name=str(discovered.get("name") or allocation_name),
                                cached=True,
                                manual=False,
                                config=runtime_config,
                            )
                            self.live_slurm_allocations[cache_key] = allocation
                            print(f"[Sol remote] Reusing discovered allocation {allocation['job_id']}.", flush=True)

        if allocation is None:
            print("[Sol remote] Requesting reusable Slurm allocation...", flush=True)
            submit_command, allocation_root, allocation_name = self.build_remote_allocation_submit_command_fn(
                self.config,
                remote_helper_dir,
            )
            submit_completed = self.run_ssh_shell_fn(self.config, submit_command, False, None)
            if submit_completed.returncode != 0:
                raise RuntimeError(
                    "Remote Slurm allocation submission failed.\n"
                    f"Stdout:\n{submit_completed.stdout}\n\nStderr:\n{submit_completed.stderr}"
                )
            try:
                submission = json.loads((submit_completed.stdout or "").strip())
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Remote Slurm allocation submission did not return valid JSON.\n"
                    f"Stdout:\n{submit_completed.stdout}\n\nStderr:\n{submit_completed.stderr}"
                ) from exc
            allocation = allocation_record(
                job_id=str(submission["job_id"]),
                cache_key=cache_key,
                allocation_root=str(submission.get("allocation_root") or allocation_root.as_posix()),
                batch_script=str(submission.get("batch_script") or ""),
                heartbeat_path=str(submission.get("heartbeat_path") or allocation_root / "notebook-heartbeat.txt"),
                heartbeat_timeout_s=submission.get("heartbeat_timeout_s"),
                slurm_log_pattern=str(submission.get("slurm_log_pattern") or ""),
                name=str(submission.get("name") or allocation_name),
                cached=True,
                manual=False,
                config=runtime_config,
            )
            self.live_slurm_allocations[cache_key] = allocation
            created_now = True

        last_signature: tuple[str, str, str] | None = None
        try:
            while True:
                self.refresh_heartbeat(str(allocation.get("heartbeat_path") or ""), warn=False)
                status = self.query_remote_slurm_job_state_fn(self.config, str(allocation["job_id"]))
                state = status.get("state", "UNKNOWN")
                reason = str(status.get("reason") or "").strip()
                location = str(status.get("location") or "").strip()
                status_signature = (state, reason, location)
                if status_signature != last_signature:
                    detail = ""
                    if state == "PENDING" and reason:
                        detail = f" reason={reason}"
                    elif location and state not in {"UNKNOWN", "PENDING"}:
                        detail = f" where={location}"
                    print(f"[Sol remote] Allocation {allocation['job_id']}: {state}{detail}", flush=True)
                    last_signature = status_signature

                if state == "RUNNING":
                    allocation.update(
                        allocation_record(
                            job_id=str(allocation["job_id"]),
                            cache_key=str(allocation["cache_key"]),
                            allocation_root=str(allocation["allocation_root"]),
                            batch_script=str(allocation["batch_script"]),
                            heartbeat_path=str(allocation["heartbeat_path"]),
                            heartbeat_timeout_s=allocation.get("heartbeat_timeout_s"),
                            slurm_log_pattern=str(allocation.get("slurm_log_pattern") or ""),
                            name=str(allocation.get("name") or ""),
                            cached=bool(allocation.get("cached", False)),
                            manual=bool(allocation.get("manual", False)),
                            config=runtime_config,
                            state=state,
                            reason=reason,
                            location=location,
                        )
                    )
                    self.live_slurm_allocations[cache_key] = allocation
                    return allocation
                if state in REMOTE_SLURM_TERMINAL_OK or state in REMOTE_SLURM_TERMINAL_FAIL:
                    self.live_slurm_allocations.pop(cache_key, None)
                    raise RuntimeError(
                        "Reusable Slurm allocation terminated before it became runnable.\n"
                        f"Job id: {allocation['job_id']}\n"
                        f"State: {state}\n"
                        f"Reason: {reason}\n"
                        f"Location: {location}"
                    )
                self.sleep_fn(5.0)
        except KeyboardInterrupt:
            if created_now:
                print(
                    f"[Sol remote] Interrupt received; cancelling new allocation {allocation['job_id']}...",
                    flush=True,
                )
                self.run_ssh_shell_fn(
                    self.config,
                    self.build_remote_cancel_command_fn(str(allocation["job_id"])),
                    False,
                    None,
                )
                self.live_slurm_allocations.pop(cache_key, None)
            raise

    def release_allocation(self) -> bool:
        """Cancel and forget the cached or remotely-discovered reusable Slurm allocation."""
        cache_key = self.allocation_cache_key_fn(self.config)
        allocation = self.live_slurm_allocations.pop(cache_key, None)
        if allocation is None:
            discover_command, _allocation_root, _allocation_name = self.build_remote_allocation_discovery_command_fn(
                self.config,
                None,
            )
            discover_completed = self.run_ssh_shell_fn(self.config, discover_command, False, None)
            if discover_completed.returncode != 0:
                print(f"[Sol remote] Allocation discovery stderr: {(discover_completed.stderr or '').strip()}", flush=True)
                return False
            discovered_text = (discover_completed.stdout or "").strip()
            if discovered_text:
                try:
                    discovered = json.loads(discovered_text)
                except json.JSONDecodeError:
                    discovered = None
                if isinstance(discovered, dict) and discovered.get("job_id") not in (None, ""):
                    allocation = {"job_id": str(discovered["job_id"])}
            if allocation is None:
                print("[Sol remote] No cached or discovered reusable allocation for this config.", flush=True)
                return False
        job_id = str(allocation["job_id"])
        print(f"[Sol remote] Releasing reusable allocation {job_id}...", flush=True)
        completed = self.run_ssh_shell_fn(
            self.config,
            self.build_remote_cancel_command_fn(job_id),
            False,
            None,
        )
        if completed.returncode != 0:
            print(f"[Sol remote] scancel stderr: {(completed.stderr or '').strip()}", flush=True)
            return False
        print(f"[Sol remote] Cancellation requested for allocation {job_id}.", flush=True)
        return True
