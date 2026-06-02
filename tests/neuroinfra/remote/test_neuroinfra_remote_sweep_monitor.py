"""Smoke tests for extracted live remote sweep monitoring helpers."""

from __future__ import annotations

import subprocess

from neuroinfra.remote.sweep_monitor import (
    RemoteSweepMonitorHooks,
    monitor_remote_sweep,
)


def _completed(*, returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ssh", "bash", "-lc", "test"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _hooks(
    *,
    poll_status_fn,
    synced_count_fn=None,
    poll_calls=None,
    progress_messages=None,
    sleep_calls=None,
    refresh_calls=None,
    sync_calls=None,
    cancel_calls=None,
    monotonic_fn=None,
) -> RemoteSweepMonitorHooks:
    progress_messages = progress_messages if progress_messages is not None else []
    poll_calls = poll_calls if poll_calls is not None else []
    sleep_calls = sleep_calls if sleep_calls is not None else []
    refresh_calls = refresh_calls if refresh_calls is not None else []
    sync_calls = sync_calls if sync_calls is not None else []
    cancel_calls = cancel_calls if cancel_calls is not None else []
    monotonic_counter = {"value": -10.0}

    def _default_monotonic() -> float:
        monotonic_counter["value"] += 10.0
        return float(monotonic_counter["value"])

    return RemoteSweepMonitorHooks(
        refresh_remote_leases_fn=lambda *, warn=False: refresh_calls.append(bool(warn)),
        poll_status_fn=lambda **kwargs: poll_calls.append(dict(kwargs)) or poll_status_fn(**kwargs),
        sync_finished_items_fn=lambda status: sync_calls.append(status.get("state")),
        cancel_job_fn=lambda: cancel_calls.append("cancel") or _completed(returncode=0),
        synced_count_fn=synced_count_fn or (lambda: 0),
        progress_write=progress_messages.append,
        sleep_fn=lambda seconds: sleep_calls.append(float(seconds)),
        monotonic_fn=monotonic_fn or _default_monotonic,
    )


def main() -> None:
    success_statuses = iter(
        [
            {"state": "UNKNOWN", "done": False, "progress_payload": {}},
            {
                "state": "RUNNING",
                "done": False,
                "progress_current_ms": 2,
                "progress_total_ms": 8,
                "progress_payload": {
                    "running_items": [{"label": "A"}],
                    "pending_labels": ["B", "C"],
                },
            },
            {
                "state": "COMPLETED",
                "done": True,
                "progress_current_ms": 8,
                "progress_total_ms": 8,
                "summary_exists": True,
                "progress_payload": {
                    "finished_items": [
                        {"label": "A", "ok": True},
                        {"label": "B", "ok": False},
                    ],
                },
            },
        ]
    )
    progress_messages: list[str] = []
    poll_calls: list[dict[str, object]] = []
    refresh_calls: list[bool] = []
    sync_calls: list[str] = []
    success_monotonic_values = iter([1.0, 0.0, 2.0, 3.0, 10.0, 11.0])
    success_result = monitor_remote_sweep(
        job_id="77",
        poll_interval_s=1.0,
        log_poll_interval_s=5.0,
        live_status=True,
        hooks=_hooks(
            poll_status_fn=lambda **_kwargs: next(success_statuses),
            synced_count_fn=lambda: 3,
            poll_calls=poll_calls,
            progress_messages=progress_messages,
            refresh_calls=refresh_calls,
            sync_calls=sync_calls,
            monotonic_fn=lambda: next(success_monotonic_values),
        ),
    )
    assert success_result.final_status is not None
    assert success_result.final_status["state"] == "COMPLETED"
    assert poll_calls[:3] == [
        {"refresh_heartbeat": True, "include_sacct": False},
        {"refresh_heartbeat": False, "include_sacct": True},
        {"refresh_heartbeat": True, "include_sacct": False},
    ]
    assert sync_calls == ["RUNNING", "COMPLETED"]
    assert any("Sweep job 77: RUNNING (2/8 items)" in message for message in progress_messages)
    assert any("Sweep job 77: COMPLETED (8/8 items)" in message for message in progress_messages)
    print("remote sweep monitor success path: OK")

    pending_statuses = iter(
        [
            {
                "state": "PENDING",
                "done": False,
                "reason": "Resources",
                "progress_payload": {},
            },
            {
                "state": "COMPLETED",
                "done": True,
                "progress_payload": {},
            },
        ]
    )
    pending_messages: list[str] = []
    monitor_remote_sweep(
        job_id="78",
        poll_interval_s=1.0,
        log_poll_interval_s=5.0,
        live_status=True,
        hooks=_hooks(
            poll_status_fn=lambda **_kwargs: next(pending_statuses),
            progress_messages=pending_messages,
        ),
    )
    assert any("reason=Resources" in message for message in pending_messages)
    print("remote sweep monitor pending reason path: OK")

    cancel_calls: list[str] = []
    try:
        monitor_remote_sweep(
            job_id="79",
            poll_interval_s=1.0,
            log_poll_interval_s=5.0,
            live_status=False,
            hooks=_hooks(
                poll_status_fn=lambda **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
                cancel_calls=cancel_calls,
            ),
        )
        raise AssertionError("Expected KeyboardInterrupt to propagate")
    except KeyboardInterrupt as exc:
        assert "Interrupted remote sweep and requested cancellation for job 79." in str(exc)
    assert cancel_calls == ["cancel"]
    print("remote sweep monitor interrupt cancel path: OK")


if __name__ == "__main__":
    main()
