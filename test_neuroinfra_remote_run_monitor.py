"""Smoke tests for extracted live remote single-run monitoring helpers."""

from __future__ import annotations

import subprocess

from neuroinfra.remote.run_monitor import (
    RemoteRunMonitorHooks,
    monitor_remote_run,
)


class _ProgressProbe:
    def __init__(self, total: int, desc: str) -> None:
        self.total = total
        self.desc = desc
        self.updates: list[int] = []
        self.closed = False

    def update_to(self, value: int) -> None:
        self.updates.append(int(value))

    def close(self) -> None:
        self.closed = True


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
    progress_messages=None,
    sleep_calls=None,
    refresh_calls=None,
    cancel_calls=None,
    partial_sync_calls=None,
    progress_instances=None,
    remote_status_has_artifacts_fn=None,
    time_fn=None,
    monotonic_fn=None,
) -> RemoteRunMonitorHooks:
    progress_messages = progress_messages if progress_messages is not None else []
    sleep_calls = sleep_calls if sleep_calls is not None else []
    refresh_calls = refresh_calls if refresh_calls is not None else []
    cancel_calls = cancel_calls if cancel_calls is not None else []
    partial_sync_calls = partial_sync_calls if partial_sync_calls is not None else []
    progress_instances = progress_instances if progress_instances is not None else []
    monotonic_counter = {"value": -10.0}

    def _default_monotonic() -> float:
        monotonic_counter["value"] += 10.0
        return float(monotonic_counter["value"])

    return RemoteRunMonitorHooks(
        refresh_remote_leases_fn=lambda *, warn=False: refresh_calls.append(bool(warn)),
        poll_status_fn=poll_status_fn,
        cancel_job_fn=lambda: cancel_calls.append("cancel") or _completed(returncode=0),
        sync_partial_artifacts_fn=lambda: partial_sync_calls.append("partial") or _completed(returncode=0),
        remote_status_has_artifacts_fn=remote_status_has_artifacts_fn or (lambda status: bool(status.get("has_artifacts", False))),
        progress_bar_factory_fn=lambda total, desc: progress_instances.append(_ProgressProbe(total, desc)) or progress_instances[-1],
        filter_live_log_line_fn=lambda _kind, line: str(line).strip() or None,
        progress_write=progress_messages.append,
        sleep_fn=lambda seconds: sleep_calls.append(float(seconds)),
        monotonic_fn=monotonic_fn or _default_monotonic,
        time_fn=time_fn or (lambda: 0.0),
    )


def main() -> None:
    progress_messages: list[str] = []
    progress_instances: list[_ProgressProbe] = []
    success_statuses = iter(
        [
            {
                "state": "RUNNING",
                "done": False,
                "ok": False,
                "summary_exists": False,
                "stdout_tail": "line1\n",
            },
            {
                "state": "RUNNING",
                "done": False,
                "ok": False,
                "summary_exists": False,
                "progress_total_ms": 1000,
                "progress_current_ms": 500,
                "stdout_tail": "line1\nline2\n",
            },
            {
                "state": "COMPLETED",
                "done": True,
                "ok": True,
                "summary_exists": True,
                "progress_total_ms": 1000,
                "progress_current_ms": 1000,
                "stdout_tail": "line1\nline2\n",
            },
        ]
    )
    success_result = monitor_remote_run(
        job_id="42",
        poll_interval_s=1.0,
        log_poll_interval_s=5.0,
        live_status=True,
        live_logs=True,
        missing_artifact_retry_limit=3,
        hooks=_hooks(
            poll_status_fn=lambda **_kwargs: next(success_statuses),
            progress_messages=progress_messages,
            progress_instances=progress_instances,
        ),
    )
    assert success_result.final_status is not None
    assert success_result.final_status["state"] == "COMPLETED"
    assert len(success_result.poll_transcript) == 3
    assert progress_instances and progress_instances[0].updates == [500, 1000]
    assert progress_instances[0].closed
    assert any("Job 42: RUNNING" in message for message in progress_messages)
    assert any("[Sol remote][stdout] line1" == message for message in progress_messages)
    assert any("[Sol remote][stdout] line2" == message for message in progress_messages)
    assert sum(1 for message in progress_messages if message == "[Sol remote][stdout] line1") == 1
    print("remote run monitor success path: OK")

    retry_sleep_calls: list[float] = []
    retry_statuses = iter(
        [
            {"state": "FAILED", "done": True, "ok": False, "summary_exists": False, "has_artifacts": False},
            {"state": "FAILED", "done": True, "ok": False, "summary_exists": False, "has_artifacts": False},
            {"state": "FAILED", "done": True, "ok": False, "summary_exists": False, "has_artifacts": True},
        ]
    )
    retry_result = monitor_remote_run(
        job_id="43",
        poll_interval_s=1.0,
        log_poll_interval_s=5.0,
        live_status=False,
        live_logs=False,
        missing_artifact_retry_limit=2,
        hooks=_hooks(
            poll_status_fn=lambda **_kwargs: next(retry_statuses),
            sleep_calls=retry_sleep_calls,
            remote_status_has_artifacts_fn=lambda status: bool(status.get("has_artifacts", False)),
        ),
    )
    assert retry_result.final_status is not None
    assert retry_result.final_status["has_artifacts"] is True
    assert retry_sleep_calls == [3.0, 3.0]
    print("remote run monitor missing-artifact retry: OK")

    cancel_calls: list[str] = []
    partial_sync_calls: list[str] = []
    interrupt_events = [KeyboardInterrupt(), {"state": "CANCELLED", "done": True, "ok": False}]

    def _interrupt_poll(**_kwargs):
        event = interrupt_events.pop(0)
        if isinstance(event, BaseException):
            raise event
        return event

    try:
        monitor_remote_run(
            job_id="44",
            poll_interval_s=1.0,
            log_poll_interval_s=5.0,
            live_status=False,
            live_logs=False,
            missing_artifact_retry_limit=0,
            hooks=_hooks(
                poll_status_fn=_interrupt_poll,
                cancel_calls=cancel_calls,
                partial_sync_calls=partial_sync_calls,
                time_fn=lambda: 0.0,
            ),
        )
        raise AssertionError("Expected KeyboardInterrupt to propagate")
    except KeyboardInterrupt as exc:
        assert "Interrupted remote Sol run and requested cancellation for job 44." in str(exc)
    assert cancel_calls == ["cancel"]
    assert partial_sync_calls == ["partial"]
    print("remote run monitor interrupt cancel path: OK")


if __name__ == "__main__":
    main()
