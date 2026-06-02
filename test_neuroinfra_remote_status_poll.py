"""Smoke tests for extracted remote JSON status polling helpers."""

from __future__ import annotations

import subprocess

from neuroinfra.remote.status_poll import (
    RemoteJSONPollHooks,
    poll_remote_json_status,
)


def _completed(stdout: str, *, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ssh", "bash", "-lc", "test"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def main() -> None:
    timing_calls: list[tuple[str, float]] = []
    sleep_calls: list[float] = []

    completed = poll_remote_json_status(
        "poll-command",
        poll_json_retries=3,
        error_prefix="Remote status poll",
        hooks=RemoteJSONPollHooks(
            run_command_fn=lambda command, timeout_s: _completed('{"state":"RUNNING"}\n'),
            record_timing_fn=lambda key, started: timing_calls.append((key, started)),
            sleep_fn=lambda seconds: sleep_calls.append(seconds),
            perf_counter_fn=lambda: 1.0,
        ),
    )
    assert completed == {"state": "RUNNING"}
    assert timing_calls == [("poll_s", 1.0)]
    assert sleep_calls == []

    attempts = iter([
        _completed("{not-json\n"),
        _completed('{"state":"COMPLETED"}\n'),
    ])
    retry_sleep_calls: list[float] = []
    retry_completed = poll_remote_json_status(
        "retry-command",
        poll_json_retries=3,
        error_prefix="Remote retry poll",
        hooks=RemoteJSONPollHooks(
            run_command_fn=lambda command, timeout_s: next(attempts),
            record_timing_fn=lambda key, started: None,
            sleep_fn=lambda seconds: retry_sleep_calls.append(seconds),
            perf_counter_fn=lambda: 2.0,
        ),
    )
    assert retry_completed == {"state": "COMPLETED"}
    assert retry_sleep_calls == [0.5]

    try:
        poll_remote_json_status(
            "fail-command",
            poll_json_retries=2,
            error_prefix="Remote fail poll",
            hooks=RemoteJSONPollHooks(
                run_command_fn=lambda command, timeout_s: _completed("", returncode=1, stderr="boom"),
                record_timing_fn=lambda key, started: None,
                sleep_fn=lambda seconds: None,
                perf_counter_fn=lambda: 3.0,
            ),
        )
        raise AssertionError("Expected nonzero remote poll to raise")
    except RuntimeError as exc:
        assert "Remote fail poll failed." in str(exc)
        assert "boom" in str(exc)

    invalid_attempts = iter([
        _completed("{still-bad\n"),
        _completed("{still-bad\n"),
    ])
    try:
        poll_remote_json_status(
            "invalid-command",
            poll_json_retries=2,
            error_prefix="Remote invalid poll",
            hooks=RemoteJSONPollHooks(
                run_command_fn=lambda command, timeout_s: next(invalid_attempts),
                record_timing_fn=lambda key, started: None,
                sleep_fn=lambda seconds: None,
                perf_counter_fn=lambda: 4.0,
            ),
        )
        raise AssertionError("Expected invalid JSON poll to raise")
    except RuntimeError as exc:
        assert "Remote invalid poll did not return valid JSON." in str(exc)

    print("neuroinfra remote status poll smoke test: OK")


if __name__ == "__main__":
    main()
