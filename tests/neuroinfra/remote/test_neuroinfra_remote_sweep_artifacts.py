"""Smoke tests for extracted remote sweep final sync and artifact handling."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import subprocess
import tempfile

from neuroinfra.remote.sweep_artifacts import (
    RemoteSweepArtifactHooks,
    finalize_remote_sweep_artifacts,
)


def _completed(*, returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ssh", "bash", "-lc", "test"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _read_json_if_present(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _make_hooks(
    *,
    sync_remote_result_dir_fn,
    sync_remote_sweep_compact_items_fn,
    recover_local_sweep_summary_fn,
    progress_messages=None,
    timing_calls=None,
    refresh_calls=None,
) -> RemoteSweepArtifactHooks:
    progress_messages = progress_messages if progress_messages is not None else []
    timing_calls = timing_calls if timing_calls is not None else []
    refresh_calls = refresh_calls if refresh_calls is not None else []
    return RemoteSweepArtifactHooks(
        sync_remote_result_dir_fn=sync_remote_result_dir_fn,
        sync_remote_sweep_compact_items_fn=sync_remote_sweep_compact_items_fn,
        read_json_if_present_fn=_read_json_if_present,
        recover_local_sweep_summary_fn=recover_local_sweep_summary_fn,
        remote_sweep_metadata_files_fn=lambda: ("summary.json", "sim_progress.json"),
        remote_sweep_item_sync_files_fn=lambda _config: ("summary.json", "soma_spikes.npz"),
        remote_sweep_item_diagnostic_files_fn=lambda: ("stdout.txt", "stderr.txt"),
        local_sweep_item_sync_complete_fn=lambda result_dir: (
            (Path(result_dir) / "summary.json").exists()
            and (Path(result_dir) / "soma_spikes.npz").exists()
        ),
        local_result_dir_has_diagnostics_fn=lambda result_dir: any(
            (Path(result_dir) / name).exists()
            for name in ("stdout.txt", "stderr.txt", "bootstrap.log")
        ),
        progress_write=progress_messages.append,
        refresh_remote_leases_fn=lambda **_kwargs: refresh_calls.append("refresh"),
        record_timing_fn=lambda key, started: timing_calls.append((key, started)),
        perf_counter_fn=lambda: 1.0,
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        sweep_dir = tmpdir_path / "sweep"
        item_runs_dir = sweep_dir / "item_runs"
        ready_item_dir = item_runs_dir / "item_ready"
        ready_item_dir.mkdir(parents=True, exist_ok=True)
        (ready_item_dir / "summary.json").write_text("{}")
        (ready_item_dir / "soma_spikes.npz").write_bytes(b"payload")

        progress_messages: list[str] = []
        timing_calls: list[tuple[str, float]] = []
        refresh_calls: list[str] = []
        bulk_entries_seen: list[dict[str, object]] = []

        sweep_summary = {
            "completed_items": [
                {"label": "item_ready", "ok": True, "result_dir": "/remote/item_ready"},
                {"label": "item_missing", "ok": True, "result_dir": "/remote/item_missing"},
            ],
            "failed_items": [
                {"label": "item_failed", "ok": False, "result_dir": "/remote/item_failed"},
            ],
        }

        def _metadata_sync(_config, *, local_result_dir, **_kwargs):
            local_result_dir = Path(local_result_dir)
            local_result_dir.mkdir(parents=True, exist_ok=True)
            (local_result_dir / "summary.json").write_text(json.dumps(sweep_summary))
            return _completed(returncode=0, stdout="metadata ok")

        def _bulk_sync(_config, *, local_sweep_dir, entries):
            bulk_entries_seen.extend(dict(entry) for entry in entries)
            local_sweep_dir = Path(local_sweep_dir)
            missing_dir = local_sweep_dir / "item_runs" / "item_missing"
            missing_dir.mkdir(parents=True, exist_ok=True)
            (missing_dir / "summary.json").write_text("{}")
            (missing_dir / "soma_spikes.npz").write_bytes(b"payload")
            failed_dir = local_sweep_dir / "item_runs" / "item_failed"
            failed_dir.mkdir(parents=True, exist_ok=True)
            (failed_dir / "stderr.txt").write_text("diagnostic")
            return _completed(returncode=0, stdout="bulk ok")

        success_result = finalize_remote_sweep_artifacts(
            {"remote_sync_compress": True},
            final_status={"progress_payload": {"finished_items": [{"label": "item_missing"}]}},
            local_sweep_dir=sweep_dir,
            local_runs_dir=item_runs_dir,
            remote_sweep_root=PurePosixPath("/remote/sweep"),
            sweep_label="sweep",
            manifest_items=[
                {"label": "item_ready", "result_dir": "/remote/item_ready"},
                {"label": "item_missing", "result_dir": "/remote/item_missing"},
                {"label": "item_failed", "result_dir": "/remote/item_failed"},
            ],
            item_status_by_label={"item_live": {"state": "RUNNING"}},
            hooks=_make_hooks(
                sync_remote_result_dir_fn=_metadata_sync,
                sync_remote_sweep_compact_items_fn=_bulk_sync,
                recover_local_sweep_summary_fn=lambda *_args, **_kwargs: {},
                progress_messages=progress_messages,
                timing_calls=timing_calls,
                refresh_calls=refresh_calls,
            ),
        )
        assert success_result.final_sync.returncode == 0
        assert success_result.sweep_summary == sweep_summary
        assert [entry["label"] for entry in bulk_entries_seen] == ["item_missing", "item_failed"]
        assert success_result.item_status_by_label["item_live"]["state"] == "RUNNING"
        assert success_result.item_status_by_label["item_missing"]["ok"] is True
        assert success_result.item_status_by_label["item_failed"]["ok"] is False
        assert (sweep_dir / "sim_progress.json").exists()
        assert (sweep_dir / "sync_stdout.txt").read_text() == "[sweep-metadata]\nmetadata ok[sweep-items-bulk]\nbulk ok"
        assert (sweep_dir / "sync_stderr.txt").read_text() == ""
        assert timing_calls == [("sync_s", 1.0)]
        assert len(refresh_calls) == 4
        assert progress_messages == ["[OBGPU load] Syncing compact artifacts for 2 sweep items in one stream..."]
        print("remote sweep artifact bulk-finalization path: OK")

        recovered_dir = tmpdir_path / "recovered"
        recovered_timing_calls: list[tuple[str, float]] = []
        recovered_result = finalize_remote_sweep_artifacts(
            {},
            final_status=None,
            local_sweep_dir=recovered_dir,
            local_runs_dir=recovered_dir / "item_runs",
            remote_sweep_root=PurePosixPath("/remote/recovered"),
            sweep_label="recovered",
            manifest_items=[],
            item_status_by_label={},
            hooks=_make_hooks(
                sync_remote_result_dir_fn=lambda *_args, **_kwargs: _completed(
                    returncode=1,
                    stderr="metadata failed",
                ),
                sync_remote_sweep_compact_items_fn=lambda *_args, **_kwargs: _completed(returncode=0),
                recover_local_sweep_summary_fn=lambda *_args, **_kwargs: {"completed_items": []},
                timing_calls=recovered_timing_calls,
            ),
        )
        assert recovered_result.final_sync.returncode == 0
        assert "sufficient to recover a sweep summary" in recovered_result.final_sync.stderr
        assert recovered_timing_calls == [("sync_s", 1.0)]
        print("remote sweep artifact recovered-summary path: OK")

        missing_dir = tmpdir_path / "missing"
        missing_result = finalize_remote_sweep_artifacts(
            {},
            final_status=None,
            local_sweep_dir=missing_dir,
            local_runs_dir=missing_dir / "item_runs",
            remote_sweep_root=PurePosixPath("/remote/missing"),
            sweep_label="missing",
            manifest_items=[],
            item_status_by_label={"item_live": {"state": "RUNNING"}},
            hooks=_make_hooks(
                sync_remote_result_dir_fn=lambda *_args, **_kwargs: _completed(returncode=0),
                sync_remote_sweep_compact_items_fn=lambda *_args, **_kwargs: _completed(returncode=0),
                recover_local_sweep_summary_fn=lambda *_args, **_kwargs: {},
            ),
        )
        assert missing_result.final_sync.returncode == 1
        assert missing_result.sweep_summary == {}
        assert missing_result.item_status_by_label == {"item_live": {"state": "RUNNING"}}
        assert "could not fetch summary metadata" in missing_result.final_sync.stderr
        print("remote sweep artifact missing-summary path: OK")


if __name__ == "__main__":
    main()
