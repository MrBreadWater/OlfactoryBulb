"""Smoke tests for standardized remote helper-cache utilities."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import neuroinfra.remote.helper_bundle as helper_bundle
import neuroinfra.remote.helper_cache as helper_cache
import obgpu_experiment_helpers as hlp


def main() -> None:
    entries = (
        helper_bundle.HelperBundleEntry("submit.py", Path(__file__)),
        helper_bundle.HelperBundleEntry("pkg/worker.py", Path(__file__)),
    )
    signature = helper_bundle.helper_bundle_signature(entries)
    results_root = PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs")
    remote_dir = helper_cache.helper_cache_dir(results_root=results_root, signature=signature[:20])
    assert remote_dir == PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/.obgpu-helper-cache") / signature[:20]
    assert helper_cache.helper_cache_manifest_path(remote_dir) == remote_dir / "manifest.json"
    assert helper_cache.helper_cache_runtime_key(
        connection_key="user@host:22",
        results_root=results_root,
        signature=signature[:20],
    ) == "user@host:22::/remote/OlfactoryBulb/results/notebook_runs::{}".format(signature[:20])

    probe_command = helper_cache.helper_cache_probe_command(remote_dir / "manifest.json")
    assert "cat" in probe_command
    assert "manifest.json" in probe_command
    assert helper_cache.helper_cache_probe_matches('{"signature":"abc"}', expected_signature="abc") is True
    assert helper_cache.helper_cache_probe_matches('{"signature":"abc"}', expected_signature="def") is False
    assert helper_cache.helper_cache_probe_matches("not json", expected_signature="abc") is False

    mkdir_targets = helper_cache.helper_cache_mkdir_targets(remote_dir=remote_dir, entries=entries)
    assert mkdir_targets == (
        remote_dir.as_posix(),
        (remote_dir / "pkg").as_posix(),
    )

    helper_sources, manifest_payload, manifest_path = helper_cache.helper_cache_upload_payload(
        remote_dir=remote_dir,
        entries=entries,
        signature=signature,
    )
    assert sorted(helper_sources.keys()) == ["pkg/worker.py", "submit.py"]
    assert manifest_payload["signature"] == signature
    assert manifest_payload["parent_dirs"] == ["pkg"]
    assert manifest_path == remote_dir / "manifest.json"

    helper_entries = hlp._remote_helper_bundle_entries()
    helper_signature = hlp._remote_helper_signature()
    helper_remote_dir = hlp._remote_helper_cache_dir(
        {
            "remote_host": "user@host",
            "remote_results_root": "/remote/OlfactoryBulb/results/notebook_runs",
        }
    )
    assert helper_remote_dir == PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/.obgpu-helper-cache") / helper_signature
    assert hlp._remote_helper_cache_runtime_key(
        {
            "remote_host": "user@host",
            "remote_results_root": "/remote/OlfactoryBulb/results/notebook_runs",
        }
    ) == helper_cache.helper_cache_runtime_key(
        connection_key=hlp._paramiko_connection_key(
            {
                "remote_host": "user@host",
                "remote_results_root": "/remote/OlfactoryBulb/results/notebook_runs",
            }
        ),
        results_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs"),
        signature=helper_signature,
    )
    _, wrapper_manifest_payload, wrapper_manifest_path = helper_cache.helper_cache_upload_payload(
        remote_dir=helper_remote_dir,
        entries=helper_entries,
        signature=helper_signature,
    )
    assert wrapper_manifest_payload["files"] == sorted(hlp._remote_helper_sources().keys())
    assert wrapper_manifest_path == helper_remote_dir / "manifest.json"

    print("neuroinfra remote helper cache smoke test: OK")


if __name__ == "__main__":
    main()
