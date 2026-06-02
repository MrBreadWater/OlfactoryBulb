"""Smoke tests for standardized Git publication helpers."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import subprocess

import neuroinfra.remote.git_sync as git_sync
import obgpu_experiment_helpers as hlp


def main() -> None:
    repo_root = hlp.REPO_ROOT
    head_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()
    parent_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD^"],
        cwd=repo_root,
        text=True,
    ).strip()

    assert git_sync.resolve_local_git_head(repo_root) == hlp._resolve_local_git_head()
    assert git_sync.resolve_local_git_branch(repo_root) == hlp._resolve_local_git_branch()
    assert git_sync.resolve_local_git_upstream_ref(repo_root) == hlp._resolve_local_git_upstream_ref()
    assert git_sync.git_rev_parse(repo_root, "HEAD") == head_sha == hlp._git_rev_parse("HEAD")
    assert git_sync.git_ref_points_to_commit(repo_root, "HEAD", head_sha) is True
    assert hlp._git_ref_points_to_commit("HEAD", head_sha) is True

    candidates = git_sync.local_git_sync_base_candidates(repo_root, head_sha, max_count=32)
    assert candidates == hlp._local_git_sync_base_candidates(head_sha, max_count=32)
    assert head_sha not in candidates
    if candidates:
        assert git_sync.git_ref_is_ancestor(repo_root, candidates[0], head_sha)
        assert hlp._git_ref_is_ancestor(candidates[0], head_sha)

    tracking_ref = git_sync.remote_notebook_tracking_ref_for_source("refs/heads/Speedups")
    assert tracking_ref == "refs/obgpu-notebook-sync/heads/Speedups"
    assert tracking_ref == hlp._remote_notebook_tracking_ref_for_source("refs/heads/Speedups")
    assert git_sync.remote_notebook_tracking_ref_for_source("refs/obgpu-notebook-sync/tmp") is None

    fetch_command = git_sync.build_remote_git_bundle_fetch_command(
        remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        remote_bundle_path="/tmp/example.bundle",
        source_ref="refs/heads/Speedups",
        remote_git_ref="abcdef1234567890",
    )
    assert fetch_command == hlp._build_remote_git_bundle_fetch_command(
        remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        remote_bundle_path="/tmp/example.bundle",
        source_ref="refs/heads/Speedups",
        remote_git_ref="abcdef1234567890",
    )
    assert "refs/obgpu-notebook-sync/abcdef1234567890" in fetch_command
    assert "refs/obgpu-notebook-sync/heads/Speedups" in fetch_command

    repo_probe = git_sync.build_remote_git_repo_probe_command(PurePosixPath("/remote/OlfactoryBulb"))
    assert repo_probe == hlp._build_remote_git_repo_probe_command(PurePosixPath("/remote/OlfactoryBulb"))
    assert "remote_repo_root does not exist" in repo_probe
    assert "is not a git work tree" in repo_probe

    bundle_path, source_ref = git_sync.create_git_bundle_for_commit(
        repo_root,
        head_sha,
        exclude_ref=parent_sha,
    )
    try:
        assert bundle_path.exists()
        assert bundle_path.stat().st_size > 0
        assert source_ref
        verify = subprocess.run(
            ["git", "bundle", "verify", str(bundle_path)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert verify.returncode == 0, verify.stderr
    finally:
        bundle_path.unlink(missing_ok=True)

    wrapper_bundle_path, wrapper_source_ref = hlp._create_git_bundle_for_commit(
        head_sha,
        exclude_ref=parent_sha,
    )
    try:
        assert wrapper_bundle_path.exists()
        assert wrapper_bundle_path.stat().st_size > 0
        assert wrapper_source_ref == source_ref or wrapper_source_ref.startswith("refs/")
    finally:
        wrapper_bundle_path.unlink(missing_ok=True)

    print("neuroinfra remote git sync smoke test: OK")


if __name__ == "__main__":
    main()
