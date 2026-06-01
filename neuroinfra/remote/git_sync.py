"""Reusable local-side Git helpers for remote publication workflows."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path, PurePosixPath
import shlex


def _run_git(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one Git command rooted at ``repo_root``."""
    return subprocess.run(
        args,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def resolve_local_git_head(repo_root: Path) -> str | None:
    """Return the current local git HEAD commit or ``None`` when unavailable."""
    completed = _run_git(repo_root, ["git", "rev-parse", "HEAD"])
    if completed.returncode != 0:
        return None
    head = (completed.stdout or "").strip()
    return head or None


def resolve_local_git_branch(repo_root: Path) -> str | None:
    """Return the current local branch name or ``None`` when detached."""
    completed = _run_git(repo_root, ["git", "branch", "--show-current"])
    if completed.returncode != 0:
        return None
    branch = (completed.stdout or "").strip()
    return branch or None


def resolve_local_git_upstream_ref(repo_root: Path) -> str | None:
    """Return the current branch upstream ref, or ``None`` when unavailable."""
    completed = _run_git(
        repo_root,
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
    )
    if completed.returncode != 0:
        return None
    upstream = (completed.stdout or "").strip()
    return upstream or None


def git_rev_parse(repo_root: Path, ref_name: str) -> str | None:
    """Resolve one local git ref to a commit SHA."""
    completed = _run_git(repo_root, ["git", "rev-parse", ref_name])
    if completed.returncode != 0:
        return None
    sha = (completed.stdout or "").strip()
    return sha or None


def git_ref_points_to_commit(repo_root: Path, ref_name: str, commit_sha: str) -> bool:
    """Return whether one local git ref currently resolves to the requested commit."""
    return git_rev_parse(repo_root, ref_name) == commit_sha


def git_ref_is_ancestor(repo_root: Path, ancestor_ref: str, descendant_ref: str) -> bool:
    """Return whether one git ref is an ancestor of another."""
    completed = _run_git(
        repo_root,
        ["git", "merge-base", "--is-ancestor", ancestor_ref, descendant_ref],
    )
    return completed.returncode == 0


def git_merged_ref_shas(repo_root: Path, commit_sha: str, *, max_count: int = 128) -> list[str]:
    """Return ancestor ref tips already merged into one commit."""
    completed = _run_git(
        repo_root,
        [
            "git",
            "for-each-ref",
            "--merged={}".format(commit_sha),
            "--format=%(objectname)",
            "refs/heads",
            "refs/remotes",
        ],
    )
    if completed.returncode != 0:
        return []
    shas: list[str] = []
    seen: set[str] = set()
    for line in (completed.stdout or "").splitlines():
        sha = line.strip()
        if not sha or sha == commit_sha or sha in seen:
            continue
        seen.add(sha)
        shas.append(sha)
        if len(shas) >= int(max_count):
            break
    return shas


def local_git_sync_base_candidates(repo_root: Path, commit_sha: str, *, max_count: int = 500) -> list[str]:
    """Return local ancestor SHAs to test as possible remote bundle bases."""
    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(ref_name: str | None) -> None:
        if not ref_name:
            return
        sha = git_rev_parse(repo_root, ref_name)
        if not sha or sha == commit_sha or sha in seen:
            return
        if not git_ref_is_ancestor(repo_root, sha, commit_sha):
            return
        seen.add(sha)
        candidates.append(sha)

    for sha in git_merged_ref_shas(repo_root, commit_sha, max_count=min(int(max_count), 128)):
        add_candidate(sha)

    completed = _run_git(
        repo_root,
        ["git", "rev-list", "--first-parent", "--max-count={}".format(int(max_count)), "{}^".format(commit_sha)],
    )
    if completed.returncode == 0:
        for line in (completed.stdout or "").splitlines():
            add_candidate(line.strip())

    add_candidate(resolve_local_git_upstream_ref(repo_root))
    return candidates


def create_git_bundle_for_commit(
    repo_root: Path,
    commit_sha: str,
    *,
    exclude_ref: str | None = None,
) -> tuple[Path, str]:
    """Create a temporary git bundle for the requested commit."""
    branch_name = resolve_local_git_branch(repo_root)
    temp_ref: str | None = None
    source_ref: str

    if branch_name and git_ref_points_to_commit(repo_root, branch_name, commit_sha):
        source_ref = "refs/heads/{}".format(branch_name)
    else:
        temp_ref = "refs/obgpu-notebook-sync/{}".format(commit_sha)
        updated = _run_git(repo_root, ["git", "update-ref", temp_ref, commit_sha])
        if updated.returncode != 0:
            raise RuntimeError(
                "Could not create a temporary git ref for the remote sync bundle.\n"
                "Commit: {}\n"
                "Stderr:\n{}".format(commit_sha, updated.stderr)
            )
        source_ref = temp_ref

    bundle_handle = tempfile.NamedTemporaryFile(prefix="obgpu-sol-sync-", suffix=".bundle", delete=False)
    bundle_path = Path(bundle_handle.name)
    bundle_handle.close()

    try:
        bundle_args = ["git", "bundle", "create", str(bundle_path), source_ref]
        if (
            exclude_ref
            and not git_ref_points_to_commit(repo_root, exclude_ref, commit_sha)
            and git_ref_is_ancestor(repo_root, exclude_ref, commit_sha)
        ):
            bundle_args.append("^{}".format(exclude_ref))
        created = _run_git(repo_root, bundle_args)
        if created.returncode != 0:
            raise RuntimeError(
                "Could not create a git bundle for the remote backend.\n"
                "Source ref: {}\n"
                "Stderr:\n{}".format(source_ref, created.stderr)
            )
        return bundle_path, source_ref
    except Exception:
        try:
            bundle_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    finally:
        if temp_ref is not None:
            _run_git(repo_root, ["git", "update-ref", "-d", temp_ref])


def remote_notebook_tracking_ref_for_source(source_ref: str) -> str | None:
    """Return the stable remote notebook ref for one published local branch tip."""
    branch_prefix = "refs/heads/"
    if not source_ref.startswith(branch_prefix):
        return None
    branch_name = source_ref[len(branch_prefix):].strip("/")
    if not branch_name:
        return None
    return "refs/obgpu-notebook-sync/heads/{}".format(branch_name)


def build_remote_git_repo_probe_command(remote_repo_root: PurePosixPath) -> str:
    """Build a remote shell command that verifies the configured repo exists."""
    repo_root = remote_repo_root.as_posix()
    quoted_repo = shlex.quote(repo_root)
    missing_message = shlex.quote("remote_repo_root does not exist: {}".format(repo_root))
    not_git_message = shlex.quote("remote_repo_root is not a git work tree: {}".format(repo_root))
    return (
        "if ! test -d {repo}; then printf '%s\\n' {missing} >&2; exit 2; fi; "
        "if ! git -C {repo} rev-parse --is-inside-work-tree >/dev/null 2>&1; "
        "then printf '%s\\n' {not_git} >&2; exit 3; fi"
    ).format(repo=quoted_repo, missing=missing_message, not_git=not_git_message)


def build_remote_git_bundle_fetch_command(
    *,
    remote_repo_root: PurePosixPath,
    remote_bundle_path: str,
    source_ref: str,
    remote_git_ref: str,
) -> str:
    """Build the remote git fetch command used to publish one local bundle."""
    remote_private_ref = "refs/obgpu-notebook-sync/{}".format(remote_git_ref)
    fetch_refspecs = ["{}:{}".format(source_ref, remote_private_ref)]
    tracking_ref = remote_notebook_tracking_ref_for_source(source_ref)
    if tracking_ref and tracking_ref != remote_private_ref:
        fetch_refspecs.append("{}:{}".format(source_ref, tracking_ref))
    fetch_refspec_args = " ".join(shlex.quote(refspec) for refspec in fetch_refspecs)
    remote_git_lock = shlex.quote((remote_repo_root / ".obgpu-git.lock").as_posix())
    fetch_body = (
        "git -C {repo} fetch --force --no-tags {bundle} {refspecs}"
        " && git -C {repo} cat-file -e {commit}"
        " && rm -f {bundle}"
    ).format(
        repo=shlex.quote(remote_repo_root.as_posix()),
        bundle=shlex.quote(remote_bundle_path),
        refspecs=fetch_refspec_args,
        commit=shlex.quote(remote_git_ref + "^{commit}"),
    )
    return (
        "if command -v flock >/dev/null 2>&1; then "
        "touch {lock} && flock {lock} bash -lc {body}; "
        "else {body}; fi"
    ).format(lock=remote_git_lock, body=shlex.quote(fetch_body))
