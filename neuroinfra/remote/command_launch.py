"""Reusable local-side helpers for launching uploaded remote Python helpers."""

from __future__ import annotations

from base64 import b64encode
from pathlib import Path, PurePosixPath
import shlex


def remote_helper_script_path(
    remote_helper_dir: PurePosixPath | None,
    script_name: str,
) -> PurePosixPath | None:
    """Return the uploaded helper-script path when a cache directory is available."""
    if remote_helper_dir is None:
        return None
    return remote_helper_dir / str(script_name)


def remote_python_exec_prefix() -> str:
    """Return the remote shell prefix that resolves python3/python and execs it."""
    return (
        'REMOTE_PYTHON="$(command -v python3 || command -v python || true)"'
        ' && test -n "$REMOTE_PYTHON"'
        ' && exec "$REMOTE_PYTHON"'
    )


def _shell_join(parts: list[str]) -> str:
    """Portable equivalent of ``shlex.join`` for local command assembly."""
    return " ".join(shlex.quote(str(part)) for part in parts)


def build_remote_python_file_command(script_path: PurePosixPath, argv: list[str]) -> str:
    """Build a remote shell command that executes one uploaded helper script."""
    return remote_python_exec_prefix() + " " + _shell_join([script_path.as_posix(), *argv])


def build_remote_python_inline_command(script_path: Path, argv: list[str]) -> str:
    """Build a remote shell command that executes one helper script inline."""
    helper_b64 = b64encode(script_path.read_bytes()).decode("ascii")
    python_exec = (
        remote_python_exec_prefix()
        + " -c "
        + shlex.quote(
            'import base64,sys; '
            'script_b64=sys.argv[1]; '
            'script_path=sys.argv[2]; '
            'sys.argv=sys.argv[2:]; '
            'namespace={"__name__":"__main__","__file__":script_path}; '
            'exec(compile(base64.b64decode(script_b64).decode("utf-8"), script_path, "exec"), namespace)'
        )
    )
    return python_exec + " " + _shell_join([helper_b64, str(script_path), *argv])


def build_remote_touch_command(path_value: str | PurePosixPath) -> str:
    """Build a remote command that refreshes one heartbeat path."""
    path = PurePosixPath(str(path_value))
    return (
        f"mkdir -p {shlex.quote(path.parent.as_posix())} && "
        f"touch {shlex.quote(path.as_posix())}"
    )
