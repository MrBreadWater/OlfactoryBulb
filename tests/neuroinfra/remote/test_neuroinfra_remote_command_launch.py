"""Smoke tests for standardized remote helper command builders."""

from __future__ import annotations

from base64 import b64encode
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import neuroinfra.remote.command_launch as command_launch
import obgpu_experiment_helpers as hlp


def main() -> None:
    assert 'command -v python3' in command_launch.remote_python_exec_prefix()
    assert command_launch.remote_helper_script_path(None, "helper.py") is None
    assert command_launch.remote_helper_script_path(
        PurePosixPath("/remote/cache"),
        "helper.py",
    ) == PurePosixPath("/remote/cache/helper.py")

    file_command = command_launch.build_remote_python_file_command(
        PurePosixPath("/remote/cache/helper.py"),
        ["--flag", "a b"],
    )
    assert '/remote/cache/helper.py' in file_command
    assert "'a b'" in file_command

    touch_command = command_launch.build_remote_touch_command("/remote/cache/heartbeat.txt")
    assert "mkdir -p /remote/cache" in touch_command
    assert "touch /remote/cache/heartbeat.txt" in touch_command

    with TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        helper_script = tmp / "helper.py"
        helper_script.write_text("print('hello')\n")
        helper_b64 = b64encode(helper_script.read_bytes()).decode("ascii")
        inline_command = command_launch.build_remote_python_inline_command(
            helper_script,
            ["--value", "a b"],
        )
        assert helper_b64 in inline_command
        assert str(helper_script) in inline_command
        assert "'a b'" in inline_command

        assert hlp._build_remote_python_inline_command(helper_script, ["--value", "a b"]) == inline_command

    assert (
        hlp._remote_helper_script_path(PurePosixPath("/remote/cache"), "helper.py")
        == command_launch.remote_helper_script_path(PurePosixPath("/remote/cache"), "helper.py")
    )
    assert hlp._remote_python_exec_prefix() == command_launch.remote_python_exec_prefix()
    assert (
        hlp._build_remote_python_file_command(PurePosixPath("/remote/cache/helper.py"), ["--flag", "a b"])
        == file_command
    )
    assert hlp._build_remote_touch_command("/remote/cache/heartbeat.txt") == touch_command

    print("neuroinfra remote command launch smoke test: OK")


if __name__ == "__main__":
    main()
