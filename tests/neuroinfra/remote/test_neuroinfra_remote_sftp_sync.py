"""Smoke tests for standardized SFTP sync planning and transfer loops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import neuroinfra.remote.sftp_sync as sftp_sync


@dataclass
class _FakeEntry:
    filename: str
    st_mode: int
    st_size: int = 0


class _FakeProgress:
    def __init__(self) -> None:
        self.updates: list[int] = []
        self.closed = False

    def update_to(self, value: int) -> None:
        self.updates.append(int(value))

    def close(self) -> None:
        self.closed = True


class _FakeSFTP:
    def __init__(self) -> None:
        self.files = {
            "/remote/result/a.txt": b"alpha",
            "/remote/result/nested/b.txt": b"beta",
            "/remote/result/nested/c.txt": b"gamma",
        }
        self.dirs = {
            "/remote/result": [
                _FakeEntry("a.txt", 0o100644, len(self.files["/remote/result/a.txt"])),
                _FakeEntry("nested", 0o040755, 0),
            ],
            "/remote/result/nested": [
                _FakeEntry("b.txt", 0o100644, len(self.files["/remote/result/nested/b.txt"])),
                _FakeEntry("c.txt", 0o100644, len(self.files["/remote/result/nested/c.txt"])),
            ],
        }

    def listdir_attr(self, remote_dir: str):
        return list(self.dirs.get(remote_dir, []))

    def stat(self, remote_path: str):
        if remote_path in self.files:
            return _FakeEntry(Path(remote_path).name, 0o100644, len(self.files[remote_path]))
        raise FileNotFoundError(remote_path)

    def get(self, remote_path: str, local_path: str, callback=None):
        payload = self.files[remote_path]
        Path(local_path).write_bytes(payload)
        if callback is not None:
            callback(len(payload), len(payload))


def _hooks(progress_log: list[_FakeProgress], messages: list[str]) -> sftp_sync.SFTPSyncHooks:
    return sftp_sync.SFTPSyncHooks(
        progress_factory=lambda _total: progress_log.append(_FakeProgress()) or progress_log[-1],
        progress_write=lambda message: messages.append(str(message)),
        format_bytes=lambda value: f"{int(value)} B",
        render_progress_bar=lambda current, total: f"[{int(current)}/{int(total)}]",
        replace_file_via_temp_copy=lambda copy_fn, local_path: copy_fn(local_path),
    )


def main() -> None:
    sftp = _FakeSFTP()

    with TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        tree_items = sftp_sync.collect_tree_transfer_items(sftp, "/remote/result", tmp / "tree")
        assert [item.remote_path for item in tree_items] == [
            "/remote/result/a.txt",
            "/remote/result/nested/b.txt",
            "/remote/result/nested/c.txt",
        ]

        selected_items = sftp_sync.collect_selected_transfer_items(
            sftp,
            "/remote/result",
            tmp / "selected",
            ["a.txt", "missing.txt", "nested"],
        )
        assert [item.remote_path for item in selected_items] == ["/remote/result/a.txt"]

        progress_log: list[_FakeProgress] = []
        messages: list[str] = []
        sftp_sync.sftp_copy_tree(
            sftp,
            "/remote/result",
            tmp / "tree_copy",
            hooks=_hooks(progress_log, messages),
        )
        assert (tmp / "tree_copy" / "a.txt").read_bytes() == b"alpha"
        assert (tmp / "tree_copy" / "nested" / "b.txt").read_bytes() == b"beta"
        assert progress_log and progress_log[0].closed is True
        assert any("Syncing 3 files" in message for message in messages)

        progress_log.clear()
        messages.clear()
        sftp_sync.sftp_copy_files(
            sftp,
            "/remote/result",
            tmp / "selected_copy",
            ["a.txt", "nested/b.txt", "missing.txt"],
            hooks=_hooks(progress_log, messages),
        )
        assert (tmp / "selected_copy" / "a.txt").read_bytes() == b"alpha"
        assert (tmp / "selected_copy" / "nested/b.txt").read_bytes() == b"beta"
        assert not (tmp / "selected_copy" / "missing.txt").exists()
        assert any("Syncing selected" in message for message in messages)

    print("neuroinfra remote sftp sync smoke test: OK")


if __name__ == "__main__":
    main()
