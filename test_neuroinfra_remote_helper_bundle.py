"""Smoke tests for the standardized neuroinfra remote helper-bundle utilities."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import neuroinfra.remote.helper_bundle as helper_bundle
import obgpu_experiment_helpers as hlp


def main() -> None:
    with TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        root_a = tmp / "submit.py"
        nested_b = tmp / "pkg" / "worker.py"
        nested_b.parent.mkdir(parents=True, exist_ok=True)
        root_a.write_text("print('a')\n")
        nested_b.write_text("print('b')\n")

        entries = (
            helper_bundle.HelperBundleEntry("submit.py", root_a),
            helper_bundle.HelperBundleEntry("pkg/worker.py", nested_b),
        )
        mapping = helper_bundle.bundle_entries_by_path(entries)
        assert mapping["submit.py"] == root_a
        assert mapping["pkg/worker.py"] == nested_b
        assert helper_bundle.helper_bundle_parent_dirs(entries) == ("pkg",)
        manifest = helper_bundle.helper_bundle_manifest(entries)
        assert manifest["files"] == ["pkg/worker.py", "submit.py"]
        assert manifest["parent_dirs"] == ["pkg"]
        assert len(str(manifest["signature"])) == 64

        try:
            helper_bundle.bundle_entries_by_path(
                (
                    helper_bundle.HelperBundleEntry("dup.py", root_a),
                    helper_bundle.HelperBundleEntry("dup.py", nested_b),
                )
            )
            raise AssertionError("duplicate relative paths should fail")
        except ValueError:
            pass

        try:
            helper_bundle.normalize_helper_relative_path("../escape.py")
            raise AssertionError("parent traversal should fail")
        except ValueError:
            pass

    helper_sources = hlp._remote_helper_sources()
    helper_manifest = helper_bundle.helper_bundle_manifest(
        hlp._remote_helper_bundle_entries(),
        signature=hlp._remote_helper_signature(),
    )
    assert helper_sources["slurm_common.py"] == hlp.REPO_ROOT / "tools" / "remote" / "slurm_common.py"
    assert helper_sources["neuroinfra/remote_script_common.py"] == hlp.REPO_ROOT / "neuroinfra" / "remote_script_common.py"
    assert helper_sources["neuroinfra/remote_script_polling.py"] == hlp.REPO_ROOT / "neuroinfra" / "remote_script_polling.py"
    assert helper_sources["neuroinfra/remote_script_allocations.py"] == hlp.REPO_ROOT / "neuroinfra" / "remote_script_allocations.py"
    assert sorted(helper_sources.keys()) == helper_manifest["files"]
    assert helper_manifest["parent_dirs"] == ["neuroinfra"]
    assert helper_manifest["signature"] == hlp._remote_helper_signature()
    print("neuroinfra remote helper bundle smoke test: OK")


if __name__ == "__main__":
    main()
