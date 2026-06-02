"""Compatibility smoke tests for the first-wave neuroinfra artifact extraction."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import neuroinfra.artifacts.output_paths as new_paths
import neuroinfra.artifacts.result_artifacts as new_artifacts
import olfactorybulb.output_paths as old_paths
import olfactorybulb.result_artifacts as old_artifacts


def main() -> None:
    assert old_paths.make_timestamp.__module__ == new_paths.make_timestamp.__module__
    assert old_paths.label_with_timestamp.__module__ == new_paths.label_with_timestamp.__module__
    assert old_artifacts.find_soma_trace_artifact.__module__ == new_artifacts.find_soma_trace_artifact.__module__
    assert old_artifacts.load_saved_result_artifact.__module__ == new_artifacts.load_saved_result_artifact.__module__

    with TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        trace_path = tmp / "soma_vs.npz"
        t = np.asarray([0.0, 0.1, 0.2, 0.3], dtype=float)
        v = np.asarray([-60.0, -55.0, 20.0, -62.0], dtype=float)
        traces = [("MC1[0].soma", t, v)]
        old_artifacts.save_soma_trace_artifact(traces, trace_path)
        located = new_artifacts.find_soma_trace_artifact(tmp)
        assert located == trace_path
        loaded = new_artifacts.load_soma_trace_artifact(trace_path)
        assert len(loaded) == 1
        assert loaded[0][0] == "MC1[0].soma"
        np.testing.assert_allclose(loaded[0][1], t)
        np.testing.assert_allclose(loaded[0][2], v, atol=1e-5)

        label, timestamp = old_paths.configure_output_env("compat_probe", results_base=tmp)
        assert label.endswith(timestamp)
        info_path = new_paths.write_run_info(tmp / label, {"ok": True, "timestamp": timestamp})
        assert info_path.exists()

    print("neuroinfra artifacts compatibility smoke test: OK")


if __name__ == "__main__":
    main()
