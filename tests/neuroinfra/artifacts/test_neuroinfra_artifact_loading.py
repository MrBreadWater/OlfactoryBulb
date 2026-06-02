"""Smoke tests for extracted local artifact-loading helpers."""

from __future__ import annotations

import pickle
from pathlib import Path
import tempfile

from neuroinfra.artifacts.loading import (
    ArtifactLoadingHooks,
    LazyResult,
    load_local_artifact_plan,
)


class _ProgressProbe:
    def __init__(self) -> None:
        self.updates: list[int] = []
        self.closed = False

    def update_to(self, value: int) -> None:
        self.updates.append(int(value))

    def close(self) -> None:
        self.closed = True


def main() -> None:
    lazy_attempts: list[str] = []

    def _flaky_loader():
        lazy_attempts.append("attempt")
        if len(lazy_attempts) == 1:
            raise RuntimeError("transient")
        return [("MC0", [0.0], [-65.0])]

    lazy_result = LazyResult({"soma_vs": []}, lazy_loaders={"soma_vs": _flaky_loader})
    try:
        _ = lazy_result["soma_vs"]
        raise AssertionError("Expected first lazy load attempt to fail")
    except RuntimeError:
        pass
    assert "soma_vs" in lazy_result._lazy_loaders
    assert lazy_result["soma_vs"][0][0] == "MC0"
    assert "soma_vs" not in lazy_result._lazy_loaders
    print("artifact loading LazyResult retry: OK")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / "input_times.pkl"
        lfp_path = tmpdir_path / "lfp.pkl"
        with input_path.open("wb") as handle:
            pickle.dump([1, 2, 3], handle)
        with lfp_path.open("wb") as handle:
            pickle.dump(([0.0, 0.1], [1.0, 2.0]), handle)

        progress_messages: list[str] = []
        progress_instances: list[_ProgressProbe] = []
        result = {
            "input_times": [],
            "lfp_t": [],
            "lfp": [],
        }

        def _apply_loaded(target, key, loaded):
            if key == "lfp":
                target["lfp_t"], target["lfp"] = loaded
            else:
                target[key] = loaded

        hooks = ArtifactLoadingHooks(
            load_pickle_fn=lambda path: pickle.load(open(path, "rb")),
            apply_loaded_fn=_apply_loaded,
            progress_factory_fn=lambda total, desc: progress_instances.append(_ProgressProbe()) or progress_instances[-1],
            progress_write=progress_messages.append,
            format_bytes_fn=lambda value: f"{int(value)} B",
            render_progress_bar_fn=lambda loaded, total: f"{loaded}/{total}",
            perf_counter_fn=lambda: 1.0,
        )
        load_timings, load_total_seconds = load_local_artifact_plan(
            result,
            [("input_times", input_path), ("lfp", lfp_path)],
            hooks=hooks,
            progress=True,
        )
        assert result["input_times"] == [1, 2, 3]
        assert result["lfp_t"] == [0.0, 0.1]
        assert result["lfp"] == [1.0, 2.0]
        assert set(load_timings) == {"input_times.pkl", "lfp.pkl"}
        assert load_total_seconds == 0.0
        assert progress_instances and progress_instances[0].closed
        assert progress_instances[0].updates
        assert any("Loading 2 local result files" in message for message in progress_messages)
        print("artifact loading local plan execution: OK")


if __name__ == "__main__":
    main()
