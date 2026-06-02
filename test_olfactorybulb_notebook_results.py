"""Focused tests for olfactory-bulb notebook result-loading adapters."""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np

from neuroinfra.artifacts.loading import ArtifactLoadingHooks
from neuroinfra.artifacts.result_view import ResultViewHooks

from olfactorybulb.notebook_results import (
    NotebookResultHooks,
    apply_loaded_result_artifact,
    load_result,
    set_lazy_result_artifact_path,
)


def _load_pickle(path: str | Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _write_pickle(path: Path, payload) -> None:
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        result_dir = Path(tmp_dir)
        messages = []

        (result_dir / "summary.json").write_text(json.dumps({"label": "demo", "ok": True}))
        (result_dir / "run_info.json").write_text(json.dumps({"remote": {"deferred_remote_artifacts": []}}))
        _write_pickle(result_dir / "input_times.pkl", [{"segment": "MC1", "times": [1.0, 2.0]}])
        _write_pickle(result_dir / "lfp.pkl", (np.array([0.0, 1.0]), np.array([2.0, 3.0])))
        _write_pickle(result_dir / "soma_vs.pkl", {"MC1": [0.1, 0.2]})

        result_view_hooks = ResultViewHooks(
            read_json_if_present_fn=lambda path: json.loads(Path(path).read_text()) if Path(path).exists() else None,
            standard_result_artifact_sizes_fn=lambda path: {},
            local_sync_artifact_is_usable_fn=lambda path: Path(path).exists() and Path(path).stat().st_size > 0,
            sync_deferred_artifact_fn=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected remote sync")),
            load_pickle_fn=_load_pickle,
            set_lazy_artifact_path_fn=set_lazy_result_artifact_path,
            local_lazy_notice_fn=lambda key, path: f"lazy-local:{key}:{Path(path).name}",
            remote_lazy_notice_fn=lambda key, path: f"lazy-remote:{key}:{Path(path).name}",
            progress_write=messages.append,
        )
        artifact_loading_hooks = ArtifactLoadingHooks(
            load_pickle_fn=_load_pickle,
            apply_loaded_fn=apply_loaded_result_artifact,
            progress_factory_fn=lambda total_bytes, desc: None,
            progress_write=messages.append,
            format_bytes_fn=lambda size: f"{int(size)} B",
            render_progress_bar_fn=lambda loaded, total: f"{loaded}/{total}",
        )
        hooks = NotebookResultHooks(
            find_soma_trace_artifact_fn=lambda path: Path(path) / "soma_vs.pkl",
            preferred_soma_trace_artifact_name_fn=lambda: "soma_vs.pkl",
            soma_trace_artifact_candidates_fn=lambda: ("soma_vs.pkl", "soma_vs.npz"),
            result_view_hooks=result_view_hooks,
            artifact_loading_hooks=artifact_loading_hooks,
        )

        eager = load_result(hooks, result_dir, progress=True)
        assert eager["summary"]["label"] == "demo"
        assert eager["input_times"][0]["segment"] == "MC1"
        assert np.allclose(eager["lfp_t"], np.array([0.0, 1.0]))
        assert np.allclose(eager["lfp"], np.array([2.0, 3.0]))
        assert eager["soma_vs"]["MC1"] == [0.1, 0.2]
        assert "soma_vs_file" not in eager
        assert isinstance(eager["artifact_sizes"], dict)
        assert "Local file timings" in messages[-1]

        messages.clear()
        lazy = load_result(hooks, result_dir, lazy_soma_vs=True, progress=True)
        assert lazy["soma_vs"] == {"MC1": [0.1, 0.2]}
        assert lazy["soma_vs_file"] == result_dir / "soma_vs.pkl"
        assert lazy["artifact_sizes"]["soma_vs.pkl"] == (result_dir / "soma_vs.pkl").stat().st_size
        assert any(message == "lazy-local:soma_vs:soma_vs.pkl" for message in messages)
        assert any("Lazy-loading soma_vs" in message for message in messages)
        assert any("Loaded soma_vs in" in message for message in messages)

    print("olfactorybulb notebook results: OK")


if __name__ == "__main__":
    main()
