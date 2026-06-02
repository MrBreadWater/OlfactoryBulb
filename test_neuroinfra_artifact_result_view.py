"""Smoke tests for extracted result-view planning and lazy artifact wiring."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import tempfile

from neuroinfra.artifacts.loading import LazyResult
from neuroinfra.artifacts.result_view import (
    ResultArtifactBinding,
    ResultViewHooks,
    attach_lazy_artifact_loaders,
    plan_result_view,
)


def _read_json_if_present(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _result_factory(*, result_dir: Path, summary: dict | None, run_info: dict | None, artifact_sizes: dict[str, int]) -> LazyResult:
    return LazyResult(
        {
            "result_dir": result_dir,
            "summary": summary,
            "run_info": run_info,
            "artifact_sizes": artifact_sizes,
            "input_times": [],
            "soma_vs": [],
            "lfp_t": [],
            "lfp": [],
        }
    )


def _load_pickle(path: str | Path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        result_dir = tmpdir_path / "result"
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "summary.json").write_text(json.dumps({"ok": True}))
        (result_dir / "run_info.json").write_text(
            json.dumps(
                {
                    "remote": {
                        "deferred_remote_artifacts": ["soma_vs.pkl", "lfp.pkl"],
                    }
                }
            )
        )
        with (result_dir / "input_times.pkl").open("wb") as handle:
            pickle.dump([1, 2, 3], handle)
        with (result_dir / "soma_vs_local.pkl").open("wb") as handle:
            pickle.dump([("MC0", [0.0], [-65.0])], handle)

        progress_messages: list[str] = []
        sync_calls: list[tuple[Path, str]] = []
        lazy_path_calls: list[tuple[str, str]] = []

        def _sync_deferred_artifact(result_dir_arg, *, run_info, filename):
            sync_calls.append((Path(result_dir_arg), filename))
            path = Path(result_dir_arg) / filename
            with path.open("wb") as handle:
                pickle.dump((["t"], ["v"]), handle)
            return path

        hooks = ResultViewHooks(
            read_json_if_present_fn=_read_json_if_present,
            standard_result_artifact_sizes_fn=lambda path: {
                child.name: int(child.stat().st_size)
                for child in sorted(Path(path).iterdir())
                if child.is_file()
            },
            local_sync_artifact_is_usable_fn=lambda path: Path(path).exists() and Path(path).stat().st_size > 0,
            sync_deferred_artifact_fn=_sync_deferred_artifact,
            load_pickle_fn=_load_pickle,
            set_lazy_artifact_path_fn=lambda result, key, path: (
                lazy_path_calls.append((key, path.name)),
                result.__setitem__(f"{key}_file", path),
            )[-1],
            local_lazy_notice_fn=lambda key, path: f"lazy-local:{key}:{path.name}",
            remote_lazy_notice_fn=lambda key, path: f"lazy-remote:{key}:{path.name}",
            progress_write=progress_messages.append,
        )

        eager_plan = plan_result_view(
            result_dir,
            result_factory_fn=_result_factory,
            artifact_bindings=[
                ResultArtifactBinding("input_times", result_dir / "input_times.pkl"),
                ResultArtifactBinding(
                    "lfp",
                    None,
                    deferred_remote_name="lfp.npz",
                    deferred_remote_names=("lfp.pkl",),
                ),
            ],
            lazy_keys=set(),
            hooks=hooks,
        )
        assert eager_plan.summary == {"ok": True}
        assert eager_plan.run_info is not None
        assert eager_plan.remote_payload["deferred_remote_artifacts"] == ["soma_vs.pkl", "lfp.pkl"]
        assert eager_plan.load_plan[0][0] == "input_times"
        assert eager_plan.load_plan[1][0] == "lfp"
        assert sync_calls == [(result_dir, "lfp.pkl")]
        assert eager_plan.artifact_sizes["lfp.pkl"] > 0
        print("artifact result-view eager planning path: OK")

        progress_messages.clear()
        sync_calls.clear()
        lazy_path_calls.clear()
        lazy_plan = plan_result_view(
            result_dir,
            result_factory_fn=_result_factory,
            artifact_bindings=[
                ResultArtifactBinding("soma_vs", result_dir / "soma_vs_local.pkl", deferred_remote_name="soma_vs.pkl"),
                ResultArtifactBinding("lfp", None, deferred_remote_name="lfp.pkl"),
            ],
            lazy_keys={"soma_vs", "lfp"},
            hooks=hooks,
        )
        assert lazy_plan.load_plan == []
        assert lazy_plan.lazy_local_paths["soma_vs"].name == "soma_vs_local.pkl"
        assert lazy_plan.lazy_remote_names["lfp"] == "lfp.pkl"
        assert sync_calls == []
        attach_lazy_artifact_loaders(lazy_plan, hooks=hooks, progress=True)
        assert lazy_path_calls == [("soma_vs", "soma_vs_local.pkl"), ("lfp", "lfp.pkl")]
        assert progress_messages == [
            "lazy-local:soma_vs:soma_vs_local.pkl",
            "lazy-remote:lfp:lfp.pkl",
        ]
        assert lazy_plan.result["soma_vs"][0][0] == "MC0"
        assert sync_calls == []
        assert lazy_plan.result["lfp"] == (["t"], ["v"])
        assert sync_calls == [(result_dir, "lfp.pkl")]
        print("artifact result-view lazy attachment path: OK")


if __name__ == "__main__":
    main()
