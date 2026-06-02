"""Reusable result-view planning and lazy-artifact wiring helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, MutableMapping

from .loading import LazyResult


@dataclass(frozen=True)
class ResultArtifactBinding:
    """One result artifact that may be loaded eagerly or lazily."""

    key: str
    local_path: Path | None
    deferred_remote_name: str | None = None
    deferred_remote_names: tuple[str, ...] = ()


@dataclass
class ResultViewPlan:
    """Planned eager and lazy artifact work for one result directory."""

    result: MutableMapping[str, Any]
    result_dir: Path
    summary: dict[str, Any] | None
    run_info: dict[str, Any] | None
    remote_payload: dict[str, Any]
    artifact_sizes: dict[str, int]
    load_plan: list[tuple[str, Path]]
    lazy_local_paths: dict[str, Path]
    lazy_remote_names: dict[str, str]


@dataclass(frozen=True)
class ResultViewHooks:
    """Callbacks injected by the notebook-facing caller for result-view assembly."""

    read_json_if_present_fn: Callable[[str | Path], dict[str, Any] | None]
    standard_result_artifact_sizes_fn: Callable[[str | Path], dict[str, int]]
    local_sync_artifact_is_usable_fn: Callable[[str | Path], bool]
    sync_deferred_artifact_fn: Callable[..., Path]
    load_pickle_fn: Callable[[str | Path], Any]
    set_lazy_artifact_path_fn: Callable[[MutableMapping[str, Any], str, Path], None]
    local_lazy_notice_fn: Callable[[str, Path], str | None]
    remote_lazy_notice_fn: Callable[[str, Path], str | None]
    progress_write: Callable[[str], None]


def plan_result_view(
    result_dir: str | Path,
    *,
    result_factory_fn: Callable[..., MutableMapping[str, Any]],
    artifact_bindings: list[ResultArtifactBinding],
    lazy_keys: set[str] | None,
    hooks: ResultViewHooks,
) -> ResultViewPlan:
    """Read result metadata and plan eager plus deferred artifact loading."""
    result_dir = Path(result_dir)
    summary = hooks.read_json_if_present_fn(result_dir / "summary.json")
    run_info = hooks.read_json_if_present_fn(result_dir / "run_info.json")
    remote_payload_value = (run_info or {}).get("remote") if isinstance(run_info, dict) else {}
    remote_payload = remote_payload_value if isinstance(remote_payload_value, dict) else {}
    deferred_remote_artifacts = set(remote_payload.get("deferred_remote_artifacts") or [])
    artifact_sizes = hooks.standard_result_artifact_sizes_fn(result_dir)
    result = result_factory_fn(
        result_dir=result_dir,
        summary=summary,
        run_info=run_info,
        artifact_sizes=artifact_sizes,
    )

    lazy_keys = set(lazy_keys or ())
    load_plan: list[tuple[str, Path]] = []
    lazy_local_paths: dict[str, Path] = {}
    lazy_remote_names: dict[str, str] = {}

    for binding in artifact_bindings:
        local_path = Path(binding.local_path) if binding.local_path is not None else None
        remote_candidates = tuple(str(name) for name in binding.deferred_remote_names if name)
        if binding.deferred_remote_name:
            remote_candidates = (str(binding.deferred_remote_name), *remote_candidates)
        remote_name = next(
            (name for name in remote_candidates if name in deferred_remote_artifacts),
            None,
        )
        local_usable = (
            local_path is not None
            and hooks.local_sync_artifact_is_usable_fn(local_path)
        )

        if not local_usable and remote_name is not None and binding.key not in lazy_keys:
            local_path = hooks.sync_deferred_artifact_fn(
                result_dir,
                run_info=run_info,
                filename=remote_name,
            )
            artifact_sizes[local_path.name] = int(local_path.stat().st_size)
            local_usable = hooks.local_sync_artifact_is_usable_fn(local_path)

        if local_usable and local_path is not None and binding.key not in lazy_keys:
            load_plan.append((binding.key, local_path))
            continue

        if binding.key not in lazy_keys:
            continue

        if local_usable and local_path is not None:
            lazy_local_paths[binding.key] = local_path
        elif remote_name is not None:
            lazy_remote_names[binding.key] = remote_name

    return ResultViewPlan(
        result=result,
        result_dir=result_dir,
        summary=summary,
        run_info=run_info,
        remote_payload=remote_payload,
        artifact_sizes=artifact_sizes,
        load_plan=load_plan,
        lazy_local_paths=lazy_local_paths,
        lazy_remote_names=lazy_remote_names,
    )


def attach_lazy_artifact_loaders(
    plan: ResultViewPlan,
    *,
    hooks: ResultViewHooks,
    progress: bool = True,
) -> None:
    """Attach local and deferred lazy loaders onto one planned result mapping."""
    result = plan.result
    if not isinstance(result, LazyResult):
        raise TypeError("attach_lazy_artifact_loaders requires a LazyResult-compatible result mapping")

    for key, local_path in plan.lazy_local_paths.items():
        hooks.set_lazy_artifact_path_fn(result, key, local_path)
        result._lazy_loaders[key] = lambda path=local_path: hooks.load_pickle_fn(path)
        if progress:
            message = hooks.local_lazy_notice_fn(key, local_path)
            if message:
                hooks.progress_write(message)

    for key, remote_name in plan.lazy_remote_names.items():
        local_path = plan.result_dir / remote_name
        hooks.set_lazy_artifact_path_fn(result, key, local_path)
        result._lazy_loaders[key] = (
            lambda path=local_path, info=plan.run_info, directory=plan.result_dir, filename=remote_name:
            hooks.load_pickle_fn(
                hooks.sync_deferred_artifact_fn(directory, run_info=info, filename=filename)
            )
        )
        if progress:
            message = hooks.remote_lazy_notice_fn(key, local_path)
            if message:
                hooks.progress_write(message)
