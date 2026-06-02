"""Reusable local artifact-loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, MutableMapping


class LazyResult(dict):
    """Result dict that loads selected heavy payloads on first access."""

    def __init__(self, *args: Any, lazy_loaders: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._lazy_loaders = dict(lazy_loaders or {})

    def _ensure_loaded(self, key: str) -> None:
        if key not in self._lazy_loaders:
            return
        loader = self._lazy_loaders[key]
        value = loader()
        dict.__setitem__(self, key, value)
        self._lazy_loaders.pop(key, None)

    def __getitem__(self, key: str) -> Any:
        self._ensure_loaded(key)
        return dict.__getitem__(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._lazy_loaders:
            self._ensure_loaded(key)
        return dict.get(self, key, default)

    def __contains__(self, key: object) -> bool:
        return dict.__contains__(self, key) or key in self._lazy_loaders


@dataclass(frozen=True)
class ArtifactLoadingHooks:
    """Callbacks injected by the notebook-facing caller for artifact loading."""

    load_pickle_fn: Callable[[str | Path], Any]
    apply_loaded_fn: Callable[[MutableMapping[str, Any], str, Any], None]
    progress_factory_fn: Callable[[int, str], Any | None]
    progress_write: Callable[[str], None]
    format_bytes_fn: Callable[[int | float], str]
    render_progress_bar_fn: Callable[[int, int], str]
    perf_counter_fn: Callable[[], float] = time.perf_counter


def load_local_artifact_plan(
    result: MutableMapping[str, Any],
    load_plan: list[tuple[str, Path]],
    *,
    hooks: ArtifactLoadingHooks,
    progress: bool = True,
    progress_desc: str = "[OBGPU load] Load result files",
) -> tuple[dict[str, float], float]:
    """Load one ordered local artifact plan into an existing result mapping."""
    load_timings: dict[str, float] = {}
    load_started = hooks.perf_counter_fn()
    total_bytes = sum(path.stat().st_size for _key, path in load_plan)
    loaded_bytes = 0
    progress_bar = hooks.progress_factory_fn(total_bytes, progress_desc) if progress and load_plan else None

    if load_plan and progress:
        hooks.progress_write(
            f"[OBGPU load] Loading {len(load_plan)} local result files ({hooks.format_bytes_fn(total_bytes)})..."
        )

    try:
        for index, (key, path) in enumerate(load_plan, start=1):
            file_size = path.stat().st_size
            if progress:
                hooks.progress_write(
                    f"[OBGPU load] Loading {index}/{len(load_plan)}: {path.name} ({hooks.format_bytes_fn(file_size)})"
                )
            started = hooks.perf_counter_fn()
            loaded = hooks.load_pickle_fn(path)
            elapsed_s = hooks.perf_counter_fn() - started
            load_timings[path.name] = round(elapsed_s, 3)
            hooks.apply_loaded_fn(result, key, loaded)
            loaded_bytes += file_size
            if progress_bar is not None:
                progress_bar.update_to(loaded_bytes)
            if progress:
                hooks.progress_write(
                    f"[OBGPU load] {hooks.render_progress_bar_fn(loaded_bytes, total_bytes)} "
                    f"{hooks.format_bytes_fn(loaded_bytes)} / {hooks.format_bytes_fn(total_bytes)} "
                    f"(loaded {path.name} in {elapsed_s:.1f}s)",
                )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return load_timings, round(hooks.perf_counter_fn() - load_started, 3)
