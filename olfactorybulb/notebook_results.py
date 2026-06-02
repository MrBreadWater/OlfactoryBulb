"""Concrete olfactory-bulb notebook result-loading adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, MutableMapping

import numpy as np

from neuroinfra.artifacts.loading import ArtifactLoadingHooks, LazyResult as _LazyResult
from neuroinfra.artifacts.loading import load_local_artifact_plan
from neuroinfra.artifacts.result_view import (
    ResultArtifactBinding,
    ResultFieldSpec,
    ResultViewHooks,
    ResultViewSchema,
    attach_lazy_artifact_loaders,
    plan_result_view,
)


class LazyResult(_LazyResult):
    """Result dict that tracks lazy OBGPU soma-trace loads with progress messages."""

    def _ensure_loaded(self, key: str) -> None:
        if key not in self._lazy_loaders:
            return
        progress_write = getattr(self, "_progress_write", None)
        if callable(progress_write):
            progress_write(f"[OBGPU load] Lazy-loading {key}...")
        started = time.perf_counter()
        try:
            super()._ensure_loaded(key)
        finally:
            elapsed_s = time.perf_counter() - started
        if key == "soma_vs":
            soma_path = dict.get(self, "soma_vs_file")
            artifact_sizes = dict.get(self, "artifact_sizes")
            if isinstance(soma_path, Path) and soma_path.exists() and isinstance(artifact_sizes, dict):
                artifact_sizes[soma_path.name] = int(soma_path.stat().st_size)
        if callable(progress_write):
            progress_write(f"[OBGPU load] Loaded {key} in {elapsed_s:.1f}s")


def _apply_loaded_lfp_payload(result: MutableMapping[str, Any], loaded: Any) -> None:
    """Apply one loaded LFP payload into the standard result mapping."""
    lfp_t, lfp = loaded
    result["lfp_t"] = np.asarray(lfp_t, dtype=float)
    result["lfp"] = np.asarray(lfp, dtype=float)


_OBGPU_RESULT_VIEW_SCHEMA = ResultViewSchema(
    fields=(
        ResultFieldSpec("input_times", default_factory=list),
        ResultFieldSpec("soma_vs", default_factory=list, lazy_path_key="soma_vs_file"),
        ResultFieldSpec("soma_spikes", default_factory=dict),
        ResultFieldSpec("voltage_summary", default_factory=dict),
        ResultFieldSpec("gc_output_events", default_factory=list),
        ResultFieldSpec("lfp_t", default_factory=lambda: np.array([])),
        ResultFieldSpec(
            "lfp",
            default_factory=lambda: np.array([]),
            apply_loaded_fn=_apply_loaded_lfp_payload,
        ),
    ),
    result_type=LazyResult,
)


@dataclass(frozen=True)
class NotebookResultHooks:
    """Hooks for constructing the concrete olfactory-bulb result view."""

    find_soma_trace_artifact_fn: Callable[[str | Path], Path | None]
    preferred_soma_trace_artifact_name_fn: Callable[[], str]
    soma_trace_artifact_candidates_fn: Callable[[], tuple[str, ...]]
    result_view_hooks: ResultViewHooks
    artifact_loading_hooks: ArtifactLoadingHooks


def apply_loaded_result_artifact(result: MutableMapping[str, Any], key: str, loaded: Any) -> None:
    """Apply one loaded artifact payload through the standard OBGPU result schema."""
    _OBGPU_RESULT_VIEW_SCHEMA.apply_loaded_artifact(result, key, loaded)


def set_lazy_result_artifact_path(result: MutableMapping[str, Any], key: str, path: Path) -> None:
    """Store one lazy artifact path using the standard OBGPU result schema."""
    _OBGPU_RESULT_VIEW_SCHEMA.set_lazy_artifact_path(result, key, path)


def _make_result_view(
    *,
    result_dir: Path,
    summary: dict[str, Any] | None,
    run_info: dict[str, Any] | None,
    artifact_sizes: dict[str, int],
) -> LazyResult:
    """Build the standard OBGPU notebook result mapping before artifact loads."""
    return _OBGPU_RESULT_VIEW_SCHEMA.create_result(
        result_dir=result_dir,
        summary=summary,
        run_info=run_info,
        artifact_sizes=artifact_sizes,
    )


def load_result(
    hooks: NotebookResultHooks,
    run_or_dir: Any,
    *,
    lazy_soma_vs: bool = False,
    progress: bool = True,
) -> MutableMapping[str, Any]:
    """Load the standard saved outputs for one olfactory-bulb notebook run."""
    result_dir = Path(getattr(run_or_dir, "result_dir", run_or_dir))
    soma_path = hooks.find_soma_trace_artifact_fn(result_dir)
    view_plan = plan_result_view(
        result_dir,
        result_factory_fn=_make_result_view,
        artifact_bindings=[
            ResultArtifactBinding("input_times", result_dir / "input_times.pkl"),
            ResultArtifactBinding(
                "soma_vs",
                soma_path,
                deferred_remote_name=hooks.preferred_soma_trace_artifact_name_fn(),
                deferred_remote_names=hooks.soma_trace_artifact_candidates_fn(),
            ),
            ResultArtifactBinding("gc_output_events", result_dir / "gc_output_events.pkl"),
            ResultArtifactBinding("lfp", result_dir / "lfp.pkl"),
            ResultArtifactBinding("soma_spikes", result_dir / "soma_spikes.npz"),
            ResultArtifactBinding("voltage_summary", result_dir / "voltage_summary.npz"),
        ],
        lazy_keys={"soma_vs"} if lazy_soma_vs else set(),
        hooks=hooks.result_view_hooks,
    )
    result = view_plan.result
    if isinstance(result, LazyResult):
        result._progress_write = hooks.result_view_hooks.progress_write

    load_timings, load_total_seconds = load_local_artifact_plan(
        result,
        view_plan.load_plan,
        hooks=hooks.artifact_loading_hooks,
        progress=progress,
    )

    if lazy_soma_vs:
        attach_lazy_artifact_loaders(
            view_plan,
            hooks=hooks.result_view_hooks,
            progress=progress,
        )

    result["load_timing_seconds"] = load_timings
    result["load_total_seconds"] = load_total_seconds
    if load_timings and progress:
        timing_summary = ", ".join(
            f"{name}={seconds:.2f}s"
            for name, seconds in sorted(load_timings.items(), key=lambda item: item[1], reverse=True)
        )
        hooks.result_view_hooks.progress_write(f"[OBGPU load] Local file timings: {timing_summary}")

    return result
