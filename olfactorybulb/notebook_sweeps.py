"""Concrete olfactory-bulb notebook sweep helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class NotebookSweepHooks:
    """Hooks for constructing concrete olfactory-bulb notebook sweep helpers."""

    sweeps_base: str | Path
    default_results_base: str | Path
    make_timestamp_fn: Callable[[], str]
    safe_name_fn: Callable[[Any], str]
    json_ready_fn: Callable[[Any], Any]
    resolve_git_head_fn: Callable[[], str | None]
    load_result_fn: Callable[..., Any]
    save_sweep_fn: Callable[..., Path]
    load_sweep_fn: Callable[..., dict[str, Any]]
    list_sweeps_fn: Callable[..., list[Path]]
    save_animation_fn: Callable[..., Path]
    save_sweep_animation_stream_fn: Callable[..., Path]
    animate_sweep_plots_fn: Callable[..., dict[str, Path]]
    build_sweep_plot_callable_fn: Callable[[Any], tuple[Any, str]]
    normalize_sweep_plot_spec_fn: Callable[[Any], Any]
    is_deprecated_sweep_animation_spec_fn: Callable[[Any], bool]
    deprecated_plot_names: tuple[str, ...]
    progress_factory_fn: Callable[[int, str], Any | None]
    progress_write_fn: Callable[[str], None]
    run_parameter_sweep_fn: Callable[..., dict[str, Any]]
    run_grid_sweep_fn: Callable[..., dict[str, Any]]


def save_sweep(
    hooks: NotebookSweepHooks,
    sweep: dict[str, Any],
    *,
    name: str | None = None,
    base_dir: str | Path | None = None,
) -> Path:
    """Persist one completed sweep using the standard notebook directory policy."""
    return hooks.save_sweep_fn(
        sweep,
        name=name,
        base_dir=base_dir or hooks.sweeps_base,
        timestamp_factory=hooks.make_timestamp_fn,
        safe_name=hooks.safe_name_fn,
        json_ready=hooks.json_ready_fn,
        resolve_git_head=hooks.resolve_git_head_fn,
    )


def load_sweep(
    hooks: NotebookSweepHooks,
    path: str | Path,
) -> dict[str, Any]:
    """Reload one saved sweep and lazy-load each result through the notebook loader."""
    return hooks.load_sweep_fn(
        path,
        load_result_fn=lambda result_dir: hooks.load_result_fn(result_dir, progress=False),
        safe_name=hooks.safe_name_fn,
    )


def list_sweeps(
    hooks: NotebookSweepHooks,
    *,
    prefix: str | None = None,
    base_dir: str | Path | None = None,
) -> list[Path]:
    """List saved sweeps from oldest to newest under the notebook sweep root."""
    return hooks.list_sweeps_fn(base_dir=base_dir or hooks.sweeps_base, prefix=prefix)


def save_animation(
    hooks: NotebookSweepHooks,
    anim: Any,
    name: str,
    *,
    output_dir: str | Path | None = None,
    sweep: dict[str, Any] | None = None,
    fps: int = 10,
) -> Path:
    """Save one animation using the standard notebook animation-output policy."""
    return hooks.save_animation_fn(
        anim,
        name,
        safe_name=hooks.safe_name_fn,
        output_dir=output_dir,
        sweep=sweep,
        fps=fps,
        default_output_dir_factory=lambda: Path(hooks.default_results_base) / "animations" / hooks.make_timestamp_fn(),
    )


def save_sweep_animation_stream(
    hooks: NotebookSweepHooks,
    sweep: dict[str, Any],
    plot_fn: Any,
    name: str,
    *,
    output_dir: str | Path | None = None,
    figsize: tuple[float, float] = (12.0, 5.0),
    interval: int = 100,
    title_fn: Any = None,
    close_frames: bool = True,
    fps: int = 10,
    workers: int | None = None,
) -> Path:
    """Stream-render one sweep GIF with the standard notebook progress/output policy."""
    progress = hooks.progress_factory_fn(len(sweep["items"]), f"[OBGPU load] Render {name}")
    try:
        return hooks.save_sweep_animation_stream_fn(
            sweep,
            plot_fn,
            name,
            safe_name=hooks.safe_name_fn,
            output_dir=output_dir,
            figsize=figsize,
            title_fn=title_fn,
            close_frames=close_frames,
            fps=fps,
            workers=workers,
            env_var_name="OBGPU_SWEEP_RENDER_WORKERS",
            progress_callback=(lambda current, total: progress.update_to(current)) if progress is not None else None,
            default_output_dir_factory=lambda: Path(hooks.default_results_base) / "animations" / hooks.make_timestamp_fn(),
        )
    finally:
        if progress is not None:
            progress.close()


def animate_sweep_plots(
    hooks: NotebookSweepHooks,
    sweep: dict[str, Any],
    plots: Iterable[Any],
    *,
    close_frames: bool = True,
    stream: bool = True,
    workers: int | None = None,
) -> dict[str, Path]:
    """Render one or more sweep animation artifacts using notebook defaults."""
    for raw_spec in plots:
        spec = hooks.normalize_sweep_plot_spec_fn(raw_spec)
        if hooks.is_deprecated_sweep_animation_spec_fn(spec):
            hooks.progress_write_fn(
                f"[OBGPU load] Skipping deprecated sweep animation plot {spec.name!r}."
            )
    return hooks.animate_sweep_plots_fn(
        sweep,
        plots,
        plot_builder=hooks.build_sweep_plot_callable_fn,
        safe_name=hooks.safe_name_fn,
        deprecated_names=set(hooks.deprecated_plot_names),
        close_frames=close_frames,
        stream=stream,
        workers=workers,
        env_var_name="OBGPU_SWEEP_RENDER_WORKERS",
        default_output_dir_factory=lambda: Path(hooks.default_results_base) / "animations" / hooks.make_timestamp_fn(),
    )


def run_sweep_with_animations(
    hooks: NotebookSweepHooks,
    base_config: dict[str, Any],
    sweep_path: str | list[str] | dict[str, list[Any]],
    values: list[Any] | tuple[Any, ...] | None = None,
    *,
    plots: list[Any] | None = None,
    use_grid: bool = False,
    close_frames: bool = True,
    workers: int | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Run one sweep, then optionally emit one or more animation artifacts."""
    if use_grid:
        if not isinstance(sweep_path, dict):
            raise TypeError("Grid sweeps require sweep_path to be a dict of {path: values}")
        sweep = hooks.run_grid_sweep_fn(base_config, sweep_path)
    else:
        sweep = hooks.run_parameter_sweep_fn(base_config, sweep_path, values)

    artifacts: dict[str, Path] = {}
    if plots:
        artifacts = animate_sweep_plots(
            hooks,
            sweep,
            plots,
            close_frames=close_frames,
            workers=workers,
        )
    return sweep, artifacts
