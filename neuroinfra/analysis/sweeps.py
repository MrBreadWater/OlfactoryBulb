"""Reusable sweep planning, persistence, and animation helpers."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
import json
import multiprocessing as _multiprocessing
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from matplotlib import animation
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SweepPlotSpec:
    """One sweep-animation artifact generated from an existing sweep."""

    name: str
    plot: str | Any
    plot_kwargs: dict[str, Any] | None = None
    filename: str | None = None
    figsize: tuple[float, float] = (12.0, 5.0)
    interval: int = 100
    fps: int = 10
    title_fn: Any = None


def make_sweep_plot_spec(
    plot: str | Any,
    *,
    name: str | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    filename: str | None = None,
    figsize: tuple[float, float] = (12.0, 5.0),
    interval: int = 100,
    fps: int = 10,
    title_fn: Any = None,
) -> SweepPlotSpec:
    """Build a sweep-plot spec from a plot name or custom callable."""
    if name is None:
        if isinstance(plot, str):
            name = plot
        else:
            name = getattr(plot, "__name__", "custom_plot")
    return SweepPlotSpec(
        name=str(name),
        plot=plot,
        plot_kwargs=dict(plot_kwargs or {}),
        filename=filename,
        figsize=figsize,
        interval=int(interval),
        fps=int(fps),
        title_fn=title_fn,
    )


def normalize_sweep_plot_spec(plot_spec: SweepPlotSpec | str | Any | dict[str, Any]) -> SweepPlotSpec:
    """Accept ergonomic plot-spec forms and normalize them."""
    if isinstance(plot_spec, SweepPlotSpec):
        return make_sweep_plot_spec(
            plot_spec.plot,
            name=plot_spec.name,
            plot_kwargs=plot_spec.plot_kwargs,
            filename=plot_spec.filename,
            figsize=plot_spec.figsize,
            interval=plot_spec.interval,
            fps=plot_spec.fps,
            title_fn=plot_spec.title_fn,
        )
    if isinstance(plot_spec, str) or callable(plot_spec):
        return make_sweep_plot_spec(plot_spec)
    if isinstance(plot_spec, dict):
        plot = plot_spec.get("plot")
        if plot is None:
            if "plot_fn" in plot_spec:
                plot = plot_spec["plot_fn"]
            elif "name" in plot_spec:
                plot = plot_spec["name"]
            else:
                raise ValueError("Plot-spec dict must include 'plot', 'plot_fn', or 'name'")
        return make_sweep_plot_spec(
            plot,
            name=plot_spec.get("name"),
            plot_kwargs=plot_spec.get("plot_kwargs"),
            filename=plot_spec.get("filename"),
            figsize=tuple(plot_spec.get("figsize", (12.0, 5.0))),
            interval=int(plot_spec.get("interval", 100)),
            fps=int(plot_spec.get("fps", 10)),
            title_fn=plot_spec.get("title_fn"),
        )
    raise TypeError(f"Unsupported sweep-plot spec type {type(plot_spec)!r}")


def extract_figure_from_plot_result(plot_result: Any) -> Any:
    """Best-effort extraction of a Matplotlib figure from a plot return value."""
    if plot_result is None:
        return plt.gcf()
    if hasattr(plot_result, "savefig"):
        return plot_result
    if hasattr(plot_result, "figure"):
        return plot_result.figure
    if isinstance(plot_result, tuple):
        for item in plot_result:
            if hasattr(item, "savefig"):
                return item
            if hasattr(item, "figure"):
                return item.figure
    return plt.gcf()


def build_sweep_plot_callable(
    spec: SweepPlotSpec,
    *,
    plot_resolver: Callable[[str], Any],
) -> Any:
    """Resolve one plot spec into a figure-producing callable."""
    plot_kwargs = dict(spec.plot_kwargs or {})
    if isinstance(spec.plot, str):
        plot_fn = plot_resolver(spec.plot)
    else:
        plot_fn = spec.plot

    def _wrapped(result: dict[str, Any]) -> Any:
        return extract_figure_from_plot_result(plot_fn(result, **plot_kwargs))

    return _wrapped


def is_deprecated_sweep_animation_spec(
    spec: SweepPlotSpec,
    *,
    deprecated_names: Iterable[str],
) -> bool:
    """Return whether a sweep spec names a retired animation plot."""
    names = {
        str(spec.name or ""),
        str(spec.filename or ""),
    }
    if isinstance(spec.plot, str):
        names.add(spec.plot)
    deprecated = {str(name) for name in deprecated_names}
    return any(name in deprecated for name in names)


def format_sweep_value(value: Any) -> str:
    """Format a sweep value compactly for figure titles."""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def format_sweep_value_label(sweep: dict[str, Any], value: Any) -> str:
    """Format one sweep-path/value label for animation titles."""
    path = sweep.get("path", "")
    if isinstance(path, dict):
        if isinstance(value, dict):
            return ", ".join(f"{key}={format_sweep_value(val)}" for key, val in value.items())
        return ", ".join(f"{key}={format_sweep_value(path_value)}" for key, path_value in path.items())
    if isinstance(path, (list, tuple)):
        if isinstance(value, (list, tuple)):
            pairs = zip(path, value)
            return ", ".join(f"{key}={format_sweep_value(val)}" for key, val in pairs)
        return ", ".join(str(key) for key in path)
    return f"{path} = {format_sweep_value(value)}"


def format_sweep_progress_label(frame_index: int, total_frames: int, *, width: int = 12) -> str:
    """Format one compact sweep-progress label without ASCII bar glyphs."""
    del width
    total = max(int(total_frames), 1)
    current = max(1, min(int(frame_index) + 1, total))
    percent = (current / total) * 100.0
    return f"{current}/{total} ({percent:.1f}%)"


def format_sweep_frame_title(sweep: dict[str, Any], value: Any, frame_index: int, total_frames: int) -> str:
    """Build one default animation title with value and sweep progress."""
    return (
        f"{format_sweep_value_label(sweep, value)}"
        f" | {format_sweep_progress_label(frame_index, total_frames)}"
    )


def describe_unavailable_sweep_item(item: dict[str, Any]) -> str:
    """Return a compact reason for a missing/unrenderable sweep frame."""
    load_error = item.get("load_error")
    if load_error:
        return f"Load failed: {load_error}"
    status = item.get("status")
    if isinstance(status, dict):
        for key in ("error", "reason", "state"):
            value = status.get(key)
            if value not in (None, ""):
                return f"{key}: {value}"
        if status.get("ok") is False:
            return "Run did not produce a usable local result payload."
    return "No local result payload was recovered for this sweep item."


def make_sweep_placeholder_figure(
    sweep: dict[str, Any],
    item: dict[str, Any],
    frame_index: int,
    total_frames: int,
    *,
    reason: str,
    figsize: tuple[float, float],
    title_formatter: Callable[[dict[str, Any], Any, int, int], str] = format_sweep_frame_title,
) -> Any:
    """Render an explicit placeholder frame instead of aborting partial sweeps."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    value = item.get("value")
    label = str(item.get("label") or f"item_{frame_index:03d}")
    title = title_formatter(sweep, value, frame_index, total_frames)
    ax.text(
        0.5,
        0.62,
        "Sweep item unavailable",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.46,
        f"{label}\n{title}",
        ha="center",
        va="center",
        fontsize=11,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.24,
        str(reason),
        ha="center",
        va="center",
        fontsize=10,
        wrap=True,
        transform=ax.transAxes,
    )
    fig.tight_layout()
    return fig


def fig_to_rgb_array(fig: Any) -> np.ndarray:
    """Render a matplotlib figure to an H x W x 3 uint8 numpy array."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    return np.ascontiguousarray(rgba[..., :3])


def render_sweep_frame(
    sweep: dict[str, Any],
    item: dict[str, Any],
    frame_index: int,
    total_frames: int,
    plot_fn: Any,
    *,
    figsize: tuple[float, float],
    title_fn: Any = None,
    close_frames: bool = True,
    title_formatter: Callable[[dict[str, Any], Any, int, int], str] = format_sweep_frame_title,
) -> tuple[np.ndarray, str]:
    """Render one sweep item to a frame array and title."""
    result = item.get("result") if isinstance(item, dict) else None
    value = item.get("value") if isinstance(item, dict) else None
    if title_fn is not None:
        try:
            title = title_fn(
                value,
                frame_index=frame_index,
                total_frames=total_frames,
                sweep=sweep,
            )
        except TypeError:
            title = title_fn(value)
    else:
        title = title_formatter(sweep, value, frame_index, total_frames)

    if result is None:
        fig = make_sweep_placeholder_figure(
            sweep,
            item,
            frame_index,
            total_frames,
            reason=describe_unavailable_sweep_item(item),
            figsize=figsize,
            title_formatter=title_formatter,
        )
    else:
        before_figs = set(plt.get_fignums())
        try:
            returned = plot_fn(result)
            fig = extract_figure_from_plot_result(returned)
        except Exception as exc:
            if close_frames:
                for fignum in set(plt.get_fignums()) - before_figs:
                    plt.close(fignum)
            fig = make_sweep_placeholder_figure(
                sweep,
                item,
                frame_index,
                total_frames,
                reason=f"Plot failed: {type(exc).__name__}: {exc}",
                figsize=figsize,
                title_formatter=title_formatter,
            )

    frame_rgb = fig_to_rgb_array(fig)

    if close_frames:
        plt.close(fig)

    return frame_rgb, str(title)


def resolve_sweep_item_result_dir(item: Mapping[str, Any] | Any) -> Path | None:
    """Resolve the concrete result directory for one sweep item if available."""
    if not isinstance(item, Mapping):
        return None
    run = item.get("run")
    result = item.get("result")
    if run is not None and getattr(run, "result_dir", None) is not None:
        return Path(run.result_dir)
    if isinstance(result, Mapping) and result.get("result_dir") is not None:
        return Path(result["result_dir"])
    return None


def write_sweep_info(
    sweep: Mapping[str, Any] | dict[str, Any],
    *,
    sweep_dir: str | Path,
    timestamp: str,
    json_ready: Callable[[Any], Any],
    resolve_git_head: Callable[[], str | None] | None = None,
    result_dir_resolver: Callable[[Mapping[str, Any] | Any], Path | None] = resolve_sweep_item_result_dir,
) -> Path:
    """Persist sweep metadata in a compact, reloadable directory layout."""
    sweep_dir = Path(sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "animations").mkdir(exist_ok=True)
    (sweep_dir / "figures").mkdir(exist_ok=True)
    (sweep_dir / "runs").mkdir(exist_ok=True)

    git_ref = None
    if resolve_git_head is not None:
        try:
            git_ref = resolve_git_head()
        except Exception:
            git_ref = None

    run_dirs: list[str | None] = []
    item_statuses: list[Any] = []
    item_labels: list[str] = []
    for item in sweep.get("items", []):
        result_dir = result_dir_resolver(item)
        run_dirs.append(str(result_dir) if result_dir is not None else None)
        if isinstance(item, Mapping):
            item_statuses.append(json_ready(item.get("status", {})))
            item_labels.append(str(item.get("label", "")))

    sweep_info: dict[str, Any] = {
        "path": sweep.get("path"),
        "values": [json_ready(v) for v in sweep.get("values", [])],
        "paramset": sweep.get("paramset"),
        "timestamp": timestamp,
        "git_ref": git_ref,
        "run_dirs": run_dirs,
        "n_items": len(sweep.get("items", [])),
    }
    if item_statuses:
        sweep_info["item_statuses"] = item_statuses
    if item_labels:
        sweep_info["item_labels"] = item_labels
    for key in (
        "partial",
        "failed_labels",
        "failed_without_result",
        "recovered_failed_labels",
        "missing_labels",
        "load_errors",
    ):
        if key in sweep:
            sweep_info[key] = json_ready(sweep[key])
    if sweep.get("grid") is not None:
        sweep_info["grid"] = json_ready(sweep.get("grid"))

    (sweep_dir / "sweep_info.json").write_text(json.dumps(sweep_info, indent=2, sort_keys=True))
    if isinstance(sweep, dict):
        sweep["sweep_dir"] = sweep_dir
        sweep["sweep_info"] = sweep_info
    return sweep_dir


def save_sweep(
    sweep: Mapping[str, Any] | dict[str, Any],
    *,
    base_dir: str | Path,
    timestamp_factory: Callable[[], str],
    safe_name: Callable[[Any], str],
    json_ready: Callable[[Any], Any],
    name: str | None = None,
    resolve_git_head: Callable[[], str | None] | None = None,
    result_dir_resolver: Callable[[Mapping[str, Any] | Any], Path | None] = resolve_sweep_item_result_dir,
) -> Path:
    """Persist a completed sweep together with stable per-slot run pointers."""
    base_dir = Path(base_dir)
    timestamp = str(timestamp_factory())
    path_label = sweep.get("path", "sweep")
    if isinstance(path_label, dict):
        path_label = "_".join(str(key) for key in path_label.keys())
    auto_name = safe_name(f"{path_label}_{timestamp}")
    sweep_dir = base_dir / str(name or auto_name)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "animations").mkdir(exist_ok=True)
    (sweep_dir / "figures").mkdir(exist_ok=True)
    runs_dir = sweep_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    for index, item in enumerate(sweep.get("items", [])):
        value = item.get("value") if isinstance(item, Mapping) else None
        result_dir = result_dir_resolver(item)
        value_text = safe_name(str(value)) if value is not None else str(index)
        slot = runs_dir / f"{index:02d}_{value_text}"
        slot.mkdir(exist_ok=True)
        if result_dir is None:
            continue
        (slot / "result_dir.txt").write_text(str(result_dir))
        run_info_path = result_dir / "run_info.json"
        if run_info_path.exists():
            import shutil as _shutil

            _shutil.copy2(run_info_path, slot / "run_info.json")

    return write_sweep_info(
        sweep,
        sweep_dir=sweep_dir,
        timestamp=timestamp,
        json_ready=json_ready,
        resolve_git_head=resolve_git_head,
        result_dir_resolver=result_dir_resolver,
    )


def load_sweep(
    path: str | Path,
    *,
    load_result_fn: Callable[[Path], Any],
    safe_name: Callable[[Any], str],
) -> dict[str, Any]:
    """Reconstruct a saved sweep using a caller-provided result loader."""
    sweep_dir = Path(path)
    info_path = sweep_dir / "sweep_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"No sweep_info.json found in {sweep_dir}")

    info = json.loads(info_path.read_text())
    items: list[dict[str, Any]] = []
    runs_dir = sweep_dir / "runs"
    values = list(info.get("values", []))
    run_dirs = list(info.get("run_dirs", []))
    item_statuses = list(info.get("item_statuses", []))
    item_labels = list(info.get("item_labels", []))
    for index, value in enumerate(values):
        run_dir_str = run_dirs[index] if index < len(run_dirs) else None
        status = item_statuses[index] if index < len(item_statuses) and isinstance(item_statuses[index], dict) else {}
        label = (
            str(item_labels[index])
            if index < len(item_labels) and item_labels[index] not in (None, "")
            else None
        )
        result = None
        load_error = None
        if run_dir_str is not None:
            run_dir = Path(run_dir_str)
            try:
                if run_dir.exists():
                    result = load_result_fn(run_dir)
                else:
                    slot = runs_dir / f"{index:02d}_{safe_name(str(value))}"
                    pointer_path = slot / "result_dir.txt"
                    if pointer_path.exists():
                        alt = Path(pointer_path.read_text().strip())
                        if alt.exists():
                            result = load_result_fn(alt)
            except Exception as exc:
                load_error = str(exc)
        item = {"value": value, "config": None, "run": None, "result": result, "status": status}
        if label is not None:
            item["label"] = label
        if load_error is not None:
            item["load_error"] = load_error
        items.append(item)

    inferred_missing_labels = [
        str(item.get("label") or index)
        for index, item in enumerate(items)
        if item.get("result") is None
    ]
    missing_labels = list(info.get("missing_labels", [])) or inferred_missing_labels
    item_load_errors = {
        str(item.get("label") or index): item["load_error"]
        for index, item in enumerate(items)
        if isinstance(item, dict) and item.get("load_error")
    }
    saved_load_errors = dict(info.get("load_errors", {})) if isinstance(info.get("load_errors"), dict) else {}
    load_errors = {**saved_load_errors, **item_load_errors}

    return {
        "path": info.get("path"),
        "values": values,
        "items": items,
        "sweep_dir": sweep_dir,
        "sweep_info": info,
        "paramset": info.get("paramset"),
        "grid": info.get("grid"),
        "partial": bool(info.get("partial", False) or missing_labels or load_errors),
        "failed_labels": list(info.get("failed_labels", [])),
        "failed_without_result": list(info.get("failed_without_result", [])),
        "recovered_failed_labels": list(info.get("recovered_failed_labels", [])),
        "missing_labels": missing_labels,
        "load_errors": load_errors,
    }


def list_sweeps(
    *,
    base_dir: str | Path,
    prefix: str | None = None,
) -> list[Path]:
    """List saved sweep directories from oldest to newest."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []
    return [
        path
        for path in sorted(base_dir.iterdir())
        if path.is_dir()
        and (path / "sweep_info.json").exists()
        and (prefix is None or path.name.startswith(prefix))
    ]


def compose_sweep_display_frame(
    frame_rgb: np.ndarray,
    title: str,
    *,
    figsize: tuple[float, float],
    frame_index: int | None = None,
    total_frames: int | None = None,
) -> np.ndarray:
    """Compose one frame with title and a simple embedded progress bar."""
    display_fig = plt.figure(figsize=figsize)
    header_ax = display_fig.add_axes([0.045, 0.905, 0.91, 0.085])
    image_ax = display_fig.add_axes([0.0, 0.0, 1.0, 0.895])
    header_ax.axis("off")
    image_ax.axis("off")
    image_ax.imshow(frame_rgb)

    progress_label = ""
    fraction = 0.0
    if frame_index is not None and total_frames:
        total = max(int(total_frames), 1)
        current = max(1, min(int(frame_index) + 1, total))
        fraction = current / total
        progress_label = f"{current}/{total} ({fraction * 100.0:.1f}%)"

    header_text = str(title)
    if progress_label and progress_label not in header_text:
        header_text = f"{header_text} | {progress_label}"
    header_ax.text(
        0.0,
        0.72,
        header_text,
        ha="left",
        va="center",
        fontsize=10,
        fontweight="semibold",
        color="#111111",
        transform=header_ax.transAxes,
    )
    if progress_label:
        header_ax.add_patch(
            Rectangle(
                (0.0, 0.12),
                1.0,
                0.22,
                transform=header_ax.transAxes,
                facecolor="#e6e8eb",
                edgecolor="#9aa1a9",
                linewidth=0.6,
            )
        )
        header_ax.add_patch(
            Rectangle(
                (0.0, 0.12),
                fraction,
                0.22,
                transform=header_ax.transAxes,
                facecolor="#1f77b4",
                edgecolor="none",
            )
        )
    try:
        return fig_to_rgb_array(display_fig)
    finally:
        plt.close(display_fig)


def iter_sweep_animation_frames(
    sweep: Mapping[str, Any],
    plot_fn: Any,
    *,
    figsize: tuple[float, float],
    title_fn: Any = None,
    close_frames: bool = True,
) -> Iterator[tuple[np.ndarray, str]]:
    """Yield raw frame arrays and titles for one sweep animation."""
    total_frames = len(sweep["items"])
    for frame_index, item in enumerate(sweep["items"]):
        yield render_sweep_frame(
            dict(sweep),
            item,
            frame_index,
            total_frames,
            plot_fn,
            figsize=figsize,
            title_fn=title_fn,
            close_frames=close_frames,
        )


_SWEEP_ANIMATION_WORKER_STATE: dict[str, Any] = {}


def default_sweep_animation_worker_count(
    frame_count: int,
    *,
    env_var_name: str = "NEUROINFRA_SWEEP_RENDER_WORKERS",
) -> int:
    """Choose a safe default worker count for CPU-bound frame rendering."""
    if frame_count < 4:
        return 1
    raw = os.environ.get(env_var_name)
    if raw not in (None, ""):
        try:
            requested = int(raw)
        except ValueError:
            requested = 1
        return max(1, min(frame_count, requested))
    cpu_count = os.cpu_count() or 1
    return max(1, min(frame_count, cpu_count))


def _init_sweep_animation_worker(
    sweep: Mapping[str, Any],
    plot_fn: Any,
    figsize: tuple[float, float],
    title_fn: Any,
    close_frames: bool,
) -> None:
    """Initialise per-process worker state for parallel frame rendering."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        pass
    _SWEEP_ANIMATION_WORKER_STATE.clear()
    _SWEEP_ANIMATION_WORKER_STATE.update(
        {
            "sweep": dict(sweep),
            "plot_fn": plot_fn,
            "figsize": figsize,
            "title_fn": title_fn,
            "close_frames": close_frames,
        }
    )


def _render_sweep_animation_worker_frame(frame_index: int) -> tuple[int, np.ndarray]:
    """Render one composed sweep frame in a worker process."""
    state = _SWEEP_ANIMATION_WORKER_STATE
    sweep = state["sweep"]
    frame_rgb, title = render_sweep_frame(
        sweep,
        sweep["items"][frame_index],
        frame_index,
        len(sweep["items"]),
        state["plot_fn"],
        figsize=state["figsize"],
        title_fn=state["title_fn"],
        close_frames=state["close_frames"],
    )
    return (
        frame_index,
        compose_sweep_display_frame(
            np.asarray(frame_rgb, dtype=np.uint8),
            title,
            figsize=state["figsize"],
            frame_index=frame_index,
            total_frames=len(sweep["items"]),
        ),
    )


def iter_parallel_sweep_display_frames(
    sweep: Mapping[str, Any],
    plot_fn: Any,
    *,
    figsize: tuple[float, float],
    title_fn: Any = None,
    close_frames: bool = True,
    workers: int | None = None,
    env_var_name: str = "NEUROINFRA_SWEEP_RENDER_WORKERS",
) -> Iterator[np.ndarray]:
    """Yield composed sweep display frames, rendering in parallel when possible."""
    total_frames = len(sweep["items"])
    worker_count = (
        default_sweep_animation_worker_count(total_frames, env_var_name=env_var_name)
        if workers is None
        else max(1, min(total_frames, int(workers)))
    )
    if worker_count <= 1:
        for frame_index, (frame_rgb, title) in enumerate(
            iter_sweep_animation_frames(
                sweep,
                plot_fn,
                figsize=figsize,
                title_fn=title_fn,
                close_frames=close_frames,
            )
        ):
            yield compose_sweep_display_frame(
                np.asarray(frame_rgb, dtype=np.uint8),
                title,
                figsize=figsize,
                frame_index=frame_index,
                total_frames=total_frames,
            )
        return

    try:
        context = _multiprocessing.get_context("fork")
    except ValueError:
        for frame_index, (frame_rgb, title) in enumerate(
            iter_sweep_animation_frames(
                sweep,
                plot_fn,
                figsize=figsize,
                title_fn=title_fn,
                close_frames=close_frames,
            )
        ):
            yield compose_sweep_display_frame(
                np.asarray(frame_rgb, dtype=np.uint8),
                title,
                figsize=figsize,
                frame_index=frame_index,
                total_frames=total_frames,
            )
        return

    next_submit = 0
    next_yield = 0
    completed: dict[int, np.ndarray] = {}
    pending: set[concurrent.futures.Future] = set()
    max_pending = max(worker_count * 2, worker_count)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_init_sweep_animation_worker,
        initargs=(sweep, plot_fn, figsize, title_fn, close_frames),
    ) as executor:
        while next_submit < total_frames and len(pending) < max_pending:
            pending.add(executor.submit(_render_sweep_animation_worker_frame, next_submit))
            next_submit += 1

        while pending:
            done, pending = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                frame_index, frame = future.result()
                completed[frame_index] = frame

            while next_submit < total_frames and len(pending) < max_pending:
                pending.add(executor.submit(_render_sweep_animation_worker_frame, next_submit))
                next_submit += 1

            while next_yield in completed:
                yield completed.pop(next_yield)
                next_yield += 1


def animate_sweep(
    sweep: Mapping[str, Any],
    plot_fn: Any,
    *,
    figsize: tuple[float, float] = (12.0, 5.0),
    interval: int = 100,
    title_fn: Any = None,
    close_frames: bool = True,
) -> animation.FuncAnimation:
    """Render a full in-memory animation for one completed sweep."""
    frames_rgb: list[np.ndarray] = []
    total_frames = len(sweep["items"])
    for frame_index, (frame_rgb, title) in enumerate(
        iter_sweep_animation_frames(
            sweep,
            plot_fn,
            figsize=figsize,
            title_fn=title_fn,
            close_frames=close_frames,
        )
    ):
        frames_rgb.append(
            compose_sweep_display_frame(
                np.asarray(frame_rgb, dtype=np.uint8),
                title,
                figsize=figsize,
                frame_index=frame_index,
                total_frames=total_frames,
            )
        )
    if not frames_rgb:
        raise ValueError("sweep has no items to animate")

    display_fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    display_fig.tight_layout(pad=0)
    image = ax.imshow(frames_rgb[0])

    def _update(index: int) -> list[Any]:
        image.set_data(frames_rgb[index])
        return [image]

    anim = animation.FuncAnimation(
        display_fig,
        _update,
        frames=len(frames_rgb),
        interval=interval,
        repeat=True,
    )
    plt.close(display_fig)
    return anim


def save_animation(
    anim: animation.FuncAnimation,
    name: str,
    *,
    safe_name: Callable[[Any], str],
    output_dir: str | Path | None = None,
    sweep: Mapping[str, Any] | None = None,
    fps: int = 10,
    default_output_dir_factory: Callable[[], Path] | None = None,
) -> Path:
    """Save an already-built animation as a GIF and return the written path."""
    if output_dir is None and sweep is not None and "sweep_dir" in sweep:
        output_dir = Path(sweep["sweep_dir"]) / "animations"
    elif output_dir is None and default_output_dir_factory is not None:
        output_dir = default_output_dir_factory()
    if output_dir is None:
        raise ValueError("output_dir or default_output_dir_factory is required when sweep_dir is absent")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / f"{safe_name(name)}.gif"
    writer = animation.PillowWriter(fps=max(1, int(fps)))
    anim.save(str(gif_path), writer=writer)
    return gif_path


def save_sweep_animation_stream(
    sweep: Mapping[str, Any],
    plot_fn: Any,
    name: str,
    *,
    safe_name: Callable[[Any], str],
    output_dir: str | Path | None = None,
    figsize: tuple[float, float] = (12.0, 5.0),
    title_fn: Any = None,
    close_frames: bool = True,
    fps: int = 10,
    workers: int | None = None,
    env_var_name: str = "NEUROINFRA_SWEEP_RENDER_WORKERS",
    progress_callback: Callable[[int, int], None] | None = None,
    default_output_dir_factory: Callable[[], Path] | None = None,
) -> Path:
    """Stream-render a sweep GIF without retaining all frames in memory."""
    if not sweep.get("items"):
        raise ValueError("sweep has no items to animate")
    if output_dir is None and "sweep_dir" in sweep:
        output_dir = Path(sweep["sweep_dir"]) / "animations"
    elif output_dir is None and default_output_dir_factory is not None:
        output_dir = default_output_dir_factory()
    if output_dir is None:
        raise ValueError("output_dir or default_output_dir_factory is required when sweep_dir is absent")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / f"{safe_name(name)}.gif"

    frame_count = 0
    total_frames = len(sweep["items"])
    duration_s = 1.0 / max(1, int(fps))
    try:
        import imageio.v2 as imageio

        with imageio.get_writer(
            gif_path,
            mode="I",
            fps=max(1, int(fps)),
            loop=0,
            palettesize=256,
            subrectangles=False,
        ) as writer:
            import gc as _gc

            for frame_rgb in iter_parallel_sweep_display_frames(
                sweep,
                plot_fn,
                figsize=figsize,
                title_fn=title_fn,
                close_frames=close_frames,
                workers=workers,
                env_var_name=env_var_name,
            ):
                writer.append_data(np.asarray(frame_rgb, dtype=np.uint8))
                frame_count += 1
                if progress_callback is not None:
                    progress_callback(frame_count, total_frames)
                if frame_count % 16 == 0:
                    _gc.collect()
    except ImportError:
        from PIL import Image

        frames = []
        for frame_rgb in iter_parallel_sweep_display_frames(
            sweep,
            plot_fn,
            figsize=figsize,
            title_fn=title_fn,
            close_frames=close_frames,
            workers=workers,
            env_var_name=env_var_name,
        ):
            frames.append(Image.fromarray(np.asarray(frame_rgb, dtype=np.uint8)))
            frame_count += 1
            if progress_callback is not None:
                progress_callback(frame_count, total_frames)
        if frames:
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=max(1, int(round(duration_s * 1000))),
                loop=0,
            )

    if frame_count == 0:
        raise ValueError("sweep has no items to animate")
    return gif_path


def animate_sweep_plots(
    sweep: Mapping[str, Any],
    plots: Iterable[SweepPlotSpec | str | Any | dict[str, Any]],
    *,
    plot_builder: Callable[[SweepPlotSpec], tuple[Any, str]],
    safe_name: Callable[[Any], str],
    deprecated_names: Iterable[str] = (),
    close_frames: bool = True,
    stream: bool = True,
    workers: int | None = None,
    output_dir: str | Path | None = None,
    env_var_name: str = "NEUROINFRA_SWEEP_RENDER_WORKERS",
    progress_callback: Callable[[int, int], None] | None = None,
    default_output_dir_factory: Callable[[], Path] | None = None,
) -> dict[str, Path]:
    """Render and save multiple sweep animation artifacts from one sweep."""
    artifacts: dict[str, Path] = {}
    for raw_spec in plots:
        spec = normalize_sweep_plot_spec(raw_spec)
        if is_deprecated_sweep_animation_spec(spec, deprecated_names=deprecated_names):
            continue
        plot_fn, filename = plot_builder(spec)
        if stream:
            artifacts[filename] = save_sweep_animation_stream(
                sweep,
                plot_fn,
                filename,
                safe_name=safe_name,
                output_dir=output_dir,
                figsize=spec.figsize,
                title_fn=spec.title_fn,
                close_frames=close_frames,
                fps=spec.fps,
                workers=workers,
                env_var_name=env_var_name,
                progress_callback=progress_callback,
                default_output_dir_factory=default_output_dir_factory,
            )
        else:
            anim = animate_sweep(
                sweep,
                plot_fn,
                figsize=spec.figsize,
                interval=spec.interval,
                title_fn=spec.title_fn,
                close_frames=close_frames,
            )
            artifacts[filename] = save_animation(
                anim,
                filename,
                safe_name=safe_name,
                output_dir=output_dir,
                sweep=sweep,
                fps=spec.fps,
                default_output_dir_factory=default_output_dir_factory,
            )
    return artifacts
