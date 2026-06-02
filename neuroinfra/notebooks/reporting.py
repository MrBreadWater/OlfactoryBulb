"""Generic notebook reporting and figure-output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping


def flatten_for_diff(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts into ``path -> value`` pairs for diff reporting."""
    items: dict[str, Any] = {}
    if isinstance(value, dict):
        for key in sorted(value.keys(), key=lambda item: str(item)):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.update(flatten_for_diff(value[key], next_prefix))
        return items
    items[prefix or "$"] = value
    return items


def diff_values(before: Any, after: Any) -> list[dict[str, Any]]:
    """Return value changes between two nested JSON-like structures."""
    before_flat = flatten_for_diff(before)
    after_flat = flatten_for_diff(after)
    keys = sorted(set(before_flat) | set(after_flat))
    changes = []
    for key in keys:
        before_value = before_flat.get(key)
        after_value = after_flat.get(key)
        if before_value != after_value:
            changes.append({"path": key, "before": before_value, "after": after_value})
    return changes


def format_diff_value(
    value: Any,
    *,
    json_ready_fn: Callable[[Any], Any] | None = None,
    max_len: int = 160,
) -> str:
    """Render a compact JSON string for one diff value."""
    payload = json_ready_fn(value) if json_ready_fn is not None else value
    text = json.dumps(payload, sort_keys=True)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def print_diff_section(
    title: str,
    changes: list[dict[str, Any]],
    max_items: int | None = None,
    *,
    json_ready_fn: Callable[[Any], Any] | None = None,
    write_fn: Callable[[str], None] = print,
) -> None:
    """Print a human-readable diff section using an injected write function."""
    write_fn(f"\n{title}:")
    if not changes:
        write_fn("  (no differences)")
        return

    if max_items is None:
        max_items = len(changes)

    for change in changes[:max_items]:
        before_text = format_diff_value(change["before"], json_ready_fn=json_ready_fn)
        after_text = format_diff_value(change["after"], json_ready_fn=json_ready_fn)
        write_fn(f"- {change['path']}: {before_text} -> {after_text}")

    remaining = len(changes) - max_items
    if remaining > 0:
        write_fn(f"- ... {remaining} more differences")


def save_figure(
    name: str,
    *,
    fig: Any,
    safe_name_fn: Callable[[Any], str],
    default_output_dir_factory: Callable[[], Path],
    close_figure_fn: Callable[[Any], None] | None = None,
    run_or_result: Any = None,
    output_dir: str | Path | None = None,
    sweep: Mapping[str, Any] | None = None,
    dpi: int = 200,
    close: bool = False,
) -> Path:
    """Save one figure near a run directory or in a timestamped folder."""
    resolved_output_dir: Path | None = None
    if output_dir is None and sweep is not None and "sweep_dir" in sweep:
        resolved_output_dir = Path(sweep["sweep_dir"]) / "figures"
    elif output_dir is None and run_or_result is not None:
        if hasattr(run_or_result, "result_dir"):
            resolved_output_dir = Path(getattr(run_or_result, "result_dir"))
        elif isinstance(run_or_result, Mapping) and "result_dir" in run_or_result:
            resolved_output_dir = Path(run_or_result["result_dir"])

    output_base = Path(output_dir) if output_dir is not None else (resolved_output_dir or default_output_dir_factory())
    output_base.mkdir(parents=True, exist_ok=True)
    png_path = output_base / f"{safe_name_fn(name)}.png"
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")

    if close and close_figure_fn is not None:
        close_figure_fn(fig)

    return png_path
