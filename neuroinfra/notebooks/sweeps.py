"""Reusable notebook-facing parameter-sweep planning helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable


@dataclass(frozen=True)
class SweepPlanHooks:
    """Hook bundle for domain-specific sweep config and label policy."""

    normalize_base_config_fn: Callable[[dict[str, Any]], dict[str, Any]]
    make_timestamp_fn: Callable[[], str]
    make_sweep_label_fn: Callable[[dict[str, Any], Any, str], str]
    make_item_label_fn: Callable[[dict[str, Any], Any, str, int], str]


def split_path_parts(path: Any) -> list[str]:
    """Split one dotted or indexed path into addressable components."""
    if isinstance(path, (list, tuple)):
        return list(path)
    text = str(path).replace("[", ".").replace("]", "")
    return [part for part in text.split(".") if part]


def set_nested_value(obj: Any, path: Any, value: Any) -> None:
    """Assign ``value`` inside one nested dict/list structure addressed by ``path``."""
    parts = split_path_parts(path)
    if not parts:
        raise ValueError("path must contain at least one addressable component")

    current = obj
    for index, part in enumerate(parts[:-1]):
        next_part = parts[index + 1]
        if isinstance(current, list):
            part = int(part)
            while len(current) <= part:
                current.append({} if not str(next_part).isdigit() else [])
            current = current[part]
            continue
        if part not in current or current[part] is None:
            current[part] = [] if str(next_part).isdigit() else {}
        current = current[part]

    final = parts[-1]
    if isinstance(current, list):
        final = int(final)
        while len(current) <= final:
            current.append(None)
        current[final] = value
    else:
        current[final] = value


def prepare_sweep_plan(
    hooks: SweepPlanHooks,
    base_config: dict[str, Any],
    sweep_path: str | list[str] | dict[str, list[Any]],
    values: list[Any] | tuple[Any, ...] | None = None,
    *,
    grid: bool = False,
) -> dict[str, Any]:
    """Normalize one sweep request into explicit per-item configs and labels."""
    normalized_base_config = hooks.normalize_base_config_fn(deepcopy(base_config))
    timestamp = hooks.make_timestamp_fn()
    sweep_label = hooks.make_sweep_label_fn(normalized_base_config, sweep_path, timestamp)

    items: list[dict[str, Any]] = []
    normalized_values: list[Any] = []

    if grid:
        if not isinstance(sweep_path, dict):
            raise TypeError("Grid sweeps require a dict of {path: values}")
        paths = list(sweep_path.keys())
        value_lists = list(sweep_path.values())
        iterable = [dict(zip(paths, combo)) for combo in product(*value_lists)]
    elif isinstance(sweep_path, dict):
        paths = list(sweep_path.keys())
        value_lists = list(sweep_path.values())
        lengths = [len(v) for v in value_lists]
        if len(set(lengths)) != 1:
            raise ValueError(
                "All parameter lists must have the same length for a joint sweep; "
                f"got lengths {dict(zip(paths, lengths))}"
            )
        iterable = [dict(zip(paths, combo)) for combo in zip(*value_lists)]
    else:
        if values is None:
            raise ValueError("values must be provided for single-axis sweeps")
        iterable = list(values)

    for index, value in enumerate(iterable):
        sweep_config = deepcopy(normalized_base_config)
        if isinstance(value, dict):
            for path, path_value in value.items():
                set_nested_value(sweep_config, path, path_value)
            item_value = dict(value)
        else:
            set_nested_value(sweep_config, sweep_path, value)
            item_value = value
        item_label = hooks.make_item_label_fn(normalized_base_config, sweep_path, timestamp, index)
        items.append(
            {
                "index": index,
                "value": item_value,
                "config": sweep_config,
                "label": item_label,
            }
        )
        normalized_values.append(item_value)

    return {
        "path": sweep_path,
        "values": normalized_values,
        "items": items,
        "paramset": normalized_base_config.get("paramset"),
        "timestamp": timestamp,
        "sweep_label": sweep_label,
        "base_config": normalized_base_config,
        "grid": {path: list(path_values) for path, path_values in sweep_path.items()}
        if grid and isinstance(sweep_path, dict)
        else None,
    }
