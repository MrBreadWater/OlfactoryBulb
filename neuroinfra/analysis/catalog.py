"""Reusable result-catalog helpers for analysis-facing result inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence


@dataclass(frozen=True)
class CategoryCatalogHooks:
    """Callbacks for category inference and stable ordering."""

    categorize_label_fn: Callable[[str], str]
    order_categories_fn: Callable[[Iterable[str]], list[str]]
    unknown_category: str = "other"


def group_rows_by_category(
    rows: Sequence[Any],
    *,
    label_fn: Callable[[Any], str],
    transform_row_fn: Callable[[Any], Any],
    hooks: CategoryCatalogHooks,
) -> dict[str, list[Any]]:
    """Group rows under stable category keys derived from each row label."""
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        label = str(label_fn(row))
        try:
            bucket = hooks.categorize_label_fn(label)
        except Exception:
            bucket = hooks.unknown_category
        grouped.setdefault(str(bucket), []).append(transform_row_fn(row))
    return grouped


def list_available_categories(
    *,
    label_sources: Sequence[Iterable[Any]],
    hooks: CategoryCatalogHooks,
) -> list[str]:
    """Infer stable available categories from one or more label sources."""
    inferred: list[str] = []
    for source in label_sources:
        for raw_label in source:
            label = str(raw_label)
            try:
                inferred.append(hooks.categorize_label_fn(label))
            except Exception:
                inferred.append(hooks.unknown_category)
    return hooks.order_categories_fn(inferred)


def list_unique_labels(*label_sources: Iterable[Any]) -> list[str]:
    """Return labels in first-seen order across one or more sources."""
    labels: list[str] = []
    seen: set[str] = set()
    for source in label_sources:
        for raw_label in source:
            label = str(raw_label)
            if label in seen:
                continue
            seen.add(label)
            labels.append(label)
    return labels


def ordered_names(
    names: Iterable[Any],
    *,
    preferred_order: Sequence[str],
    unknown_name: str = "other",
) -> list[str]:
    """Return names in stable preferred order with sorted overflow and unknown last."""
    seen = {str(name) for name in names}
    preferred = tuple(str(name) for name in preferred_order)
    ordered = [name for name in preferred if name in seen]
    preferred_set = set(preferred)
    ordered.extend(sorted(name for name in seen if name not in preferred_set and name != unknown_name))
    if unknown_name in seen:
        ordered.append(str(unknown_name))
    return ordered


def round_robin_limit_by_subgroup(
    rows: Sequence[Any],
    *,
    subgroup_fn: Callable[[Any], str],
    max_rows: int,
    unknown_subgroup: str = "other",
) -> list[Any]:
    """Select up to max_rows while alternating fairly across discovered subgroups."""
    if max_rows <= 0 or not rows:
        return []

    subgroups: dict[str, list[Any]] = {}
    subgroup_order: list[str] = []
    for row in rows:
        try:
            subgroup = str(subgroup_fn(row))
        except Exception:
            subgroup = str(unknown_subgroup)
        if subgroup not in subgroups:
            subgroups[subgroup] = []
            subgroup_order.append(subgroup)
        subgroups[subgroup].append(row)

    if len(subgroup_order) <= 1:
        return list(rows[:max_rows])

    selected: list[Any] = []
    indices = {subgroup: 0 for subgroup in subgroup_order}
    while len(selected) < max_rows:
        added = False
        for subgroup in subgroup_order:
            idx = indices[subgroup]
            bucket = subgroups[subgroup]
            if idx < len(bucket):
                selected.append(bucket[idx])
                indices[subgroup] = idx + 1
                added = True
                if len(selected) >= max_rows:
                    break
        if not added:
            break
    return selected
