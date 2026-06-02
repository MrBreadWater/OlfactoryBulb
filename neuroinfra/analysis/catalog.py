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
