"""Smoke tests for extracted result-catalog helpers."""

from __future__ import annotations

from neuroinfra.analysis.catalog import (
    CategoryCatalogHooks,
    group_rows_by_category,
    list_available_categories,
    list_unique_labels,
)


def main() -> None:
    hooks = CategoryCatalogHooks(
        categorize_label_fn=lambda label: "MC" if str(label).startswith("MC") else "TC" if str(label).startswith("TC") else "other",
        order_categories_fn=lambda categories: [name for name in ("MC", "TC", "other") if name in set(categories)],
        unknown_category="other",
    )

    grouped = group_rows_by_category(
        [("MC0.soma", [0.0], [-65.0]), ("TC0.soma", [0.0], [-63.0]), ("weird", [0.0], [-60.0])],
        label_fn=lambda row: row[0],
        transform_row_fn=lambda row: row[0],
        hooks=hooks,
    )
    assert grouped == {
        "MC": ["MC0.soma"],
        "TC": ["TC0.soma"],
        "other": ["weird"],
    }

    categories = list_available_categories(
        label_sources=(
            ["TC0.soma", "MC0.soma"],
            ["other"],
            ["MC1.soma"],
        ),
        hooks=hooks,
    )
    assert categories == ["MC", "TC", "other"]

    labels = list_unique_labels(["MC0", "TC0"], ["TC0", "MC1"], [])
    assert labels == ["MC0", "TC0", "MC1"]
    print("analysis catalog helpers: OK")


if __name__ == "__main__":
    main()
