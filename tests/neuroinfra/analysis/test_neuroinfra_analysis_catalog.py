"""Smoke tests for extracted result-catalog helpers."""

from __future__ import annotations

from neuroinfra.analysis.catalog import (
    CategoryCatalogHooks,
    group_rows_by_category,
    list_available_categories,
    list_unique_labels,
    ordered_group_rows,
    ordered_names,
    round_robin_limit_by_subgroup,
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

    assert ordered_names({"TC", "other", "MC", "X"}, preferred_order=("MC", "TC"), unknown_name="other") == [
        "MC",
        "TC",
        "X",
        "other",
    ]

    fair_rows = round_robin_limit_by_subgroup(
        [
            ("MC0", 0),
            ("MC1", 1),
            ("TC0", 2),
            ("TC1", 3),
            ("TC2", 4),
        ],
        subgroup_fn=lambda row: "MC" if row[0].startswith("MC") else "TC",
        max_rows=4,
    )
    assert [row[0] for row in fair_rows] == ["MC0", "TC0", "MC1", "TC1"]

    ordered_rows = ordered_group_rows(
        [
            ("MC0", 0),
            ("MC1", 1),
            ("TC0", 2),
            ("GC0", 3),
        ],
        bucket_fn=lambda row: "MT" if row[0].startswith(("MC", "TC")) else "GC",
        order_buckets_fn=lambda buckets: ordered_names(buckets, preferred_order=("MT", "GC"), unknown_name="other"),
        limit_bucket_rows_fn=lambda bucket, bucket_rows: round_robin_limit_by_subgroup(
            bucket_rows,
            subgroup_fn=lambda row: row[0][:2],
            max_rows=2 if bucket == "MT" else 1,
        ),
    )
    assert [row[0] for row in ordered_rows] == ["MC0", "TC0", "GC0"]
    print("analysis catalog helpers: OK")


if __name__ == "__main__":
    main()
