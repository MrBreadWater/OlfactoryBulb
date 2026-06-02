"""Focused tests for generic notebook reporting helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

from neuroinfra.notebooks.reporting import (
    diff_values,
    flatten_for_diff,
    format_diff_value,
    print_diff_section,
    save_figure,
)


class _FakeFigure:
    def __init__(self) -> None:
        self.calls = []

    def savefig(self, path: Path, **kwargs) -> None:
        self.calls.append((Path(path), kwargs))
        Path(path).write_text("fake-image")


def main() -> None:
    assert flatten_for_diff({"b": {"x": 2}, "a": 1}) == {"a": 1, "b.x": 2}
    assert flatten_for_diff(7) == {"$": 7}

    changes = diff_values(
        {"a": 1, "nested": {"x": 2, "y": 3}},
        {"a": 1, "nested": {"x": 4}, "new": True},
    )
    assert changes == [
        {"path": "nested.x", "before": 2, "after": 4},
        {"path": "nested.y", "before": 3, "after": None},
        {"path": "new", "before": None, "after": True},
    ]

    formatted = format_diff_value(
        {"path": Path("/tmp/demo"), "value": [1, 2, 3]},
        json_ready_fn=lambda value: {"path": str(value["path"]), "value": value["value"]},
        max_len=24,
    )
    assert formatted.endswith("...")
    assert '"path"' in formatted

    lines: list[str] = []
    print_diff_section(
        "Config diff",
        changes,
        max_items=2,
        write_fn=lines.append,
    )
    assert lines == [
        "\nConfig diff:",
        "- nested.x: 2 -> 4",
        "- nested.y: 3 -> null",
        "- ... 1 more differences",
    ]

    no_diff_lines: list[str] = []
    print_diff_section("Empty", [], write_fn=no_diff_lines.append)
    assert no_diff_lines == ["\nEmpty:", "  (no differences)"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        fallback = base / "fallback"
        figure = _FakeFigure()
        closed = []

        saved = save_figure(
            "Demo Figure",
            fig=figure,
            safe_name_fn=lambda name: str(name).lower().replace(" ", "_"),
            default_output_dir_factory=lambda: fallback,
            close_figure_fn=lambda fig: closed.append(fig),
        )
        assert saved == fallback / "demo_figure.png"
        assert saved.exists()
        assert figure.calls[0][1]["dpi"] == 200
        assert figure.calls[0][1]["bbox_inches"] == "tight"
        assert closed == []

        sweep_figure = _FakeFigure()
        sweep_dir = base / "demo_sweep"
        sweep_saved = save_figure(
            "Sweep Figure",
            fig=sweep_figure,
            safe_name_fn=lambda name: str(name).lower().replace(" ", "_"),
            default_output_dir_factory=lambda: fallback,
            close_figure_fn=lambda fig: closed.append(fig),
            sweep={"sweep_dir": str(sweep_dir)},
            close=True,
        )
        assert sweep_saved == sweep_dir / "figures" / "sweep_figure.png"
        assert sweep_saved.exists()
        assert closed == [sweep_figure]

    print("neuroinfra notebook reporting: OK")


if __name__ == "__main__":
    main()
