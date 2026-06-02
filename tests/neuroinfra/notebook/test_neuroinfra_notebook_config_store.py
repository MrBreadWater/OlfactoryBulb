"""Focused tests for generic notebook config persistence helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from neuroinfra.notebooks.config_store import (
    json_ready,
    list_json_configs,
    load_json_config,
    save_json_config,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        payload = {
            "path": base / "demo",
            "array": np.asarray([1.0, 2.0]),
            "scalar": np.float64(3.5),
            "nested": {1: np.int64(4)},
        }
        normalized = json_ready(payload)
        assert normalized["path"] == str(base / "demo")
        assert normalized["array"] == [1.0, 2.0]
        assert normalized["scalar"] == 3.5
        assert normalized["nested"]["1"] == 4

        written = save_json_config(payload, base / "alpha.json")
        assert written.exists()
        loaded = load_json_config(written)
        assert loaded["array"] == [1.0, 2.0]
        assert loaded["nested"]["1"] == 4

        save_json_config({"name": "beta"}, base / "beta.json")
        save_json_config({"name": "gamma"}, base / "gamma.txt")
        assert [path.name for path in list_json_configs(base)] == ["alpha.json", "beta.json"]
        assert [path.name for path in list_json_configs(None, default_directory=base)] == ["alpha.json", "beta.json"]
        assert list_json_configs(base / "missing") == []

    print("neuroinfra notebook config store: OK")


if __name__ == "__main__":
    main()
