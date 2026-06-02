"""Focused tests for concrete olfactory-bulb notebook config helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

from olfactorybulb.notebook_configs import (
    NotebookConfigHooks,
    config_diff,
    list_paramsets,
    list_saved_configs,
    load_config,
    save_config,
)


def _normalize_input_odors(value):
    return {int(key): payload for key, payload in value.items()}


def _resolve_effective_params(config):
    return {
        "full_param_snapshot": {
            "paramset": config.get("paramset"),
            "gaba_tau2_ms": config.get("gaba_tau2_ms"),
            "gap_mc": config.get("gap_mc"),
        }
    }


def _diff_values(before, after):
    changes = []
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            changes.append({"path": key, "before": before.get(key), "after": after.get(key)})
    return changes


def main() -> None:
    hooks = NotebookConfigHooks(
        normalize_input_odors_fn=_normalize_input_odors,
        resolve_effective_params_fn=_resolve_effective_params,
        diff_values_fn=_diff_values,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        config = {
            "paramset": "GammaSignature",
            "gaba_tau2_ms": 36.0,
            "input_odors": {"0": {"name": "Apple", "rel_conc": 0.1}},
        }
        saved = save_config(config, base / "demo.json")
        loaded = load_config(hooks, saved)
        assert loaded["input_odors"][0]["name"] == "Apple"

        save_config({"paramset": "GammaSignature", "gap_mc": 16.0}, base / "other.json")
        assert [path.name for path in list_saved_configs(base)] == ["demo.json", "other.json"]

        names = list_paramsets()
        assert "GammaSignature" in names
        with_saved = list_paramsets(include_saved=True, configs_dir=base)
        assert any(path.name == "demo.json" for path in with_saved["saved"])

        changes = config_diff(
            hooks,
            {"paramset": "GammaSignature", "gaba_tau2_ms": 36.0},
            {"paramset": "GammaSignature", "gaba_tau2_ms": 50.0},
        )
        assert changes == [{"path": "gaba_tau2_ms", "before": 36.0, "after": 50.0}]

    print("olfactorybulb notebook configs: OK")


if __name__ == "__main__":
    main()
