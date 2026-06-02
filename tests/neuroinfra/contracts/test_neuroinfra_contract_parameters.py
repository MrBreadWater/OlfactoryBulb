"""Regression tests for generic parameter contract helpers."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from neuroinfra.contracts.parameters import (
    ParameterSpec,
    campaign_search_space_paths,
    parameter_contract_snapshot,
    parameter_display_order,
    search_space_paths,
    search_space_rows,
)


log_spec = ParameterSpec(path="kar_mt_gmax", low=0.01, high=0.08, scale="log")
linear_spec = ParameterSpec(path="tc_input_weight", low=0.4, high=1.2, scale="linear", default=1.0)
int_spec = ParameterSpec(path="n_cells", low=1, high=10, scale="linear", dtype="int")

assert log_spec.clamp(0.5) == 0.08
assert linear_spec.clamp(0.1) == 0.4
assert log_spec.low_encoded() < log_spec.high_encoded()
assert 0.01 <= log_spec.default_value() <= 0.08
assert linear_spec.default_value() == 1.0
assert int_spec.decode(4.6) == 5

space = [log_spec, linear_spec]
assert search_space_paths(space) == ["kar_mt_gmax", "tc_input_weight"]
rows = search_space_rows(space)
assert rows[0]["path"] == "kar_mt_gmax"
assert rows[1]["default"] == 1.0

contract = parameter_contract_snapshot(
    version=3,
    search_space_paths=search_space_paths(space),
    runtime_parameter_keys=("kar_mt_gmax", "tc_input_weight", "gaba_gmax"),
)
assert contract == {
    "version": 3,
    "search_space_paths": ["kar_mt_gmax", "tc_input_weight"],
    "runtime_parameter_keys": ["kar_mt_gmax", "tc_input_weight", "gaba_gmax"],
}

display_order = parameter_display_order(
    {"tc_input_weight": 0.7, "extra": 2.0, "optimizer_debug": "skip"},
    preferred_paths=("kar_mt_gmax", "tc_input_weight"),
    runtime_parameter_keys=("kar_mt_gmax", "tc_input_weight", "gaba_gmax"),
)
assert display_order == ["kar_mt_gmax", "tc_input_weight", "gaba_gmax", "extra"]

with TemporaryDirectory() as tmpdir:
    campaign_dir = Path(tmpdir)
    (campaign_dir / "campaign_config.json").write_text(
        json.dumps(
            {
                "search_space": [
                    {"path": "kar_mt_gmax"},
                    {"path": "gaba_gmax"},
                ]
            }
        )
    )
    assert campaign_search_space_paths(campaign_dir, fallback=["fallback"]) == ["kar_mt_gmax", "gaba_gmax"]
    assert campaign_search_space_paths(campaign_dir / "missing", fallback=["fallback"]) == ["fallback"]
