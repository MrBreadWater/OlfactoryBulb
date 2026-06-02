"""Focused tests for generic notebook sweep-planning helpers."""

from __future__ import annotations

from neuroinfra.notebooks.sweeps import SweepPlanHooks, prepare_sweep_plan, set_nested_value, split_path_parts


def main() -> None:
    assert split_path_parts("a.b[0].c") == ["a", "b", "0", "c"]
    assert split_path_parts(["x", "1", "y"]) == ["x", "1", "y"]

    payload = {}
    set_nested_value(payload, "outer.inner[0].value", 7)
    assert payload == {"outer": {"inner": [{"value": 7}]}}
    set_nested_value(payload, ["outer", "inner", "1", "flag"], True)
    assert payload["outer"]["inner"][1]["flag"] is True

    hooks = SweepPlanHooks(
        normalize_base_config_fn=lambda cfg: {"normalized": True, **cfg},
        make_timestamp_fn=lambda: "2026-06-02T12-00-00",
        make_sweep_label_fn=lambda cfg, sweep_path, timestamp: f"{cfg['paramset']}-{timestamp}",
        make_item_label_fn=lambda cfg, sweep_path, timestamp, index: f"{cfg['paramset']}-{index:03d}",
    )

    single = prepare_sweep_plan(
        hooks,
        {"paramset": "GammaSignature"},
        "gaba_tau2_ms",
        [36.0, 50.0],
    )
    assert single["sweep_label"] == "GammaSignature-2026-06-02T12-00-00"
    assert single["values"] == [36.0, 50.0]
    assert single["items"][1]["config"]["gaba_tau2_ms"] == 50.0
    assert single["items"][1]["label"] == "GammaSignature-001"

    joint = prepare_sweep_plan(
        hooks,
        {"paramset": "GammaSignature"},
        {"gaba_tau2_ms": [36.0, 50.0], "gap_mc": [16.0, 32.0]},
    )
    assert joint["values"] == [
        {"gaba_tau2_ms": 36.0, "gap_mc": 16.0},
        {"gaba_tau2_ms": 50.0, "gap_mc": 32.0},
    ]
    assert joint["items"][0]["config"]["gap_mc"] == 16.0
    assert joint["items"][1]["config"]["gaba_tau2_ms"] == 50.0

    grid = prepare_sweep_plan(
        hooks,
        {"paramset": "GammaSignature"},
        {"gaba_tau2_ms": [36.0, 50.0], "gap_mc": [16.0, 32.0]},
        grid=True,
    )
    assert len(grid["items"]) == 4
    assert grid["grid"] == {"gaba_tau2_ms": [36.0, 50.0], "gap_mc": [16.0, 32.0]}
    assert grid["items"][2]["value"] == {"gaba_tau2_ms": 50.0, "gap_mc": 16.0}

    try:
        prepare_sweep_plan(
            hooks,
            {"paramset": "GammaSignature"},
            {"gaba_tau2_ms": [36.0], "gap_mc": [16.0, 32.0]},
        )
        raise AssertionError("expected mismatched joint sweep lengths to fail")
    except ValueError as exc:
        assert "same length" in str(exc)

    print("neuroinfra notebook sweeps: OK")


if __name__ == "__main__":
    main()
