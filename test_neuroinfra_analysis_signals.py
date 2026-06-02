"""Smoke tests for extracted named-signal registry helpers."""

from __future__ import annotations

import numpy as np

from neuroinfra.analysis.signals import (
    ResultSignalProvider,
    ResultSignalRegistry,
    keyed_trace_signal_provider,
    list_available_result_signals,
    labeled_trace_signal_provider,
    mean_aligned_row_trace,
    pattern_result_signal_provider,
    resolve_result_signal,
    suffix_variant_signal_provider,
)


def main() -> None:
    result = {
        "lfp": [1.0, 2.0],
        "lfp_t": [0.0, 1.0],
        "labels": ["MC0", "TC0"],
    }

    def _uniform_trace(times, values, dt_ms=None):
        _ = dt_ms
        return np.asarray(times, dtype=float), np.asarray(values, dtype=float)

    providers = (
        keyed_trace_signal_provider(
            "lfp",
            time_key="lfp_t",
            value_key="lfp",
            uniform_trace_fn=_uniform_trace,
        ),
        suffix_variant_signal_provider(
            base_name="input_rate",
            suffix_payloads={"": None, "_MC": ["MC"]},
            availability_fn=lambda result, _context: bool(result.get("lfp")),
            resolve_variant_fn=lambda _result, payload, _context: payload or "all",
        ),
        pattern_result_signal_provider(
            r"mean_([A-Z]+)_voltage",
            list_names_fn=lambda _result, _context: ["mean_MC_voltage"],
            resolve_match_fn=lambda _result, match, _context: match.group(1),
        ),
        labeled_trace_signal_provider(
            include_context_key="include_labels",
            list_labels_fn=lambda result: result["labels"],
            iter_rows_fn=lambda result: [("MC0", [0.0, 1.0], [3.0, 4.0]), ("TC0", [0.0, 1.0], [5.0, 6.0])],
            label_fn=lambda row: row[0],
            time_fn=lambda row: row[1],
            value_fn=lambda row: row[2],
            uniform_trace_fn=_uniform_trace,
        ),
        ResultSignalProvider(
            list_names_fn=lambda result, context: result["labels"] if context.get("include_labels") else [],
            matches_fn=lambda _signal: True,
            resolve_fn=lambda result, signal, _context: signal if signal in result["labels"] else (_ for _ in ()).throw(KeyError(signal)),
        ),
    )
    registry = ResultSignalRegistry(providers)

    assert list_available_result_signals(result, providers) == ["lfp", "input_rate", "input_rate_MC", "mean_MC_voltage"]
    assert list_available_result_signals(result, providers, include_labels=True) == [
        "lfp",
        "input_rate",
        "input_rate_MC",
        "mean_MC_voltage",
        "MC0",
        "TC0",
    ]
    assert registry.list_available(result) == ["lfp", "input_rate", "input_rate_MC", "mean_MC_voltage"]
    assert registry.list_available(result, include_labels=True) == [
        "lfp",
        "input_rate",
        "input_rate_MC",
        "mean_MC_voltage",
        "MC0",
        "TC0",
    ]
    lfp_t, lfp_v = resolve_result_signal(result, "lfp", providers)
    assert np.allclose(lfp_t, [0.0, 1.0])
    assert np.allclose(lfp_v, [1.0, 2.0])
    assert resolve_result_signal(result, "input_rate", providers) == "all"
    assert resolve_result_signal(result, "input_rate_MC", providers) == ["MC"]
    assert resolve_result_signal(result, "mean_MC_voltage", providers) == "MC"
    mc_t, mc_v = resolve_result_signal(result, "MC0", providers, include_labels=True)
    assert np.allclose(mc_t, [0.0, 1.0])
    assert np.allclose(mc_v, [3.0, 4.0])
    assert registry.resolve(result, "input_rate") == "all"
    assert registry.resolve(result, "mean_MC_voltage") == "MC"
    try:
        registry.resolve(result, "missing")
        raise AssertionError("Expected unsupported signal lookup to fail")
    except KeyError as exc:
        assert "missing" in str(exc)

    mean_t, mean_v = mean_aligned_row_trace(
        [
            ("MC0", [0.0, 1.0, 2.0], [1.0, 3.0, 5.0]),
            ("MC1", [0.0, 1.0, 2.0], [3.0, 5.0, 7.0]),
        ],
        time_fn=lambda row: row[1],
        value_fn=lambda row: row[2],
        uniform_trace_fn=_uniform_trace,
    )
    assert np.allclose(mean_t, [0.0, 1.0, 2.0])
    assert np.allclose(mean_v, [2.0, 4.0, 6.0])
    print("analysis signal registry path: OK")


if __name__ == "__main__":
    main()
