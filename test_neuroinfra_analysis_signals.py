"""Smoke tests for extracted named-signal registry helpers."""

from __future__ import annotations

from neuroinfra.analysis.signals import (
    ResultSignalProvider,
    list_available_result_signals,
    resolve_result_signal,
)


def main() -> None:
    result = {
        "lfp": [1.0, 2.0],
        "labels": ["MC0", "TC0"],
    }

    providers = (
        ResultSignalProvider(
            list_names_fn=lambda result, _context: ["lfp"] if result.get("lfp") else [],
            matches_fn=lambda signal: signal == "lfp",
            resolve_fn=lambda result, signal, _context: tuple(result["lfp"]) if signal == "lfp" else (_ for _ in ()).throw(KeyError(signal)),
        ),
        ResultSignalProvider(
            list_names_fn=lambda result, context: result["labels"] if context.get("include_labels") else [],
            matches_fn=lambda _signal: True,
            resolve_fn=lambda result, signal, _context: signal if signal in result["labels"] else (_ for _ in ()).throw(KeyError(signal)),
        ),
    )

    assert list_available_result_signals(result, providers) == ["lfp"]
    assert list_available_result_signals(result, providers, include_labels=True) == ["lfp", "MC0", "TC0"]
    assert resolve_result_signal(result, "lfp", providers) == (1.0, 2.0)
    assert resolve_result_signal(result, "MC0", providers, include_labels=True) == "MC0"
    try:
        resolve_result_signal(result, "missing", providers)
        raise AssertionError("Expected unsupported signal lookup to fail")
    except KeyError as exc:
        assert "missing" in str(exc)
    print("analysis signal registry path: OK")


if __name__ == "__main__":
    main()
