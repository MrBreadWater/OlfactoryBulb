"""Smoke tests for the concrete olfactory-bulb result-analysis data adapters."""

from __future__ import annotations

import numpy as np

from olfactorybulb.analysis_data import (
    OLFACTORY_BULB_CATEGORY_CATALOG_HOOKS,
    cell_type_of,
    collect_spike_frequency_samples,
    list_available_cell_types,
    list_available_soma_labels,
    normalize_cell_name,
    ordered_cell_types,
    saved_soma_spike_rows,
    saved_voltage_summary_signal,
    split_traces_by_type,
)


def main() -> None:
    result = {
        "soma_vs": [
            ("MC0.soma", [0.0, 1.0], [-65.0, -60.0]),
            ("h.PVCRH0.soma", [0.0, 1.0], [-58.0, -55.0]),
        ],
        "soma_spikes": {
            "labels": ["MC0.soma", "h.PVCRH0.soma"],
            "spike_times": [
                np.asarray([10.0, 20.0], dtype=float),
                np.asarray([5.0, 15.0, 25.0], dtype=float),
            ],
            "metadata": {"threshold_mv": -20.0},
        },
        "voltage_summary": {
            "cell_types": ["MC", "PVCRH"],
            "t_by_type": {"MC": [0.0, 1.0]},
            "mean_by_type": {"MC": [-64.0, -61.0]},
        },
    }

    assert normalize_cell_name("h.PVCRH0.soma") == "PVCRH0"
    assert cell_type_of("h.PVCRH0.soma") == "EPLI"
    assert ordered_cell_types({"EPLI", "GC", "MC"}) == ["MC", "GC", "EPLI"]
    assert OLFACTORY_BULB_CATEGORY_CATALOG_HOOKS.categorize_label_fn("h.PVCRH0.soma") == "EPLI"

    grouped = split_traces_by_type(result)
    assert sorted(grouped) == ["EPLI", "MC"]
    assert list_available_cell_types(result) == ["MC", "EPLI"]
    assert list_available_soma_labels(result) == ["MC0.soma", "h.PVCRH0.soma"]

    signal_t, signal_y = saved_voltage_summary_signal(result, cell_type="MC", moment="mean", dt_ms=0.1)
    np.testing.assert_allclose(signal_t[[0, -1]], [0.0, 1.0])
    np.testing.assert_allclose(signal_y[[0, -1]], [-64.0, -61.0])
    assert len(signal_t) == 11
    assert len(signal_y) == 11
    assert saved_voltage_summary_signal(result, cell_type="EPLI") is None

    saved_rows = saved_soma_spike_rows(result, cell_types=["MC"], threshold=-20.0)
    assert saved_rows is not None
    assert saved_rows[0][0] == "MC0.soma"
    np.testing.assert_allclose(saved_rows[0][1], [10.0, 20.0])

    samples = collect_spike_frequency_samples(result, cell_types=["MC"], threshold=-20.0, modulus=None)
    np.testing.assert_allclose(samples["times"], [15.0])
    np.testing.assert_allclose(samples["freqs"], [100.0])
    assert samples["labels"] == ["MC0.soma"]
    assert samples["n_traces"] == 1
    assert samples["cell_types"] == ["MC"]

    print("olfactorybulb analysis data: OK")


if __name__ == "__main__":
    main()
