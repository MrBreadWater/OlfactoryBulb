"""Smoke tests for dynamic cell-family grouping in saved artifacts/helpers.

Run with:
    MPLCONFIGDIR=/tmp/mpl conda run -n OBGPU python test_dynamic_cell_grouping.py
"""

from tempfile import TemporaryDirectory

import numpy as np

from obgpu_experiment_helpers import (
    list_available_cell_types,
    list_available_named_signals,
    list_available_soma_labels,
    split_traces_by_type,
)
from olfactorybulb.result_artifacts import (
    load_soma_spike_artifact,
    load_voltage_summary_artifact,
    save_soma_spike_artifact,
    save_voltage_summary_artifact,
)


t = np.asarray([0.0, 0.1, 0.2], dtype=float)
traces = [
    ("MC1[0].soma", t, np.asarray([-60.0, -55.0, -52.0])),
    ("EPLI0[0].soma", t, np.asarray([-58.0, -53.0, -50.0])),
    ("PVCRH_FSI1[0].soma", t, np.asarray([-59.0, -52.0, -48.0])),
    ("GC1[0].soma", t, np.asarray([-62.0, -61.0, -60.0])),
]

grouped = split_traces_by_type({"soma_vs": traces})
assert set(grouped) >= {"MC", "EPLI", "GC"}
assert len(grouped["EPLI"]) == 2
assert list_available_cell_types({"soma_vs": traces}) == ["MC", "GC", "EPLI"]
assert list_available_soma_labels({"soma_vs": traces}) == [row[0] for row in traces]

with TemporaryDirectory() as tmpdir:
    save_voltage_summary_artifact(traces, tmpdir)
    save_soma_spike_artifact(traces, tmpdir, threshold=-45.0)
    summary = load_voltage_summary_artifact(tmpdir)
    spikes = load_soma_spike_artifact(tmpdir)

assert "EPLI" in summary["cell_types"]
assert summary["n_traces"]["EPLI"] == 2
assert len(summary["mean_by_type"]["EPLI"]) == 3

saved_only_result = {
    "soma_vs": [],
    "voltage_summary": summary,
    "soma_spikes": spikes,
    "lfp_t": t,
    "lfp": np.asarray([0.0, 0.1, -0.1]),
    "input_times": [("MC1[0].tuft", np.asarray([1.0, 2.0]))],
    "gc_output_events": [{"dest_section": "MC1[0].soma", "times": np.asarray([2.0, 3.0])}],
}
available_signals = list_available_named_signals(saved_only_result, include_soma_labels=True)
assert "lfp" in available_signals
assert "input_rate" in available_signals
assert "gc_output_rate" in available_signals
assert "mean_EPLI_voltage" in available_signals
assert "EPLI0[0].soma" in available_signals

print("dynamic cell grouping smoke test: OK")
