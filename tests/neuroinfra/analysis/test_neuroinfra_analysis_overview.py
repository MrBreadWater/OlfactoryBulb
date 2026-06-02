"""Smoke tests for extracted result-overview helpers."""

from __future__ import annotations

from pathlib import Path

from neuroinfra.analysis.overview import (
    build_result_overview,
    build_result_overview_context,
    first_result_file_metadata,
    metadata_value_or_result_length,
    result_file_metadata,
    result_value_length,
)


def main() -> None:
    result = {
        "result_dir": Path("/tmp/run"),
        "summary": {
            "label": "run_001",
            "paramset": "GammaSignature",
            "nranks": 16,
            "params": {
                "tstop": 1000.0,
                "sim_dt": 0.025,
                "actual_dt": 0.1,
                "recording_period": 0.1,
            },
            "timing_seconds": {
                "run_max_rank": 12.5,
                "total_max_rank": 14.0,
            },
            "files": {
                "input_times.pkl": {"items": 612},
                "soma_vs.npz": {"items": 193},
                "lfp.pkl": {"len_1": 36000},
            },
        },
        "input_times": [1, 2],
        "soma_vs": [],
        "lfp": [0.0, 1.0, 2.0],
    }

    context = build_result_overview_context(result)
    assert result_file_metadata(context, "input_times.pkl") == {"items": 612}
    assert first_result_file_metadata(context, ("missing.pkl", "soma_vs.npz")) == {"items": 193}
    assert metadata_value_or_result_length(
        context,
        metadata=result_file_metadata(context, "input_times.pkl"),
        metadata_key="items",
        result_key="input_times",
    ) == 612
    assert metadata_value_or_result_length(
        context,
        metadata={},
        metadata_key="items",
        result_key="lfp",
    ) == 3
    assert result_value_length(result, "missing") == 0

    overview = build_result_overview(
        context,
        extra_fields={
            "n_inputs": 612,
            "n_soma_traces": 193,
        },
    )
    assert overview == {
        "result_dir": "/tmp/run",
        "label": "run_001",
        "paramset": "GammaSignature",
        "nranks": 16,
        "tstop_ms": 1000.0,
        "sim_dt_ms": 0.025,
        "actual_dt_ms": 0.1,
        "recording_period_ms": 0.1,
        "run_seconds": 12.5,
        "total_seconds": 14.0,
        "n_inputs": 612,
        "n_soma_traces": 193,
    }
    print("analysis overview helpers: OK")


if __name__ == "__main__":
    main()
