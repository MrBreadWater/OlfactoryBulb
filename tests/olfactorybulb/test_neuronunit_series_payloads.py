"""Coverage for typed series row/context payload contracts."""

from __future__ import annotations

from olfactorybulb.neuronunit.series_payloads import (
    SeriesContextPayload,
    SeriesRowTable,
    coerce_series_context_payload,
    coerce_series_row_table,
)


raw_rows = [
    {
        "cell_name": "Model1",
        "current_pA": 100.0,
        "firing_rate_Hz": 6.0,
    },
    {
        "cell_name": "Model2",
        "current_pA": 200.0,
        "firing_rate_Hz": 11.0,
    },
]

raw_context = {
    "cell_models": ["Model1", "Model2"],
    "step_duration_ms": 500.0,
    "protocol_meta": {"family": "synthetic_series"},
}

row_table = coerce_series_row_table(raw_rows)
context_payload = coerce_series_context_payload(raw_context)

raw_rows[0]["cell_name"] = "MUTATED"
raw_context["cell_models"].append("MUTATED")
raw_context["protocol_meta"]["family"] = "mutated"

assert isinstance(row_table, SeriesRowTable)
assert len(row_table) == 2
assert row_table[0]["cell_name"] == "Model1"
assert row_table[1]["current_pA"] == 200.0
assert row_table.to_rows() == [
    {"cell_name": "Model1", "current_pA": 100.0, "firing_rate_Hz": 6.0},
    {"cell_name": "Model2", "current_pA": 200.0, "firing_rate_Hz": 11.0},
]

assert isinstance(context_payload, SeriesContextPayload)
assert context_payload.to_dict() == {
    "cell_models": ["Model1", "Model2"],
    "step_duration_ms": 500.0,
    "protocol_meta": {"family": "synthetic_series"},
}

print("neuronunit_series_payloads: OK")
