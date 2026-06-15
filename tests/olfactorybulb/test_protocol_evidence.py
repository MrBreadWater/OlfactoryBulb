"""Coverage for typed protocol-evidence payload wrappers."""

from __future__ import annotations

from olfactorybulb.audit.protocol_evidence import (
    ProtocolEvidenceBundle,
    ProtocolEvidenceSeriesSpec,
    ProtocolEvidenceStyleMap,
    ProtocolEvidenceValueMap,
    coerce_protocol_evidence_bundle,
)
from olfactorybulb.neuronunit.frozen_payloads import FrozenMappingPayload


style_source = {
    "line_width": 1.6,
    "nested": {"alpha": 0.8},
}
series_spec = ProtocolEvidenceSeriesSpec(
    evidence_key="fi_curve_rows",
    x_key="current_pA",
    y_keys=("firing_rate_Hz",),
    x_unit_text="pA",
    y_unit_text="Hz",
    x_quantity_name="Injected current",
    y_quantity_name="Firing rate",
    style=style_source,
)
style_source["nested"]["alpha"] = 0.1
assert isinstance(series_spec.style, ProtocolEvidenceStyleMap)
assert series_spec.to_dict()["style"] == {
    "line_width": 1.6,
    "nested": {"alpha": 0.8},
}
visual_style = series_spec.to_visual_spec()["style"]
assert visual_style["line_width"] == 1.6
assert visual_style["x_label"] == "Injected current (pA)"
assert visual_style["y_label"] == "Firing rate (Hz)"

rows_source = [
    {"cell_name": "CellA", "current_pA": 50.0, "nested": {"rate": 3.0}},
    {"cell_name": "CellB", "current_pA": 100.0, "nested": {"rate": 5.0}},
]
bundle = ProtocolEvidenceBundle(
    values={
        "fi_curve_rows": rows_source,
        "cell_models": ["CellA", "CellB"],
        "step_duration_ms": 500.0,
    },
    series_specs=(series_spec,),
)
rows_source[0]["nested"]["rate"] = 99.0
rows_source.append({"cell_name": "CellC", "current_pA": 150.0, "nested": {"rate": 7.0}})

assert isinstance(bundle.values, ProtocolEvidenceValueMap)
assert isinstance(bundle.values["fi_curve_rows"][0], FrozenMappingPayload)
assert bundle.rows("fi_curve_rows") == [
    {"cell_name": "CellA", "current_pA": 50.0, "nested": {"rate": 3.0}},
    {"cell_name": "CellB", "current_pA": 100.0, "nested": {"rate": 5.0}},
]
assert bundle.to_dict() == {
    "fi_curve_rows": [
        {"cell_name": "CellA", "current_pA": 50.0, "nested": {"rate": 3.0}},
        {"cell_name": "CellB", "current_pA": 100.0, "nested": {"rate": 5.0}},
    ],
    "cell_models": ["CellA", "CellB"],
    "step_duration_ms": 500.0,
}

updated = bundle.with_value("target_vm_mV", -60.0)
assert updated.to_dict()["target_vm_mV"] == -60.0
assert "target_vm_mV" not in bundle.to_dict()

merged = coerce_protocol_evidence_bundle(
    bundle,
    series_specs=(
        ProtocolEvidenceSeriesSpec(
            evidence_key="fi_curve_rows",
            x_key="current_pA",
            y_keys=("alt_rate_Hz",),
            x_unit_text="pA",
            y_unit_text="Hz",
            x_quantity_name="Injected current",
            y_quantity_name="Alt firing rate",
        ),
    ),
)
assert len(merged.series_specs) == 1
assert merged.series_specs[0].y_keys == ("alt_rate_Hz",)

print("protocol_evidence: OK")
