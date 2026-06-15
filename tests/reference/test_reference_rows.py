"""Coverage for typed literature/reference row wrappers."""

from __future__ import annotations

from argparse import Namespace

from olfactorybulb.audit.reference_rows import (
    ReferenceRowRecord,
    ReferenceRowTable,
    coerce_reference_row_table,
)
from olfactorybulb.audit.reference_validation_document import ValidationDesignReviewDefaultsSpec
from olfactorybulb.audit.reference_validation_rules import ValidationRuleContext, _filter_rows


raw_rows = [
    {
        "Property": "Input Resistance",
        "Source": "Synthetic Study",
        "cell_type": "MC",
        "protocol_id": "P1",
        "note_ids": "N_SYNTHETIC",
    },
    {
        "Property": "Input Resistance",
        "Source": "Synthetic Study",
        "cell_type": "TC",
        "protocol_id": "P2",
        "note_ids": "",
    },
]

table = coerce_reference_row_table(raw_rows)
assert isinstance(table, ReferenceRowTable)
assert isinstance(table[0], ReferenceRowRecord)
assert table[0]["Property"] == "Input Resistance"
assert table[1].get("cell_type") == "TC"
assert table.to_rows() == raw_rows

context = ValidationRuleContext(
    metrics=[],
    summary={},
    args=Namespace(selected_cell_types=["MC"]),
    validation_id="synthetic_validation",
    default_group="",
    notes_path="",
    design_review_defaults=ValidationDesignReviewDefaultsSpec(status="pending"),
    protocol_result=None,
)

filtered = _filter_rows(
    table,
    {
        "filter_field": "cell_type",
        "filter_values_arg": "selected_cell_types",
    },
    args=context.args,
)
assert isinstance(filtered, ReferenceRowTable)
assert len(filtered) == 1
assert filtered[0]["protocol_id"] == "P1"

print("reference_rows: OK")
