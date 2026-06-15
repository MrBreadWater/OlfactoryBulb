"""Coverage for typed protocol metric rows and grouped summaries."""

from __future__ import annotations

from argparse import Namespace

from olfactorybulb.audit.protocol_evidence import ProtocolEvidenceBundle
from olfactorybulb.audit.reference_validation_document import ValidationDesignReviewDefaultsSpec
from olfactorybulb.audit.reference_validation_protocol_core import ProtocolRunResult
from olfactorybulb.audit.reference_validation_rules import ValidationRuleContext
from olfactorybulb.neuronunit.metric_tables import (
    MetricSummaryTable,
    MetricTable,
    MetricValueMapPayload,
    coerce_metric_table,
)


rows = [
    {
        "cell_name": "MC1",
        "cell_type": "MC",
        "rheobase_pA": 100.0,
        "zero_step_rate_Hz": 0.0,
        "label": "keep-nonnumeric",
    },
    {
        "cell_name": "MC2",
        "cell_type": "MC",
        "rheobase_pA": 110.0,
        "zero_step_rate_Hz": 0.0,
        "label": "keep-nonnumeric",
    },
    {
        "cell_name": "TC1",
        "cell_type": "TC",
        "rheobase_pA": 90.0,
        "zero_step_rate_Hz": 0.0,
        "label": "keep-nonnumeric",
    },
]


metric_table = coerce_metric_table(rows)
assert isinstance(metric_table, MetricTable)
assert len(metric_table) == 3
assert metric_table[0]["cell_name"] == "MC1"
assert metric_table[0].get("rheobase_pA") == 100.0
metric_value_map = metric_table.metric_value_map("rheobase_pA")
assert isinstance(metric_value_map, MetricValueMapPayload)
assert metric_value_map.to_dict() == {
    "MC1": 100.0,
    "MC2": 110.0,
    "TC1": 90.0,
}

summary = metric_table.summarize()
assert isinstance(summary, MetricSummaryTable)
assert set(summary) == {"MC", "TC"}
assert summary["MC"]["rheobase_pA"] == 105.0
assert summary["MC"]["zero_step_rate_Hz"] == 0.0
assert "label" not in summary["MC"]
assert summary.metric_value("TC", "rheobase_pA") == 90.0

protocol_result = ProtocolRunResult(
    metrics=rows,
    protocol_evidence=ProtocolEvidenceBundle(),
    group_field="cell_type",
)
assert isinstance(protocol_result.metrics, MetricTable)
assert protocol_result.metrics.group_field == "cell_type"
assert protocol_result.metrics[1]["cell_name"] == "MC2"

context = ValidationRuleContext(
    metrics=rows,
    summary={"MC": {"rheobase_pA": 105.0}, "TC": {"rheobase_pA": 90.0}},
    args=Namespace(reference_sigma_multiplier=2.0),
    validation_id="synthetic_validation",
    default_group="",
    notes_path="",
    design_review_defaults=ValidationDesignReviewDefaultsSpec(status="pending"),
    protocol_result=protocol_result,
)
assert isinstance(context.metrics, MetricTable)
assert isinstance(context.summary, MetricSummaryTable)
assert context.metrics[0]["cell_name"] == "MC1"
assert context.summary["MC"]["rheobase_pA"] == 105.0
