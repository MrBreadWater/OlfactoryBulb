"""Coverage for typed suite aggregation policies."""

from __future__ import annotations

from olfactorybulb.neuronunit.suite_scores import (
    SuiteAggregatePolicy,
    SuiteCaseSummary,
    build_suite_aggregate_score,
)


weighted_score = build_suite_aggregate_score(
    suite_id="synthetic.weighted_suite",
    case_summaries=[
        SuiteCaseSummary(
            check_id="case_a",
            title="Case A",
            status="PASS",
            norm_score=1.0,
            case_weight=2.0,
            case_weight_label="matched points",
        ),
        SuiteCaseSummary(
            check_id="case_b",
            title="Case B",
            status="WARN",
            norm_score=0.5,
            case_weight=6.0,
            case_weight_label="matched points",
        ),
    ],
    policy=SuiteAggregatePolicy(norm_rollup="weighted_mean"),
)

assert weighted_score.status == "WARN"
assert weighted_score.aggregate_norm_score == 0.625
assert weighted_score.score_text == "worst WARN, weighted mean norm 0.625"

weighted_evidence = weighted_score.to_evidence()
assert weighted_evidence["suite_norm_score_summary"] == {
    "mean": 0.75,
    "median": 0.75,
    "min": 0.5,
    "max": 1.0,
    "count": 2.0,
    "weighted_mean": 0.625,
    "total_weight": 8.0,
    "weight_label": "matched points",
}
assert weighted_evidence["suite_aggregate_score"] == {
    "status_rollup": "worst_case",
    "norm_rollup": "weighted_mean",
    "score_kind": "worst_case_weighted_mean_norm",
    "score_interpretation": (
        "Aggregate suite score derived from the worst detailed-case status plus "
        "the weighted mean normalized case score using the per-case aggregate weights."
    ),
    "status": "WARN",
    "score_text": "worst WARN, weighted mean norm 0.625",
    "score_value": 0.625,
}

print("neuronunit_suite_scores: OK")
