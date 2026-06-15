"""Coverage for typed suite aggregation policies."""

from __future__ import annotations

from olfactorybulb.neuronunit.suite_scores import (
    SuiteAggregatePolicy,
    SuiteCaseScorePayload,
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

equivalence_score = build_suite_aggregate_score(
    suite_id="synthetic.equivalence_suite",
    case_summaries=[
        SuiteCaseSummary(
            check_id="equiv_a",
            title="Equivalence case A",
            status="PASS",
            norm_score=1.0,
            case_score=SuiteCaseScorePayload(
                score_kind="hybrid_residual_equivalence",
                observation={"aggregate_statistical_pvalue": 0.01},
                prediction={
                    "equivalence_alpha": 0.05,
                    "statistical_test_family": "equivalence_tost",
                },
            ),
        ),
        SuiteCaseSummary(
            check_id="equiv_b",
            title="Equivalence case B",
            status="PASS",
            norm_score=0.9,
            case_score=SuiteCaseScorePayload(
                score_kind="equivalence_only",
                observation={"aggregate_statistical_pvalue": 0.03},
                prediction={
                    "equivalence_alpha": 0.05,
                    "statistical_test_family": "equivalence_tost",
                },
            ),
        ),
    ],
)

equivalence_summary = equivalence_score.to_evidence()["suite_statistical_summary"]
assert equivalence_summary == {
    "score_family_category": "equivalence",
    "statistical_test_family": "equivalence_tost",
    "rollup_method": "max",
    "rollup_source": "auto_default",
    "rollup_pvalue": 0.03,
    "available_case_count": 2,
    "total_case_count": 2,
    "score_text": "max TOST p 0.03",
    "score_interpretation": (
        "Diagnostic suite-level statistical summary derived from the case-level "
        "TOST p values. It does not replace the detailed per-case gates. "
        "(threshold 0.05)"
    ),
    "threshold": 0.05,
    "threshold_key": "equivalence_alpha",
    "threshold_direction": "le",
    "gate_passed": True,
    "case_pvalues": [0.01, 0.03],
    "case_check_ids": ["equiv_a", "equiv_b"],
}

welch_score = build_suite_aggregate_score(
    suite_id="synthetic.welch_suite",
    case_summaries=[
        SuiteCaseSummary(
            check_id="welch_a",
            title="Welch case A",
            status="PASS",
            norm_score=1.0,
            case_score=SuiteCaseScorePayload(
                score_kind="hybrid_residual_welch",
                observation={"median_welch_pvalue": 0.08},
                prediction={
                    "minimum_median_welch_pvalue": 0.05,
                    "statistical_test_family": "legacy_welch_difference",
                },
            ),
        ),
        SuiteCaseSummary(
            check_id="welch_b",
            title="Welch case B",
            status="PASS",
            norm_score=0.8,
            case_score=SuiteCaseScorePayload(
                score_kind="welch_only",
                observation={"median_welch_pvalue": 0.12},
                prediction={
                    "minimum_median_welch_pvalue": 0.05,
                    "statistical_test_family": "legacy_welch_difference",
                },
            ),
        ),
    ],
)

welch_summary = welch_score.to_evidence()["suite_statistical_summary"]
assert welch_summary == {
    "score_family_category": "welch_similarity",
    "statistical_test_family": "legacy_welch_difference",
    "rollup_method": "min",
    "rollup_source": "auto_default",
    "rollup_pvalue": 0.08,
    "available_case_count": 2,
    "total_case_count": 2,
    "score_text": "min Welch p 0.08",
    "score_interpretation": (
        "Diagnostic suite-level statistical summary derived from the case-level "
        "Welch p values. It does not replace the detailed per-case gates. "
        "(threshold 0.05)"
    ),
    "threshold": 0.05,
    "threshold_key": "minimum_median_welch_pvalue",
    "threshold_direction": "ge",
    "gate_passed": True,
    "case_pvalues": [0.08, 0.12],
    "case_check_ids": ["welch_a", "welch_b"],
}

print("neuronunit_suite_scores: OK")
