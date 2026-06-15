"""Coverage for typed suite aggregation policies."""

from __future__ import annotations

from olfactorybulb.neuronunit.suite_scores import (
    SuiteAggregatePolicy,
    SuiteCaseMappingPayload,
    SuiteCaseScorePayload,
    SuiteCaseStatisticalPayload,
    SuiteCaseSummary,
    SuiteStatisticalPolicy,
    build_suite_aggregate_score,
)


payload_source = {
    "observed": 1.5,
    "nested": {"alpha": 0.25},
    "groups": ["MC", "TC"],
}
typed_payload = SuiteCaseScorePayload(
    score_kind="synthetic_payload",
    observation=payload_source,
    prediction={"maximum": 2.0, "allowed_groups": ["MC", "TC"]},
    normalization={"norm_score": 0.75, "status": "PASS"},
)
payload_source["nested"]["alpha"] = 9.9
payload_source["groups"].append("GC")

assert isinstance(typed_payload.observation, SuiteCaseMappingPayload)
assert isinstance(typed_payload.prediction, SuiteCaseMappingPayload)
assert isinstance(typed_payload.normalization, SuiteCaseMappingPayload)
assert typed_payload.to_dict()["observation"] == {
    "observed": 1.5,
    "nested": {"alpha": 0.25},
    "groups": ["MC", "TC"],
}
assert typed_payload.to_dict()["prediction"] == {
    "maximum": 2.0,
    "allowed_groups": ["MC", "TC"],
}


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
                statistical_summary=SuiteCaseStatisticalPayload(
                    score_family_category="equivalence",
                    statistical_test_family="equivalence_tost",
                    pvalue=0.01,
                    label="TOST p",
                    default_rollup_method="max",
                    threshold=0.05,
                    threshold_key="equivalence_alpha",
                    threshold_direction="le",
                ),
            ),
        ),
        SuiteCaseSummary(
            check_id="equiv_b",
            title="Equivalence case B",
            status="PASS",
            norm_score=0.9,
            case_score=SuiteCaseScorePayload(
                score_kind="equivalence_only",
                statistical_summary=SuiteCaseStatisticalPayload(
                    score_family_category="equivalence",
                    statistical_test_family="equivalence_tost",
                    pvalue=0.03,
                    label="TOST p",
                    default_rollup_method="max",
                    threshold=0.05,
                    threshold_key="equivalence_alpha",
                    threshold_direction="le",
                ),
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
    "available_case_fraction": 1.0,
    "unsupported_case_count": 0,
    "unsupported_case_fraction": 0.0,
    "score_text": "max TOST p 0.03",
    "score_interpretation": (
        "Diagnostic suite-level statistical summary derived from the case-level "
        "TOST p values. It does not replace the detailed per-case gates. "
        "(threshold 0.05)"
    ),
    "threshold": 0.05,
    "threshold_key": "equivalence_alpha",
    "threshold_direction": "le",
    "support_gate_passed": True,
    "threshold_gate_passed": True,
    "gate_passed": True,
    "case_pvalues": [0.01, 0.03],
    "case_check_ids": ["equiv_a", "equiv_b"],
    "unsupported_case_check_ids": [],
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
                statistical_summary=SuiteCaseStatisticalPayload(
                    score_family_category="welch_similarity",
                    statistical_test_family="legacy_welch_difference",
                    pvalue=0.08,
                    label="Welch p",
                    default_rollup_method="min",
                    threshold=0.05,
                    threshold_key="minimum_median_welch_pvalue",
                    threshold_direction="ge",
                ),
            ),
        ),
        SuiteCaseSummary(
            check_id="welch_b",
            title="Welch case B",
            status="PASS",
            norm_score=0.8,
            case_score=SuiteCaseScorePayload(
                score_kind="welch_only",
                statistical_summary=SuiteCaseStatisticalPayload(
                    score_family_category="welch_similarity",
                    statistical_test_family="legacy_welch_difference",
                    pvalue=0.12,
                    label="Welch p",
                    default_rollup_method="min",
                    threshold=0.05,
                    threshold_key="minimum_median_welch_pvalue",
                    threshold_direction="ge",
                ),
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
    "available_case_fraction": 1.0,
    "unsupported_case_count": 0,
    "unsupported_case_fraction": 0.0,
    "score_text": "min Welch p 0.08",
    "score_interpretation": (
        "Diagnostic suite-level statistical summary derived from the case-level "
        "Welch p values. It does not replace the detailed per-case gates. "
        "(threshold 0.05)"
    ),
    "threshold": 0.05,
    "threshold_key": "minimum_median_welch_pvalue",
    "threshold_direction": "ge",
    "support_gate_passed": True,
    "threshold_gate_passed": True,
    "gate_passed": True,
    "case_pvalues": [0.08, 0.12],
    "case_check_ids": ["welch_a", "welch_b"],
    "unsupported_case_check_ids": [],
}

median_equivalence_score = build_suite_aggregate_score(
    suite_id="synthetic.equivalence_suite_median",
    case_summaries=[
        SuiteCaseSummary(
            check_id="equiv_a",
            title="Equivalence case A",
            status="PASS",
            norm_score=1.0,
            case_score=SuiteCaseScorePayload(
                score_kind="hybrid_residual_equivalence",
                statistical_summary=SuiteCaseStatisticalPayload(
                    score_family_category="equivalence",
                    statistical_test_family="equivalence_tost",
                    pvalue=0.01,
                    label="TOST p",
                    default_rollup_method="max",
                    threshold=0.05,
                    threshold_key="equivalence_alpha",
                    threshold_direction="le",
                ),
            ),
        ),
        SuiteCaseSummary(
            check_id="equiv_b",
            title="Equivalence case B",
            status="PASS",
            norm_score=0.9,
            case_score=SuiteCaseScorePayload(
                score_kind="equivalence_only",
                statistical_summary=SuiteCaseStatisticalPayload(
                    score_family_category="equivalence",
                    statistical_test_family="equivalence_tost",
                    pvalue=0.05,
                    label="TOST p",
                    default_rollup_method="max",
                    threshold=0.05,
                    threshold_key="equivalence_alpha",
                    threshold_direction="le",
                ),
            ),
        ),
        SuiteCaseSummary(
            check_id="equiv_c",
            title="Equivalence case C",
            status="PASS",
            norm_score=0.8,
            case_score=SuiteCaseScorePayload(
                score_kind="equivalence_only",
                statistical_summary=SuiteCaseStatisticalPayload(
                    score_family_category="equivalence",
                    statistical_test_family="equivalence_tost",
                    pvalue=0.09,
                    label="TOST p",
                    default_rollup_method="max",
                    threshold=0.05,
                    threshold_key="equivalence_alpha",
                    threshold_direction="le",
                ),
            ),
        ),
    ],
    statistical_policy=SuiteStatisticalPolicy(rollup_method="median"),
)

median_equivalence_summary = median_equivalence_score.to_evidence()["suite_statistical_summary"]
assert median_equivalence_score.to_evidence()["suite_statistical_policy"] == {"rollup_method": "median"}
assert median_equivalence_summary == {
    "score_family_category": "equivalence",
    "statistical_test_family": "equivalence_tost",
    "rollup_method": "median",
    "rollup_source": "explicit",
    "rollup_pvalue": 0.05,
    "available_case_count": 3,
    "total_case_count": 3,
    "available_case_fraction": 1.0,
    "unsupported_case_count": 0,
    "unsupported_case_fraction": 0.0,
    "score_text": "median TOST p 0.05",
    "score_interpretation": (
        "Diagnostic suite-level statistical summary derived from the case-level "
        "TOST p values. It does not replace the detailed per-case gates. "
        "(threshold 0.05)"
    ),
    "threshold": 0.05,
    "threshold_key": "equivalence_alpha",
    "threshold_direction": "le",
    "support_gate_passed": True,
    "threshold_gate_passed": True,
    "gate_passed": True,
    "case_pvalues": [0.01, 0.05, 0.09],
    "case_check_ids": ["equiv_a", "equiv_b", "equiv_c"],
    "unsupported_case_check_ids": [],
}

partial_support_score = build_suite_aggregate_score(
    suite_id="synthetic.partial_support_suite",
    case_summaries=[
        SuiteCaseSummary(
            check_id="supported_case",
            title="Supported case",
            status="PASS",
            norm_score=1.0,
            case_score=SuiteCaseScorePayload(
                score_kind="equivalence_only",
                statistical_summary=SuiteCaseStatisticalPayload(
                    score_family_category="equivalence",
                    statistical_test_family="equivalence_tost",
                    pvalue=0.02,
                    label="TOST p",
                    default_rollup_method="max",
                    threshold=0.05,
                    threshold_key="equivalence_alpha",
                    threshold_direction="le",
                ),
            ),
        ),
        SuiteCaseSummary(
            check_id="unsupported_case",
            title="Unsupported case",
            status="FAIL",
            norm_score=0.0,
            case_score=SuiteCaseScorePayload(
                score_kind="equivalence_only",
            ),
        ),
    ],
    statistical_policy=SuiteStatisticalPolicy(
        minimum_available_case_count=2,
        minimum_available_case_fraction=1.0,
    ),
)

partial_support_policy = partial_support_score.to_evidence()["suite_statistical_policy"]
assert partial_support_policy == {
    "rollup_method": "auto",
    "minimum_available_case_count": 2,
    "minimum_available_case_fraction": 1.0,
}

partial_support_summary = partial_support_score.to_evidence()["suite_statistical_summary"]
assert partial_support_summary == {
    "score_family_category": "equivalence",
    "statistical_test_family": "equivalence_tost",
    "rollup_method": "max",
    "rollup_source": "auto_default",
    "rollup_pvalue": 0.02,
    "available_case_count": 1,
    "total_case_count": 2,
    "available_case_fraction": 0.5,
    "unsupported_case_count": 1,
    "unsupported_case_fraction": 0.5,
    "score_text": "max TOST p 0.02",
    "score_interpretation": (
        "Diagnostic suite-level statistical summary derived from the case-level "
        "TOST p values. It does not replace the detailed per-case gates. "
        "(threshold 0.05) Statistical support requirements: at least 2 supported cases; "
        "supported-case fraction >= 1."
    ),
    "threshold": 0.05,
    "threshold_key": "equivalence_alpha",
    "threshold_direction": "le",
    "minimum_available_case_count": 2,
    "minimum_available_case_fraction": 1.0,
    "support_gate_passed": False,
    "threshold_gate_passed": True,
    "gate_passed": False,
    "case_pvalues": [0.02],
    "case_check_ids": ["supported_case"],
    "unsupported_case_check_ids": ["unsupported_case"],
}

weighted_support_score = build_suite_aggregate_score(
    suite_id="synthetic.weighted_support_suite",
    case_summaries=[
        SuiteCaseSummary(
            check_id="weighted_supported",
            title="Weighted supported case",
            status="PASS",
            norm_score=1.0,
            case_weight=4.0,
            case_weight_label="matched points",
            case_score=SuiteCaseScorePayload(
                score_kind="equivalence_only",
                statistical_summary=SuiteCaseStatisticalPayload(
                    score_family_category="equivalence",
                    statistical_test_family="equivalence_tost",
                    pvalue=0.02,
                    label="TOST p",
                    default_rollup_method="max",
                    threshold=0.05,
                    threshold_key="equivalence_alpha",
                    threshold_direction="le",
                ),
            ),
        ),
        SuiteCaseSummary(
            check_id="weighted_unsupported",
            title="Weighted unsupported case",
            status="FAIL",
            norm_score=0.0,
            case_weight=8.0,
            case_weight_label="matched points",
            case_score=SuiteCaseScorePayload(
                score_kind="equivalence_only",
            ),
        ),
    ],
    statistical_policy=SuiteStatisticalPolicy(
        minimum_available_case_weight=6.0,
        minimum_available_case_weight_fraction=0.5,
    ),
)

weighted_support_policy = weighted_support_score.to_evidence()["suite_statistical_policy"]
assert weighted_support_policy == {
    "rollup_method": "auto",
    "minimum_available_case_weight": 6.0,
    "minimum_available_case_weight_fraction": 0.5,
}

weighted_support_summary = weighted_support_score.to_evidence()["suite_statistical_summary"]
assert weighted_support_summary == {
    "score_family_category": "equivalence",
    "statistical_test_family": "equivalence_tost",
    "rollup_method": "max",
    "rollup_source": "auto_default",
    "rollup_pvalue": 0.02,
    "available_case_count": 1,
    "total_case_count": 2,
    "available_case_fraction": 0.5,
    "unsupported_case_count": 1,
    "unsupported_case_fraction": 0.5,
    "available_case_weight": 4.0,
    "unsupported_case_weight": 8.0,
    "total_case_weight": 12.0,
    "available_case_weight_fraction": 0.333,
    "unsupported_case_weight_fraction": 0.667,
    "weight_label": "matched points",
    "score_text": "max TOST p 0.02",
    "score_interpretation": (
        "Diagnostic suite-level statistical summary derived from the case-level "
        "TOST p values. It does not replace the detailed per-case gates. "
        "(threshold 0.05) Statistical support requirements: supported-case weight >= 6; "
        "supported-case weight fraction >= 0.5."
    ),
    "threshold": 0.05,
    "threshold_key": "equivalence_alpha",
    "threshold_direction": "le",
    "minimum_available_case_weight": 6.0,
    "minimum_available_case_weight_fraction": 0.5,
    "support_gate_passed": False,
    "weight_support_gate_passed": False,
    "threshold_gate_passed": True,
    "gate_passed": False,
    "case_pvalues": [0.02],
    "case_check_ids": ["weighted_supported"],
    "unsupported_case_check_ids": ["weighted_unsupported"],
}

missing_weight_support_score = build_suite_aggregate_score(
    suite_id="synthetic.missing_weight_support_suite",
    case_summaries=[
        SuiteCaseSummary(
            check_id="weighted_supported",
            title="Weighted supported case",
            status="PASS",
            norm_score=1.0,
            case_weight=3.0,
            case_weight_label="matched points",
            case_score=SuiteCaseScorePayload(
                score_kind="equivalence_only",
                statistical_summary=SuiteCaseStatisticalPayload(
                    score_family_category="equivalence",
                    statistical_test_family="equivalence_tost",
                    pvalue=0.02,
                    label="TOST p",
                    default_rollup_method="max",
                    threshold=0.05,
                    threshold_key="equivalence_alpha",
                    threshold_direction="le",
                ),
            ),
        ),
        SuiteCaseSummary(
            check_id="unweighted_unsupported",
            title="Unweighted unsupported case",
            status="FAIL",
            norm_score=0.0,
            case_score=SuiteCaseScorePayload(score_kind="equivalence_only"),
        ),
    ],
    statistical_policy=SuiteStatisticalPolicy(minimum_available_case_weight=2.0),
)

missing_weight_summary = missing_weight_support_score.to_evidence()["suite_statistical_summary"]
assert missing_weight_summary["support_gate_passed"] is False
assert missing_weight_summary["weight_support_gate_passed"] is False
assert missing_weight_summary["minimum_available_case_weight"] == 2.0
assert "all suite cases need per-case weights" in missing_weight_summary["score_interpretation"]
assert missing_weight_summary["unsupported_case_count"] == 1
assert missing_weight_summary["unsupported_case_fraction"] == 0.5
assert missing_weight_summary["unsupported_case_check_ids"] == ["unweighted_unsupported"]
assert "available_case_weight" not in missing_weight_summary
assert "total_case_weight" not in missing_weight_summary

print("neuronunit_suite_scores: OK")
