"""Repository audit framework.

Audits in this package are intended to be:

1. explicit about what they can and cannot guarantee,
2. machine-runnable from a raw checkout, and
3. extensible so new biological or engineering checks can be added without
   rewriting the entrypoint every time.
"""

from .criterion_math import (
    CriterionMath,
    criterion_math_for_absolute_difference,
    criterion_math_for_closed_range,
    criterion_math_for_exact_metric,
    criterion_math_for_lower_bound,
    criterion_math_for_ordering,
    criterion_math_for_reference_band,
    criterion_math_for_upper_bound,
    group_mean_symbol,
    observed_symbol_for_property,
)
from .core import (
    AuditItem,
    AuditReport,
    companion_visual_spec,
    format_report,
    make_visual_spec,
    series_visual_spec,
)
from .registry import AUDITS, get_audit_spec, iter_audit_specs

__all__ = [
    "AUDITS",
    "AuditItem",
    "AuditReport",
    "CriterionMath",
    "companion_visual_spec",
    "criterion_math_for_absolute_difference",
    "criterion_math_for_closed_range",
    "criterion_math_for_exact_metric",
    "criterion_math_for_lower_bound",
    "criterion_math_for_ordering",
    "criterion_math_for_reference_band",
    "criterion_math_for_upper_bound",
    "format_report",
    "group_mean_symbol",
    "make_visual_spec",
    "get_audit_spec",
    "iter_audit_specs",
    "observed_symbol_for_property",
    "series_visual_spec",
]
