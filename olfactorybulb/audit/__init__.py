"""Repository audit framework.

Audits in this package are intended to be:

1. explicit about what they can and cannot guarantee,
2. machine-runnable from a raw checkout, and
3. extensible so new biological or engineering checks can be added without
   rewriting the entrypoint every time.
"""

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
    "companion_visual_spec",
    "format_report",
    "make_visual_spec",
    "get_audit_spec",
    "iter_audit_specs",
    "series_visual_spec",
]
