"""Repository audit framework.

Audits in this package are intended to be:

1. explicit about what they can and cannot guarantee,
2. machine-runnable from a raw checkout, and
3. extensible so new biological or engineering checks can be added without
   rewriting the entrypoint every time.
"""

from .core import AuditItem, AuditReport, format_report
from .registry import AUDITS, get_audit_spec, iter_audit_specs

__all__ = [
    "AUDITS",
    "AuditItem",
    "AuditReport",
    "format_report",
    "get_audit_spec",
    "iter_audit_specs",
]
