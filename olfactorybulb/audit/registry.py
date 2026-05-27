"""Registry of available audits."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import importlib
from typing import Iterable


@dataclass(frozen=True)
class AuditSpec:
    audit_id: str
    title: str
    description: str
    module_path: str

    def load_module(self):
        return importlib.import_module(self.module_path)


AUDITS: "OrderedDict[str, AuditSpec]" = OrderedDict(
    [
        (
            "epli_correctness",
            AuditSpec(
                audit_id="epli_correctness",
                title="EPLI correctness audit",
                description="Audit optional EPLI morphology, slice defaults, and network-readiness constraints.",
                module_path="olfactorybulb.audit.epli_correctness",
            ),
        ),
    ]
)


def get_audit_spec(audit_id: str) -> AuditSpec:
    try:
        return AUDITS[audit_id]
    except KeyError as exc:
        known = ", ".join(AUDITS)
        raise KeyError(f"Unknown audit {audit_id!r}. Known audits: {known}") from exc


def iter_audit_specs() -> Iterable[AuditSpec]:
    return AUDITS.values()

