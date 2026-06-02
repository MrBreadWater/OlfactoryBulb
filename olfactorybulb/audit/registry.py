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
    include_in_new_sweep: bool = True

    def load_module(self):
        return importlib.import_module(self.module_path)


AUDITS: "OrderedDict[str, AuditSpec]" = OrderedDict(
    [
        (
            "env_install",
            AuditSpec(
                audit_id="env_install",
                title="Environment/install audit",
                description="Audit whether the active machine environment can run the maintained OBGPU workflow.",
                module_path="olfactorybulb.audit.env_install",
            ),
        ),
        (
            "repo_health",
            AuditSpec(
                audit_id="repo_health",
                title="Repo health audit",
                description="Run the curated maintained-surface environment, wrapper, contract, and reference-data health checks.",
                module_path="olfactorybulb.audit.repo_health",
                include_in_new_sweep=False,
            ),
        ),
        (
            "reference_dataset_status",
            AuditSpec(
                audit_id="reference_dataset_status",
                title="Reference dataset status audit",
                description="Audit that one declarative reference dataset's generated outputs exist and load cleanly.",
                module_path="olfactorybulb.audit.reference_dataset_status",
                include_in_new_sweep=False,
            ),
        ),
        (
            "burton_urban_fi",
            AuditSpec(
                audit_id="burton_urban_fi",
                title="Burton & Urban f-I validation audit",
                description="Audit MC/TC f-I, AP-shape, and spike-train metrics against Burton & Urban 2014.",
                module_path="olfactorybulb.audit.burton_urban_fi",
            ),
        ),
        (
            "epli_correctness",
            AuditSpec(
                audit_id="epli_correctness",
                title="EPLI correctness audit",
                description="Audit optional EPLI morphology, slice defaults, and network-readiness constraints.",
                module_path="olfactorybulb.audit.epli_correctness",
            ),
        ),
        (
            "gc_intrinsic_validation",
            AuditSpec(
                audit_id="gc_intrinsic_validation",
                title="Granule-cell intrinsic validation audit",
                description="Audit maintained granule-cell models against generic and subtype-aware literature references.",
                module_path="olfactorybulb.audit.gc_intrinsic_validation",
            ),
        ),
        (
            "epl_fsi_intrinsic_validation",
            AuditSpec(
                audit_id="epl_fsi_intrinsic_validation",
                title="External plexiform layer fast-spiking interneuron intrinsic validation audit",
                description="Audit the maintained synthetic EPL fast-spiking interneuron surrogate against the Burton, Malyshko, and Urban 2024 reference bundle.",
                module_path="olfactorybulb.audit.epl_fsi_intrinsic_validation",
            ),
        ),
        (
            "human_review_status",
            AuditSpec(
                audit_id="human_review_status",
                title="Human review status audit",
                description="Audit that declarative reference-validation items resolve to explicit human-review states.",
                module_path="olfactorybulb.audit.human_review_status",
            ),
        ),
        (
            "hfo_feature_contracts",
            AuditSpec(
                audit_id="hfo_feature_contracts",
                title="HFO feature/visual contract audit",
                description="Audit that HFO optimizer parameters, notebook controls, packet artifacts, and dashboard expectations share one centralized contract.",
                module_path="olfactorybulb.audit.hfo_feature_contracts",
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


def iter_new_sweep_audit_specs() -> Iterable[AuditSpec]:
    return (spec for spec in AUDITS.values() if spec.include_in_new_sweep)
