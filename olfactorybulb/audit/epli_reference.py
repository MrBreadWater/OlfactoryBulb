"""Literature-backed reference values for the EPLI audit."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundedReference:
    mean: float
    tolerance: float
    units: str
    source: str

    @property
    def low(self) -> float:
        return float(self.mean - self.tolerance)

    @property
    def high(self) -> float:
        return float(self.mean + self.tolerance)


@dataclass(frozen=True)
class EpliBehaviorReference:
    source: str
    minimum_fast_spiking_rate_hz: float
    stretch_fast_spiking_rate_hz: float
    audit_current_max_nA: float


SOMA_DIAMETER_UM = BoundedReference(
    mean=9.6,
    tolerance=0.7,
    units="um",
    source="Huang et al. 2013 (CRH+ EPL interneurons)",
)

PRIMARY_PROCESS_COUNT = BoundedReference(
    mean=3.5,
    tolerance=0.5,
    units="count",
    source="Huang et al. 2013 (CRH+ EPL interneurons)",
)

PLANAR_SPAN_UM = BoundedReference(
    mean=71.0,
    tolerance=4.5,
    units="um",
    source="Huang et al. 2013 (CRH+ EPL interneurons)",
)

BRANCHING_ZONE_MAX_UM = 30.0
PV_EPL_FRACTION = 0.914
CRH_PV_OVERLAP_FRACTION = 0.815

FAST_SPIKING_REFERENCE = EpliBehaviorReference(
    source="Huang et al. 2013, with upper-range context from Kato et al. 2013",
    minimum_fast_spiking_rate_hz=60.0,
    stretch_fast_spiking_rate_hz=77.0,
    audit_current_max_nA=2.0,
)


__all__ = [
    "BRANCHING_ZONE_MAX_UM",
    "CRH_PV_OVERLAP_FRACTION",
    "FAST_SPIKING_REFERENCE",
    "PLANAR_SPAN_UM",
    "PRIMARY_PROCESS_COUNT",
    "PV_EPL_FRACTION",
    "SOMA_DIAMETER_UM",
]
