"""Reference-band policy and provenance primitives for the NeuronUnit overhaul."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
import quantities as pq
from scipy.stats import beta as beta_distribution

from olfactorybulb.neuronunit.frozen_payloads import FrozenMappingPayload
from olfactorybulb.neuronunit.provenance import ProvenanceRecord, ValidationReview


def sigma_phrase(sigma_multiplier: float) -> str:
    if np.isclose(float(sigma_multiplier), 2.0):
        return "two standard deviations"
    if np.isclose(float(sigma_multiplier), 1.0):
        return "one standard deviation"
    return f"{float(sigma_multiplier):g} standard deviations"


@dataclass(frozen=True)
class ReferenceAcceptanceBand:
    low: float
    high: float
    mode: str
    standard_label: str
    raw_low: float
    raw_high: float
    lower_bound: float | None
    upper_bound: float | None
    description: str
    sigma_multiplier: float = 2.0


@dataclass(frozen=True)
class ReferenceBandPolicy:
    mode: str = "symmetric_sd"
    sigma_multiplier: float = 2.0
    lower_bound: float | None = None
    upper_bound: float | None = None
    quantile_low: float | None = None
    quantile_high: float | None = None
    quantile_low_label: str | None = None
    quantile_high_label: str | None = None

    def accepted_band(self, *, reference_mean: float, reference_sd: float) -> ReferenceAcceptanceBand:
        return compute_reference_acceptance_band(
            reference_mean=reference_mean,
            reference_sd=reference_sd,
            sigma_multiplier=self.sigma_multiplier,
            band_mode=self.mode,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            quantile_low=self.quantile_low,
            quantile_high=self.quantile_high,
            quantile_low_label=self.quantile_low_label,
            quantile_high_label=self.quantile_high_label,
        )


_UNIT_ALIASES: dict[str, pq.Quantity] = {
    "mV": pq.mV,
    "ms": pq.ms,
    "pA": pq.pA,
    "Hz": pq.Hz,
    "pF": pq.pF,
    "MOhm": pq.MOhm,
    "um": pq.um,
    "mV/ms": pq.mV / pq.ms,
    "Hz/nA": pq.Hz / pq.nA,
}


def quantity_unit_for_text(unit_text: str) -> pq.Quantity | None:
    normalized = str(unit_text or "").strip()
    if not normalized or normalized == "%":
        return None
    return _UNIT_ALIASES.get(normalized)


def measurement_with_unit(value: float, unit_text: str) -> float | pq.Quantity:
    unit = quantity_unit_for_text(unit_text)
    numeric = float(value)
    if unit is None:
        return numeric
    return numeric * unit


def numeric_value(value: float | pq.Quantity) -> float:
    if isinstance(value, pq.Quantity):
        return float(value.magnitude)
    return float(value)


@dataclass(frozen=True)
class ReferenceBandObservation:
    property_name: str
    group: str
    metric_key: str
    reference_mean: float
    reference_sd: float
    unit_text: str = ""
    policy: ReferenceBandPolicy = field(default_factory=ReferenceBandPolicy)
    provenance: ProvenanceRecord = field(default_factory=ProvenanceRecord)
    review: ValidationReview = field(default_factory=ValidationReview)
    prediction_unit_text: str = ""

    @property
    def resolved_prediction_unit_text(self) -> str:
        return self.prediction_unit_text or self.unit_text

    @property
    def accepted_band(self) -> ReferenceAcceptanceBand:
        return self.policy.accepted_band(
            reference_mean=self.reference_mean,
            reference_sd=self.reference_sd,
        )

    @property
    def reference_mean_measurement(self) -> float | pq.Quantity:
        return measurement_with_unit(self.reference_mean, self.unit_text)

    def normalize_prediction(self, value: float | pq.Quantity) -> float | pq.Quantity:
        prediction_unit = quantity_unit_for_text(self.resolved_prediction_unit_text)
        reference_unit = quantity_unit_for_text(self.unit_text)
        if isinstance(value, pq.Quantity):
            if reference_unit is None:
                return float(value.magnitude)
            return value.rescale(reference_unit)
        if prediction_unit is None or reference_unit is None:
            return measurement_with_unit(float(value), self.unit_text)
        return (float(value) * prediction_unit).rescale(reference_unit)

    def accepted_bounds_with_units(self) -> tuple[float | pq.Quantity, float | pq.Quantity]:
        band = self.accepted_band
        return (
            measurement_with_unit(band.low, self.unit_text),
            measurement_with_unit(band.high, self.unit_text),
        )

    def observation_payload(self) -> "ReferenceBandObservationPayload":
        return ReferenceBandObservationPayload.from_mapping({
            "property_name": self.property_name,
            "group": self.group,
            "metric_key": self.metric_key,
            "reference_mean": self.reference_mean,
            "reference_sd": self.reference_sd,
            "unit_text": self.unit_text,
            "band_mode": self.policy.mode,
        })


class ReferenceBandObservationPayload(FrozenMappingPayload):
    """Frozen observation payload for reference-band SciUnit tests."""


def compute_reference_acceptance_band(
    *,
    reference_mean: float,
    reference_sd: float,
    sigma_multiplier: float,
    band_mode: str = "symmetric_sd",
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    quantile_low: float | None = None,
    quantile_high: float | None = None,
    quantile_low_label: str | None = None,
    quantile_high_label: str | None = None,
) -> ReferenceAcceptanceBand:
    mode = str(band_mode or "symmetric_sd").strip()
    phrase = sigma_phrase(sigma_multiplier)
    if mode == "symmetric_sd":
        raw_low = float(reference_mean - reference_sd * sigma_multiplier)
        raw_high = float(reference_mean + reference_sd * sigma_multiplier)
        standard_label = "symmetric reference interval"
        description = f"the uploaded arithmetic mean plus or minus {phrase}"
    elif mode == "lognormal_sd":
        if reference_mean <= 0.0:
            raise ValueError("lognormal_sd acceptance bands require a strictly positive reference mean")
        if reference_sd < 0.0:
            raise ValueError("lognormal_sd acceptance bands require a non-negative reference standard deviation")
        variance_ratio = (float(reference_sd) / float(reference_mean)) ** 2
        sigma_log = math.sqrt(math.log1p(variance_ratio))
        mu_log = math.log(float(reference_mean)) - 0.5 * sigma_log**2
        raw_low = float(math.exp(mu_log - float(sigma_multiplier) * sigma_log))
        raw_high = float(math.exp(mu_log + float(sigma_multiplier) * sigma_log))
        standard_label = "lognormal reference interval"
        description = (
            f"the uploaded arithmetic mean and standard deviation under a lognormal reconstruction with {phrase}"
        )
    elif mode == "beta_sd":
        if reference_sd <= 0.0:
            raise ValueError("beta_sd acceptance bands require a strictly positive reference standard deviation")
        if not 0.0 < reference_mean < 1.0:
            raise ValueError("beta_sd acceptance bands require a reference mean strictly between zero and one")
        variance = float(reference_sd) ** 2
        max_variance = float(reference_mean) * (1.0 - float(reference_mean))
        if variance >= max_variance:
            raise ValueError("beta_sd acceptance bands require variance below the Bernoulli ceiling")
        common = (float(reference_mean) * (1.0 - float(reference_mean)) / variance) - 1.0
        alpha = float(reference_mean) * common
        beta = (1.0 - float(reference_mean)) * common
        central_mass = float(math.erf(abs(float(sigma_multiplier)) / math.sqrt(2.0)))
        tail_mass = (1.0 - central_mass) / 2.0
        raw_low = float(beta_distribution.ppf(tail_mass, alpha, beta))
        raw_high = float(beta_distribution.ppf(1.0 - tail_mass, alpha, beta))
        standard_label = "beta-reconstructed probability interval"
        description = f"a beta-distribution reconstruction matched to the uploaded mean and standard deviation with {phrase}"
    elif mode == "quantile_interval":
        if quantile_low is None or quantile_high is None:
            raise ValueError("quantile_interval acceptance bands require both quantile_low and quantile_high")
        raw_low = float(quantile_low)
        raw_high = float(quantile_high)
        low_label = str(quantile_low_label or "lower quantile").strip()
        high_label = str(quantile_high_label or "upper quantile").strip()
        standard_label = "reported quantile interval"
        description = f"the uploaded reported interval from {low_label} to {high_label}"
    elif mode == "binary_indicator":
        if reference_mean not in (0.0, 1.0):
            raise ValueError("binary_indicator acceptance bands require a reference mean of exactly 0 or 1")
        raw_low = raw_high = float(reference_mean)
        standard_label = "binary reference indicator"
        description = "an exact match to the uploaded binary reference indicator"
    else:
        raise ValueError(f"Unsupported reference acceptance band mode {mode!r}")

    clipped_low = raw_low if lower_bound is None else max(raw_low, float(lower_bound))
    clipped_high = raw_high if upper_bound is None else min(raw_high, float(upper_bound))
    if clipped_high < clipped_low:
        raise ValueError("Configured acceptance-band bounds clipped the interval into an empty range")
    return ReferenceAcceptanceBand(
        low=float(clipped_low),
        high=float(clipped_high),
        mode=mode,
        standard_label=standard_label,
        raw_low=float(raw_low),
        raw_high=float(raw_high),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        description=description,
        sigma_multiplier=float(sigma_multiplier),
    )


__all__ = [
    "ProvenanceRecord",
    "ReferenceAcceptanceBand",
    "ReferenceBandObservation",
    "ReferenceBandObservationPayload",
    "ReferenceBandPolicy",
    "ValidationReview",
    "compute_reference_acceptance_band",
    "measurement_with_unit",
    "numeric_value",
    "quantity_unit_for_text",
    "sigma_phrase",
]
