"""Shared evidence-formatting helpers for maintained NeuronUnit bridges."""

from __future__ import annotations

from typing import Any

import quantities as pq

from olfactorybulb.audit.core import rounded
from olfactorybulb.neuronunit.reference_bands import numeric_value
from olfactorybulb.neuronunit.scalar_observations import is_finite_scalar


def rounded_evidence_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            result[key] = rounded_evidence_mapping(dict(value))
        elif isinstance(value, list):
            result[key] = [
                rounded(numeric_value(item)) if is_finite_scalar(item) else item
                for item in value
            ]
        elif is_finite_scalar(value):
            result[key] = rounded(numeric_value(value) if isinstance(value, pq.Quantity) else float(value))
        else:
            result[key] = value
    return result


__all__ = ["rounded_evidence_mapping"]
