"""Generic visual-contract metadata helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class FrequencyGroupSpec:
    """A named population grouping used by visualization consumers."""

    label: str
    display_label: str
    cell_types: tuple[str, ...]


@dataclass(frozen=True)
class ConditionPairSpec:
    """A matched two-condition figure pair for dashboard presentation."""

    title: str
    control_file: str
    ketamine_file: str
    dom_id_suffix: str
    open_by_default: bool = False


@dataclass(frozen=True)
class DashboardTabSpec:
    """One dashboard tab and its presentation metadata."""

    key: str
    label: str
    table_heading: str
    packet_heading: str
    display_limit: int | None = None


def build_visual_contract_snapshot(
    *,
    style_version: int,
    frequency_groups: Sequence[FrequencyGroupSpec],
    fixed_condition_pairs: Sequence[ConditionPairSpec],
    dashboard_tabs: Sequence[DashboardTabSpec],
    primary_psd_name_order: Sequence[str],
    packet_files: Sequence[str],
    spectrogram_pipeline: Mapping[str, Any],
    spectrogram_window_ms: float,
    time_modulus_ms: float,
) -> dict[str, Any]:
    return {
        "style_version": int(style_version),
        "frequency_groups": [asdict(group) for group in frequency_groups],
        "fixed_condition_pairs": [asdict(pair) for pair in fixed_condition_pairs],
        "dashboard_tabs": [asdict(tab) for tab in dashboard_tabs],
        "primary_psd_name_order": list(primary_psd_name_order),
        "packet_files": list(packet_files),
        "spectrogram_pipeline": dict(spectrogram_pipeline),
        "spectrogram_window_ms": float(spectrogram_window_ms),
        "time_modulus_ms": float(time_modulus_ms),
    }


__all__ = [
    "ConditionPairSpec",
    "DashboardTabSpec",
    "FrequencyGroupSpec",
    "build_visual_contract_snapshot",
]
