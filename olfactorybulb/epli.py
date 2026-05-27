"""Shared constants and helpers for the optional EPL fast interneuron population.

The current maintained network remains MC/TC/GC-only by default. This module
defines one consistent naming/configuration layer for the future opt-in EPLI
population so runtime loading and slice building do not drift apart.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from prev_ob_models.cell_registry import resolve_cell_choice

EPLI_CELL_TYPE = "EPLI"
EPLI_GROUP_NAME = "EPLIs"
DEFAULT_EPLI_MODEL_KEY = "SyntheticEPL2026.PVCRH_FSI1"
DEFAULT_EPLI_FAMILY = "SyntheticEPL2026"
DEFAULT_EPLI_SYNAPSE_SET_NAMES = ("EPLIs__MCs", "EPLIs__TCs")
DEFAULT_EPLI_GROUP_COLOR = [0.57, 0.93, 0.37]
PRINCIPAL_PERISOMATIC_SELECTOR = "@principal_perisomatic"


def unique_extend(base_items: Sequence[str], extra_items: Iterable[str]) -> list[str]:
    """Return ``base_items`` plus missing extras in first-seen order."""
    result = list(base_items)
    for item in extra_items:
        if item not in result:
            result.append(item)
    return result


def epli_population_enabled(
    *,
    enable_epl_interneurons: bool = False,
    max_epl_interneurons: int | None = 0,
) -> bool:
    """Return whether the optional EPLI population should be present."""
    return bool(enable_epl_interneurons) and int(max_epl_interneurons or 0) > 0


def resolve_epli_model_spec(*, model: str | None = None, family: str | None = None):
    """Resolve the configured EPLI cell model choice through the registry."""
    if model is None and family is None:
        model = DEFAULT_EPLI_MODEL_KEY
    if family is None and model is None:
        family = DEFAULT_EPLI_FAMILY
    return resolve_cell_choice(model=model, family=family, role=EPLI_CELL_TYPE)


def epli_root_name_pattern(*, model: str | None = None, family: str | None = None) -> str:
    """Return the lower-case BlenderNEURON root-name pattern for the configured EPLI model."""
    spec = resolve_epli_model_spec(model=model, family=family)
    return f"{spec.class_name.lower()}*"


def extend_runtime_cell_types(
    base_cell_types: Sequence[str],
    *,
    enable_epl_interneurons: bool = False,
    max_epl_interneurons: int | None = 0,
    epl_interneuron_cell_type: str = EPLI_CELL_TYPE,
) -> list[str]:
    """Append the opt-in EPLI runtime cell type when enabled."""
    if not epli_population_enabled(
        enable_epl_interneurons=enable_epl_interneurons,
        max_epl_interneurons=max_epl_interneurons,
    ):
        return list(base_cell_types)
    return unique_extend(base_cell_types, [str(epl_interneuron_cell_type)])


def extend_runtime_synapse_sets(
    base_synapse_sets: Sequence[str],
    *,
    enable_epl_interneurons: bool = False,
    max_epl_interneurons: int | None = 0,
    epl_interneuron_synapse_sets: Sequence[str] = DEFAULT_EPLI_SYNAPSE_SET_NAMES,
) -> list[str]:
    """Append EPLI reciprocal synapse-set names when the population is enabled."""
    if not epli_population_enabled(
        enable_epl_interneurons=enable_epl_interneurons,
        max_epl_interneurons=max_epl_interneurons,
    ):
        return list(base_synapse_sets)
    return unique_extend(base_synapse_sets, list(epl_interneuron_synapse_sets))


def default_slice_group_names(*, include_epli: bool = False) -> list[str]:
    """Return the canonical slice-builder group names."""
    groups = ["MCs", "TCs", "GCs"]
    if include_epli:
        groups.append(EPLI_GROUP_NAME)
    return groups


def default_slice_group_colors(*, include_epli: bool = False) -> dict[str, list[float]]:
    """Return display colors for canonical slice-builder groups."""
    colors = {
        "MCs": [0.15, 0.71, 0.96],
        "TCs": [1.0, 0.73, 0.82],
        "GCs": [1.0, 0.80, 0.11],
    }
    if include_epli:
        colors[EPLI_GROUP_NAME] = list(DEFAULT_EPLI_GROUP_COLOR)
    return colors


def default_slice_synapse_blueprints(*, include_epli: bool = False) -> list[dict[str, object]]:
    """Return canonical reciprocal synapse-set definitions for slice building."""
    blueprints = [
        {
            "group_from": "GCs",
            "group_to": "MCs",
            "max_distance": 5,
            "section_pattern_source": "*apic*",
            "section_pattern_dest": "*dend*",
            "synapse_name_dest": "GabaSyn",
            "synapse_params_dest": {"gmax": 0.005, "tau1": 1, "tau2": 100},
            "is_reciprocal": True,
            "synapse_name_source": "AmpaNmdaSyn",
            "synapse_params_source": {"gmax": 0.1},
            "create_spines": False,
            "spine_neck_diameter": 0.2,
            "spine_head_diameter": 1,
            "spine_name_prefix": "Spine",
            "conduction_velocity": 1,
            "initial_weight": 1,
            "threshold": 0,
        },
        {
            "group_from": "GCs",
            "group_to": "TCs",
            "max_distance": 5,
            "section_pattern_source": "*apic*",
            "section_pattern_dest": "*dend*",
            "synapse_name_dest": "GabaSyn",
            "synapse_params_dest": {"gmax": 0.005, "tau1": 1, "tau2": 100},
            "is_reciprocal": True,
            "synapse_name_source": "AmpaNmdaSyn",
            "synapse_params_source": {"gmax": 0.1},
            "create_spines": False,
            "spine_neck_diameter": 0.2,
            "spine_head_diameter": 1,
            "spine_name_prefix": "Spine",
            "conduction_velocity": 1,
            "initial_weight": 1,
            "threshold": 0,
        },
    ]
    if include_epli:
        for target_group in ("MCs", "TCs"):
            blueprints.append(
                {
                    "group_from": EPLI_GROUP_NAME,
                    "group_to": target_group,
                    "max_distance": 20,
                    "section_pattern_source": "*dend*",
                    "section_pattern_dest": PRINCIPAL_PERISOMATIC_SELECTOR,
                    "synapse_name_dest": "GabaSyn",
                    "synapse_params_dest": {"gmax": 0.005, "tau1": 1, "tau2": 20},
                    "is_reciprocal": True,
                    "synapse_name_source": "AmpaNmdaSyn",
                    "synapse_params_source": {"gmax": 0.1},
                    "create_spines": False,
                    "spine_neck_diameter": 0.2,
                    "spine_head_diameter": 1,
                    "spine_name_prefix": "Spine",
                    "conduction_velocity": 1,
                    "initial_weight": 1,
                    "threshold": 0,
                }
            )
    return blueprints


__all__ = [
    "DEFAULT_EPLI_FAMILY",
    "DEFAULT_EPLI_GROUP_COLOR",
    "DEFAULT_EPLI_MODEL_KEY",
    "DEFAULT_EPLI_SYNAPSE_SET_NAMES",
    "EPLI_CELL_TYPE",
    "EPLI_GROUP_NAME",
    "PRINCIPAL_PERISOMATIC_SELECTOR",
    "default_slice_group_colors",
    "default_slice_group_names",
    "default_slice_synapse_blueprints",
    "epli_root_name_pattern",
    "epli_population_enabled",
    "extend_runtime_cell_types",
    "extend_runtime_synapse_sets",
    "resolve_epli_model_spec",
    "unique_extend",
]
