"""Registry of locally available olfactory bulb cell model families.

The current network runtime is built around Birgiolas 2020 slice JSON files,
but the repo also contains several published single-cell templates that are
useful as candidate interneuron models. This registry provides a stable,
string-keyed way to discover and instantiate them without hard-coded imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CellModelSpec:
    key: str
    family: str
    role: str
    source_title: str
    citation: str
    module_path: str
    class_name: str
    morphology_style: str
    target_use: str
    network_ready: bool
    notes: str = ""

    @property
    def import_path(self) -> str:
        return f"{self.module_path}.{self.class_name}"

    def load_class(self):
        module = import_module(self.module_path)
        return getattr(module, self.class_name)

    def instantiate(self):
        return self.load_class()()


def _birgiolas_specs() -> List[CellModelSpec]:
    models: List[CellModelSpec] = []
    for role, count in (("MC", 5), ("TC", 5), ("GC", 5)):
        for index in range(1, count + 1):
            class_name = f"{role}{index}"
            models.append(
                CellModelSpec(
                    key=f"Birgiolas2020.{class_name}",
                    family="Birgiolas2020",
                    role=role,
                    source_title="Birgiolas (2020)",
                    citation="Birgiolas (2020) dissertation-derived fitted olfactory bulb cells",
                    module_path="prev_ob_models.Birgiolas2020.isolated_cells",
                    class_name=class_name,
                    morphology_style="reconstructed_fitted",
                    target_use="current_network",
                    network_ready=True,
                    notes="Current slice-builder and network JSON are built against this family.",
                )
            )
    return models


CELL_MODEL_SPECS: Tuple[CellModelSpec, ...] = tuple(
    _birgiolas_specs()
    + [
        CellModelSpec(
            key="Short2016.PGC",
            family="Short2016",
            role="PGC",
            source_title="Short et al. (2016)",
            citation="Short et al. (2016) Respiration Gates Sensory Input Responses in the Mitral Cell Layer of the Olfactory Bulb",
            module_path="prev_ob_models.Short2016.isolated_cells_obgpu",
            class_name="PGC",
            morphology_style="stylized_multicompartment",
            target_use="fast_inhibitory_proxy",
            network_ready=False,
            notes=(
                "Published glomerular inhibitory template with two dendrite/gemmule branches. "
                "Useful as a fast local inhibitory proxy, but not a true EPL PV reconstruction."
            ),
        ),
        CellModelSpec(
            key="Short2016.ETC",
            family="Short2016",
            role="ETC",
            source_title="Short et al. (2016)",
            citation="Short et al. (2016) Respiration Gates Sensory Input Responses in the Mitral Cell Layer of the Olfactory Bulb",
            module_path="prev_ob_models.Short2016.isolated_cells_obgpu",
            class_name="ETC",
            morphology_style="stylized_multicompartment",
            target_use="feedforward_excitation_control",
            network_ready=False,
            notes="Published external tufted template for glomerular feedforward excitation controls.",
        ),
        CellModelSpec(
            key="LiCleland2013.PGC",
            family="LiCleland2013",
            role="PGC",
            source_title="Li and Cleland (2013)",
            citation="Li and Cleland (2013) A two-layer biophysical model of cholinergic neuromodulation in olfactory bulb",
            module_path="prev_ob_models.LiCleland2013.isolated_cells_obgpu",
            class_name="PGC",
            morphology_style="stylized_multicompartment",
            target_use="fast_inhibitory_proxy",
            network_ready=False,
            notes="Published periglomerular template; closest local inhibitory candidate family already packaged in the repo.",
        ),
        CellModelSpec(
            key="LiCleland2013.GC",
            family="LiCleland2013",
            role="GC",
            source_title="Li and Cleland (2013)",
            citation="Li and Cleland (2013) A two-layer biophysical model of cholinergic neuromodulation in olfactory bulb",
            module_path="prev_ob_models.LiCleland2013.isolated_cells_obgpu",
            class_name="GC",
            morphology_style="stylized_multicompartment",
            target_use="published_interneuron_reference",
            network_ready=False,
            notes="Published granule template from the Li/Cleland cholinergic model.",
        ),
    ]
)

CELL_MODELS_BY_KEY: Dict[str, CellModelSpec] = {spec.key: spec for spec in CELL_MODEL_SPECS}

CELL_MODEL_ALIASES: Dict[str, str] = {
    spec.class_name: spec.key
    for spec in CELL_MODEL_SPECS
    if spec.family == "Birgiolas2020"
}

DEFAULT_FAMILY_MODELS: Dict[Tuple[str, str], str] = {
    ("Birgiolas2020", "MC"): "Birgiolas2020.MC1",
    ("Birgiolas2020", "TC"): "Birgiolas2020.TC1",
    ("Birgiolas2020", "GC"): "Birgiolas2020.GC1",
    ("Short2016", "PGC"): "Short2016.PGC",
    ("Short2016", "ETC"): "Short2016.ETC",
    ("LiCleland2013", "PGC"): "LiCleland2013.PGC",
    ("LiCleland2013", "GC"): "LiCleland2013.GC",
}


def canonical_cell_model_key(cell_model: str) -> str:
    if cell_model in CELL_MODELS_BY_KEY:
        return cell_model
    if cell_model in CELL_MODEL_ALIASES:
        return CELL_MODEL_ALIASES[cell_model]
    raise KeyError(f"Unknown cell model {cell_model!r}")


def get_cell_model_spec(cell_model: str) -> CellModelSpec:
    return CELL_MODELS_BY_KEY[canonical_cell_model_key(cell_model)]


def load_cell_class(cell_model: str):
    return get_cell_model_spec(cell_model).load_class()


def instantiate_cell(cell_model: str):
    return get_cell_model_spec(cell_model).instantiate()


def list_cell_models(
    *,
    family: Optional[str] = None,
    role: Optional[str] = None,
    target_use: Optional[str] = None,
    network_ready: Optional[bool] = None,
) -> List[CellModelSpec]:
    result = []
    for spec in CELL_MODEL_SPECS:
        if family is not None and spec.family != family:
            continue
        if role is not None and spec.role != role:
            continue
        if target_use is not None and spec.target_use != target_use:
            continue
        if network_ready is not None and spec.network_ready != network_ready:
            continue
        result.append(spec)
    return result


def list_cell_families() -> List[str]:
    return sorted({spec.family for spec in CELL_MODEL_SPECS})


def resolve_family_role_model(family: str, role: str) -> CellModelSpec:
    key = DEFAULT_FAMILY_MODELS.get((family, role))
    if key is None:
        raise KeyError(f"No default model registered for family={family!r}, role={role!r}")
    return get_cell_model_spec(key)


def resolve_cell_choice(
    *,
    model: Optional[str] = None,
    family: Optional[str] = None,
    role: Optional[str] = None,
) -> CellModelSpec:
    if model:
        return get_cell_model_spec(model)
    if family and role:
        return resolve_family_role_model(family, role)
    raise ValueError("Provide either model=<key> or both family=<family> and role=<role>")


def describe_cell_models(
    specs: Optional[Sequence[CellModelSpec]] = None,
) -> List[Mapping[str, str]]:
    specs = list(CELL_MODEL_SPECS if specs is None else specs)
    return [
        {
            "key": spec.key,
            "family": spec.family,
            "role": spec.role,
            "source_title": spec.source_title,
            "citation": spec.citation,
            "morphology_style": spec.morphology_style,
            "target_use": spec.target_use,
            "network_ready": str(spec.network_ready),
            "notes": spec.notes,
        }
        for spec in specs
    ]


def list_fast_inhibitory_proxy_models() -> List[CellModelSpec]:
    return list_cell_models(target_use="fast_inhibitory_proxy")


__all__ = [
    "CELL_MODEL_SPECS",
    "CELL_MODELS_BY_KEY",
    "CellModelSpec",
    "canonical_cell_model_key",
    "describe_cell_models",
    "get_cell_model_spec",
    "instantiate_cell",
    "list_cell_families",
    "list_cell_models",
    "list_fast_inhibitory_proxy_models",
    "load_cell_class",
    "resolve_cell_choice",
    "resolve_family_role_model",
]
