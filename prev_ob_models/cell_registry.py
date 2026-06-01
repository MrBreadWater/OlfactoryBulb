"""Registry of locally available olfactory bulb cell model families."""

from __future__ import annotations

from neuroinfra.models.registry import CellModelRegistry, CellModelSpec


def _birgiolas_specs() -> list[CellModelSpec]:
    models: list[CellModelSpec] = []
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


CELL_MODEL_SPECS: tuple[CellModelSpec, ...] = tuple(
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
        CellModelSpec(
            key="SyntheticEPL2026.PVCRH_FSI1",
            family="SyntheticEPL2026",
            role="EPLI",
            source_title="Synthetic PV/CRH-overlap EPL FSI surrogate (2026)",
            citation=(
                "Literature-constrained synthetic axonless EPL fast-spiking interneuron "
                "surrogate derived from Huang et al. (2013), Kato et al. (2013), "
                "and Burton et al. (2024)"
            ),
            module_path="prev_ob_models.SyntheticEPL2026.isolated_cells",
            class_name="PVCRH_FSI1",
            morphology_style="literature_constrained_synthetic",
            target_use="fast_epl_inhibitory_surrogate",
            network_ready=False,
            notes=(
                "Compact axonless multipolar EPL interneuron surrogate for isolated-cell "
                "and future slice-builder work. Not yet wired into the live network."
            ),
        ),
    ]
)

CELL_MODEL_ALIASES: dict[str, str] = {
    spec.class_name: spec.key
    for spec in CELL_MODEL_SPECS
    if spec.family == "Birgiolas2020"
}

DEFAULT_FAMILY_MODELS: dict[tuple[str, str], str] = {
    ("Birgiolas2020", "MC"): "Birgiolas2020.MC1",
    ("Birgiolas2020", "TC"): "Birgiolas2020.TC1",
    ("Birgiolas2020", "GC"): "Birgiolas2020.GC1",
    ("Short2016", "PGC"): "Short2016.PGC",
    ("Short2016", "ETC"): "Short2016.ETC",
    ("LiCleland2013", "PGC"): "LiCleland2013.PGC",
    ("LiCleland2013", "GC"): "LiCleland2013.GC",
    ("SyntheticEPL2026", "EPLI"): "SyntheticEPL2026.PVCRH_FSI1",
}

CELL_MODEL_REGISTRY = CellModelRegistry(
    CELL_MODEL_SPECS,
    aliases=CELL_MODEL_ALIASES,
    default_models=DEFAULT_FAMILY_MODELS,
)

CELL_MODELS_BY_KEY = CELL_MODEL_REGISTRY.by_key


def canonical_cell_model_key(cell_model: str) -> str:
    return CELL_MODEL_REGISTRY.canonical_model_key(cell_model)


def get_cell_model_spec(cell_model: str) -> CellModelSpec:
    return CELL_MODEL_REGISTRY.get_spec(cell_model)


def load_cell_class(cell_model: str):
    return CELL_MODEL_REGISTRY.load_class(cell_model)


def instantiate_cell(cell_model: str):
    return CELL_MODEL_REGISTRY.instantiate(cell_model)


def list_cell_models(
    *,
    family: str | None = None,
    role: str | None = None,
    target_use: str | None = None,
    network_ready: bool | None = None,
) -> list[CellModelSpec]:
    return CELL_MODEL_REGISTRY.list_models(
        family=family,
        role=role,
        target_use=target_use,
        network_ready=network_ready,
    )


def list_cell_families() -> list[str]:
    return CELL_MODEL_REGISTRY.list_families()


def resolve_family_role_model(family: str, role: str) -> CellModelSpec:
    return CELL_MODEL_REGISTRY.resolve_family_role_model(family, role)


def resolve_cell_choice(
    *,
    model: str | None = None,
    family: str | None = None,
    role: str | None = None,
) -> CellModelSpec:
    return CELL_MODEL_REGISTRY.resolve_cell_choice(model=model, family=family, role=role)


def describe_cell_models(
    specs: list[CellModelSpec] | tuple[CellModelSpec, ...] | None = None,
) -> list[dict[str, str]]:
    return list(CELL_MODEL_REGISTRY.describe_models(specs))


def list_fast_inhibitory_proxy_models() -> list[CellModelSpec]:
    return CELL_MODEL_REGISTRY.list_by_target_use("fast_inhibitory_proxy")


__all__ = [
    "CELL_MODEL_ALIASES",
    "CELL_MODEL_REGISTRY",
    "CELL_MODEL_SPECS",
    "CELL_MODELS_BY_KEY",
    "DEFAULT_FAMILY_MODELS",
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
