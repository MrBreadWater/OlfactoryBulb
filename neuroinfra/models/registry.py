"""Generic registry primitives for discoverable neuroscience cell models."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class CellModelSpec:
    """One discoverable cell-model definition in a neuroscience model catalog."""

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


class CellModelRegistry:
    """Generic registry for string-keyed cell-model catalogs."""

    def __init__(
        self,
        specs: Sequence[CellModelSpec],
        *,
        aliases: Mapping[str, str] | None = None,
        default_models: Mapping[tuple[str, str], str] | None = None,
    ) -> None:
        self._specs = tuple(specs)
        self._by_key = {spec.key: spec for spec in self._specs}
        self._aliases = dict(aliases or {})
        self._default_models = dict(default_models or {})

    @property
    def specs(self) -> tuple[CellModelSpec, ...]:
        return self._specs

    @property
    def by_key(self) -> dict[str, CellModelSpec]:
        return dict(self._by_key)

    @property
    def aliases(self) -> dict[str, str]:
        return dict(self._aliases)

    @property
    def default_models(self) -> dict[tuple[str, str], str]:
        return dict(self._default_models)

    def canonical_model_key(self, cell_model: str) -> str:
        if cell_model in self._by_key:
            return cell_model
        if cell_model in self._aliases:
            return self._aliases[cell_model]
        raise KeyError(f"Unknown cell model {cell_model!r}")

    def get_spec(self, cell_model: str) -> CellModelSpec:
        return self._by_key[self.canonical_model_key(cell_model)]

    def load_class(self, cell_model: str):
        return self.get_spec(cell_model).load_class()

    def instantiate(self, cell_model: str):
        return self.get_spec(cell_model).instantiate()

    def list_models(
        self,
        *,
        family: str | None = None,
        role: str | None = None,
        target_use: str | None = None,
        network_ready: bool | None = None,
    ) -> list[CellModelSpec]:
        result = []
        for spec in self._specs:
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

    def list_families(self) -> list[str]:
        return sorted({spec.family for spec in self._specs})

    def resolve_family_role_model(self, family: str, role: str) -> CellModelSpec:
        key = self._default_models.get((family, role))
        if key is None:
            raise KeyError(f"No default model registered for family={family!r}, role={role!r}")
        return self.get_spec(key)

    def resolve_cell_choice(
        self,
        *,
        model: str | None = None,
        family: str | None = None,
        role: str | None = None,
    ) -> CellModelSpec:
        if model:
            return self.get_spec(model)
        if family and role:
            return self.resolve_family_role_model(family, role)
        raise ValueError("Provide either model=<key> or both family=<family> and role=<role>")

    def describe_models(
        self,
        specs: Sequence[CellModelSpec] | None = None,
    ) -> list[Mapping[str, str]]:
        selected_specs = list(self._specs if specs is None else specs)
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
            for spec in selected_specs
        ]

    def list_by_target_use(self, target_use: str) -> list[CellModelSpec]:
        return self.list_models(target_use=target_use)
