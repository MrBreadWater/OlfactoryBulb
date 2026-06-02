"""Smoke tests for the extracted neuroinfra cell-model registry layer."""

from __future__ import annotations

import sys
import types

from neuroinfra.models.registry import CellModelRegistry, CellModelSpec
from prev_ob_models.cell_registry import (
    CELL_MODEL_REGISTRY,
    canonical_cell_model_key,
    describe_cell_models,
    get_cell_model_spec,
    list_cell_families,
    list_fast_inhibitory_proxy_models,
    resolve_family_role_model,
)


def main() -> None:
    dummy_module = types.ModuleType("dummy_registry_module")

    class ExampleCell:
        pass

    class ExampleOtherCell:
        pass

    dummy_module.ExampleCell = ExampleCell
    dummy_module.ExampleOtherCell = ExampleOtherCell
    sys.modules[dummy_module.__name__] = dummy_module
    try:
        specs = (
            CellModelSpec(
                key="DummyFamily.ExampleCell",
                family="DummyFamily",
                role="IN",
                source_title="Dummy source",
                citation="Dummy citation",
                module_path=dummy_module.__name__,
                class_name="ExampleCell",
                morphology_style="synthetic",
                target_use="unit_test",
                network_ready=False,
            ),
            CellModelSpec(
                key="DummyFamily.ExampleOtherCell",
                family="DummyFamily",
                role="PN",
                source_title="Dummy source",
                citation="Dummy citation",
                module_path=dummy_module.__name__,
                class_name="ExampleOtherCell",
                morphology_style="synthetic",
                target_use="unit_test",
                network_ready=True,
            ),
        )
        registry = CellModelRegistry(
            specs,
            aliases={"ExampleCell": "DummyFamily.ExampleCell"},
            default_models={("DummyFamily", "IN"): "DummyFamily.ExampleCell"},
        )
        assert registry.canonical_model_key("ExampleCell") == "DummyFamily.ExampleCell"
        assert registry.get_spec("DummyFamily.ExampleCell").class_name == "ExampleCell"
        assert registry.load_class("ExampleCell") is ExampleCell
        assert isinstance(registry.instantiate("ExampleCell"), ExampleCell)
        assert registry.list_families() == ["DummyFamily"]
        assert registry.resolve_family_role_model("DummyFamily", "IN").class_name == "ExampleCell"
        assert registry.resolve_cell_choice(model="ExampleCell").class_name == "ExampleCell"
        assert registry.resolve_cell_choice(family="DummyFamily", role="IN").class_name == "ExampleCell"
        assert len(registry.list_models(target_use="unit_test")) == 2
        description = registry.describe_models()
        assert description[0]["key"] == "DummyFamily.ExampleCell"
        assert description[1]["network_ready"] == "True"
    finally:
        sys.modules.pop(dummy_module.__name__, None)

    assert canonical_cell_model_key("MC1") == "Birgiolas2020.MC1"
    assert get_cell_model_spec("MC1") is CELL_MODEL_REGISTRY.get_spec("MC1")
    assert resolve_family_role_model("Birgiolas2020", "MC").key == "Birgiolas2020.MC1"
    assert "Birgiolas2020" in list_cell_families()
    assert any(spec.role == "PGC" for spec in list_fast_inhibitory_proxy_models())
    described = describe_cell_models(CELL_MODEL_REGISTRY.list_models(family="SyntheticEPL2026"))
    assert described[0]["key"] == "SyntheticEPL2026.PVCRH_FSI1"
    print("neuroinfra model registry smoke test: OK")


if __name__ == "__main__":
    main()
