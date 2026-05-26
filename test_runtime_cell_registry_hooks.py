"""Import-level smoke tests for configurable cell/runtime hooks.

Run with:
    conda run -n OBGPU python test_runtime_cell_registry_hooks.py
"""

from types import SimpleNamespace

from olfactorybulb.epli import DEFAULT_EPLI_SYNAPSE_SET_NAMES
from olfactorybulb.model import resolve_cell_factory
from olfactorybulb.model import OlfactoryBulb
from olfactorybulb.paramsets.base import ParameterSetBase


params = ParameterSetBase()

assert params.cell_types == ["MC", "GC", "TC"]
assert params.chemical_synapse_sets == ["GCs__MCs", "GCs__TCs"]
assert params.gc_kar_synapse_sets == ["GCs__MCs", "GCs__TCs"]
assert params.gc_output_event_synapse_sets == ["GCs__MCs", "GCs__TCs"]

assert resolve_cell_factory("MC1").__name__ == "MC1"
assert resolve_cell_factory("SyntheticEPL2026.PVCRH_FSI1").__name__ == "PVCRH_FSI1"

dummy_default = SimpleNamespace(params=params)
assert OlfactoryBulb.get_configured_cell_types(dummy_default) == ["MC", "GC", "TC"]
assert OlfactoryBulb.get_configured_chemical_synapse_sets(dummy_default) == ["GCs__MCs", "GCs__TCs"]

epli_params = ParameterSetBase()
epli_params.enable_epl_interneurons = True
epli_params.max_epl_interneurons = 4
dummy_epli = SimpleNamespace(params=epli_params)
assert OlfactoryBulb.get_configured_cell_types(dummy_epli) == ["MC", "GC", "TC", "EPLI"]
assert OlfactoryBulb.get_configured_chemical_synapse_sets(dummy_epli) == ["GCs__MCs", "GCs__TCs", *DEFAULT_EPLI_SYNAPSE_SET_NAMES]

print("runtime cell registry hook smoke test: OK")
