"""Import-level smoke tests for configurable cell/runtime hooks.

Run with:
    conda run -n OBGPU python test_runtime_cell_registry_hooks.py
"""

from olfactorybulb.model import resolve_cell_factory
from olfactorybulb.paramsets.base import ParameterSetBase


params = ParameterSetBase()

assert params.cell_types == ["MC", "GC", "TC"]
assert params.chemical_synapse_sets == ["GCs__MCs", "GCs__TCs"]
assert params.gc_kar_synapse_sets == ["GCs__MCs", "GCs__TCs"]
assert params.gc_output_event_synapse_sets == ["GCs__MCs", "GCs__TCs"]

assert resolve_cell_factory("MC1").__name__ == "MC1"
assert resolve_cell_factory("SyntheticEPL2026.PVCRH_FSI1").__name__ == "PVCRH_FSI1"

print("runtime cell registry hook smoke test: OK")
