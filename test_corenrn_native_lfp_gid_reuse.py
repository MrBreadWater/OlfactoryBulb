"""Regression test for exact-handle CoreNEURON native LFP gid reuse.

Run with:
    MPLCONFIGDIR=/tmp/mpl /opt/miniconda3/envs/OBGPU/bin/python test_corenrn_native_lfp_gid_reuse.py
"""

from neuron import h
from types import SimpleNamespace

from olfactorybulb.model import OBNeuronNode, OlfactoryBulb, section_cell_type


class _DummyCell:
    def __init__(self, soma):
        self.soma = soma


bulb = object.__new__(OlfactoryBulb)
bulb._native_lfp_cell_gids = {}
bulb._native_lfp_gid_source = {}
bulb._registered_source_section_gids = {}
bulb._cell_type_by_model_id = {}
bulb._next_lfp_report_gid = 1503000000
bulb.bn_server = OBNeuronNode(server_end="Package")
bulb.params = SimpleNamespace(lfp_include_cell_types=None, lfp_exclude_cell_types=[])

soma = h.Section(name="MC1[0].soma")
soma.nseg = 1
cell = _DummyCell(soma)
bulb._cell_type_by_model_id[id(cell)] = "MC"

bulb._registered_source_section_gids[soma.name()] = 43367062

gid = OlfactoryBulb.get_cell_report_gid(bulb, cell)
assert gid == 43367062
assert bulb._next_lfp_report_gid == 1503000000

# Cached lookup should be stable.
assert OlfactoryBulb.get_cell_report_gid(bulb, cell) == 43367062
assert OlfactoryBulb.should_include_cell_in_lfp(bulb, cell)

other_soma = h.Section(name="TC1[0].soma")
other_soma.nseg = 1
new_cell = _DummyCell(other_soma)
bulb._cell_type_by_model_id[id(new_cell)] = "TC"
new_gid = OlfactoryBulb.get_cell_report_gid(bulb, new_cell)
assert new_gid == 1503000000
assert bulb._next_lfp_report_gid == 1503000001
assert section_cell_type("PVCRH1[0].soma") == "PVCRH"
bulb.params.lfp_exclude_cell_types = ["TC"]
assert not OlfactoryBulb.should_include_cell_in_lfp(bulb, new_cell)
bulb.params.lfp_exclude_cell_types = []
bulb.params.lfp_include_cell_types = ["MC"]
assert OlfactoryBulb.should_include_cell_in_lfp(bulb, cell)
assert not OlfactoryBulb.should_include_cell_in_lfp(bulb, new_cell)

print("coreneuron native lfp gid reuse: OK")
