"""Regression test for exact-handle CoreNEURON native LFP gid reuse.

Run with:
    MPLCONFIGDIR=/tmp/mpl /opt/miniconda3/envs/OBGPU/bin/python test_corenrn_native_lfp_gid_reuse.py
"""

from neuron import h

from olfactorybulb.model import OBNeuronNode, OlfactoryBulb


class _DummyCell:
    def __init__(self, soma):
        self.soma = soma


bulb = object.__new__(OlfactoryBulb)
bulb._native_lfp_cell_gids = {}
bulb._native_lfp_gid_source = {}
bulb._registered_source_section_gids = {}
bulb._next_lfp_report_gid = 1503000000
bulb.bn_server = OBNeuronNode(server_end="Package")

soma = h.Section(name="MC1[0].soma")
soma.nseg = 1
cell = _DummyCell(soma)

bulb._registered_source_section_gids[soma.name()] = 43367062

gid = OlfactoryBulb.get_cell_report_gid(bulb, cell)
assert gid == 43367062
assert bulb._next_lfp_report_gid == 1503000000

# Cached lookup should be stable.
assert OlfactoryBulb.get_cell_report_gid(bulb, cell) == 43367062

other_soma = h.Section(name="TC1[0].soma")
other_soma.nseg = 1
new_cell = _DummyCell(other_soma)
new_gid = OlfactoryBulb.get_cell_report_gid(bulb, new_cell)
assert new_gid == 1503000000
assert bulb._next_lfp_report_gid == 1503000001

print("coreneuron native lfp gid reuse: OK")
