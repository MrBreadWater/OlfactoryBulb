"""Regression test for CoreNEURON native LFP gid reuse.

Run with:
    MPLCONFIGDIR=/tmp/mpl /opt/miniconda3/envs/OBGPU/bin/python test_corenrn_native_lfp_gid_reuse.py
"""

from olfactorybulb.model import OlfactoryBulb


class _DummySoma:
    def __init__(self, name: str):
        self._name = name

    def name(self) -> str:
        return self._name


class _DummyCell:
    def __init__(self, name: str):
        self.soma = _DummySoma(name)


bulb = object.__new__(OlfactoryBulb)
bulb._native_lfp_cell_gids = {}
bulb._native_lfp_gid_source = {"MC1[0]": 43367062}
bulb._next_lfp_report_gid = 1503000000

existing = _DummyCell("MC1[0].soma")
gid = OlfactoryBulb.get_cell_report_gid(bulb, existing)
assert gid == 43367062
assert bulb._next_lfp_report_gid == 1503000000

# Cached lookup should be stable.
assert OlfactoryBulb.get_cell_report_gid(bulb, existing) == 43367062

new_cell = _DummyCell("TC1[0].soma")
new_gid = OlfactoryBulb.get_cell_report_gid(bulb, new_cell)
assert new_gid == 1503000000
assert bulb._next_lfp_report_gid == 1503000001

print("coreneuron native lfp gid reuse: OK")
