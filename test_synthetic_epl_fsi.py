"""Smoke test for the synthetic EPL fast interneuron surrogate.

Run with:
    conda run -n OBGPU python test_synthetic_epl_fsi.py
"""

from prev_ob_models.SyntheticEPL2026.isolated_cells import PVCRH_FSI1


PVCRH_FSI1._instance_counter = 0
cell = PVCRH_FSI1()

assert len(cell.primary_dendrites) == 4
assert len(cell.branch_dendrites) == 8
assert len(cell.dend) == 12
assert not hasattr(cell, "axon")
assert 60.0 <= cell.planar_dendritic_span_um <= 80.0
assert abs(cell.soma.L - 9.6) < 1e-6
assert abs(cell.soma.diam - 9.6) < 1e-6
assert str(cell.soma) == "PVCRH_FSI1[0].soma"
assert str(cell.primary_dendrites[0]) == "PVCRH_FSI1[0].dend_primary_0"
assert str(cell.branch_dendrites[0]) == "PVCRH_FSI1[0].dend_branch_0"

print("synthetic EPL FSI smoke test: OK")
