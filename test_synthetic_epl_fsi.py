"""Smoke test for the synthetic EPL fast interneuron surrogate.

Run with:
    conda run -n OBGPU python test_synthetic_epl_fsi.py
"""

from olfactorybulb.audit.neuron_protocols import simulate_soma_step_response
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
assert not hasattr(cell.soma, "gbar_Ih")

second_cell = PVCRH_FSI1()
assert str(second_cell.soma) == "PVCRH_FSI1[2].soma"

rest = simulate_soma_step_response(cell, amp_nA=0.0)
assert not rest.has_nan
assert len(rest.step_spike_times_ms) == 0

PVCRH_FSI1._instance_counter = 0
spiking_cell = PVCRH_FSI1()
fast = simulate_soma_step_response(spiking_cell, amp_nA=2.0)
assert not fast.has_nan
assert fast.step_rate_hz >= 60.0

print("synthetic EPL FSI smoke test: OK")
