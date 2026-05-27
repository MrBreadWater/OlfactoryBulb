"""Regression test for shared-node source gid aliasing.

Run with:
    MPLCONFIGDIR=/tmp/mpl /opt/miniconda3/envs/OBGPU/bin/python test_shared_source_gid_alias.py
"""

from neuron import h

from olfactorybulb.model import OBNeuronNode


node = OBNeuronNode(server_end="Package")

soma = h.Section(name="AliasCell[0].soma")
dend = h.Section(name="AliasCell[0].dend[0]")
dend.connect(soma(1.0), 0.0)
soma.nseg = 1
dend.nseg = 1

soma_end_key = node.source_handle_key(soma, 1.0)
dend_root_key = node.source_handle_key(dend, 0.0)
soma_mid_key = node.source_handle_key(soma, 0.5)

assert bool(soma_end_key == dend_root_key)
assert not bool(soma_mid_key == soma_end_key)

node.source_gid_alias_map = {
    801091747: 633119557,
}

assert node.apply_source_gid_alias(801091747) == 633119557
assert node.apply_source_gid_alias(633119557) == 633119557

print("shared source gid aliasing: OK")
