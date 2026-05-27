"""Import-level smoke tests for configurable cell/runtime hooks.

Run with:
    conda run -n OBGPU python test_runtime_cell_registry_hooks.py
"""

from types import SimpleNamespace

from olfactorybulb.epli import DEFAULT_EPLI_SYNAPSE_SET_NAMES
from blenderneuron.nrn.neuronnode import NeuronNode
from olfactorybulb.model import OBNeuronNode, resolve_cell_factory
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


original_create_synapses = NeuronNode.create_synapses


def fake_parent_create_synapses(self, syn_set):
    self.synapse_sets[syn_set["name"]] = [("forward_nc", "forward_syn", None, None, "recip_nc", "recip_syn")]


try:
    NeuronNode.create_synapses = fake_parent_create_synapses
    node = object.__new__(OBNeuronNode)
    node.synapse_sets = {}
    node.cell_source_gids = {}
    node.rank_section_name = lambda section_name: section_name
    node.segment_gid = lambda section_name, seg_i, create_spine: int(seg_i) + (100 if create_spine else 0)
    node.apply_source_gid_alias = lambda gid: int(gid)

    syn_set = {
        "name": "EPLIs__TCs",
        "entries": [
            {
                "source_section": "PVCRH_FSI1[0].dend_primary_0",
                "source_seg_i": 2,
                "dest_section": "TC4[0].soma",
                "dest_seg_i": 0,
                "create_spine": False,
                "is_reciprocal": True,
            }
        ],
    }
    returned_synapses = node.create_synapses(syn_set)
    assert returned_synapses is node.synapse_sets["EPLIs__TCs"]
    assert node.cell_source_gids["PVCRH_FSI1[0]"] == 2
    assert node.cell_source_gids["TC4[0]"] == 0
finally:
    NeuronNode.create_synapses = original_create_synapses


class FakeNetCon:
    def __init__(self, weight):
        self.weight = [weight]


scaling_params = SimpleNamespace(epli_gaba_weight_scale=2.5, epli_ampa_weight_scale=1.75)
scaling_dummy = object.__new__(OlfactoryBulb)
scaling_dummy.params = scaling_params
forward_nc = FakeNetCon(4.0)
reciprocal_nc = FakeNetCon(8.0)
OlfactoryBulb.apply_reciprocal_weight_scales(
    scaling_dummy,
    "EPLIs__TCs",
    [(forward_nc, None, None, None, reciprocal_nc, None)],
)
assert forward_nc.weight[0] == 10.0
assert reciprocal_nc.weight[0] == 14.0

print("runtime cell registry hook smoke test: OK")
