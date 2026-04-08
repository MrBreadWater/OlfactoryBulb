import os
os.environ["NEURON_MODULE_OPTIONS"] = "-nogui"
from olfactorybulb.paramsets.base import ParameterSetBase
from olfactorybulb.model import OlfactoryBulb
from olfactorybulb.parse_topology import build_adjacency_list
from neuron import h
import prev_ob_models.Birgiolas2020.isolated_cells as isolated_cells
from dataclasses import dataclass, field
import json

@dataclass
class ModifiedParameterSet(ParameterSetBase):
    topology: dict[str, list] = field(default_factory={"add_connections": [], "connections_weights": []})
    swap_cell_types: list[dict] = field(default_factory=[])

test_params = ModifiedParameterSet(topology={
        "add_connections": [
            {
                "pre_cell_type":  "MC",
                "pre_cell_index": 0,        # MC5[0]
                "post_cell_type": "GC",
                "post_cell_index": 100,     # GC5[26]
                "synapse_type":   "Exp2Syn",
                "weight":         0.05,
                "delay":          1.0,
                "post_section":   "GC5[26].apic[8]",
                "post_loc":       0.5,
            }
        ],
        "remove_connections": [
            {
                "pre_cell_type":   "MC",
                "pre_cell_index":  7,       # MC5[14]
                "post_cell_type":  "GC",
                "post_cell_index": 100,     # GC5[26]
                "synapse_type":    "AmpaNmdaSyn",
            }
        ]
    }, 
    swap_cell_types=[
        {
            "original_cell_type": "MC",
            "cell_index":         0,        # MC5[0]
            "replacement_type":   "MC3",
        }
    ])

def perform_cell_type_swaps(bulb: OlfactoryBulb, swap_list: list, adj_list: dict):
    pass

def add_connection(bulb, connection_config, adj_list: dict, update_adj_list: bool = True):
    pass

def set_connection_weight(bulb, remove_config, adj_list: dict, update_adj_list: bool = True):
    pass

def build_synapse_map(bulb):
    """
    bulb: an initialized OlfactoryBulb instance
    Returns a dict mapping each live NEURON synapse object → its JSON entry
    """

    h = bulb.h
    # ── 1. Build inverse mpimap: rank-local cell name → original JSON cell name ──
    inv_mpimap = {}
    for json_cell_name, info in bulb.mpimap.items():
        if info['rank'] == bulb.mpirank:
            inv_mpimap[info['name']] = json_cell_name
    # e.g. inv_mpimap["GC3[0]"] = "GC3[314]"
    def to_json_section(rank_section_name):
        """Convert a rank-local section name like 'GC3[0].apic[8]' → 'GC3[314].apic[8]'"""
        cell_id  = rank_section_name[:rank_section_name.find(']') + 1]   # 'GC3[0]'
        sec_part = rank_section_name[rank_section_name.find('.') + 1:]   # 'apic[8]'
        json_cell = inv_mpimap.get(cell_id, cell_id)                     # 'GC3[314]'
        return json_cell + '.' + sec_part
    # ── 2. Load JSON entries and index by (dest_section, rounded dest_x) ──
    syn_sets = []
    for fname in ['GCs__MCs.json', 'GCs__TCs.json']:
        with open(bulb.slice_dir + '/' + fname) as f:
            syn_sets.extend(json.load(f)['entries'])
    # Index by the GabaSyn (dest) side — section name + position
    gaba_index = {}
    for entry in syn_sets:
        key = (entry['dest_section'], round(entry['dest_x'], 4))
        gaba_index[key] = entry
    # Index by the AmpaNmda (source) side
    ampa_index = {}
    for entry in syn_sets:
        key = (entry['source_section'], round(entry['source_x'], 4))
        ampa_index[key] = entry
    # ── 3. Walk live synapse objects and cross-reference ──
    synapse_map = {}
    for syn in h.GabaSyn:          # inhibitory synapses sitting on MC/TC dendrites
        seg      = syn.get_segment()
        sec_name = to_json_section(seg.sec.name())
        key      = (sec_name, round(seg.x, 4))
        entry    = gaba_index.get(key)
        if entry:
            synapse_map[syn] = {
                'type': 'GabaSyn (GC→MC/TC inhibition)',
                'json_entry': entry,
                'pre_json_section':  entry['source_section'],  # GC apic
                'post_json_section': entry['dest_section'],     # MC/TC dend
            }
    for syn in h.AmpaNmdaSyn:      # excitatory synapses sitting on GC apic dendrites
        seg      = syn.get_segment()
        sec_name = to_json_section(seg.sec.name())
        key      = (sec_name, round(seg.x, 4))
        entry    = ampa_index.get(key)
        if entry:
            synapse_map[syn] = {
                'type': 'AmpaNmdaSyn (MC/TC→GC excitation)',
                'json_entry': entry,
                'pre_json_section':  entry['dest_section'],    # MC/TC dend (the driver)
                'post_json_section': entry['source_section'],  # GC apic (the receiver)
            }
    return synapse_map

bulb = OlfactoryBulb(params=test_params, autorun=False)





