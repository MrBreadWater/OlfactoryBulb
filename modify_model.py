import os
os.environ["NEURON_MODULE_OPTIONS"] = "-nogui"
from olfactorybulb.paramsets.base import ParameterSetBase
from olfactorybulb.model import OlfactoryBulb
from olfactorybulb.parse_topology import build_adjacency_list
from neuron import h
import prev_ob_models.Birgiolas2020.isolated_cells as isolated_cells
from dataclasses import dataclass, field
import json

@dataclass(frozen = True)
class SynapseKey:
    x_name: str       # source section name
    x_location: float # source section location
    y_name: str       # destination section name
    y_location: float # destination section location
    synapse_type: str # ampa or gaba synapse

    def __str__(self):
        return f"{self.x_name}({self.x_location}) =({synapse_type})=> {self.y_name}(self.y_location)"

def make_synapse_key(entry: dict, source_to_dest: bool) -> SynapseKey:
    """ where entry is a json entry """
    if source_to_dest:
        return SynapseKey(entry['source_section'], entry['source_x'], entry['dest_section'], entry['dest_x'], 'GabaSyn')
    else:
        return SynapseKey(entry['dest_section'], entry['dest_x'], entry['source_section'], entry['source_x'], 'AmpaNmdaSyn')

@dataclass
class ModifiedParameterSet(ParameterSetBase):
    topology: dict[str, list] = field(default_factory={"add_connections": [], "connections_weights": []})
    swap_cell_types: list[dict] = field(default_factory=[])

def perform_cell_type_swaps(bulb: OlfactoryBulb, swap_list: list, synapse_map: dict):
    cell_type_re = re.compile(r'^([A-Z]+)')
    for swap in swap_list:
        json_cell_id = swap['cell']
        to_class = swap['to_class']

        mapping = bulb.mpimap.get(json_cell_id)
        if mapping is None or mapping['rank'] != bulb.mpirank:
            continue

        hoc_cell_id = mapping['name']
        cell_type = cell_type_re.match(json_cell_id).group(1)

        cell_obj = None
        for cell in bulb.cells.get(cell_type, []):
            if cell.soma.name().startswith(hoc_cell_id + '.'):
                cell_obj = cell
                break

        if cell_obj is None:
            print(f"Warning: could not find live cell object for {json_cell_id} (hoc: {hoc_cell_id})")
            continue

        target_class = getattr(isolated_cells, to_class, None)
        if target_class is None:
            raise ValueError(f"Unkown cell class '{to_class}' in isolated_cells")

        temp_cell = target_class()
        cell_obj.set_model_params(temp_cell.param_values)
        del temp_cell

        cell_prefix = json_cell_id + '.'
        connected_synapses = {
                key: entry,
                for key, entry in synapse_map.items() if key.x_name.startswith(cell_prefix) or key.y_name.startswith(cell_prefix)
        }

        print(f"Swapped {json_cell_id} -> {to_class}.\nFound {len(connected_synapses)} connected synapse(s) in synapse_map.")


def add_synaptic_connection(bulb, connection_config):
    """ connection config just needs the new synapse parameters and the precell segement name and postcell segment name """
    # create the post synaptic point process on the *target section* at the desired location
    # create the netcon connecting the pre synaptic volatage to that synapse
    # keep references to both objects in python, this is critical. NEURON's garabge collector will destroy them if no python variable holds a reference, even if the simluation is still running
    h = bulb.h

    post_seg = eval('h.' + connection_config['post_section_name'] + '(0.5)')
    syn = getattr(h, connection_config['synapse_type'])(post_seg)

    for attr, val in connection_config.get('syn_params', {}).items():
        setattr(syn, attr, val)

    pre_seg = eval('h.' + connection_config['pre_section_name'] + '(0.5)')

    netcon = h.NetCon(
            pre_seg._ref_v,
            syn,
            connection_config.get('threshold', 0.0),
            connection_config.get('delay', 0.5),
            connection_config.get('weight', 1.0),
            sec=pre_seg.sec
    )

    if not hasattr(bulb, 'added_connections'):
        bulb.added_connections = []
    bulb.added_connections.append((syn, netcon))

def modify_synaptic_connection(bulb: OlfactoryBulb, synapse_map: dict, synapse_key: SynapseKey, modification: dict):
    # use synaps map and synapse key to iterate over netcons and identify the netcon which has the target synapse key pointer
    # then perform in place modifications of the netcon and synapse point process
    entry = synapse_map.get(synapse_key)
    if entry is None:
        raise KeyError(f"SynapseKey not found in synapse_map: {synapse_key}")

    netcon = get_connection_netcon(bulb, synapse_map, synapse_key)

    for attr, val in modification.get('netcon', {}).items():
        setattr(netcon, attr, val)

    syn = entry['synapse_pointer']
    for attr, val in modification.get('synapse', {}).items():
        setattr(syn, attr, val)

def get_connection_netcon(bulb: OlfactoryBulb, synapse_map: dict, key: SynapseKey):
    for netcon in bulb.h.NetCon:
        if netcon.syn() == synapse_map[key]['synapse_pointer']:
            return netcon
    raise Exception(f"Queried model netcon for non-existent synapse key: {key}")

def json_name_to_hoc_name(json_section_name, bulb):
    """
    Converts a JSON section name like 'GC3[314].apic[8]' to its rank-local
    HOC section name like 'GC3[0].apic[8]' using the bulb's mpimap.
    Returns None if the cell is not assigned to the current rank.
    """
    mpimap = bulb.mpimap
    bracket_end  = json_section_name.find(']') + 1
    json_cell_id = json_section_name[:bracket_end]     # e.g. 'GC3[314]'
    sec_part     = json_section_name[bracket_end + 1:] # e.g. 'apic[8]'
    mapping = mpimap.get(json_cell_id)

    assert mapping is not None
    assert mapping['rank'] == bulb.mpirank

    return mapping['name'] + '.' + sec_part # e.g. 'GC3[0].apic[8]'

def build_synapse_map(bulb):
    h = bulb.h

    gcs_mcs_entries = json.load(open(os.path.join(bulb.slice_dir, 'GCs__MCs.json')))['entries']
    gcs_tcs_entries = json.load(open(os.path.join(bulb.slice_dir, 'GCs__TCs.json')))['entries']

    # same order NEURON loaded them
    all_entries = gcs_mcs_entries + gcs_tcs_entries  

    gaba_syns = list(h.GabaSyn)
    ampa_syns = list(h.AmpaNmdaSyn)

    assert len(gaba_syns) == len(ampa_syns) == len(all_entries) == 3989

    synapse_map = {}
    for entry, gaba_syn, ampa_syn in zip(all_entries, gaba_syns, ampa_syns):
        gaba_key = make_synapse_key(entry, True)
        ampa_key = make_synapse_key(entry, False)

        synapse_map[gaba_key] = {
            'json_data': entry,
            'synapse_pointer': gaba_syn
        }
        synapse_map[ampa_key] = {
            'json_data': entry,
            'synapse_pointer': ampa_syn
        }

    return synapse_map

bulb = OlfactoryBulb(params=test_params, autorun=False)
topology = build_synapse_map(bulb)



    




