from olfactorybulb.paramsets.base import ParameterSetBase
from olfactorybulb.model import OlfactoryBulb

def export_model_to_hoc(model_parameters, output_filename="ob_model.hoc"):
    bulb_model = OlfactoryBulb(params=model_parameters, autorun=False)
    hoc = bulb_model.h

    file_lines = []
    section_lines = ["\n// SECTIONS"]
    geometry_lines = ["\n// GEOMETRY"]
    topology_lines = ["\n// TOPOLOGY"]
    biophysics_lines = ["\n// BIOPHYSICS"]
    synapse_lines = ["\n// SYNAPSES"]
    gap_junction_lines = ["\n// GAP JUNCTIONS"]

    # setup file header
    file_lines.append("load_file(\"stdrun.hoc\")")


    for section in hoc.allsec():
        # 1. export sections
        section_lines.append(f"create {section.name()}")

        # 2. topology
        parent = section.parentseg()
        if parent:
            topology_lines.append(f"connect {section.name()}(0), {parent.sec.name()}({parent.x})")

        # 3. geometry
        geometry_lines.append(f"{section.name()} {{")
        geometry_lines.append(f"    nseg = {section.nseg}")
        geometry_lines.append(f"    L = {section.L}")

        number_of_points = int(hoc.n3d(sec=section))
        if number_of_points > 0:
            geometry_lines.append(f"    pt3dclear()")
            for i in range(n3d):
                x = hoc.x3d(i, sec=section)
                y = hoc.y3d(i, sec=section)
                z = hoc.z3d(i, sec=section)
                d = hoc.diam3d(i, sec=section)
                geometry_lines.append(f"    pt3dadd({x}, {y}, {z}, {d})")
        else:
            geometry_lines.append(f"    diam = {section.diam}")

        geometry_lines.append("}")

        # 4. biophysics
        biophysics_lines.append(f"{section.name()} {{")
        biophysics_lines.append(f"    Ra = {section.Ra}")
        biophysics_lines.append(f"    cm = {section.cm}")

        # insert mechanisms
        for segment in section:
            for mechanism in segment:
                mechanism_name = mechanism.name()
                if mechanism_name in ['morphology', 'capacitance']:
                    continue 

                biophysics_lines.append(f"    insert {mechanism_name}")

                for variable in dir(mechanism):
                    if variable.startswith('_') or variable in ['name', 'segment']:
                        continue

                    try:
                        value = getattr(mechanism, variable)
                        if isinstance(value, (int, float)):
                            biophysics_lines.append(f"    {variable}_{mechanism_name} = {value}")
                    except:
                        pass
            break  # only need first segment for mechanism names
        biophysics_lines.append("}")

    # 5. export synapses
    synapse_lines.append("objref syn_list")
    synapse_lines.append("syn_list = new List()")

    for synapse_type in ['Exp2Syn', 'AmpaNmdaSyn', 'GabaSyn']:
        if hasattr(hoc, synapse_type):
            for i, synapse in enumerate(getattr(hoc, synapse_type)):
                segment = synapse.get_segment()
                if segment:
                    synapse_lines.append(f"// {synapse_type}[{i}]")
                    synapse_lines.append(f"{segment.sec.name()} {{")
                    synapse_lines.append(f"    objref syn_{synapse_type}_{i}")
                    synapse_lines.append(f"    syn_{synapse_type}_{i} = new {synapse_type}({segment.x})")

                    # export synapse parameters
                    for attribute in ['tau1', 'tau2', 'e', 'gmax']:
                        if hasattr(synapse, attribute):
                            synapse_lines.append(f"    syn_{synapse_type}_{i}.{attribute} = {getattr(synapse, attribute)}")
                    synapse_lines.append("}")

    # 6. export gap junctions
    for i, gap_junction in enumerate(bulb_model.gjs):
        segment = gap_junction.get_segment()
        if segment:
            gap_junction_lines.append(f"{segment.sec.name()} {{")
            gap_junction_lines.append(f"    objref gj_{i}")
            gap_junction_lines.append(f"    gj_{i} = new GapJunction({segment.x})")
            gap_junction_lines.append(f"    gj_{i}.g = {gap_junction.g}")
            gap_junction_lines.append("}")


    with open(filename, 'w') as file:
        file.write('\n'.join(file_lines))
        file.write('\n'.join(section_lines))
        file.write('\n'.join(geometry_lines))
        file.write('\n'.join(topology_lines))
        file.write('\n'.join(biophysics_lines))
        file.write('\n'.join(synapse_lines))
        file.write('\n'.join(gap_junction_lines))

    return filename

class ModelParameters(ParameterSetBase):
    tstop = 500
    rnd_seed = 42

import sys
if __name__ == "__main__":
    parameters = ModelParameters()

    try:
        output_filename = sys.argv[1]
        export_model_to_hoc(parameters, output_filename)
    except:
        export_model_to_hoc(parameters)







