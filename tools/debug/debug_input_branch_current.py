import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from mpi4py import MPI

repo_root = os.environ.get("OB_DEBUG_REPO_ROOT")
if repo_root:
    repo_root = str(Path(repo_root).resolve())
    os.chdir(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from olfactorybulb.model import OlfactoryBulb
from olfactorybulb.output_paths import configure_output_env


def main():
    label, _timestamp = configure_output_env(
        os.environ.get("OB_DEBUG_LABEL", "debug_input_branch_current"),
        comm=MPI.COMM_WORLD,
    )
    tstop = float(os.environ.get("OB_DEBUG_TSTOP_MS", "0.1"))
    target_rank = int(os.environ.get("OB_DEBUG_TARGET_RANK", "0"))
    target_section = os.environ.get("OB_DEBUG_TARGET_SECTION", "TC3[2].apic[2]")
    use_coreneuron = os.environ.get("OB_DEBUG_CORENEURON", "0") == "1"
    use_gpu = os.environ.get("OB_DEBUG_GPU", "0") == "1"

    ob = OlfactoryBulb("GammaSignature", autorun=False)
    h = ob.h
    pc = ob.pc
    ob.params.coreneuron = SimpleNamespace(enable=use_coreneuron, gpu=use_gpu, file_mode=False, verbose=0)
    resolve_segment = getattr(ob, "resolve_segment", None)
    if resolve_segment is None:
        def resolve_segment(seg_name):
            normalized_name = seg_name.replace("(1)", "(.999)")
            return eval(normalized_name, {"h": h})

    input_syns = []
    for item in ob.inputs:
        syn = item[0]
        seg = syn.get_segment()
        sec_name = seg.sec.name().split("h.", 1)[-1] if seg is not None else "NONE"
        if sec_name == target_section:
            input_syns.append(syn)

    pc.setup_transfer()
    h.cvode.use_fast_imem(1)
    h.cvode_active(0)
    h.dt = ob.params.sim_dt
    pc.set_maxstep(1)
    h.stdinit()
    if use_coreneuron:
        ob.prepare_corenrn_native_lfp()
    pc.psolve(tstop)

    payload = None
    if int(ob.mpirank) == target_rank:
        section = resolve_segment(f"h.{target_section}(0.999)").sec
        payload = {
            "rank": int(ob.mpirank),
            "section": target_section,
            "t": float(h.t),
            "dt": float(h.dt),
            "n_inputs": len(input_syns),
            "input_syn_i_sum": sum(float(syn.i) for syn in input_syns),
            "input_syn_is": [float(syn.i) for syn in input_syns],
            "input_syn_gmax_like": [float(getattr(syn, "i", 0.0)) for syn in input_syns],
            "section_v_segments": [float(seg.v) for seg in section],
            "section_imem_sum": sum(float(seg.i_membrane_) for seg in section),
            "section_imem_segments": [float(seg.i_membrane_) for seg in section],
        }

    gathered = pc.py_gather(payload, 0)
    if int(ob.mpirank) == 0:
        result = next((item for item in gathered if item is not None), None)
        out = Path("results") / label / "branch_current.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(result)

    pc.barrier()
    sys.exit(0)


if __name__ == "__main__":
    main()
