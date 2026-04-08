import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from mpi4py import MPI


repo_root = os.environ.get("OB_DEBUG_REPO_ROOT")
if repo_root:
    repo_root = str(Path(repo_root).resolve())
    os.chdir(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from LFPsimpy import SectionLfpLineMethod
from olfactorybulb.model import OlfactoryBulb
from olfactorybulb.output_paths import configure_output_env


def main():
    label, _timestamp = configure_output_env(
        os.environ.get("OB_DEBUG_LABEL", "debug_lfp_cell_segments"),
        comm=MPI.COMM_WORLD,
    )
    tstop = float(os.environ.get("OB_DEBUG_TSTOP_MS", "0.1"))
    sample_dt = float(os.environ.get("OB_DEBUG_SAMPLE_DT_MS", str(tstop)))
    target_rank = int(os.environ.get("OB_DEBUG_TARGET_RANK", "0"))
    target_cell = os.environ["OB_DEBUG_TARGET_CELL"]

    ob = OlfactoryBulb("GammaSignature", autorun=False)
    ob.results_dir = os.path.join("results", label)
    ob.params.parallel_timeout = 0.0
    ob.params.enable_lfp = False
    ob.params.legacy_parallel_dt = True
    ob.params.coreneuron = SimpleNamespace(enable=False, gpu=False, file_mode=False, verbose=0)

    h = ob.h
    pc = ob.pc

    electrode_location = getattr(ob.params, "lfp_electrode_location", (116, 1078, -61))
    electrode_kwargs = getattr(
        ob,
        "_electrode_kwargs",
        {"x": electrode_location[0], "y": electrode_location[1], "z": electrode_location[2]},
    )
    geom = SimpleNamespace(
        h=h,
        elec_x=electrode_kwargs["x"],
        elec_y=electrode_kwargs["y"],
        elec_z=electrode_kwargs["z"],
    )

    iter_cell_models = getattr(ob, "iter_cell_models", None)
    if iter_cell_models is None:
        def iter_cell_models():
            for cells in ob.cells.values():
                for cell_model in cells:
                    yield cell_model

    get_cell_sections = getattr(ob, "get_cell_sections", None)
    if get_cell_sections is None:
        def get_cell_sections(cell_model):
            sec_list = h.SectionList()
            sec_list.wholetree(sec=cell_model.soma)
            return list(sec_list)

    get_cell_name = getattr(ob, "get_cell_name", None)
    if get_cell_name is None:
        def get_cell_name(cell_model):
            return cell_model.soma.name().split(".", 1)[0]

    resolve_segment = getattr(ob, "resolve_segment", None)
    if resolve_segment is None:
        def resolve_segment(seg_name):
            return eval(seg_name, {"h": h})

    target_sections = []
    if int(ob.mpirank) == target_rank:
        for cell_model in iter_cell_models():
            cell_name = get_cell_name(cell_model)
            if cell_name != target_cell:
                continue
            for sec in get_cell_sections(cell_model):
                tr = float(SectionLfpLineMethod(geom, sec).transfer_resistance)
                segs = []
                for seg in sec:
                    seg_addr = f"h.{sec.name()}({float(seg.x):.15g})"
                    segs.append(
                        {
                            "section": sec.name(),
                            "segment": seg_addr,
                            "x": float(seg.x),
                            "node_index": int(seg.node_index()),
                            "factor": tr,
                        }
                    )
                target_sections.append({"section": sec.name(), "factor": tr, "segments": segs})
            break

    h.tstop = tstop
    pc.setup_transfer()
    h.cvode.use_fast_imem(1)
    h.cvode_active(0)
    h.dt = ob.params.sim_dt
    pc.set_maxstep(1)
    h.stdinit()

    gathered_samples = []
    for target in np.arange(sample_dt, tstop + 1e-12, sample_dt):
        pc.psolve(float(target))
        local_sample = None
        if int(ob.mpirank) == target_rank and target_sections:
            sections = []
            total = 0.0
            for sec_info in target_sections:
                seg_entries = []
                sec_total = 0.0
                for seg_info in sec_info["segments"]:
                    seg = resolve_segment(seg_info["segment"])
                    current = float(seg.i_membrane_)
                    contribution = seg_info["factor"] * current
                    sec_total += contribution
                    seg_entries.append(
                        {
                            "section": seg_info["section"],
                            "segment": seg_info["segment"],
                            "x": seg_info["x"],
                            "node_index": seg_info["node_index"],
                            "current": current,
                            "factor": seg_info["factor"],
                            "contribution": contribution,
                        }
                    )
                total += sec_total
                sections.append(
                    {
                        "section": sec_info["section"],
                        "factor": sec_info["factor"],
                        "section_total": sec_total,
                        "segments": seg_entries,
                    }
                )
            local_sample = {
                "rank": int(ob.mpirank),
                "cell": target_cell,
                "t": float(h.t),
                "dt": float(h.dt),
                "total": total,
                "sections": sections,
            }

        gathered = pc.py_gather(local_sample, 0)
        if int(ob.mpirank) == 0:
            sample = next((item for item in gathered if item is not None), None)
            gathered_samples.append(sample)

    if int(ob.mpirank) == 0:
        results_dir = Path("results") / label
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "segments.json").write_text(json.dumps(gathered_samples, indent=2))
        print({"samples": len(gathered_samples), "target_rank": target_rank, "target_cell": target_cell})


if __name__ == "__main__":
    main()
