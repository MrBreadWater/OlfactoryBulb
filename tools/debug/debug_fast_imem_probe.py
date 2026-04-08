import os
import sys
import json
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
        os.environ.get("OB_DEBUG_LABEL", "debug_fast_imem_probe"),
        comm=MPI.COMM_WORLD,
    )
    tstop = float(os.environ.get("OB_DEBUG_TSTOP_MS", "0.1"))
    sample_dt = float(os.environ.get("OB_DEBUG_SAMPLE_DT_MS", str(tstop)))
    use_coreneuron = os.environ.get("OB_DEBUG_CORENEURON", "0") == "1"
    use_gpu = os.environ.get("OB_DEBUG_GPU", "0") == "1"

    ob = OlfactoryBulb("GammaSignature", autorun=False)
    ob.results_dir = os.path.join("results", label)
    ob.params.parallel_timeout = 0.0
    ob.params.enable_lfp = False
    ob.params.legacy_parallel_dt = True
    ob.params.coreneuron = SimpleNamespace(enable=use_coreneuron, gpu=use_gpu, file_mode=False, verbose=0)

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

    section_terms = []
    for cell_model in iter_cell_models():
        for sec in get_cell_sections(cell_model):
            tr = float(SectionLfpLineMethod(geom, sec).transfer_resistance)
            section_terms.append((cell_model.soma.name().split(".", 1)[0], sec, tr))

    h.tstop = tstop
    pc.setup_transfer()
    h.cvode.use_fast_imem(1)
    h.cvode_active(0)
    h.dt = ob.params.sim_dt
    pc.set_maxstep(1)
    h.stdinit()
    if use_coreneuron:
        ob.prepare_corenrn_native_lfp()
    gathered = []
    for target in np.arange(sample_dt, tstop + 1e-12, sample_dt):
        pc.psolve(float(target))

        local_total = 0.0
        local_cells = {}
        for cell_name, sec, tr in section_terms:
            sec_total = tr * sum(seg.i_membrane_ for seg in sec)
            local_total += sec_total
            local_cells[cell_name] = local_cells.get(cell_name, 0.0) + sec_total

        sample = pc.py_gather(
            {
                "rank": int(ob.mpirank),
                "t": float(h.t),
                "dt": float(h.dt),
                "local_total": float(local_total),
                "local_cells": local_cells,
            },
            0,
        )
        if ob.mpirank == 0:
            gathered.append(sample)

    if ob.mpirank == 0:
        results_dir = Path("results") / label
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "probe.json").write_text(json.dumps(gathered, indent=2, default=str))
        print(gathered)


if __name__ == "__main__":
    main()
