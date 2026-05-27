import os
import sys
import pickle
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
        os.environ.get("OB_DEBUG_LABEL", "debug_percell_lfp"),
        comm=MPI.COMM_WORLD,
    )
    tstop = float(os.environ.get("OB_DEBUG_TSTOP_MS", "1.0"))
    sample_dt = float(os.environ.get("OB_DEBUG_SAMPLE_DT_MS", "0.1"))
    use_coreneuron = os.environ.get("OB_DEBUG_CORENEURON", "0") == "1"
    use_gpu = os.environ.get("OB_DEBUG_GPU", "0") == "1"

    ob = OlfactoryBulb("GammaSignature", autorun=False)
    ob.results_dir = os.path.join("results", label)
    ob.params.parallel_timeout = 0.0
    ob.params.enable_lfp = False
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

    cell_sections = []
    for cell_model in iter_cell_models():
        cell_name = cell_model.soma.name().split(".", 1)[0]
        sec_terms = []
        for sec in get_cell_sections(cell_model):
            tr = float(SectionLfpLineMethod(geom, sec).transfer_resistance)
            sec_terms.append((sec, tr))
        cell_sections.append((cell_name, sec_terms))

    h.tstop = tstop
    pc.setup_transfer()
    h.cvode.use_fast_imem(1)
    h.cvode_active(0)
    h.dt = ob.params.sim_dt
    pc.set_maxstep(1)
    if not use_coreneuron:
        h.steps_per_ms = 1.0 / ob.params.sim_dt
        h.setdt()
    h.stdinit()
    actual_dt = float(h.dt)
    if use_coreneuron:
        ob.prepare_corenrn_native_lfp()

    local_cell_names = [cell_name for cell_name, _ in cell_sections]
    gathered_names = pc.py_gather(local_cell_names, 0)
    if ob.mpirank == 0:
        all_cell_names = []
        for rank_names in gathered_names:
            all_cell_names.extend(rank_names)
        per_cell = {cell_name: [0.0] for cell_name in sorted(all_cell_names)}
    else:
        per_cell = {}
    times = [0.0]

    target_times = np.arange(sample_dt, tstop + 1e-12, sample_dt)
    for target in target_times:
        pc.psolve(float(target))
        local = {}
        for cell_name, sec_terms in cell_sections:
            value = 0.0
            for sec, tr in sec_terms:
                value += tr * sum(seg.i_membrane_ for seg in sec)
            local[cell_name] = value
        gathered = pc.py_gather((float(h.t), local), 0)
        if ob.mpirank == 0:
            t = gathered[0][0]
            merged = {}
            for _, rank_local in gathered:
                for cell_name, value in rank_local.items():
                    merged[cell_name] = merged.get(cell_name, 0.0) + value
            times.append(t)
            for cell_name in per_cell:
                per_cell[cell_name].append(merged.get(cell_name, 0.0))

    if ob.mpirank == 0:
        get_results_dir = getattr(ob, "get_results_dir", None)
        if get_results_dir is None:
            results_dir = Path(getattr(ob, "results_dir", os.path.join("results", label)))
        else:
            results_dir = Path(get_results_dir())
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "percell_lfp_debug.pkl", "wb") as f:
            pickle.dump((times, per_cell), f)
        print(
            {
                "actual_dt": actual_dt,
                "times_len": len(times),
                "cells": len(per_cell),
                "sample_cells": list(sorted(per_cell))[:5],
            }
        )


if __name__ == "__main__":
    main()
