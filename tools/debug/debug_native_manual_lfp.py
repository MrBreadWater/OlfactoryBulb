import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from mpi4py import MPI

from LFPsimpy import SectionLfpLineMethod
from olfactorybulb.model import OlfactoryBulb
from olfactorybulb.output_paths import configure_output_env


def main():
    label, _timestamp = configure_output_env(
        os.environ.get("OB_DEBUG_LABEL", "debug_native_manual_lfp"),
        comm=MPI.COMM_WORLD,
    )
    ob = OlfactoryBulb("GammaSignature", autorun=False)
    ob.results_dir = os.path.join("results", label)
    ob.params.coreneuron = SimpleNamespace(enable=True, gpu=False, file_mode=False, verbose=0)
    ob.params.parallel_timeout = 0.0
    ob.params.enable_lfp = True
    ob.params.legacy_parallel_dt = True

    h = ob.h
    pc = ob.pc

    tstop = float(os.environ.get("OB_DEBUG_TSTOP_MS", "1.0"))
    sample_dt = float(os.environ.get("OB_DEBUG_SAMPLE_DT_MS", "0.1"))

    geom = SimpleNamespace(
        h=h,
        elec_x=ob._electrode_kwargs["x"],
        elec_y=ob._electrode_kwargs["y"],
        elec_z=ob._electrode_kwargs["z"],
    )

    section_terms = []
    for cell_model in ob.iter_cell_models():
        for sec in ob.get_cell_sections(cell_model):
            tr = float(SectionLfpLineMethod(geom, sec).transfer_resistance)
            section_terms.append((sec, tr))

    h.tstop = tstop
    pc.setup_transfer()
    h.cvode.use_fast_imem(1)
    h.cvode_active(0)
    h.dt = ob.params.sim_dt
    pc.set_maxstep(1)
    h.stdinit()
    actual_dt = float(h.dt)
    ob.prepare_corenrn_native_lfp()

    manual_times = [0.0]
    manual_values = [0.0]

    target_times = np.arange(sample_dt, tstop + 1e-12, sample_dt)
    for target in target_times:
        pc.psolve(float(target))
        local_value = 0.0
        for sec, tr in section_terms:
            local_value += tr * sum(seg.i_membrane_ for seg in sec)
        gathered = pc.py_gather((float(h.t), local_value), 0)
        if ob.mpirank == 0:
            t = gathered[0][0]
            value = sum(v for _, v in gathered)
            manual_times.append(t)
            manual_values.append(value)

    if ob.mpirank == 0:
        results_dir = Path(ob.get_results_dir())
        with open(results_dir / "lfp_manual_debug.pkl", "wb") as f:
            pickle.dump((manual_times, manual_values), f)
        print(
            {
                "actual_dt": actual_dt,
                "manual_len": len(manual_values),
                "manual_head": manual_values[:10],
                "manual_absmax": float(np.max(np.abs(manual_values))),
            }
        )


if __name__ == "__main__":
    main()
