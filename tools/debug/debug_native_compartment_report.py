import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from mpi4py import MPI

from olfactorybulb.model import OlfactoryBulb
from olfactorybulb.output_paths import configure_output_env


def write_report_config(report_conf_path, report_name, report_type, report_on, unit, report_dt, tstop, gids):
    with open(report_conf_path, "wb") as f:
        metadata = (
            f"1\n"
            f"{report_name} All {report_type} {report_on} {unit} SONATA All Center "
            f"{float(report_dt):g} 0.0 {float(tstop):g} {len(gids)} 8 None\n"
        )
        f.write(metadata.encode("ascii"))
        if gids:
            f.write(np.asarray(gids, dtype=np.int32).tobytes())
        f.write(b"\n")
        f.write(b"1\n")
        f.write(b"All 0\n")
        f.write(b"out.h5\n")


def main():
    label, _timestamp = configure_output_env(
        os.environ.get("OB_DEBUG_LABEL", "debug_native_compartment_report"),
        comm=MPI.COMM_WORLD,
    )
    target_cell = os.environ.get("OB_DEBUG_TARGET_CELL", "TC3[2]")
    report_on = os.environ.get("OB_DEBUG_REPORT_ON", "v")
    report_type = os.environ.get("OB_DEBUG_REPORT_TYPE", "compartment")
    unit = os.environ.get("OB_DEBUG_REPORT_UNIT", "mV" if report_on == "v" else "nA")
    tstop = float(os.environ.get("OB_DEBUG_TSTOP_MS", "1.0"))
    report_dt = float(os.environ.get("OB_DEBUG_REPORT_DT_MS", "0.1"))
    use_gpu = os.environ.get("OB_DEBUG_GPU", "0") == "1"
    gid_kind = os.environ.get("OB_DEBUG_GID_KIND", "report")

    ob = OlfactoryBulb("GammaSignature", autorun=False)
    ob.results_dir = os.path.join("results", label)
    ob.params.parallel_timeout = 0.0
    ob.params.enable_lfp = True
    ob.params.legacy_parallel_dt = True
    ob.params.coreneuron = SimpleNamespace(enable=True, gpu=use_gpu, file_mode=False, verbose=0)

    if gid_kind != "source":
        ob.register_corenrn_native_lfp_mappings()

    local_target_gid = None
    if gid_kind == "source":
        local_target_gid = ob.bn_server.cell_source_gids.get(target_cell)
    else:
        for cell_model in ob.iter_cell_models():
            if ob.get_cell_name(cell_model) == target_cell:
                local_target_gid = ob.get_cell_report_gid(cell_model)
                break
    results_dir = Path(ob.get_results_dir())
    gathered_target_gids = ob.pc.py_gather(local_target_gid, 0)
    if ob.mpirank == 0:
        target_gid = next((gid for gid in gathered_target_gids if gid is not None), None)
        if target_gid is None:
            raise KeyError(target_cell)
        results_dir.mkdir(parents=True, exist_ok=True)
        report_conf = results_dir / f"{report_name(report_on)}.report.conf"
        sim_conf = results_dir / f"{report_name(report_on)}.sim.conf"
        write_report_config(report_conf, f"{report_name(report_on)}.tsv", report_type, report_on, unit, report_dt, tstop, [target_gid])
        with open(sim_conf, "w") as f:
            f.write(f"outpath='{results_dir}'\n")
            f.write(f"report-conf='{report_conf}'\n")
    else:
        sim_conf = results_dir / f"{report_name(report_on)}.sim.conf"

    ob.pc.barrier()

    from neuron import coreneuron

    coreneuron.enable = True
    coreneuron.gpu = use_gpu
    coreneuron.file_mode = False
    coreneuron.verbose = 0
    coreneuron.cell_permute = 1 if use_gpu else 0
    coreneuron.sim_config = str(sim_conf)
    h = ob.h
    pc = ob.pc
    h.tstop = tstop
    pc.setup_transfer()
    h.cvode.use_fast_imem(1)
    h.cvode_active(0)
    h.dt = ob.params.sim_dt
    pc.set_maxstep(1)
    h.stdinit()
    pc.psolve(tstop)

    if ob.mpirank == 0:
        report_path = results_dir / f"{report_name(report_on)}.tsv"
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
                for _ in range(20):
                    line = f.readline()
                    if not line:
                        break
                    print(line.rstrip())
        else:
            print({"error": "report file not found", "path": str(report_path)})


def report_name(report_on):
    return f"{report_on}_native"


if __name__ == "__main__":
    main()
