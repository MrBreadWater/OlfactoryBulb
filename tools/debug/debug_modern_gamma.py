import argparse
import json
import os
import time
from pathlib import Path

from mpi4py import MPI

import olfactorybulb.model as obmodel
from olfactorybulb.model import OlfactoryBulb
from olfactorybulb.output_paths import label_with_timestamp, sync_timestamp


def log(rank, phase, extra=None):
    payload = {"rank": rank, "phase": phase}
    if extra:
        payload.update(extra)
    print(json.dumps(payload), flush=True)


def main():
    from neuron import coreneuron

    parser = argparse.ArgumentParser()
    parser.add_argument("--paramset", default="GammaSignature")
    parser.add_argument("--tstop", type=float, default=None)
    parser.add_argument("--skip-save", action="store_true")
    parser.add_argument("--skip-lfp", action="store_true")
    parser.add_argument("--disable-inputs", action="store_true")
    parser.add_argument("--mc-gap", type=float, default=None)
    parser.add_argument("--tc-gap", type=float, default=None)
    parser.add_argument("--sparse-partrans", action="store_true")
    parser.add_argument("--results-dir", default="results/debug_modern")
    parser.add_argument("--coreneuron", action="store_true")
    parser.add_argument("--coreneuron-gpu", action="store_true")
    parser.add_argument("--coreneuron-file-mode", action="store_true")
    parser.add_argument("--coreneuron-verbose", type=int, default=0)
    parser.add_argument("--disable-status-report", action="store_true")
    parser.add_argument("--disable-lfp-electrode", action="store_true")
    parser.add_argument("--parallel-timeout", type=float, default=None)
    args, _unknown = parser.parse_known_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()
    run_timestamp = sync_timestamp(comm=comm)
    results_dir_path = Path(args.results_dir)
    args.results_dir = str(
        results_dir_path.parent / label_with_timestamp(results_dir_path.name, run_timestamp)
    )

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)

    params_cls = getattr(obmodel, args.paramset)
    params = params_cls()
    if args.disable_inputs:
        params.input_odors = {}
    if args.mc_gap is not None:
        params.gap_juction_gmax["MC"] = args.mc_gap
    if args.tc_gap is not None:
        params.gap_juction_gmax["TC"] = args.tc_gap
    if args.tstop is not None:
        params.tstop = args.tstop
    if args.disable_status_report:
        params.enable_status_report = False
    if args.disable_lfp_electrode:
        params.enable_lfp = False
    if args.parallel_timeout is not None:
        params.parallel_timeout = args.parallel_timeout

    log(
        rank,
        "build_start",
        {
            "nranks": nranks,
            "disable_inputs": args.disable_inputs,
            "mc_gap": params.gap_juction_gmax.get("MC"),
            "tc_gap": params.gap_juction_gmax.get("TC"),
            "tstop": params.tstop,
        },
    )
    build_start = time.perf_counter()
    ob = OlfactoryBulb(params, autorun=False)
    if args.sparse_partrans and hasattr(ob.h, "nrn_sparse_partrans"):
        ob.h.nrn_sparse_partrans = 1
    build_elapsed = time.perf_counter() - build_start
    log(rank, "build_done", {"seconds": build_elapsed, "local_cells": {k: len(v) for k, v in ob.cells.items()}})

    if rank == 0:
        ob.results_dir = args.results_dir
    comm.Barrier()

    coreneuron.enable = args.coreneuron
    coreneuron.gpu = args.coreneuron_gpu
    coreneuron.file_mode = args.coreneuron_file_mode
    coreneuron.verbose = args.coreneuron_verbose
    coreneuron.cell_permute = 1 if args.coreneuron_gpu else 0
    log(
        rank,
        "coreneuron_config",
        {
            "enable": coreneuron.enable,
            "gpu": coreneuron.gpu,
            "file_mode": coreneuron.file_mode,
            "verbose": coreneuron.verbose,
            "cell_permute": coreneuron.cell_permute,
        },
    )

    log(rank, "run_start", {"tstop": ob.params.tstop})
    run_start = time.perf_counter()
    ob.run(ob.params.tstop)
    run_elapsed = time.perf_counter() - run_start
    log(rank, "run_done", {"seconds": run_elapsed, "t": ob.h.t})

    if not args.skip_save:
        log(
            rank,
            "save_start",
            {
                "v_vectors": len(ob.v_vectors),
                "input_vectors": len(ob.input_vectors),
                "t_vec_len": int(ob.t_vec.size()),
            },
        )
        save_start = time.perf_counter()
        ob.save_recorded_vectors()
        save_elapsed = time.perf_counter() - save_start
        log(rank, "save_done", {"seconds": save_elapsed})

    if not args.skip_lfp:
        if rank == 0:
            log(rank, "lfp_start")
        lfp_start = time.perf_counter()
        t, values = ob.get_lfp()
        lfp_elapsed = time.perf_counter() - lfp_start
        if rank == 0:
            log(rank, "lfp_done", {"seconds": lfp_elapsed, "samples": len(t), "values": len(values)})

    if nranks > 1:
        try:
            from olfactorybulb.database import database

            database.close()
        except Exception:
            pass

    try:
        from neuron import h

        h.quit()
    except Exception:
        pass


if __name__ == "__main__":
    main()
