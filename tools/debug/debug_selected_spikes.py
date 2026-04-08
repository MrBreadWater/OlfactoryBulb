import argparse
import json
import os
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--paramset", default="GammaSignature")
    parser.add_argument("--cells", nargs="+", default=["MC4[0]", "MC5[10]", "TC3[2]"])
    parser.add_argument("--tstop", type=float, default=30.0)
    parser.add_argument("--sim-dt", type=float, default=None)
    parser.add_argument("--disable-status-report", action="store_true")
    parser.add_argument("--disable-lfp", action="store_true")
    parser.add_argument("--disable-gap-junctions", action="store_true")
    parser.add_argument("--disable-reciprocal-synapses", action="store_true")
    parser.add_argument("--skip-reciprocal-synapse-creation", action="store_true")
    parser.add_argument("--no-odor-inputs", action="store_true")
    parser.add_argument("--coreneuron", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args, _unknown = parser.parse_known_args()

    if args.repo_root is not None:
        repo_root = os.path.abspath(args.repo_root)
        os.chdir(repo_root)
        sys.path.insert(0, repo_root)

    from mpi4py import MPI
    from neuron import h

    try:
        from neuron import coreneuron
    except ImportError:
        coreneuron = None

    import olfactorybulb.model as obmodel
    from olfactorybulb.model import OlfactoryBulb

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    params = getattr(obmodel, args.paramset)()
    params.coreneuron = SimpleNamespace(
        enable=args.coreneuron,
        gpu=args.gpu,
        file_mode=False,
        verbose=0,
    )
    if args.sim_dt is not None:
        params.sim_dt = args.sim_dt
        params.recording_period = args.sim_dt
    if args.disable_status_report:
        params.enable_status_report = False
    if args.disable_lfp:
        params.enable_lfp = False
    if args.disable_gap_junctions:
        params.gap_juction_gmax = {"MC": 0, "TC": 0}
    if args.disable_reciprocal_synapses:
        params.synapse_properties = {
            "AmpaNmdaSyn": {"gmax": 0, "ltpinvl": 0, "ltdinvl": 0},
            "GabaSyn": {
                "gmax": 0,
                "tau2": params.synapse_properties["GabaSyn"]["tau2"],
                "ltpinvl": 0,
                "ltdinvl": 0,
            },
        }
    if args.skip_reciprocal_synapse_creation:
        params.enable_reciprocal_synapses = False
    if args.no_odor_inputs:
        params.input_odors = {}

    if coreneuron is not None:
        coreneuron.enable = args.coreneuron
        coreneuron.gpu = args.gpu
        coreneuron.verbose = 0
        coreneuron.cell_permute = 1 if args.gpu else 0

    ob = OlfactoryBulb(params, autorun=False)
    ob.params.tstop = args.tstop

    trace_vectors = []
    result = {"rank": rank, "cells": []}

    for canonical_cell in args.cells:
        rank_cell = ob.bn_server.rank_section_name(canonical_cell)
        if rank_cell is None:
            continue
        soma = eval(f"h.{rank_cell}.soma", {"h": h})
        t_vec = h.Vector()
        t_vec.record(h._ref_t, sec=soma)
        v_vec = h.Vector()
        v_vec.record(soma(0.5)._ref_v, sec=soma)
        trace_vectors.append((canonical_cell, rank_cell, t_vec, v_vec))

    ob.run(args.tstop)

    for canonical_cell, rank_cell, t_vec, v_vec in trace_vectors:
        t = list(t_vec)
        v = list(v_vec)
        spike_times = []
        for i in range(len(v) - 1):
            if v[i] < 0.0 and v[i + 1] >= 0.0:
                spike_times.append(t[i + 1])
        result["cells"].append(
            {
                "canonical_cell": canonical_cell,
                "dt": h.dt,
                "rank_cell": rank_cell,
                "spike_times": spike_times,
                "peak_t": t[int(max(range(len(v)), key=lambda j: v[j]))] if v else None,
                "peak_v": max(v) if v else None,
            }
        )

    gathered = comm.gather(result, root=0)
    if rank == 0:
        print(json.dumps(gathered, indent=2, sort_keys=True))

    if comm.Get_size() > 1:
        try:
            from olfactorybulb.database import database

            database.close()
        except Exception:
            pass

    try:
        h.quit()
    except Exception:
        pass


if __name__ == "__main__":
    main()
