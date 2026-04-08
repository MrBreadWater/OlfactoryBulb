import argparse
import json
import os
import sys

import numpy as np


def disable_cvode(h):
    for fn in [
        lambda: h.cvode_active(0),
        lambda: h.cvode.active(0),
        lambda: h.CVode().active(0),
    ]:
        try:
            fn()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--target-rank", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--tstop", type=float, default=40.0)
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

    from prev_ob_models.Birgiolas2020.isolated_cells import MC4, MC5, TC3

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    pc = h.ParallelContext()
    h.load_file("stdrun.hoc")
    disable_cvode(h)
    h.steps_per_ms = 1.0 / args.dt
    h.dt = args.dt
    h.setdt()

    if coreneuron is not None:
        coreneuron.enable = args.coreneuron
        coreneuron.gpu = args.gpu
        coreneuron.verbose = 0
        coreneuron.cell_permute = 1 if args.gpu else 0

    result = {"rank": rank, "cells": []}
    recorded = []

    if rank == args.target_rank:
        cases = [
            ("MC4", MC4, 3, 23.452630291606667, 0.2),
            ("MC5", MC5, 2, 22.329288763864717, 0.2),
            ("TC3", TC3, 2, 29.173870377042306, 0.8),
        ]
        for name, cls, apic_index, event_time, weight in cases:
            cell = cls()
            disable_cvode(h)
            h.steps_per_ms = 1.0 / args.dt
            h.dt = args.dt
            h.setdt()
            seg = cell.cell.apic[apic_index](1.0)
            syn = h.Exp2Syn(seg)
            syn.tau1 = 6
            syn.tau2 = 12
            vs = h.VecStim()
            vec = h.Vector([event_time])
            vs.play(vec)
            nc = h.NetCon(vs, syn, 0, 0, weight)
            t_vec = h.Vector().record(h._ref_t, sec=cell.soma)
            v_vec = h.Vector().record(cell.soma(0.5)._ref_v, sec=cell.soma)
            recorded.append((name, t_vec, v_vec, vs, vec, nc, syn))

    pc.setup_transfer()
    pc.set_maxstep(1)
    h.stdinit()
    pc.psolve(args.tstop)

    for name, t_vec, v_vec, *_keepalive in recorded:
        t = np.array(t_vec)
        v = np.array(v_vec)
        peak_i = int(np.argmax(v))
        result["cells"].append(
            {
                "cell": name,
                "peak_t": float(t[peak_i]),
                "peak_v": float(v[peak_i]),
                "crossings": [
                    float(t[i + 1])
                    for i in np.where((v[:-1] < 0) & (v[1:] >= 0))[0][:3]
                ],
            }
        )

    gathered = comm.gather(result, root=0)
    if rank == 0:
        print(json.dumps(gathered, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
