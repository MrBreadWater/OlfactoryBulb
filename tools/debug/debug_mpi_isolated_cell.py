import argparse
import json
import os
import sys

import numpy as np


def disable_cvode(h):
    try:
        h.cvode_active(0)
    except Exception:
        pass
    try:
        h.cvode.active(0)
    except Exception:
        pass
    try:
        h.CVode().active(0)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--cell-class", required=True)
    parser.add_argument("--apic-index", type=int, required=True)
    parser.add_argument("--event-time", type=float, default=None)
    parser.add_argument("--weight", type=float, default=0.0)
    parser.add_argument("--tau1", type=float, default=6.0)
    parser.add_argument("--tau2", type=float, default=12.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--tstop", type=float, default=40.0)
    parser.add_argument("--target-rank", type=int, default=1)
    parser.add_argument("--no-input", action="store_true")
    parser.add_argument("--input-strategy", choices=["vecstim", "scheduled"], default="vecstim")
    parser.add_argument("--input-delay", type=float, default=None)
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

    from prev_ob_models.Birgiolas2020 import isolated_cells

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    result = {
        "rank": rank,
        "active": False,
    }

    pc = h.ParallelContext()
    h.load_file("stdrun.hoc")
    disable_cvode(h)
    h.steps_per_ms = 1.0 / args.dt
    h.dt = args.dt

    if coreneuron is not None:
        coreneuron.enable = args.coreneuron
        coreneuron.gpu = args.gpu
        coreneuron.verbose = 0
        coreneuron.cell_permute = 1 if args.gpu else 0

    if rank == args.target_rank:
        cell = getattr(isolated_cells, args.cell_class)()
        disable_cvode(h)
        h.steps_per_ms = 1.0 / args.dt
        h.dt = args.dt
        h.setdt()
        if not args.no_input:
            if args.event_time is None:
                raise ValueError("--event-time is required unless --no-input is used")
            seg = cell.cell.apic[args.apic_index](1.0)
            syn = h.Exp2Syn(seg)
            syn.tau1 = args.tau1
            syn.tau2 = args.tau2
            if args.input_strategy == "vecstim":
                vs = h.VecStim()
                event_vec = h.Vector([args.event_time])
                vs.play(event_vec)
                nc = h.NetCon(vs, syn, 0, 0, args.weight)
            else:
                input_delay = args.input_delay
                if input_delay is None:
                    input_delay = 2.0 * args.dt
                nc = h.NetCon(None, syn)
                nc.delay = input_delay
                nc.weight[0] = args.weight

                def schedule_event(nc_ref=nc, event_time=float(args.event_time)):
                    nc_ref.event(event_time)

                fih = h.FInitializeHandler(1, schedule_event)
                result["scheduled_event_time"] = float(args.event_time)
                result["netcon_delay"] = float(input_delay)
        t = h.Vector().record(h._ref_t, sec=cell.soma)
        v = h.Vector().record(cell.soma(0.5)._ref_v, sec=cell.soma)
        result["active"] = True
    else:
        t = None
        v = None

    pc.setup_transfer()
    pc.set_maxstep(1)
    h.stdinit()
    pc.psolve(args.tstop)

    if t is not None and v is not None:
        t_np = np.array(t)
        v_np = np.array(v)
        peak_i = int(np.argmax(v_np))
        result.update(
            {
                "len": int(v_np.size),
                "peak_t": float(t_np[peak_i]),
                "peak_v": float(v_np[peak_i]),
                "crossings": [
                    float(t_np[i + 1])
                    for i in np.where((v_np[:-1] < 0) & (v_np[1:] >= 0))[0][:3]
                ],
                "first10": v_np[:10].tolist(),
            }
        )

    gathered = comm.gather(result, root=0)
    if rank == 0:
        print(json.dumps(gathered, indent=2, sort_keys=True))

    try:
        h.quit()
    except Exception:
        pass


if __name__ == "__main__":
    main()
