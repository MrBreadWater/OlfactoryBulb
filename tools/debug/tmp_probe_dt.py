import json

from mpi4py import MPI

import olfactorybulb.model as obmodel
from olfactorybulb.model import OlfactoryBulb


def log(rank, phase, **extra):
    payload = {"rank": rank, "phase": phase}
    payload.update(extra)
    print(json.dumps(payload), flush=True)


def main():
    from neuron import coreneuron

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    params = obmodel.GammaSignature()
    params.tstop = 1.0
    params.enable_status_report = False
    params.enable_lfp = False
    params.parallel_timeout = 0

    ob = OlfactoryBulb(params, autorun=False)
    h = ob.h

    coreneuron.enable = True
    coreneuron.gpu = True
    coreneuron.verbose = 2
    coreneuron.cell_permute = 1

    h.steps_per_ms = 1.0 / ob.params.sim_dt
    h.dt = ob.params.sim_dt
    h.setdt()
    h.tstop = ob.params.tstop
    log(rank, "before_setup_transfer", dt=float(h.dt), steps_per_ms=float(h.steps_per_ms), tstop=float(h.tstop))

    ob.pc.setup_transfer()
    log(rank, "after_setup_transfer", dt=float(h.dt), steps_per_ms=float(h.steps_per_ms), t=float(h.t))
    h.cvode_active(0)
    log(rank, "after_cvode_active", dt=float(h.dt), steps_per_ms=float(h.steps_per_ms), t=float(h.t))
    ob.pc.set_maxstep(1)
    log(rank, "after_set_maxstep", dt=float(h.dt), steps_per_ms=float(h.steps_per_ms), t=float(h.t))
    log(rank, "before_stdinit", dt=float(h.dt), steps_per_ms=float(h.steps_per_ms), t=float(h.t))

    h.stdinit()
    log(rank, "after_stdinit", dt=float(h.dt), steps_per_ms=float(h.steps_per_ms), t=float(h.t))

    ob.pc.psolve(h.tstop)
    log(rank, "after_psolve", dt=float(h.dt), steps_per_ms=float(h.steps_per_ms), t=float(h.t))


if __name__ == "__main__":
    main()
