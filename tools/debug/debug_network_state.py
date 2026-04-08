import argparse
import json
import os
import sys
from types import SimpleNamespace


BIRGIOLAS_GLOBALS = [
    "tau_CaPool",
    "cainf_CaPool",
    "Ybeta_KCa",
    "thinf_Na",
    "qinf_Na",
    "Cdur_AmpaNmdaSyn",
    "Alpha_AmpaNmdaSyn",
    "Beta_AmpaNmdaSyn",
    "E_AmpaNmdaSyn",
    "ampatau_AmpaNmdaSyn",
    "gampafactor_AmpaNmdaSyn",
    "sighalf_AmpaNmdaSyn",
    "sigslope_AmpaNmdaSyn",
    "sighalf_GabaSyn",
    "sigslope_GabaSyn",
]


def count_point_processes(h, mech_name):
    mech = getattr(h, mech_name, None)
    if mech is None:
        return None
    try:
        return sum(1 for _ in mech)
    except TypeError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--paramset", default="GammaSignature")
    parser.add_argument("--disable-status-report", action="store_true")
    parser.add_argument("--disable-lfp", action="store_true")
    parser.add_argument("--disable-gap-junctions", action="store_true")
    parser.add_argument("--disable-reciprocal-synapses", action="store_true")
    parser.add_argument("--skip-reciprocal-synapse-creation", action="store_true")
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

    if coreneuron is not None:
        coreneuron.enable = args.coreneuron
        coreneuron.gpu = args.gpu
        coreneuron.verbose = 0
        coreneuron.cell_permute = 1 if args.gpu else 0

    ob = OlfactoryBulb(params, autorun=False)

    result = {
        "rank": rank,
        "cell_counts": {cell_type: len(cells) for cell_type, cells in ob.cells.items()},
        "inputs_len": len(ob.inputs),
        "input_vectors_len": len(ob.input_vectors),
        "gjs_len": len(ob.gjs),
        "point_process_counts": {
            mech: count_point_processes(h, mech)
            for mech in ["AmpaNmdaSyn", "GabaSyn", "GapJunction", "Exp2Syn", "VecStim"]
        },
        "globals": {
            name: getattr(h, name)
            for name in BIRGIOLAS_GLOBALS
            if hasattr(h, name)
        },
    }

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
