import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--paramset", default="GammaSignature")
    parser.add_argument(
        "--cells",
        nargs="+",
        default=["MC4[0]", "MC5[10]", "TC3[2]"],
    )
    parser.add_argument("--disable-status-report", action="store_true")
    parser.add_argument("--disable-lfp", action="store_true")
    parser.add_argument("--disable-reciprocal-synapses", action="store_true")
    parser.add_argument("--disable-gap-junctions", action="store_true")
    args, _unknown = parser.parse_known_args()

    if args.repo_root is not None:
        repo_root = os.path.abspath(args.repo_root)
        os.chdir(repo_root)
        sys.path.insert(0, repo_root)

    from mpi4py import MPI
    from neuron import h
    import olfactorybulb.model as obmodel
    from olfactorybulb.model import OlfactoryBulb

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    params = getattr(obmodel, args.paramset)()
    if args.disable_status_report:
        params.enable_status_report = False
    if args.disable_lfp:
        params.enable_lfp = False
    if args.disable_gap_junctions:
        params.gap_juction_gmax = {"MC": 0, "TC": 0}
    if args.disable_reciprocal_synapses:
        params.synapse_properties = {
            "AmpaNmdaSyn": {"gmax": 0, "ltpinvl": 0, "ltdinvl": 0},
            "GabaSyn": {"gmax": 0, "tau2": params.synapse_properties["GabaSyn"]["tau2"], "ltpinvl": 0, "ltdinvl": 0},
        }

    ob = OlfactoryBulb(params, autorun=False)
    model_inputsegs = ob.get_model_inputsegs()

    payload = {
        "rank": rank,
        "tau_CaPool": float(h.tau_CaPool) if hasattr(h, "tau_CaPool") else None,
        "targets": [],
    }
    for cell_name in args.cells:
        rank_cell = ob.bn_server.rank_section_name(cell_name)
        model_class = cell_name[: cell_name.find("[")]
        input_seg = model_inputsegs[model_class]
        seg_info = None
        if rank_cell is not None:
            seg_name = f"h.{rank_cell}.{input_seg}".replace("(1)", "(.999)")
            seg = eval(seg_name, {"h": h})
            sec = seg.sec
            pts = []
            for i in range(min(int(h.n3d(sec=sec)), 5)):
                pts.append(
                    [
                        float(h.x3d(i, sec=sec)),
                        float(h.y3d(i, sec=sec)),
                        float(h.z3d(i, sec=sec)),
                        float(h.diam3d(i, sec=sec)),
                    ]
                )
            seg_info = {
                "section": str(sec),
                "L": float(sec.L),
                "nseg": int(sec.nseg),
                "Ra": float(sec.Ra),
                "cm": float(sec.cm),
                "g_pas": float(sec.g_pas),
                "ena": float(sec.ena),
                "ek": float(sec.ek),
                "diam_05": float(sec(0.5).diam),
                "diam_1": float(sec(1.0).diam),
                "gbar_Na_05": float(sec(0.5).gbar_Na) if hasattr(sec(0.5), "gbar_Na") else None,
                "gbar_Kd_05": float(sec(0.5).gbar_Kd) if hasattr(sec(0.5), "gbar_Kd") else None,
                "gbar_Ih_05": float(sec(0.5).gbar_Ih) if hasattr(sec(0.5), "gbar_Ih") else None,
                "gbar_CaT_05": float(sec(0.5).gbar_CaT) if hasattr(sec(0.5), "gbar_CaT") else None,
                "n3d": int(h.n3d(sec=sec)),
                "pts": pts,
            }
        payload["targets"].append(
            {
                "canonical_cell": cell_name,
                "rank_cell": rank_cell,
                "input_seg": input_seg,
                "canonical_seg": f"h.{cell_name}.{input_seg}",
                "rank_seg": None if rank_cell is None else f"h.{rank_cell}.{input_seg}",
                "stable_gid": int(ob.stable_hash(f"h.{cell_name}.{input_seg}")),
                "seg_info": seg_info,
            }
        )

    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
