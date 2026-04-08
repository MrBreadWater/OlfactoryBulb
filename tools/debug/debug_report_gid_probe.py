import json
import os
from collections import Counter
from pathlib import Path

from mpi4py import MPI
from olfactorybulb.model import OlfactoryBulb
from olfactorybulb.output_paths import configure_output_env
from olfactorybulb.paramsets.case_studies import GammaSignature


def main():
    label, _timestamp = configure_output_env(
        os.environ.get("OB_DEBUG_LABEL", "debug_report_gid_probe"),
        comm=MPI.COMM_WORLD,
    )
    ob = OlfactoryBulb(GammaSignature(), autorun=False)
    local_cells = list(ob.iter_cell_models())
    non_none = []
    none_names = []
    gid_exists_yes = []
    gid_exists_no = []

    for cell in local_cells:
        gid = ob.get_cell_report_gid(cell)
        if gid is None:
            none_names.append(ob.get_cell_name(cell))
        else:
            gid = int(gid)
            cell_name = ob.get_cell_name(cell)
            non_none.append((gid, cell_name))
            if ob.pc.gid_exists(gid):
                gid_exists_yes.append((gid, cell_name))
            else:
                gid_exists_no.append((gid, cell_name))

    summary = {
        "rank": int(ob.mpirank),
        "local_cells": len(local_cells),
        "bn_server_cell_source_gids": len(getattr(ob.bn_server, "cell_source_gids", {})),
        "native_lfp_gid_source": len(getattr(ob, "_native_lfp_gid_source", {})),
        "report_gid_non_none": len(non_none),
        "report_gid_none": len(none_names),
        "non_none_types": Counter(name.split("[")[0] for _, name in non_none),
        "none_types": Counter(name.split("[")[0] for name in none_names),
        "gid_exists_yes": len(gid_exists_yes),
        "gid_exists_no": len(gid_exists_no),
        "gid_exists_yes_types": Counter(name.split("[")[0] for _, name in gid_exists_yes),
        "gid_exists_no_types": Counter(name.split("[")[0] for _, name in gid_exists_no),
    }

    all_summaries = ob.pc.py_gather(summary, 0)
    all_non_none = ob.pc.py_gather(non_none, 0)
    all_none = ob.pc.py_gather(none_names, 0)

    if ob.mpirank == 0:
        result = {
            "summaries": all_summaries,
            "unique_non_none": len({gid for chunk in all_non_none for gid, _ in chunk}),
            "unique_none": len({name for chunk in all_none for name in chunk}),
            "sample_none": sorted({name for chunk in all_none for name in chunk})[:20],
            "all_non_none": sorted(
                [(int(gid), name) for chunk in all_non_none for gid, name in chunk],
                key=lambda item: item[0],
            ),
        }
        out = Path("results") / label / "debug_report_gid_probe.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str))

    ob.pc.barrier()


if __name__ == "__main__":
    main()
