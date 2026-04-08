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
        os.environ.get("OB_DEBUG_LABEL", "debug_soma_gid_conflicts"),
        comm=MPI.COMM_WORLD,
    )
    ob = OlfactoryBulb(GammaSignature(), autorun=False)
    local_rows = []

    for cell_model in ob.iter_cell_models():
        cell_name = ob.get_cell_name(cell_model)
        report_gid = ob.get_cell_report_gid(cell_model)
        soma_seg_gids = []
        for seg_id, _seg in enumerate(cell_model.soma.allseg()):
            soma_sec_name = cell_model.soma.name()
            soma_seg_gids.append((seg_id, ob.bn_server.segment_gid(soma_sec_name, seg_id, False)))

        local_rows.append(
            {
                "cell": cell_name,
                "report_gid": int(report_gid) if report_gid is not None else None,
                "soma_conflict": any(gid == report_gid for _seg_id, gid in soma_seg_gids),
                "soma_seg_gids": soma_seg_gids,
            }
        )

    all_rows = ob.pc.py_gather(local_rows, 0)
    if ob.mpirank == 0:
        flat = [row for chunk in all_rows for row in chunk]
        conflicts = [row for row in flat if row["soma_conflict"]]
        result = {
            "total_cells": len(flat),
            "conflict_count": len(conflicts),
            "conflict_types": Counter(row["cell"].split("[")[0] for row in conflicts),
            "sample_conflicts": conflicts[:20],
        }
        out = Path("results") / label / "debug_soma_gid_conflicts.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))

    ob.pc.barrier()


if __name__ == "__main__":
    main()
