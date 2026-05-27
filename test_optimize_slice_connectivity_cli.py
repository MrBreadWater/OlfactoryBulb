"""CLI smoke test for the offline slice connectivity optimizer.

Run with:
    MPLCONFIGDIR=/tmp/mpl /opt/miniconda3/envs/OBGPU/bin/python test_optimize_slice_connectivity_cli.py
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


with tempfile.TemporaryDirectory(prefix="ob_slice_opt_") as tmpdir:
    json_out = Path(tmpdir) / "gc_mc_reference.json"
    completed = subprocess.run(
        [
            sys.executable,
            "tools/optimize_slice_connectivity.py",
            "reference",
            "--slice",
            "DorsalColumnSlice",
            "--synapse-set",
            "GCs__MCs",
            "--source-patterns",
            "*apic*",
            "--target-patterns",
            "*dend*",
            "--max-distances",
            "5",
            "--use-radii",
            "true",
            "--max-syns-per-pts",
            "2",
            "--top-n",
            "1",
            "--json-out",
            str(json_out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "GCs->MCs" in completed.stdout
    payload = json.loads(json_out.read_text())
    top = payload["top_results"][0]
    assert top["spec"]["source_pattern"] == "*apic*"
    assert top["spec"]["target_pattern"] == "*dend*"
    assert top["spec"]["max_distance_um"] == 5.0
    assert top["spec"]["use_radius"] is True
    assert top["spec"]["max_syns_per_pt"] == 2
    assert top["score"] > 0.99

print("slice connectivity optimizer CLI smoke test: OK")
