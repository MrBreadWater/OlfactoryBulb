"""Build one permanent provisional EPLI slice asset from the canonical slice.

This tool is intentionally simple: it does not rerun Blender. It copies the
maintained canonical DorsalColumnSlice geometry, inserts a deterministic set of
synthetic EPL fast interneurons, and generates explicit reciprocal synapse sets
for exploratory TC-focused and MTC-broad tests.

Run with:
    MPLCONFIGDIR=/tmp/mpl /opt/miniconda3/envs/OBGPU/bin/python \
        tools/build_provisional_epli_slice.py
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from olfactorybulb.slice_connectivity_optimizer import (
    GroupGeometry,
    SectionGeometry,
    build_candidate_pool,
    compute_metrics_from_pairs,
    load_group_geometry,
    resolve_slice_dir,
    select_pairs_from_pool,
)
from prev_ob_models.SyntheticEPL2026.isolated_cells import PVCRH_FSI1


def _section_to_dict(h, sec) -> dict:
    coords = []
    radii = []
    point_count = int(h.n3d(sec=sec))
    for index in range(point_count):
        coords.extend(
            [
                float(h.x3d(index, sec=sec)),
                float(h.y3d(index, sec=sec)),
                float(h.z3d(index, sec=sec)),
            ]
        )
        radii.append(float(h.diam3d(index, sec=sec)) / 2.0)

    parentseg = sec.parentseg()
    return {
        "name": str(sec.name()),
        "nseg": int(sec.nseg),
        "point_count": point_count,
        "coords": coords,
        "radii": radii,
        "parent_connection_loc": float(parentseg.x) if parentseg is not None else None,
        "connection_end": int(sec.orientation()),
        "children": [_section_to_dict(h, child) for child in sec.children()],
    }


def _walk_sections(root_dict: dict) -> list[dict]:
    sections = []
    stack = [root_dict]
    while stack:
        current = stack.pop()
        sections.append(current)
        stack.extend(reversed(current.get("children", [])))
    return sections


def _serialize_epli_roots(*, tc_roots: list[dict], offset_radius_um: float) -> list[dict]:
    from neuron import h

    PVCRH_FSI1._instance_counter = 0
    epli_roots = []
    total = len(tc_roots)
    for index, tc_root in enumerate(tc_roots):
        coords = tc_root["coords"]
        soma_x, soma_y, soma_z = float(coords[3]), float(coords[4]), float(coords[5])
        angle = (2.0 * math.pi * index) / max(total, 1)
        x = soma_x + offset_radius_um * math.cos(angle)
        y = soma_y + offset_radius_um * math.sin(angle)
        z = soma_z

        cell = PVCRH_FSI1()
        cell.position(x, y, z)
        epli_roots.append(_section_to_dict(h, cell.soma))
    return epli_roots


def _group_geometry_from_roots(group_name: str, roots: list[dict]) -> GroupGeometry:
    sections = []
    cell_names = []
    for root in roots:
        cell_names.append(root["name"].rsplit(".", 1)[0])
        for section_dict in _walk_sections(root):
            sections.append(SectionGeometry.from_dict(section_dict))
    sections_by_name = {section.name: section for section in sections}
    return GroupGeometry(
        name=group_name,
        sections=sections,
        sections_by_name=sections_by_name,
        cell_names=sorted(dict.fromkeys(cell_names)),
    )


def _pair_to_entry(
    pair,
    *,
    dest_syn: str = "GabaSyn",
    dest_syn_params: str = "{'gmax': 0.005, 'tau1': 1, 'tau2': 100}",
    source_syn: str = "AmpaNmdaSyn",
    source_syn_params: str = "{'gmax': 0.1}",
    synaptic_delay_ms: float = 0.5,
    conduction_velocity_um_per_ms: float = 1000.0,
) -> dict:
    propagation_delay = float(pair.distance_um) / float(conduction_velocity_um_per_ms)
    return {
        "source_section": pair.source.section_name,
        "source_x": float(pair.source.x),
        "source_seg_i": int(pair.source.segment_index),
        "dest_section": pair.dest.section_name,
        "dest_x": float(pair.dest.x),
        "dest_seg_i": int(pair.dest.segment_index),
        "dest_syn": dest_syn,
        "dest_syn_params": dest_syn_params,
        "delay": float(synaptic_delay_ms + propagation_delay),
        "weight": 1,
        "threshold": 0,
        "create_spine": False,
        "is_reciprocal": True,
        "source_syn": source_syn,
        "source_syn_params": source_syn_params,
    }


def _build_synapse_entries(
    *,
    epli_group: GroupGeometry,
    target_group: GroupGeometry,
    source_pattern: str,
    target_pattern: str,
    max_distance_um: float,
    max_syns_per_pt: int,
) -> tuple[list[dict], dict]:
    pool = build_candidate_pool(
        epli_group,
        target_group,
        source_patterns=[source_pattern],
        target_patterns=[target_pattern],
        max_distance_um=max_distance_um,
        use_radius=True,
    )
    pairs = select_pairs_from_pool(
        pool,
        max_distance_um=max_distance_um,
        max_syns_per_pt=max_syns_per_pt,
    )
    metrics = compute_metrics_from_pairs(
        pairs,
        label=f"{epli_group.name}__{target_group.name}",
        source_group=epli_group,
        target_group=target_group,
    )
    entries = [_pair_to_entry(pair) for pair in pairs]
    return entries, metrics.to_dict()


def build_slice(
    *,
    source_slice: str,
    output_slice: str,
    offset_radius_um: float,
) -> dict:
    source_dir = resolve_slice_dir(source_slice)
    output_dir = resolve_slice_dir(output_slice)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(source_dir, output_dir)

    tc_data = json.loads((source_dir / "TCs.json").read_text())
    epli_roots = _serialize_epli_roots(tc_roots=tc_data["roots"], offset_radius_um=offset_radius_um)

    group_template = json.loads((source_dir / "GCs.json").read_text())
    epli_group_data = {
        key: value
        for key, value in group_template.items()
        if key != "roots" and key != "name"
    }
    epli_group_data["name"] = "EPLIs"
    epli_group_data["roots"] = epli_roots
    (output_dir / "EPLIs.json").write_text(json.dumps(epli_group_data, indent=2))

    epli_group = _group_geometry_from_roots("EPLIs", epli_roots)
    tc_group = load_group_geometry(source_dir, "TCs")
    mc_group = load_group_geometry(source_dir, "MCs")

    tc_entries, tc_metrics = _build_synapse_entries(
        epli_group=epli_group,
        target_group=tc_group,
        source_pattern="*dend*",
        target_pattern="*soma*",
        max_distance_um=10.0,
        max_syns_per_pt=1,
    )
    (output_dir / "EPLIs__TCs.json").write_text(
        json.dumps({"name": "EPLIs->TCs", "entries": tc_entries}, indent=2)
    )

    mc_entries, mc_metrics = _build_synapse_entries(
        epli_group=epli_group,
        target_group=mc_group,
        source_pattern="*dend*",
        target_pattern="*dend*",
        max_distance_um=8.0,
        max_syns_per_pt=1,
    )
    (output_dir / "EPLIs__MCs.json").write_text(
        json.dumps({"name": "EPLIs->MCs", "entries": mc_entries}, indent=2)
    )

    summary = {
        "source_slice": str(source_dir),
        "output_slice": str(output_dir),
        "offset_radius_um": float(offset_radius_um),
        "epli_count": len(epli_group.cell_names),
        "epli_tc_metrics": tc_metrics,
        "epli_mc_metrics": mc_metrics,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-slice", default="DorsalColumnSlice")
    parser.add_argument("--output-slice", default="DorsalColumnSliceEPLIProvisional")
    parser.add_argument("--offset-radius-um", type=float, default=12.0)
    parser.add_argument("--summary-json", default="")
    args = parser.parse_args()

    summary = build_slice(
        source_slice=args.source_slice,
        output_slice=args.output_slice,
        offset_radius_um=float(args.offset_radius_um),
    )

    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
