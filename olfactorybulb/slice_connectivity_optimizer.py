"""Offline slice-connectivity metrics and optimization utilities.

This module evaluates candidate synapse rules directly from exported slice JSON
geometry. That keeps optimization loops out of Blender and makes it possible to
validate candidate rules against the maintained canonical slice connectivity.

Primary use cases:

1. Recover known GC->MC / GC->TC rule families on ``DorsalColumnSlice`` as an
   internal positive control.
2. Search plausible synapse-rule candidates for opt-in populations such as
   ``EPLIs`` on an already-exported slice.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from fnmatch import fnmatch
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - exercised only in environments without scipy
    cKDTree = None

from olfactorybulb import slices


def resolve_slice_dir(slice_name_or_path: str | Path) -> Path:
    """Resolve a slice name or path to the on-disk slice directory."""
    path = Path(slice_name_or_path).expanduser()
    if path.exists():
        return path.resolve()
    return (Path(slices.__file__).resolve().parent / str(slice_name_or_path)).resolve()


def _safe_float(value: float | None, default: float = 0.0) -> float:
    return default if value is None or not math.isfinite(value) else float(value)


def _normalize_hist(counter: Counter[str]) -> dict[str, float]:
    total = float(sum(counter.values()))
    if total <= 0:
        return {}
    return {key: float(value) / total for key, value in counter.items()}


def _hist_l1(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return sum(abs(a.get(key, 0.0) - b.get(key, 0.0)) for key in keys) * 0.5


def _family_fraction(hist: dict[str, float], preferred_family: str) -> float:
    total = 0.0
    for family, fraction in hist.items():
        if family == preferred_family or family.startswith(f"{preferred_family}_"):
            total += float(fraction)
    return total


def _log_count_penalty(a: float, b: float) -> float:
    return abs(math.log1p(max(0.0, a)) - math.log1p(max(0.0, b)))


def _relative_penalty(a: float | None, b: float | None, *, floor: float = 1e-6) -> float:
    a_val = _safe_float(a)
    b_val = _safe_float(b)
    scale = max(abs(b_val), floor)
    return abs(a_val - b_val) / scale


def _section_family(section_name: str) -> str:
    suffix = section_name.split(".", 1)[1] if "." in section_name else section_name
    if suffix.startswith("dend_primary"):
        return "dend_primary"
    if suffix.startswith("dend_branch"):
        return "dend_branch"
    if suffix.startswith("soma"):
        return "soma"
    if suffix.startswith("apic"):
        return "apic"
    if suffix.startswith("dend"):
        return "dend"
    if suffix.startswith("axon"):
        return "axon"
    return suffix.split("[", 1)[0]


def _root_cell_name(section_name: str) -> str:
    return section_name.rsplit(".", 1)[0] if "." in section_name else section_name


def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch(name, pattern) for pattern in patterns)


def _sorted_unique(items: Iterable[str]) -> list[str]:
    return sorted(dict.fromkeys(items))


@dataclass(frozen=True)
class SectionGeometry:
    name: str
    cell_name: str
    family: str
    coords: np.ndarray
    radii: np.ndarray
    nseg: int
    arclengths: np.ndarray
    length_um: float

    @classmethod
    def from_dict(cls, section_dict: dict[str, Any]) -> "SectionGeometry":
        coords = np.asarray(section_dict.get("coords", []), dtype=float).reshape(-1, 3)
        radii = np.asarray(section_dict.get("radii", []), dtype=float)
        if len(radii) != len(coords):
            if len(radii) == 0 and len(coords) > 0:
                radii = np.zeros(len(coords), dtype=float)
            else:
                raise ValueError(f"Section {section_dict.get('name')} has mismatched coords/radii lengths.")

        if len(coords) <= 1:
            arclengths = np.zeros(len(coords), dtype=float)
        else:
            diffs = coords[1:] - coords[:-1]
            segment_lengths = np.sqrt(np.sum(np.square(diffs), axis=1))
            arclengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))

        length_um = float(arclengths[-1]) if len(arclengths) > 0 else 0.0
        return cls(
            name=str(section_dict["name"]),
            cell_name=_root_cell_name(str(section_dict["name"])),
            family=_section_family(str(section_dict["name"])),
            coords=coords,
            radii=radii,
            nseg=max(1, int(section_dict.get("nseg", 1) or 1)),
            arclengths=arclengths,
            length_um=length_um,
        )

    def point_at_fraction(self, x: float) -> tuple[np.ndarray, float]:
        if len(self.coords) == 0:
            return np.zeros(3, dtype=float), 0.0
        if len(self.coords) == 1 or self.length_um <= 0:
            return self.coords[0].copy(), float(self.radii[0])

        x = min(max(float(x), 0.0), 1.0)
        target_len = x * self.length_um
        upper = int(np.searchsorted(self.arclengths, target_len, side="right"))
        upper = min(max(1, upper), len(self.coords) - 1)
        lower = upper - 1

        lo_len = self.arclengths[lower]
        hi_len = self.arclengths[upper]
        if hi_len <= lo_len:
            alpha = 0.0
        else:
            alpha = (target_len - lo_len) / (hi_len - lo_len)

        point = self.coords[lower] + alpha * (self.coords[upper] - self.coords[lower])
        radius = float(self.radii[lower] + alpha * (self.radii[upper] - self.radii[lower]))
        return point, radius

    def iter_terminals(self) -> Iterable["SynapseTerminal"]:
        if len(self.coords) == 0:
            return

        total_len = self.length_um
        for point_index, (coord, radius) in enumerate(zip(self.coords, self.radii, strict=True)):
            if total_len > 0:
                x = float(self.arclengths[point_index] / total_len)
            elif len(self.coords) > 1:
                x = float(point_index) / float(len(self.coords) - 1)
            else:
                x = 0.0
            seg_index = min(int(math.floor(self.nseg * x)), self.nseg - 1)
            yield SynapseTerminal(
                loc=np.asarray(coord, dtype=float),
                radius=float(radius),
                section_name=self.name,
                point_index=int(point_index),
                x=float(x),
                segment_index=int(seg_index),
                cell_name=self.cell_name,
                family=self.family,
            )


@dataclass(frozen=True)
class SynapseTerminal:
    loc: np.ndarray
    radius: float
    section_name: str
    point_index: int
    x: float
    segment_index: int
    cell_name: str
    family: str

    @property
    def loc_key(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self.loc)


@dataclass(frozen=True)
class ConnectivityPair:
    source: SynapseTerminal
    dest: SynapseTerminal
    distance_um: float


@dataclass
class GroupGeometry:
    name: str
    sections: list[SectionGeometry]
    sections_by_name: dict[str, SectionGeometry]
    cell_names: list[str]


@dataclass
class ConnectivityMetrics:
    label: str
    source_group: str
    target_group: str
    entry_count: int
    total_source_cells: int
    total_target_cells: int
    connected_source_cells: int
    connected_target_cells: int
    source_coverage: float
    target_coverage: float
    mean_entries_per_source_total: float
    mean_entries_per_source_connected: float
    mean_entries_per_target_total: float
    mean_entries_per_target_connected: float
    median_distance_um: float | None
    mean_distance_um: float | None
    p90_distance_um: float | None
    source_family_fraction: dict[str, float] = field(default_factory=dict)
    target_family_fraction: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "source_group": self.source_group,
            "target_group": self.target_group,
            "entry_count": self.entry_count,
            "total_source_cells": self.total_source_cells,
            "total_target_cells": self.total_target_cells,
            "connected_source_cells": self.connected_source_cells,
            "connected_target_cells": self.connected_target_cells,
            "source_coverage": self.source_coverage,
            "target_coverage": self.target_coverage,
            "mean_entries_per_source_total": self.mean_entries_per_source_total,
            "mean_entries_per_source_connected": self.mean_entries_per_source_connected,
            "mean_entries_per_target_total": self.mean_entries_per_target_total,
            "mean_entries_per_target_connected": self.mean_entries_per_target_connected,
            "median_distance_um": self.median_distance_um,
            "mean_distance_um": self.mean_distance_um,
            "p90_distance_um": self.p90_distance_um,
            "source_family_fraction": dict(self.source_family_fraction),
            "target_family_fraction": dict(self.target_family_fraction),
        }


@dataclass(frozen=True)
class SearchSpec:
    source_group: str
    target_group: str
    source_pattern: str
    target_pattern: str
    max_distance_um: float
    use_radius: bool = True
    max_syns_per_pt: int = 1

    @property
    def label(self) -> str:
        return (
            f"{self.source_group}->{self.target_group} "
            f"src={self.source_pattern} dst={self.target_pattern} "
            f"d<={self.max_distance_um:g} use_radius={int(self.use_radius)} "
            f"max_syns_per_pt={self.max_syns_per_pt}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_group": self.source_group,
            "target_group": self.target_group,
            "source_pattern": self.source_pattern,
            "target_pattern": self.target_pattern,
            "max_distance_um": self.max_distance_um,
            "use_radius": self.use_radius,
            "max_syns_per_pt": self.max_syns_per_pt,
        }


@dataclass
class SearchResult:
    spec: SearchSpec
    metrics: ConnectivityMetrics
    score: float
    penalties: dict[str, float]
    objective: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "metrics": self.metrics.to_dict(),
            "score": self.score,
            "penalties": dict(self.penalties),
            "objective": self.objective,
        }


@dataclass(frozen=True)
class TerminalCollection:
    group_name: str
    patterns: tuple[str, ...]
    terminals: list[SynapseTerminal]
    points: np.ndarray
    radii: np.ndarray
    max_radius: float


@dataclass(frozen=True)
class CandidatePool:
    source_group: str
    target_group: str
    source_pattern: str
    target_pattern: str
    use_radius: bool
    max_distance_um: float
    source_terminals: list[SynapseTerminal]
    target_terminals: list[SynapseTerminal]
    pairs_sorted: list[ConnectivityPair]


def _walk_sections(root_dict: dict[str, Any]) -> Iterable[dict[str, Any]]:
    stack = [root_dict]
    while stack:
        section = stack.pop()
        yield section
        children = list(section.get("children", []))
        stack.extend(reversed(children))


def load_group_geometry(slice_name_or_path: str | Path, group_name: str) -> GroupGeometry:
    slice_dir = resolve_slice_dir(slice_name_or_path)
    group_path = slice_dir / f"{group_name}.json"
    data = json.loads(group_path.read_text())

    sections: list[SectionGeometry] = []
    cell_names: list[str] = []
    for root in data.get("roots", []):
        cell_names.append(_root_cell_name(str(root["name"])))
        for section_dict in _walk_sections(root):
            sections.append(SectionGeometry.from_dict(section_dict))

    sections_by_name = {section.name: section for section in sections}
    return GroupGeometry(
        name=group_name,
        sections=sections,
        sections_by_name=sections_by_name,
        cell_names=_sorted_unique(cell_names),
    )


def load_slice_geometry(slice_name_or_path: str | Path) -> dict[str, GroupGeometry]:
    slice_dir = resolve_slice_dir(slice_name_or_path)
    groups = {}
    for group_file in sorted(slice_dir.glob("*.json")):
        if "__" in group_file.stem or group_file.stem == "glom_cells":
            continue
        groups[group_file.stem] = load_group_geometry(slice_dir, group_file.stem)
    return groups


def suggest_section_patterns(group: GroupGeometry) -> list[str]:
    """Return a small pattern search space derived from observed section families."""
    families = _sorted_unique(section.family for section in group.sections)
    patterns: list[str] = []

    if "dend_primary" in families or "dend_branch" in families:
        patterns.append("*dend*")
    for family in families:
        patterns.append(f"*{family}*")
    if "apic" in families:
        patterns.append("*apic*")
    if "dend" in families:
        patterns.append("*dend*")
    if "soma" in families:
        patterns.append("*soma*")
    if "axon" in families:
        patterns.append("*axon*")
    return _sorted_unique(patterns)


def _build_kdtree(points: np.ndarray):
    if cKDTree is not None:
        return cKDTree(points)
    return None


def _query_ball(tree, points: np.ndarray, point: np.ndarray, radius: float) -> list[int]:
    if tree is not None:
        return list(tree.query_ball_point(point, radius))
    diffs = points - point.reshape(1, 3)
    dists = np.sqrt(np.sum(np.square(diffs), axis=1))
    return [int(index) for index, dist in enumerate(dists) if dist <= radius]


def _collect_terminals(group: GroupGeometry, patterns: Sequence[str]) -> TerminalCollection:
    terminals = [
        terminal
        for section in group.sections
        if _matches_any(section.name, patterns)
        for terminal in section.iter_terminals()
    ]
    if len(terminals) == 0:
        points = np.empty((0, 3), dtype=float)
        radii = np.empty((0,), dtype=float)
    else:
        points = np.asarray([terminal.loc for terminal in terminals], dtype=float)
        radii = np.asarray([terminal.radius for terminal in terminals], dtype=float)
    max_radius = float(np.max(radii)) if len(radii) > 0 else 0.0
    return TerminalCollection(
        group_name=group.name,
        patterns=tuple(patterns),
        terminals=terminals,
        points=points,
        radii=radii,
        max_radius=max_radius,
    )


def _query_ball_many(tree, points: np.ndarray, query_points: np.ndarray, radii: np.ndarray) -> list[list[int]]:
    if len(query_points) == 0:
        return []
    if tree is not None:
        try:
            matches = tree.query_ball_point(query_points, radii, workers=-1)
        except TypeError:
            matches = tree.query_ball_point(query_points, radii)
        return [list(match) for match in matches]
    return [_query_ball(None, points, point, radius) for point, radius in zip(query_points, radii, strict=True)]


def build_candidate_pool(
    source_group: GroupGeometry,
    target_group: GroupGeometry,
    *,
    source_patterns: Sequence[str],
    target_patterns: Sequence[str],
    max_distance_um: float,
    use_radius: bool = True,
) -> CandidatePool:
    source_collection = _collect_terminals(source_group, source_patterns)
    target_collection = _collect_terminals(target_group, target_patterns)

    if len(source_collection.terminals) == 0 or len(target_collection.terminals) == 0:
        return CandidatePool(
            source_group=source_group.name,
            target_group=target_group.name,
            source_pattern="|".join(source_patterns),
            target_pattern="|".join(target_patterns),
            use_radius=use_radius,
            max_distance_um=float(max_distance_um),
            source_terminals=source_collection.terminals,
            target_terminals=target_collection.terminals,
            pairs_sorted=[],
        )

    tree = _build_kdtree(target_collection.points)
    search_radii = np.full(len(source_collection.terminals), float(max_distance_um), dtype=float)
    if use_radius:
        search_radii = search_radii + target_collection.max_radius + source_collection.radii

    candidate_indices_by_source = _query_ball_many(
        tree,
        target_collection.points,
        source_collection.points,
        search_radii,
    )

    pair_heap: list[tuple[float, ConnectivityPair]] = []
    for source_index, target_indices in enumerate(candidate_indices_by_source):
        if len(target_indices) == 0:
            continue

        source_terminal = source_collection.terminals[source_index]
        target_index_array = np.asarray(target_indices, dtype=int)
        target_points = target_collection.points[target_index_array]
        deltas = target_points - source_terminal.loc.reshape(1, 3)
        distances = np.sqrt(np.sum(np.square(deltas), axis=1))

        if use_radius:
            distances = np.maximum(
                0.0,
                distances - source_terminal.radius - target_collection.radii[target_index_array],
            )

        target_terminals = [target_collection.terminals[index] for index in target_index_array]
        for target_terminal, distance in zip(target_terminals, distances, strict=True):
            if source_terminal.cell_name == target_terminal.cell_name:
                continue
            distance = float(distance)
            if distance > max_distance_um:
                continue
            pair_heap.append(
                (
                    distance,
                    ConnectivityPair(
                        source=source_terminal,
                        dest=target_terminal,
                        distance_um=distance,
                    ),
                )
            )

    pair_heap.sort(key=lambda item: item[0])
    return CandidatePool(
        source_group=source_group.name,
        target_group=target_group.name,
        source_pattern="|".join(source_patterns),
        target_pattern="|".join(target_patterns),
        use_radius=bool(use_radius),
        max_distance_um=float(max_distance_um),
        source_terminals=source_collection.terminals,
        target_terminals=target_collection.terminals,
        pairs_sorted=[pair for _distance, pair in pair_heap],
    )


def select_pairs_from_pool(pool: CandidatePool, *, max_distance_um: float, max_syns_per_pt: int = 1) -> list[ConnectivityPair]:
    if max_distance_um > pool.max_distance_um + 1e-9:
        raise ValueError(
            f"Requested max_distance_um={max_distance_um:g} exceeds pool max_distance_um={pool.max_distance_um:g}."
        )

    used_source_points: dict[tuple[float, float, float], int] = {}
    used_target_points: dict[tuple[float, float, float], int] = {}
    accepted: list[ConnectivityPair] = []

    for pair in pool.pairs_sorted:
        if pair.distance_um > max_distance_um:
            break
        source_key = pair.source.loc_key
        target_key = pair.dest.loc_key
        if used_source_points.get(source_key, 0) >= max_syns_per_pt:
            continue
        if used_target_points.get(target_key, 0) >= max_syns_per_pt:
            continue
        used_source_points[source_key] = used_source_points.get(source_key, 0) + 1
        used_target_points[target_key] = used_target_points.get(target_key, 0) + 1
        accepted.append(pair)

    return accepted


def compute_metrics_from_pairs(
    pairs: Sequence[ConnectivityPair],
    *,
    label: str,
    source_group: GroupGeometry,
    target_group: GroupGeometry,
) -> ConnectivityMetrics:
    source_counts = Counter(pair.source.cell_name for pair in pairs)
    target_counts = Counter(pair.dest.cell_name for pair in pairs)
    source_family = Counter(pair.source.family for pair in pairs)
    target_family = Counter(pair.dest.family for pair in pairs)
    distances = np.asarray([pair.distance_um for pair in pairs], dtype=float)

    connected_source_cells = len(source_counts)
    connected_target_cells = len(target_counts)
    total_source_cells = len(source_group.cell_names)
    total_target_cells = len(target_group.cell_names)

    median_distance = float(np.median(distances)) if len(distances) > 0 else None
    mean_distance = float(np.mean(distances)) if len(distances) > 0 else None
    p90_distance = float(np.percentile(distances, 90)) if len(distances) > 0 else None

    return ConnectivityMetrics(
        label=label,
        source_group=source_group.name,
        target_group=target_group.name,
        entry_count=len(pairs),
        total_source_cells=total_source_cells,
        total_target_cells=total_target_cells,
        connected_source_cells=connected_source_cells,
        connected_target_cells=connected_target_cells,
        source_coverage=float(connected_source_cells / total_source_cells) if total_source_cells > 0 else 0.0,
        target_coverage=float(connected_target_cells / total_target_cells) if total_target_cells > 0 else 0.0,
        mean_entries_per_source_total=float(len(pairs) / total_source_cells) if total_source_cells > 0 else 0.0,
        mean_entries_per_source_connected=float(len(pairs) / connected_source_cells) if connected_source_cells > 0 else 0.0,
        mean_entries_per_target_total=float(len(pairs) / total_target_cells) if total_target_cells > 0 else 0.0,
        mean_entries_per_target_connected=float(len(pairs) / connected_target_cells) if connected_target_cells > 0 else 0.0,
        median_distance_um=median_distance,
        mean_distance_um=mean_distance,
        p90_distance_um=p90_distance,
        source_family_fraction=_normalize_hist(source_family),
        target_family_fraction=_normalize_hist(target_family),
    )


def _observed_pairs_from_entries(
    slice_name_or_path: str | Path,
    synapse_set_name: str,
    *,
    groups: dict[str, GroupGeometry] | None = None,
) -> tuple[list[ConnectivityPair], GroupGeometry, GroupGeometry]:
    slice_dir = resolve_slice_dir(slice_name_or_path)
    if groups is None:
        groups = load_slice_geometry(slice_dir)

    source_group_name, target_group_name = synapse_set_name.split("__", 1)
    source_group = groups[source_group_name]
    target_group = groups[target_group_name]

    section_index = {}
    for group in groups.values():
        section_index.update(group.sections_by_name)

    data = json.loads((slice_dir / f"{synapse_set_name}.json").read_text())
    pairs: list[ConnectivityPair] = []
    for entry in data.get("entries", []):
        source_section = section_index[entry["source_section"]]
        target_section = section_index[entry["dest_section"]]
        source_loc, source_radius = source_section.point_at_fraction(float(entry["source_x"]))
        target_loc, target_radius = target_section.point_at_fraction(float(entry["dest_x"]))
        distance = float(np.linalg.norm(source_loc - target_loc))
        distance = max(0.0, distance - source_radius - target_radius)
        pairs.append(
            ConnectivityPair(
                source=SynapseTerminal(
                    loc=source_loc,
                    radius=source_radius,
                    section_name=source_section.name,
                    point_index=int(entry.get("source_seg_i", 0)),
                    x=float(entry["source_x"]),
                    segment_index=int(entry.get("source_seg_i", 0)),
                    cell_name=source_section.cell_name,
                    family=source_section.family,
                ),
                dest=SynapseTerminal(
                    loc=target_loc,
                    radius=target_radius,
                    section_name=target_section.name,
                    point_index=int(entry.get("dest_seg_i", 0)),
                    x=float(entry["dest_x"]),
                    segment_index=int(entry.get("dest_seg_i", 0)),
                    cell_name=target_section.cell_name,
                    family=target_section.family,
                ),
                distance_um=distance,
            )
        )

    return pairs, source_group, target_group


def observed_metrics_for_synapse_set(
    slice_name_or_path: str | Path,
    synapse_set_name: str,
    *,
    groups: dict[str, GroupGeometry] | None = None,
) -> ConnectivityMetrics:
    pairs, source_group, target_group = _observed_pairs_from_entries(
        slice_name_or_path,
        synapse_set_name,
        groups=groups,
    )
    return compute_metrics_from_pairs(
        pairs,
        label=f"observed:{synapse_set_name}",
        source_group=source_group,
        target_group=target_group,
    )


def evaluate_search_spec(
    slice_name_or_path: str | Path,
    spec: SearchSpec,
    *,
    groups: dict[str, GroupGeometry] | None = None,
    pool: CandidatePool | None = None,
) -> tuple[list[ConnectivityPair], ConnectivityMetrics]:
    if groups is None:
        groups = load_slice_geometry(slice_name_or_path)
    source_group = groups[spec.source_group]
    target_group = groups[spec.target_group]
    if pool is None:
        pool = build_candidate_pool(
            source_group,
            target_group,
            source_patterns=[spec.source_pattern],
            target_patterns=[spec.target_pattern],
            max_distance_um=spec.max_distance_um,
            use_radius=spec.use_radius,
        )
    pairs = select_pairs_from_pool(pool, max_distance_um=spec.max_distance_um, max_syns_per_pt=spec.max_syns_per_pt)
    metrics = compute_metrics_from_pairs(pairs, label=spec.label, source_group=source_group, target_group=target_group)
    return pairs, metrics


def score_against_reference(candidate: ConnectivityMetrics, reference: ConnectivityMetrics) -> tuple[float, dict[str, float]]:
    penalties = {
        "entry_count": 2.5 * _log_count_penalty(candidate.entry_count, reference.entry_count),
        "source_coverage": 2.0 * _relative_penalty(candidate.source_coverage, reference.source_coverage, floor=0.01),
        "target_coverage": 2.0 * _relative_penalty(candidate.target_coverage, reference.target_coverage, floor=0.01),
        "entries_per_source_total": 1.5 * _relative_penalty(
            candidate.mean_entries_per_source_total,
            reference.mean_entries_per_source_total,
            floor=0.01,
        ),
        "entries_per_target_total": 1.5 * _relative_penalty(
            candidate.mean_entries_per_target_total,
            reference.mean_entries_per_target_total,
            floor=0.01,
        ),
        "distance_median": 1.0 * _relative_penalty(candidate.median_distance_um, reference.median_distance_um, floor=0.1),
        "distance_p90": 0.75 * _relative_penalty(candidate.p90_distance_um, reference.p90_distance_um, floor=0.1),
        "source_family": 1.5 * _hist_l1(candidate.source_family_fraction, reference.source_family_fraction),
        "target_family": 1.5 * _hist_l1(candidate.target_family_fraction, reference.target_family_fraction),
    }
    total_penalty = float(sum(penalties.values()))
    score = 1.0 / (1.0 + total_penalty)
    return score, penalties


def score_epli_candidate(
    candidate: ConnectivityMetrics,
    *,
    reference: ConnectivityMetrics,
    preferred_source_family: str = "dend",
    preferred_target_family: str = "dend",
) -> tuple[float, dict[str, float]]:
    source_family_bonus = _family_fraction(candidate.source_family_fraction, preferred_source_family)
    target_family_bonus = _family_fraction(candidate.target_family_fraction, preferred_target_family)
    source_soma_fraction = _family_fraction(candidate.source_family_fraction, "soma")
    target_soma_fraction = _family_fraction(candidate.target_family_fraction, "soma")
    structural_fit = source_family_bonus * target_family_bonus
    local_scale = max(_safe_float(reference.median_distance_um, 10.0), 1.0)
    locality_bonus = 1.0 / (1.0 + (_safe_float(candidate.median_distance_um, local_scale) / local_scale))

    target_density = candidate.mean_entries_per_target_total
    reference_density = max(reference.mean_entries_per_target_total, 0.1)
    overconnect_penalty = max(0.0, target_density - (2.0 * reference_density)) / (2.0 * reference_density)

    penalties = {
        "zero_entries": 8.0 if candidate.entry_count <= 0 else 0.0,
        "low_source_coverage": 2.5 * max(0.0, 0.5 - candidate.source_coverage),
        "low_target_coverage": 2.5 * max(0.0, 0.35 - candidate.target_coverage),
        "overconnect": 2.0 * overconnect_penalty,
        "nonpreferred_target": 6.0 * max(0.0, 1.0 - target_family_bonus),
        "nonpreferred_source": 5.0 * max(0.0, 1.0 - source_family_bonus),
        "soma_source": 4.0 * source_soma_fraction,
        "soma_target": 5.0 * target_soma_fraction,
        "nonlocal": 1.0 * max(0.0, 0.5 - locality_bonus),
    }
    reward = structural_fit * (
        min(math.log1p(max(candidate.entry_count, 0)), math.log(21.0)) / math.log(21.0)
        + 1.5 * candidate.source_coverage
        + 1.5 * candidate.target_coverage
        + 0.75 * locality_bonus
    )
    total_penalty = float(sum(penalties.values()))
    score = reward / (1.0 + total_penalty)
    return score, penalties


def grid_search_against_reference(
    slice_name_or_path: str | Path,
    *,
    reference_synapse_set: str,
    source_patterns: Sequence[str],
    target_patterns: Sequence[str],
    max_distances_um: Sequence[float],
    use_radii: Sequence[bool] = (True,),
    max_syns_per_pts: Sequence[int] = (1,),
) -> list[SearchResult]:
    groups = load_slice_geometry(slice_name_or_path)
    reference = observed_metrics_for_synapse_set(slice_name_or_path, reference_synapse_set, groups=groups)
    source_group, target_group = reference_synapse_set.split("__", 1)
    max_distance_search = float(max(max_distances_um))
    pool_cache: dict[tuple[str, str, bool], CandidatePool] = {}

    results: list[SearchResult] = []
    for source_pattern, target_pattern, max_distance_um, use_radius, max_syns_per_pt in itertools.product(
        source_patterns,
        target_patterns,
        max_distances_um,
        use_radii,
        max_syns_per_pts,
    ):
        pool_key = (source_pattern, target_pattern, bool(use_radius))
        if pool_key not in pool_cache:
            pool_cache[pool_key] = build_candidate_pool(
                groups[source_group],
                groups[target_group],
                source_patterns=[source_pattern],
                target_patterns=[target_pattern],
                max_distance_um=max_distance_search,
                use_radius=bool(use_radius),
            )
        spec = SearchSpec(
            source_group=source_group,
            target_group=target_group,
            source_pattern=source_pattern,
            target_pattern=target_pattern,
            max_distance_um=float(max_distance_um),
            use_radius=bool(use_radius),
            max_syns_per_pt=int(max_syns_per_pt),
        )
        _, metrics = evaluate_search_spec(slice_name_or_path, spec, groups=groups, pool=pool_cache[pool_key])
        score, penalties = score_against_reference(metrics, reference)
        results.append(SearchResult(spec=spec, metrics=metrics, score=score, penalties=penalties, objective=reference_synapse_set))

    results.sort(key=lambda result: result.score, reverse=True)
    return results


def grid_search_epli_candidates(
    slice_name_or_path: str | Path,
    *,
    source_group: str = "EPLIs",
    target_group: str = "MCs",
    source_patterns: Sequence[str],
    target_patterns: Sequence[str],
    max_distances_um: Sequence[float],
    use_radii: Sequence[bool] = (True,),
    max_syns_per_pts: Sequence[int] = (1,),
    reference_synapse_set: str = "GCs__MCs",
) -> list[SearchResult]:
    groups = load_slice_geometry(slice_name_or_path)
    reference = observed_metrics_for_synapse_set("DorsalColumnSlice", reference_synapse_set)
    max_distance_search = float(max(max_distances_um))
    pool_cache: dict[tuple[str, str, bool], CandidatePool] = {}
    results: list[SearchResult] = []

    for source_pattern, target_pattern, max_distance_um, use_radius, max_syns_per_pt in itertools.product(
        source_patterns,
        target_patterns,
        max_distances_um,
        use_radii,
        max_syns_per_pts,
    ):
        pool_key = (source_pattern, target_pattern, bool(use_radius))
        if pool_key not in pool_cache:
            pool_cache[pool_key] = build_candidate_pool(
                groups[source_group],
                groups[target_group],
                source_patterns=[source_pattern],
                target_patterns=[target_pattern],
                max_distance_um=max_distance_search,
                use_radius=bool(use_radius),
            )
        spec = SearchSpec(
            source_group=source_group,
            target_group=target_group,
            source_pattern=source_pattern,
            target_pattern=target_pattern,
            max_distance_um=float(max_distance_um),
            use_radius=bool(use_radius),
            max_syns_per_pt=int(max_syns_per_pt),
        )
        _, metrics = evaluate_search_spec(slice_name_or_path, spec, groups=groups, pool=pool_cache[pool_key])
        score, penalties = score_epli_candidate(metrics, reference=reference)
        results.append(
            SearchResult(
                spec=spec,
                metrics=metrics,
                score=score,
                penalties=penalties,
                objective=f"epli_heuristic:{reference_synapse_set}",
            )
        )

    results.sort(key=lambda result: result.score, reverse=True)
    return results


def pretty_top_results(results: Sequence[SearchResult], *, top_n: int = 10) -> str:
    lines = []
    for rank, result in enumerate(results[:top_n], start=1):
        metrics = result.metrics
        lines.append(
            f"{rank:>2}. score={result.score:.4f} | {result.spec.label} | "
            f"entries={metrics.entry_count} src_cov={metrics.source_coverage:.3f} "
            f"dst_cov={metrics.target_coverage:.3f} "
            f"src_mean={metrics.mean_entries_per_source_total:.2f} "
            f"dst_mean={metrics.mean_entries_per_target_total:.2f} "
            f"dist50={_safe_float(metrics.median_distance_um):.2f}"
        )
    return "\n".join(lines)


__all__ = [
    "ConnectivityMetrics",
    "CandidatePool",
    "GroupGeometry",
    "SearchResult",
    "SearchSpec",
    "TerminalCollection",
    "build_candidate_pool",
    "evaluate_search_spec",
    "grid_search_against_reference",
    "grid_search_epli_candidates",
    "load_group_geometry",
    "load_slice_geometry",
    "observed_metrics_for_synapse_set",
    "pretty_top_results",
    "resolve_slice_dir",
    "score_against_reference",
    "score_epli_candidate",
    "select_pairs_from_pool",
    "suggest_section_patterns",
]
