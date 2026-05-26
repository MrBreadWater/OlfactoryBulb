from __future__ import annotations

import argparse
import math
import pickle
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


REPO = Path("/home/alek/OlfactoryBulb")
DEFAULT_OUTPUT_DIR = REPO / "media/website_header_blenderneuron_style_v22"
DEFAULT_ACTIVITY_RUN = REPO / "results/notebook_runs/obgpu_experiment_GammaSignature_fast_20260520_035424"
WIDTH = 2280
HEIGHT = 720
SUPERSAMPLE = 2
GIF_COLORS = 144
VERTICAL_FOREGROUND_SCALE = 1.20
ACTIVITY_PERIOD_MS = 200.0
ACTIVITY_PROFILE_BINS = 400
ACTIVITY_SKIP_MS = 400.0
ACTIVITY_PROPAGATION_DELAY_MS = 52.0
ACTIVITY_PACKET_SIGMA_MS = 2.85
BRANCH_EVENT_PROFILE_BINS = 400
ACTIVITY_TRACE_WINDOW_MS = 104.0
ACTIVITY_TRACE_TAU_MS = 58.0
SOMA_RESPONSE_DELAY_MS = 30.0
SOMA_AFTERHYPERPOLARIZATION_DELAY_MS = 5.5
SOMA_AFTERHYPERPOLARIZATION_WINDOW_MS = 30.0
SOMA_AFTERHYPERPOLARIZATION_TAU_MS = 9.5
AXON_EMISSION_DELAY_MS = 10.5
AXON_PACKET_GAIN = 1.28
AXON_EXTENSION_SEGMENTS = 7
SOMA_REFERENCE_RADII = {
    "MC": 5.3,
    "TC": 3.9,
    "GC": 1.9,
}
SOMA_DISPLAY_RADII = {
    "MC": 9.8 * SUPERSAMPLE,
    "TC": 8.0 * SUPERSAMPLE,
    "GC": 6.3 * SUPERSAMPLE,
}
SOMA_DISPLAY_EMPHASIS = {
    "MC": 1.03,
    "TC": 1.00,
    "GC": 0.95,
}
DOF_FOCUS_Z = 0.02
DOF_SHARP_ZONE = 0.07
DOF_FULL_ZONE = 0.34
# ICON site cues: ASU maroon/gold accents plus cyan/green activity from
# the existing header; keep the background true black.
INK = np.array([0, 0, 0], dtype=float)
MAROON = np.array([140, 29, 64], dtype=float)
GOLD = np.array([241, 161, 67], dtype=float)
TEAL = np.array([21, 168, 152], dtype=float)
CYAN = np.array([8, 232, 232], dtype=float)
GREEN = np.array([104, 200, 8], dtype=float)
WHITE = np.array([255, 255, 255], dtype=float)
BG = tuple(int(channel) for channel in INK)
TYPE_COLORS = {
    "MC": MAROON,
    "TC": GOLD,
    "GC": TEAL,
}


@dataclass(frozen=True)
class SwcNode:
    node_id: int
    kind: int
    xyz: np.ndarray
    radius: float
    parent_id: int


@dataclass
class Morphology:
    name: str
    cell_type: str
    nodes: dict[int, SwcNode]
    children: dict[int, list[int]]
    root_id: int
    distances: dict[int, float]
    max_distance: float


@dataclass(frozen=True)
class ActivityProfile:
    label: str
    cell_type: str
    values: np.ndarray
    event_phases_ms: tuple[float, ...]
    event_strengths: tuple[float, ...]
    peak_phase_ms: float
    score: float


@dataclass(frozen=True)
class BranchInputProfile:
    target_cell: str
    section_type: str
    section_index: int
    source_type: str
    profile: ActivityProfile
    event_count: int


@dataclass(frozen=True)
class PlacedMorph:
    morphology: Morphology
    center: tuple[float, float]
    scale: float
    yaw: float
    pitch: float
    roll: float
    color: np.ndarray
    alpha: float
    width_scale: float
    distance_offset: float = 0.0
    z_bias: float = 0.0
    contact_points: tuple[tuple[float, float, float], ...] = ()
    activity_profile: ActivityProfile | None = None


@dataclass(frozen=True)
class RenderSegment:
    x0: float
    y0: float
    x1: float
    y1: float
    z: float
    width: float
    distance: float
    color: np.ndarray
    alpha: float
    cell_type: str
    neurite_kind: int
    flow_direction: float
    activity_profile: ActivityProfile | None
    branch_phase_ms: float
    branch_gain: float


@dataclass(frozen=True)
class RenderNode:
    x: float
    y: float
    z: float
    radius: float
    morph_radius: float
    distance: float
    color: np.ndarray
    cell_type: str
    terminal: bool
    soma: bool
    activity_profile: ActivityProfile | None
    neurite_kind: int
    flow_direction: float
    branch_phase_ms: float
    branch_gain: float


@dataclass
class SceneCache:
    base: Image.Image
    segments: list[RenderSegment]
    nodes: list[RenderNode]


def mix(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def brighten(color: np.ndarray, factor: float, lift: float = 0.0) -> np.ndarray:
    return np.clip(color * factor + lift, 0, 255)


def rgba(color: np.ndarray, alpha: float) -> tuple[int, int, int, int]:
    rgb = np.clip(color, 0, 255).astype(np.uint8)
    return int(rgb[0]), int(rgb[1]), int(rgb[2]), int(np.clip(alpha, 0, 255))


def swc_rows(path: Path) -> list[SwcNode]:
    nodes: list[SwcNode] = []
    for line in path.read_text(errors="ignore").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        if len(parts) < 7:
            continue
        nodes.append(
            SwcNode(
                node_id=int(float(parts[0])),
                kind=int(float(parts[1])),
                xyz=np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=float),
                radius=max(0.25, float(parts[5])),
                parent_id=int(float(parts[6])),
            )
        )
    return nodes


def load_morphology(name: str, cell_type: str, path: str | Path) -> Morphology:
    rows = swc_rows(Path(path))
    nodes = {row.node_id: row for row in rows}
    children = {row.node_id: [] for row in rows}
    root_candidates: list[int] = []
    for row in rows:
        if row.parent_id < 0 or row.parent_id not in nodes:
            root_candidates.append(row.node_id)
        else:
            children[row.parent_id].append(row.node_id)
    root_id = root_candidates[0] if root_candidates else rows[0].node_id

    distances: dict[int, float] = {root_id: 0.0}
    stack = [root_id]
    while stack:
        node_id = stack.pop()
        parent = nodes[node_id]
        for child_id in children.get(node_id, []):
            child = nodes[child_id]
            distances[child_id] = distances[node_id] + float(np.linalg.norm(child.xyz - parent.xyz))
            stack.append(child_id)
    for row in rows:
        distances.setdefault(row.node_id, 0.0)
    max_distance = max(distances.values()) if distances else 1.0
    return Morphology(name, cell_type, nodes, children, root_id, distances, max(max_distance, 1e-6))


def circular_gaussian(values: np.ndarray, sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return values.astype(float, copy=True)
    radius = max(1, int(math.ceil(4.0 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-(offsets * offsets) / (2.0 * sigma_bins * sigma_bins))
    kernel /= np.sum(kernel)
    smoothed = np.zeros_like(values, dtype=float)
    for weight, offset in zip(kernel, offsets):
        smoothed += float(weight) * np.roll(values, int(offset))
    return smoothed


def normalized_profile(values: np.ndarray, low_pct: float = 15.0, high_pct: float = 99.0) -> tuple[np.ndarray, float]:
    values = np.nan_to_num(values.astype(float, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    low = float(np.percentile(values, low_pct))
    high = float(np.percentile(values, high_pct))
    spread = max(high - low, 1e-9)
    return np.clip((values - low) / spread, 0.0, 1.0), spread


def cell_type_from_label(label: str) -> str:
    match = re.match(r"([A-Z]+)", label)
    return match.group(1) if match else "UNK"


def cell_key_from_label(label: str) -> str | None:
    match = re.search(r"([A-Z]+\d+\[\d+\])", label)
    return match.group(1) if match else None


def parse_section_ref(text: str) -> tuple[str, str, int] | None:
    match = re.search(r"([A-Z]+\d+\[\d+\])\.(\w+)\[(\d+)\]", text)
    if not match:
        return None
    return match.group(1), match.group(2), int(match.group(3))


def circular_phase_distance(a_ms: float, b_ms: float, period_ms: float = ACTIVITY_PERIOD_MS) -> float:
    return abs(((a_ms - b_ms + 0.5 * period_ms) % period_ms) - 0.5 * period_ms)


def profile_events(values: np.ndarray, period_ms: float, max_events: int = 3) -> tuple[tuple[float, ...], tuple[float, ...]]:
    threshold = max(0.36, float(np.percentile(values, 82.0)))
    candidates: list[tuple[float, float]] = []
    for idx, value in enumerate(values):
        prev_value = values[(idx - 1) % values.size]
        next_value = values[(idx + 1) % values.size]
        if value >= threshold and value >= prev_value and value >= next_value:
            candidates.append((idx / values.size * period_ms, float(value)))
    if not candidates:
        peak_idx = int(np.argmax(values))
        candidates.append((peak_idx / values.size * period_ms, float(values[peak_idx])))

    chosen: list[tuple[float, float]] = []
    for phase_ms, strength in sorted(candidates, key=lambda item: item[1], reverse=True):
        if all(circular_phase_distance(phase_ms, other_phase, period_ms) >= 18.0 for other_phase, _ in chosen):
            chosen.append((phase_ms, strength))
        if len(chosen) >= max_events:
            break
    chosen.sort(key=lambda item: item[0])
    phases = tuple(phase for phase, _ in chosen)
    strengths = tuple(max(0.34, min(1.0, strength)) for _, strength in chosen)
    return phases, strengths


def event_times_to_profile(
    label: str,
    cell_type: str,
    event_times_ms: Iterable[float],
    *,
    period_ms: float,
    bins: int,
    max_events: int = 4,
) -> ActivityProfile | None:
    times = np.asarray(list(event_times_ms), dtype=np.float32)
    times = times[np.isfinite(times)]
    times = times[times >= ACTIVITY_SKIP_MS]
    if times.size < 2:
        return None

    phases = np.mod(times, period_ms)
    bin_idx = np.floor(phases / period_ms * bins).astype(np.int32) % bins
    hist = np.bincount(bin_idx, minlength=bins).astype(float)
    hist = circular_gaussian(hist, sigma_bins=bins * 1.2 / period_ms)
    activity, spread = normalized_profile(hist, low_pct=8.0, high_pct=99.7)
    activity = np.clip(activity, 0.0, 1.0) ** 0.74
    event_phases_ms, event_strengths = profile_events(activity, period_ms, max_events=max_events)
    if not event_phases_ms:
        return None
    score = float(spread + min(times.size / 90.0, 1.4))
    return ActivityProfile(label, cell_type, activity, event_phases_ms, event_strengths, float(np.argmax(activity)) / bins * period_ms, score)


@lru_cache(maxsize=4)
def load_branch_input_profiles(
    run_dir: str = str(DEFAULT_ACTIVITY_RUN),
    period_ms: float = ACTIVITY_PERIOD_MS,
    bins: int = BRANCH_EVENT_PROFILE_BINS,
) -> tuple[BranchInputProfile, ...]:
    run_path = Path(run_dir)
    grouped: dict[tuple[str, str, int, str], list[float]] = {}

    gc_path = run_path / "gc_output_events.pkl"
    if gc_path.exists():
        with gc_path.open("rb") as handle:
            events = pickle.load(handle)
        for event in events:
            if not isinstance(event, dict) or not event.get("times"):
                continue
            dest = parse_section_ref(str(event.get("dest_section", "")))
            if dest is None:
                continue
            target_cell, section_type, section_index = dest
            delay = float(event.get("delay") or 0.0)
            key = (target_cell, section_type, section_index, "GC")
            grouped.setdefault(key, []).extend(float(t) + delay for t in event["times"])

    input_path = run_path / "input_times.pkl"
    if input_path.exists():
        with input_path.open("rb") as handle:
            input_events = pickle.load(handle)
        for label, times in input_events:
            if not times:
                continue
            dest = parse_section_ref(str(label))
            if dest is None:
                continue
            target_cell, section_type, section_index = dest
            key = (target_cell, section_type, section_index, "INPUT")
            grouped.setdefault(key, []).extend(float(t) for t in times)

    profiles: list[BranchInputProfile] = []
    for (target_cell, section_type, section_index, source_type), times in grouped.items():
        cell_type = cell_type_from_label(target_cell)
        source_label = "GC output" if source_type == "GC" else "input"
        label = f"{source_label} -> {target_cell}.{section_type}[{section_index}]"
        max_events = 5 if source_type == "GC" else 4
        profile = event_times_to_profile(label, cell_type, times, period_ms=period_ms, bins=bins, max_events=max_events)
        if profile is None:
            continue
        profiles.append(
            BranchInputProfile(
                target_cell=target_cell,
                section_type=section_type,
                section_index=section_index,
                source_type=source_type,
                profile=profile,
                event_count=len(times),
            )
        )
    profiles.sort(key=lambda item: (item.target_cell, item.section_type, item.section_index, -item.event_count))
    return tuple(profiles)


def branch_input_cells() -> set[str]:
    return {profile.target_cell for profile in load_branch_input_profiles()}


def fold_trace_activity(
    label: str,
    trace_index: int,
    times: Iterable[float],
    volts: Iterable[float],
    *,
    period_ms: float,
    bins: int,
) -> ActivityProfile | None:
    t = np.asarray(times, dtype=np.float32)
    v = np.asarray(volts, dtype=np.float32)
    finite = np.isfinite(t) & np.isfinite(v)
    if int(np.count_nonzero(finite)) < 16:
        return None

    t = t[finite]
    v = v[finite]
    start_ms = min(float(t[0]) + ACTIVITY_SKIP_MS, max(float(t[0]), float(t[-1]) - period_ms))
    mask = t >= start_ms
    if int(np.count_nonzero(mask)) < bins:
        mask = t >= float(t[0])
    folded_t = t[mask]
    folded_v = v[mask]
    if folded_t.size < bins:
        return None

    phases = np.mod(folded_t, period_ms)
    bin_idx = np.floor(phases / period_ms * bins).astype(np.int32) % bins

    rest = float(np.percentile(folded_v, 15.0))
    ceiling = float(np.percentile(folded_v, 99.4))
    depol = np.clip((folded_v - rest) / max(ceiling - rest, 1e-6), 0.0, 1.0)
    sums = np.bincount(bin_idx, weights=depol, minlength=bins)
    counts = np.bincount(bin_idx, minlength=bins)
    depol_profile = sums / np.maximum(counts, 1)
    depol_profile = circular_gaussian(depol_profile, sigma_bins=bins * 3.2 / period_ms)
    depol_norm, depol_spread = normalized_profile(depol_profile)

    crossings = np.where((v[:-1] < -20.0) & (v[1:] >= -20.0))[0] + 1
    spike_times = t[crossings]
    spike_times = spike_times[spike_times >= start_ms]
    spike_hist = np.zeros(bins, dtype=float)
    if spike_times.size:
        spike_phase = np.mod(spike_times, period_ms)
        spike_idx = np.floor(spike_phase / period_ms * bins).astype(np.int32) % bins
        spike_hist = np.bincount(spike_idx, minlength=bins).astype(float)
        spike_hist = circular_gaussian(spike_hist, sigma_bins=bins * 4.8 / period_ms)
    spike_norm, spike_spread = normalized_profile(spike_hist)

    spike_weight = 0.0 if spike_times.size == 0 else 0.48
    activity = (1.0 - spike_weight) * depol_norm + spike_weight * spike_norm
    activity = circular_gaussian(activity, sigma_bins=bins * 1.4 / period_ms)
    activity, _ = normalized_profile(activity, low_pct=8.0, high_pct=99.2)
    activity = np.clip(activity, 0.0, 1.0) ** 0.82

    cycles = max((float(folded_t[-1]) - float(folded_t[0])) / period_ms, 1.0)
    spike_rate = float(spike_times.size) / cycles
    score = depol_spread + 0.14 * min(spike_rate, 2.5) + 0.05 * min(spike_spread, 1.0)
    peak_phase_ms = float(np.argmax(activity)) / bins * period_ms
    event_phases_ms, event_strengths = profile_events(activity, period_ms)
    unique_label = f"{label} #{trace_index:03d}"
    return ActivityProfile(
        unique_label,
        cell_type_from_label(label),
        activity,
        event_phases_ms,
        event_strengths,
        peak_phase_ms,
        float(score),
    )


@lru_cache(maxsize=4)
def load_activity_profiles(
    run_dir: str = str(DEFAULT_ACTIVITY_RUN),
    period_ms: float = ACTIVITY_PERIOD_MS,
    bins: int = ACTIVITY_PROFILE_BINS,
) -> tuple[ActivityProfile, ...]:
    soma_path = Path(run_dir) / "soma_vs.pkl"
    with soma_path.open("rb") as handle:
        traces = pickle.load(handle)
    profiles: list[ActivityProfile] = []
    for trace_index, (label, times, volts) in enumerate(traces):
        profile = fold_trace_activity(str(label), trace_index, times, volts, period_ms=period_ms, bins=bins)
        if profile is not None and profile.cell_type in TYPE_COLORS and math.isfinite(profile.score):
            profiles.append(profile)
    if not profiles:
        raise RuntimeError(f"No usable activity profiles found in {soma_path}")
    return tuple(profiles)


def select_activity_profiles(counts: dict[str, int]) -> dict[str, list[ActivityProfile]]:
    profiles = load_activity_profiles()
    input_cells = branch_input_cells()
    selected: dict[str, list[ActivityProfile]] = {}
    for cell_type, count in counts.items():
        candidates = sorted(
            (profile for profile in profiles if profile.cell_type == cell_type and profile.score > 0.015),
            key=lambda profile: profile.score,
            reverse=True,
        )
        if not candidates:
            selected[cell_type] = []
            continue
        pool = candidates[: min(90, len(candidates))]
        chosen: list[ActivityProfile] = []
        while len(chosen) < count and pool:
            def rank(profile: ActivityProfile) -> float:
                value = profile.score
                if cell_key_from_label(profile.label) in input_cells:
                    value *= 1.16
                if chosen:
                    nearest_phase = min(circular_phase_distance(profile.peak_phase_ms, other.peak_phase_ms) for other in chosen)
                    phase_bonus = 0.52 + 0.48 * min(1.0, nearest_phase / 42.0)
                    base_label = profile.label.split(" #", 1)[0]
                    label_bonus = 0.74 if any(other.label.split(" #", 1)[0] == base_label for other in chosen) else 1.0
                    value *= phase_bonus * label_bonus
                return value

            best = max(pool, key=rank)
            chosen.append(best)
            pool.remove(best)
        while len(chosen) < count:
            chosen.append(candidates[len(chosen) % len(candidates)])
        selected[cell_type] = chosen
    return selected


def attach_activity_profiles(placed: list[PlacedMorph]) -> list[PlacedMorph]:
    counts: dict[str, int] = {}
    for item in placed:
        counts[item.morphology.cell_type] = counts.get(item.morphology.cell_type, 0) + 1
    profiles_by_type = select_activity_profiles(counts)
    used: dict[str, int] = {cell_type: 0 for cell_type in counts}
    attached: list[PlacedMorph] = []
    for item in placed:
        cell_type = item.morphology.cell_type
        profiles = profiles_by_type.get(cell_type, [])
        profile = profiles[used[cell_type] % len(profiles)] if profiles else None
        used[cell_type] += 1
        attached.append(replace(item, activity_profile=profile))
    return attached


def circular_time_delta_ms(a_ms: float, b_ms: float, period_ms: float = ACTIVITY_PERIOD_MS) -> float:
    return ((a_ms - b_ms + 0.5 * period_ms) % period_ms) - 0.5 * period_ms


def packet_activity(
    profile: ActivityProfile | None,
    loop_ms: float,
    distance: float,
    flow_direction: float,
    branch_phase_ms: float,
    branch_gain: float,
) -> float:
    if profile is None or not profile.event_phases_ms:
        return 0.0
    best = 0.0
    for event_ms, strength in zip(profile.event_phases_ms, profile.event_strengths):
        if flow_direction >= 0.0:
            propagation_ms = ACTIVITY_PROPAGATION_DELAY_MS * distance
        else:
            propagation_ms = ACTIVITY_PROPAGATION_DELAY_MS * (1.0 - distance)
        expected_ms = event_ms + propagation_ms + branch_phase_ms
        delta_ms = circular_time_delta_ms(loop_ms, expected_ms)
        core = math.exp(-(delta_ms * delta_ms) / (2.0 * ACTIVITY_PACKET_SIGMA_MS * ACTIVITY_PACKET_SIGMA_MS))
        skirt = math.exp(-(delta_ms * delta_ms) / (2.0 * (ACTIVITY_PACKET_SIGMA_MS * 1.75) ** 2))
        best = max(best, strength * branch_gain * (0.90 * core + 0.10 * skirt))
    return min(1.0, best)


def profile_value(profile: ActivityProfile | None, loop_ms: float) -> float:
    if profile is None or profile.values.size == 0:
        return 0.0
    phase_ms = loop_ms % ACTIVITY_PERIOD_MS
    pos = phase_ms / ACTIVITY_PERIOD_MS * profile.values.size
    idx0 = int(math.floor(pos)) % profile.values.size
    idx1 = (idx0 + 1) % profile.values.size
    frac = pos - math.floor(pos)
    return float((1.0 - frac) * profile.values[idx0] + frac * profile.values[idx1])


def delayed_profile_value(profile: ActivityProfile | None, loop_ms: float, delay_ms: float) -> float:
    return profile_value(profile, loop_ms - delay_ms)


def event_decay_activity(
    profile: ActivityProfile | None,
    loop_ms: float,
    *,
    delay_ms: float,
    window_ms: float,
    tau_ms: float,
    gain: float = 1.0,
) -> float:
    if profile is None or not profile.event_phases_ms:
        return 0.0
    best = 0.0
    for event_ms, strength in zip(profile.event_phases_ms, profile.event_strengths):
        elapsed_ms = circular_time_delta_ms(loop_ms, event_ms + delay_ms)
        if 0.0 <= elapsed_ms <= window_ms:
            best = max(best, strength * gain * math.exp(-elapsed_ms / tau_ms))
    return min(1.0, best)


def trace_activity(
    profile: ActivityProfile | None,
    loop_ms: float,
    distance: float,
    flow_direction: float,
    branch_phase_ms: float,
    branch_gain: float,
) -> float:
    if profile is None or not profile.event_phases_ms:
        return 0.0
    best = 0.0
    if flow_direction >= 0.0:
        propagation_ms = ACTIVITY_PROPAGATION_DELAY_MS * distance
    else:
        propagation_ms = ACTIVITY_PROPAGATION_DELAY_MS * (1.0 - distance)
    for event_ms, strength in zip(profile.event_phases_ms, profile.event_strengths):
        expected_ms = event_ms + propagation_ms + branch_phase_ms
        elapsed_ms = circular_time_delta_ms(loop_ms, expected_ms)
        if 0.0 <= elapsed_ms <= ACTIVITY_TRACE_WINDOW_MS:
            best = max(best, strength * branch_gain * math.exp(-elapsed_ms / ACTIVITY_TRACE_TAU_MS))
    return min(1.0, best)


def neurite_flow_direction(kind: int) -> float:
    return 1.0 if kind == 2 else -1.0


def segment_kind(child: SwcNode, parent: SwcNode) -> int:
    if child.kind == 2 or parent.kind == 2:
        return 2
    if child.kind in (3, 4):
        return child.kind
    return parent.kind


def rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    ya, pi, ro = map(math.radians, (yaw, pitch, roll))
    cy, sy = math.cos(ya), math.sin(ya)
    cp, sp = math.cos(pi), math.sin(pi)
    cr, sr = math.cos(ro), math.sin(ro)
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return rz @ ry @ rx


def normalized_coords(morph: Morphology) -> dict[int, np.ndarray]:
    coords = np.array([node.xyz for node in morph.nodes.values()], dtype=float)
    root = morph.nodes[morph.root_id].xyz
    centered = coords - root
    span = np.ptp(centered, axis=0)
    scale = max(float(np.max(span[:2])), float(span[2]) * 1.4, 1.0)
    remapped: dict[int, np.ndarray] = {}
    for node_id, node in morph.nodes.items():
        xyz = (node.xyz - root) / scale
        xyz = np.array([xyz[0], xyz[1], xyz[2] * 1.55], dtype=float)
        remapped[node_id] = xyz
    return remapped


def stable_unit(*parts: object) -> float:
    text = "|".join(str(part) for part in parts)
    h = 2166136261
    for char in text:
        h ^= ord(char)
        h = (h * 16777619) & 0xFFFFFFFF
    return h / 0xFFFFFFFF


def unit_vector(x: float, y: float) -> tuple[float, float]:
    length = math.hypot(x, y)
    if length <= 1e-9:
        return 1.0, 0.0
    return x / length, y / length


def bezier3(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    t: float,
) -> np.ndarray:
    u = 1.0 - t
    return (
        (u * u * u) * p0
        + (3.0 * u * u * t) * p1
        + (3.0 * u * t * t) * p2
        + (t * t * t) * p3
    )


def chaikin_closed(points: list[np.ndarray], refinements: int = 2) -> list[np.ndarray]:
    smoothed = [point.astype(float, copy=True) for point in points]
    for _ in range(max(0, refinements)):
        refined: list[np.ndarray] = []
        for idx, point in enumerate(smoothed):
            nxt = smoothed[(idx + 1) % len(smoothed)]
            refined.append(0.75 * point + 0.25 * nxt)
            refined.append(0.25 * point + 0.75 * nxt)
        smoothed = refined
    return smoothed


def primary_branch_ids(morph: Morphology) -> dict[int, int]:
    branch_for: dict[int, int] = {morph.root_id: morph.root_id}
    for child_id in morph.children.get(morph.root_id, []):
        stack = [child_id]
        while stack:
            node_id = stack.pop()
            branch_for[node_id] = child_id
            stack.extend(morph.children.get(node_id, []))
    for node_id in morph.nodes:
        branch_for.setdefault(node_id, node_id)
    return branch_for


def branch_timing(morph: Morphology, branch_id: int, distance_offset: float) -> tuple[float, float]:
    phase_unit = stable_unit(morph.name, branch_id, round(distance_offset, 3), "phase")
    gain_unit = stable_unit(morph.name, branch_id, round(distance_offset, 3), "gain")
    phase_ms = (phase_unit - 0.5) * 13.0
    gain = 0.80 + 0.42 * gain_unit
    return phase_ms, gain


def branch_screen_order(
    morph: Morphology,
    branch_for: dict[int, int],
    projected: dict[int, np.ndarray],
) -> dict[int, int]:
    root = projected.get(morph.root_id)
    if root is None:
        return {}
    entries: list[tuple[float, float, int]] = []
    for branch_id in sorted(set(branch_for.values())):
        if branch_id == morph.root_id or branch_id not in projected:
            continue
        point = projected[branch_id]
        angle = math.atan2(float(point[1] - root[1]), float(point[0] - root[0]))
        radius = math.hypot(float(point[0] - root[0]), float(point[1] - root[1]))
        entries.append((angle, radius, branch_id))
    return {branch_id: idx for idx, (_, _, branch_id) in enumerate(sorted(entries))}


def input_profiles_for_cell(target_cell: str, section_type: str) -> list[BranchInputProfile]:
    profiles = [
        profile
        for profile in load_branch_input_profiles()
        if profile.target_cell == target_cell and profile.section_type == section_type
    ]
    profiles.sort(key=lambda item: (-item.event_count, item.section_index, item.source_type))
    return profiles


def choose_actual_branch_input(
    item: PlacedMorph,
    branch_id: int,
    branch_rank: int,
    neurite_kind: int,
) -> ActivityProfile | None:
    target_cell = cell_key_from_label(item.activity_profile.label) if item.activity_profile else None
    if target_cell is None or item.morphology.cell_type not in ("MC", "TC"):
        return None

    preferred_sections = ("apic", "dend") if neurite_kind == 4 else ("dend", "apic")
    for section_type in preferred_sections:
        candidates = input_profiles_for_cell(target_cell, section_type)
        if not candidates:
            continue
        section_offset = int(stable_unit(target_cell, item.morphology.name, branch_id, round(item.distance_offset, 3), section_type) * len(candidates))
        chosen = candidates[(branch_rank + section_offset) % len(candidates)]
        return chosen.profile
    return None


def compatible_source_profiles(item: PlacedMorph, placed: list[PlacedMorph]) -> list[PlacedMorph]:
    target_type = item.morphology.cell_type
    if target_type == "GC":
        allowed = {"MC", "TC"}
    elif target_type in ("MC", "TC"):
        allowed = {"GC"}
    else:
        allowed = {"MC", "TC", "GC"}
    candidates = [
        other
        for other in placed
        if other is not item and other.activity_profile is not None and other.morphology.cell_type in allowed
    ]
    if candidates:
        return candidates
    return [other for other in placed if other is not item and other.activity_profile is not None]


def choose_compatible_source_profile(
    item: PlacedMorph,
    placed: list[PlacedMorph],
    branch_id: int,
) -> tuple[ActivityProfile | None, float, float]:
    candidates = compatible_source_profiles(item, placed)
    if not candidates:
        return item.activity_profile, 0.0, 1.0

    def rank(other: PlacedMorph) -> tuple[float, float]:
        dist = math.hypot(other.center[0] - item.center[0], other.center[1] - item.center[1])
        phase_gap = 0.0
        if item.activity_profile is not None and other.activity_profile is not None:
            phase_gap = circular_phase_distance(item.activity_profile.peak_phase_ms, other.activity_profile.peak_phase_ms)
        return dist, -phase_gap

    ranked = sorted(candidates, key=rank)
    window = ranked[: min(4, len(ranked))]
    index = int(stable_unit(item.morphology.name, branch_id, round(item.distance_offset, 3), "source") * len(window))
    source = window[index % len(window)]
    delay_ms = 4.0 + 16.0 * stable_unit(
        item.morphology.name,
        source.morphology.name,
        branch_id,
        round(item.distance_offset, 3),
        "source-delay",
    )
    gain = 0.78 + 0.34 * stable_unit(source.morphology.name, branch_id, "source-gain")
    return source.activity_profile, delay_ms, gain


def choose_branch_activity(
    item: PlacedMorph,
    placed: list[PlacedMorph],
    branch_id: int,
    branch_rank: int,
    neurite_kind: int,
) -> tuple[ActivityProfile | None, float, float]:
    if neurite_kind == 2:
        return item.activity_profile, 0.0, 1.0
    actual_input = choose_actual_branch_input(item, branch_id, branch_rank, neurite_kind)
    if actual_input is not None:
        phase_ms = -6.0 + 12.0 * stable_unit(actual_input.label, branch_id, round(item.distance_offset, 3), "site-phase")
        gain = 0.86 + 0.34 * stable_unit(actual_input.label, branch_id, "site-gain")
        return actual_input, phase_ms, gain
    return choose_compatible_source_profile(item, placed, branch_id)


def project_scene(placed: Iterable[PlacedMorph], width: int, height: int) -> tuple[list[RenderSegment], list[RenderNode]]:
    segments: list[RenderSegment] = []
    nodes_out: list[RenderNode] = []
    placed_list = list(placed)
    for item in placed_list:
        morph = item.morphology
        coords = normalized_coords(morph)
        branch_for = primary_branch_ids(morph)
        axon_distance_scale = max(
            (
                morph.distances[node_id]
                for node_id, node in morph.nodes.items()
                if node.kind == 2
            ),
            default=morph.max_distance,
        )
        axon_distance_scale = max(axon_distance_scale, 1e-6)
        matrix = rotation_matrix(item.yaw, item.pitch, item.roll)
        projected: dict[int, np.ndarray] = {}
        for node_id, xyz in coords.items():
            pos = matrix @ xyz
            sx = item.center[0] + width * item.scale * pos[0]
            sy = item.center[1] - height * item.scale * pos[1]
            path_weight = morph.distances[node_id] / morph.max_distance
            path_weight = path_weight ** 1.65
            for target_x, target_y, strength in item.contact_points:
                tx = width * target_x
                ty = height * target_y
                dx = tx - sx
                dy = ty - sy
                screen_dist = math.hypot(dx / width, dy / height)
                local_pull = strength * path_weight * math.exp(-(screen_dist**2) / (2.0 * 0.32**2))
                sx += dx * local_pull
                sy += dy * local_pull
            projected[node_id] = np.array([sx, sy, pos[2] + item.z_bias], dtype=float)
        branch_order = branch_screen_order(morph, branch_for, projected)

        for node_id, node in morph.nodes.items():
            parent_id = node.parent_id
            if parent_id < 0 or parent_id not in morph.nodes:
                continue
            p0 = projected[parent_id]
            p1 = projected[node_id]
            parent_node = morph.nodes[parent_id]
            radius = 0.5 * (node.radius + parent_node.radius)
            kind = segment_kind(node, parent_node)
            distance_scale = axon_distance_scale if kind == 2 else morph.max_distance
            dist = 0.5 * (morph.distances[node_id] + morph.distances.get(parent_id, 0.0)) / distance_scale
            branch_id = branch_for.get(node_id, node_id)
            base_phase_ms, base_gain = branch_timing(morph, branch_id, item.distance_offset)
            activity_profile, route_phase_ms, route_gain = choose_branch_activity(
                item,
                placed_list,
                branch_id,
                branch_order.get(branch_id, 0),
                kind,
            )
            width_px = SUPERSAMPLE * max(1.5, item.width_scale * (0.72 + 0.36 * math.sqrt(radius)) + 1.25)
            segments.append(
                RenderSegment(
                    x0=float(p0[0]),
                    y0=float(p0[1]),
                    x1=float(p1[0]),
                    y1=float(p1[1]),
                    z=float(0.5 * (p0[2] + p1[2])),
                    width=float(width_px),
                    distance=float((dist + item.distance_offset) % 1.0),
                    color=item.color,
                    alpha=item.alpha,
                    cell_type=morph.cell_type,
                    neurite_kind=kind,
                    flow_direction=neurite_flow_direction(kind),
                    activity_profile=activity_profile,
                    branch_phase_ms=base_phase_ms + route_phase_ms,
                    branch_gain=base_gain * route_gain,
                )
            )
            if kind == 2 and not morph.children.get(node_id):
                dx = float(p1[0] - p0[0])
                dy = float(p1[1] - p0[1])
                seg_len = math.hypot(dx, dy)
                if seg_len > 1e-6:
                    root = projected.get(morph.root_id, p0)
                    extend_len = max(seg_len * 7.2, item.scale * width * 0.10)
                    extend_len = min(extend_len, item.scale * width * 0.24)
                    ux, uy = unit_vector(dx, dy)
                    outx, outy = unit_vector(float(p1[0] - root[0]), float(p1[1] - root[1]))
                    dirx, diry = unit_vector(0.52 * ux + 0.48 * outx, 0.52 * uy + 0.48 * outy)
                    side_sign = -1.0 if stable_unit(morph.name, node_id, "axon-side") < 0.5 else 1.0
                    side_x, side_y = -diry * side_sign, dirx * side_sign
                    bend_mag = extend_len * (0.20 + 0.12 * stable_unit(morph.name, node_id, "axon-bend"))
                    sweep_mag = extend_len * (0.07 + 0.10 * stable_unit(morph.name, node_id, "axon-sweep"))
                    control1 = np.array(
                        [
                            float(p1[0] + dirx * extend_len * 0.18 + side_x * bend_mag * 1.18),
                            float(p1[1] + diry * extend_len * 0.18 + side_y * bend_mag * 1.18),
                            float(p1[2] - 0.003),
                        ],
                        dtype=float,
                    )
                    control2 = np.array(
                        [
                            float(p1[0] + dirx * extend_len * 0.62 + side_x * bend_mag * 1.28 + outx * sweep_mag),
                            float(p1[1] + diry * extend_len * 0.62 + side_y * bend_mag * 1.28 + outy * sweep_mag),
                            float(p1[2] - 0.006),
                        ],
                        dtype=float,
                    )
                    end = np.array(
                        [
                            float(p1[0] + dirx * extend_len + side_x * bend_mag * 0.78 + outx * sweep_mag * 1.22),
                            float(p1[1] + diry * extend_len + side_y * bend_mag * 0.78 + outy * sweep_mag * 1.22),
                            float(p1[2] - 0.009),
                        ],
                        dtype=float,
                    )
                    curve_start = np.array([float(p1[0]), float(p1[1]), float(p1[2])], dtype=float)
                    curve_points = [
                        bezier3(curve_start, control1, control2, end, step / AXON_EXTENSION_SEGMENTS)
                        for step in range(AXON_EXTENSION_SEGMENTS + 1)
                    ]
                    for ext_idx, (c0, c1) in enumerate(zip(curve_points[:-1], curve_points[1:]), start=1):
                        frac = ext_idx / AXON_EXTENSION_SEGMENTS
                        width_scale = 1.14 - 0.18 * frac
                        segments.append(
                            RenderSegment(
                                x0=float(c0[0]),
                                y0=float(c0[1]),
                                x1=float(c1[0]),
                                y1=float(c1[1]),
                                z=float(0.5 * (c0[2] + c1[2])),
                                width=float(width_px * width_scale),
                                distance=float(min(1.0, dist + 0.06 + 0.94 * frac)),
                                color=item.color,
                                alpha=item.alpha * (0.99 - 0.03 * frac),
                                cell_type=morph.cell_type,
                                neurite_kind=2,
                                flow_direction=1.0,
                                activity_profile=activity_profile,
                                branch_phase_ms=base_phase_ms + route_phase_ms,
                                branch_gain=base_gain * route_gain * 1.06,
                            )
                        )

        for node_id, node in morph.nodes.items():
            p = projected[node_id]
            degree = len(morph.children.get(node_id, [])) + int(node.parent_id in morph.nodes)
            if node.parent_id in morph.nodes:
                node_kind = segment_kind(node, morph.nodes[node.parent_id])
            elif morph.children.get(node_id):
                child = morph.nodes[morph.children[node_id][0]]
                node_kind = segment_kind(child, node)
            else:
                node_kind = node.kind
            distance_scale = axon_distance_scale if node_kind == 2 else morph.max_distance
            branch_id = branch_for.get(node_id, node_id)
            base_phase_ms, base_gain = branch_timing(morph, branch_id, item.distance_offset)
            if node_id == morph.root_id or node.kind == 1:
                activity_profile = item.activity_profile
                route_phase_ms = 0.0
                route_gain = 1.0
            else:
                activity_profile, route_phase_ms, route_gain = choose_branch_activity(
                    item,
                    placed_list,
                    branch_id,
                    branch_order.get(branch_id, 0),
                    node_kind,
                )
            nodes_out.append(
                RenderNode(
                    x=float(p[0]),
                    y=float(p[1]),
                    z=float(p[2]),
                    radius=float(max(1.4, item.width_scale * (1.0 + 0.55 * math.sqrt(node.radius)))),
                    morph_radius=float(node.radius),
                    distance=float((morph.distances[node_id] / distance_scale + item.distance_offset) % 1.0),
                    color=item.color,
                    cell_type=morph.cell_type,
                    terminal=degree <= 1,
                    soma=node_id == morph.root_id or node.kind == 1,
                    activity_profile=activity_profile,
                    neurite_kind=node_kind,
                    flow_direction=neurite_flow_direction(node_kind),
                    branch_phase_ms=base_phase_ms + route_phase_ms,
                    branch_gain=base_gain * route_gain,
                )
            )
    segments.sort(key=lambda seg: seg.z)
    nodes_out.sort(key=lambda node: node.z)
    return segments, nodes_out


def ellipse(draw: ImageDraw.ImageDraw, x: float, y: float, r: float, fill: tuple[int, int, int, int]) -> None:
    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)


def ellipse_xy(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    rx: float,
    ry: float,
    *,
    fill: tuple[int, int, int, int] | None = None,
    outline: tuple[int, int, int, int] | None = None,
    width: int = 1,
) -> None:
    draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=fill, outline=outline, width=max(1, int(round(width))))


def polygon_xy(
    draw: ImageDraw.ImageDraw,
    points: list[np.ndarray],
    *,
    fill: tuple[int, int, int, int] | None = None,
    outline: tuple[int, int, int, int] | None = None,
    width: int = 1,
) -> None:
    xy = [(float(point[0]), float(point[1])) for point in points]
    draw.polygon(xy, fill=fill, outline=outline)
    if outline is not None and width > 1:
        draw.line(xy + [xy[0]], fill=outline, width=max(1, int(round(width))), joint="curve")


def soma_body_axes(node: RenderNode) -> tuple[float, float]:
    # The morphologies are normalized independently for layout, so use
    # cell-type-biased target sizes and let the SWC soma radius only nudge
    # within-type variation rather than dictate cross-type scaling.
    reference = SOMA_REFERENCE_RADII.get(node.cell_type, 3.5)
    target = SOMA_DISPLAY_RADII.get(node.cell_type, 7.5 * SUPERSAMPLE)
    emphasis = SOMA_DISPLAY_EMPHASIS.get(node.cell_type, 1.0)
    relative_size = float(np.clip(node.morph_radius / reference, 0.78, 1.22))
    radius = target * emphasis * relative_size
    if node.cell_type == "GC":
        return radius * 0.94, radius * 1.18
    if node.cell_type == "MC":
        return radius * 1.02, radius * 1.06
    if node.cell_type == "TC":
        return radius * 0.98, radius * 1.02
    return radius, radius


def soma_shape_points(node: RenderNode, rx: float, ry: float, scale: float = 1.0) -> list[np.ndarray]:
    if node.cell_type == "GC":
        return []
    if node.cell_type == "MC":
        control = [
            np.array([0.00, -1.16]),
            np.array([0.62, -0.82]),
            np.array([0.90, -0.12]),
            np.array([0.70, 0.56]),
            np.array([0.20, 1.04]),
            np.array([-0.34, 1.00]),
            np.array([-0.82, 0.38]),
            np.array([-0.92, -0.34]),
            np.array([-0.48, -0.96]),
        ]
    else:
        control = [
            np.array([0.00, -1.08]),
            np.array([0.84, -0.54]),
            np.array([0.96, 0.10]),
            np.array([0.62, 0.82]),
            np.array([0.02, 1.02]),
            np.array([-0.66, 0.76]),
            np.array([-0.98, 0.00]),
            np.array([-0.62, -0.78]),
        ]
    smoothed = chaikin_closed(control, refinements=2)
    return [
        np.array(
            [
                node.x + point[0] * rx * scale,
                node.y + point[1] * ry * scale,
            ],
            dtype=float,
        )
        for point in smoothed
    ]


def draw_soma_shape(
    draw: ImageDraw.ImageDraw,
    node: RenderNode,
    rx: float,
    ry: float,
    *,
    fill: tuple[int, int, int, int] | None = None,
    outline: tuple[int, int, int, int] | None = None,
    width: int = 1,
    scale: float = 1.0,
) -> None:
    if node.cell_type == "GC":
        ellipse_xy(draw, node.x, node.y, rx * scale, ry * scale, fill=fill, outline=outline, width=width)
        return
    polygon_xy(draw, soma_shape_points(node, rx, ry, scale=scale), fill=fill, outline=outline, width=width)


def depth_defocus(z: float) -> float:
    delta = abs(z - DOF_FOCUS_Z)
    if delta <= DOF_SHARP_ZONE:
        return 0.0
    t = float(np.clip((delta - DOF_SHARP_ZONE) / max(1e-6, DOF_FULL_ZONE - DOF_SHARP_ZONE), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def draw_soft_line(
    draw: ImageDraw.ImageDraw,
    seg: RenderSegment,
    color: np.ndarray,
    alpha: float,
    width: float,
) -> None:
    draw.line(
        (seg.x0, seg.y0, seg.x1, seg.y1),
        fill=rgba(color, alpha),
        width=max(1, int(round(width))),
        joint="curve",
    )


def draw_packet_line(
    draw: ImageDraw.ImageDraw,
    seg: RenderSegment,
    color: np.ndarray,
    alpha: float,
    width: float,
    active: float,
    extra_length: float = 0.0,
) -> None:
    portion = min(1.0, 0.62 + 0.30 * active + extra_length)
    cx = 0.5 * (seg.x0 + seg.x1)
    cy = 0.5 * (seg.y0 + seg.y1)
    x0 = cx + (seg.x0 - cx) * portion
    y0 = cy + (seg.y0 - cy) * portion
    x1 = cx + (seg.x1 - cx) * portion
    y1 = cy + (seg.y1 - cy) * portion
    draw.line(
        (x0, y0, x1, y1),
        fill=rgba(color, alpha),
        width=max(1, int(round(width))),
        joint="curve",
    )


def render_bounds(
    segments: list[RenderSegment],
    nodes: list[RenderNode],
) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for seg in segments:
        pad = seg.width * 0.5
        xs.extend((seg.x0 - pad, seg.x0 + pad, seg.x1 - pad, seg.x1 + pad))
        ys.extend((seg.y0 - pad, seg.y0 + pad, seg.y1 - pad, seg.y1 + pad))
    for node in nodes:
        if node.soma:
            rx, ry = soma_body_axes(node)
            xs.extend((node.x - rx, node.x + rx))
            ys.extend((node.y - ry, node.y + ry))
        else:
            xs.extend((node.x - node.radius, node.x + node.radius))
            ys.extend((node.y - node.radius, node.y + node.radius))
    if not xs or not ys:
        return 0.0, 0.0, 0.0, 0.0
    return min(xs), min(ys), max(xs), max(ys)


def center_geometry(
    segments: list[RenderSegment],
    nodes: list[RenderNode],
    width: int,
    height: int,
) -> tuple[list[RenderSegment], list[RenderNode]]:
    left, top, right, bottom = render_bounds(segments, nodes)
    dx = width * 0.5 - (left + right) * 0.5
    dy = height * 0.5 - (top + bottom) * 0.5
    centered_segments = [
        replace(seg, x0=seg.x0 + dx, y0=seg.y0 + dy, x1=seg.x1 + dx, y1=seg.y1 + dy)
        for seg in segments
    ]
    centered_nodes = [replace(node, x=node.x + dx, y=node.y + dy) for node in nodes]
    return centered_segments, centered_nodes


def stretch_geometry_y(
    segments: list[RenderSegment],
    nodes: list[RenderNode],
    height: int,
    scale: float,
) -> tuple[list[RenderSegment], list[RenderNode]]:
    if abs(scale - 1.0) < 1e-6:
        return segments, nodes
    center_y = height * 0.5

    def stretch(y: float) -> float:
        return center_y + (y - center_y) * scale

    stretched_segments = [
        replace(seg, y0=stretch(seg.y0), y1=stretch(seg.y1))
        for seg in segments
    ]
    stretched_nodes = [replace(node, y=stretch(node.y)) for node in nodes]
    return stretched_segments, stretched_nodes


def background(width: int, height: int, style: str) -> Image.Image:
    del style
    return Image.new("RGBA", (width, height), BG + (255,))


def render_base(scene: SceneCache, width: int, height: int) -> Image.Image:
    image = scene.base.copy()
    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow, "RGBA")
    base = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    base_draw = ImageDraw.Draw(base, "RGBA")
    depth_soft = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    depth_soft_draw = ImageDraw.Draw(depth_soft, "RGBA")
    soma_shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    soma_shadow_draw = ImageDraw.Draw(soma_shadow, "RGBA")
    soma_base = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    soma_draw = ImageDraw.Draw(soma_base, "RGBA")
    soma_soft = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    soma_soft_draw = ImageDraw.Draw(soma_soft, "RGBA")
    for seg in scene.segments:
        depth = np.clip((seg.z + 0.7) / 1.4, 0.0, 1.0)
        defocus = depth_defocus(seg.z)
        neutral = mix(np.array([174, 188, 197], dtype=float), np.array([24, 38, 47], dtype=float), 0.43 + 0.31 * depth)
        tint = mix(neutral, seg.color, 0.46 + 0.13 * depth)
        if seg.neurite_kind == 2:
            tint = mix(tint, MAROON, 0.24)
        draw_soft_line(shadow_draw, seg, np.array([145, 157, 164], dtype=float), 34 * seg.alpha, seg.width * 3.2)
        draw_soft_line(base_draw, seg, tint, 178 * seg.alpha * (0.68 + 0.32 * depth) * (1.0 - 0.28 * defocus), seg.width * 1.05)
        draw_soft_line(base_draw, seg, np.array([255, 255, 255], dtype=float), 20 * seg.alpha * (1.0 - 0.38 * defocus), max(1.0, seg.width * 0.18))
        if defocus > 0.02:
            draw_soft_line(
                depth_soft_draw,
                seg,
                tint,
                112 * seg.alpha * defocus,
                seg.width * (1.22 + 0.70 * defocus),
            )
    for node in scene.nodes:
        if not node.soma:
            continue
        rx, ry = soma_body_axes(node)
        defocus = depth_defocus(node.z)
        body = mix(np.array([13, 18, 21], dtype=float), node.color, 0.62)
        draw_soma_shape(soma_shadow_draw, node, rx, ry, fill=rgba(node.color, 54), scale=1.34)
        draw_soma_shape(soma_draw, node, rx, ry, fill=rgba(body, 255))
        if defocus > 0.02:
            draw_soma_shape(soma_soft_draw, node, rx, ry, fill=rgba(body, 108 * defocus), scale=1.08 + 0.14 * defocus)
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=2.2 * SUPERSAMPLE))
    depth_soft = depth_soft.filter(ImageFilter.GaussianBlur(radius=1.85 * SUPERSAMPLE))
    soma_shadow = soma_shadow.filter(ImageFilter.GaussianBlur(radius=5.5 * SUPERSAMPLE))
    soma_soft = soma_soft.filter(ImageFilter.GaussianBlur(radius=2.45 * SUPERSAMPLE))
    image = Image.alpha_composite(image, shadow)
    image = Image.alpha_composite(image, base)
    image = Image.alpha_composite(image, depth_soft)
    image = Image.alpha_composite(image, soma_shadow)
    image = Image.alpha_composite(image, soma_base)
    image = Image.alpha_composite(image, soma_soft)
    return image


def render_frame(
    scene: SceneCache,
    phase: float,
    *,
    width: int,
    height: int,
    mode: str,
) -> Image.Image:
    image = scene.base.copy()
    trace = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    trace_soft = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    bloom = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    active_soft = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    core = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    soma_glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    soma_soft = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    soma_core = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    spark = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    trace_draw = ImageDraw.Draw(trace, "RGBA")
    trace_soft_draw = ImageDraw.Draw(trace_soft, "RGBA")
    bloom_draw = ImageDraw.Draw(bloom, "RGBA")
    glow_draw = ImageDraw.Draw(glow, "RGBA")
    active_soft_draw = ImageDraw.Draw(active_soft, "RGBA")
    core_draw = ImageDraw.Draw(core, "RGBA")
    soma_glow_draw = ImageDraw.Draw(soma_glow, "RGBA")
    soma_soft_draw = ImageDraw.Draw(soma_soft, "RGBA")
    soma_core_draw = ImageDraw.Draw(soma_core, "RGBA")
    spark_draw = ImageDraw.Draw(spark, "RGBA")
    if mode == "single_arbor_signal":
        pulse_color = CYAN
    elif mode == "split_wavefront":
        pulse_color = GOLD
    elif mode == "layered_exchange":
        pulse_color = CYAN
    else:
        pulse_color = TEAL
    loop_ms = phase * ACTIVITY_PERIOD_MS

    for seg in scene.segments:
        is_axon = seg.neurite_kind == 2
        defocus = depth_defocus(seg.z)
        segment_phase_ms = seg.branch_phase_ms + (AXON_EMISSION_DELAY_MS if is_axon else 0.0)
        segment_gain = seg.branch_gain * (AXON_PACKET_GAIN if is_axon else 1.0)
        pulse_tint = mix(pulse_color, seg.color, 0.45)
        if is_axon:
            pulse_tint = mix(mix(MAROON, GOLD, 0.54), WHITE, 0.18)
        if mode in ("dense_microcircuit", "layered_exchange") and seg.cell_type == "MC":
            pulse_tint = mix(MAROON, GOLD, 0.32)
        elif mode in ("dense_microcircuit", "layered_exchange") and seg.cell_type == "TC":
            pulse_tint = GOLD
        elif mode in ("dense_microcircuit", "layered_exchange"):
            pulse_tint = mix(TEAL, GREEN, 0.22)
        if is_axon:
            pulse_tint = mix(pulse_tint, WHITE, 0.16)
        memory = trace_activity(
            seg.activity_profile,
            loop_ms,
            seg.distance,
            seg.flow_direction,
            segment_phase_ms,
            segment_gain,
        )
        memory = min(1.0, memory ** (0.62 if is_axon else 0.72) * (1.16 if is_axon else 1.0))
        if memory > (0.028 if is_axon else 0.040):
            trace_tint = mix(pulse_tint, np.array([92, 104, 108], dtype=float), 0.40)
            draw_soft_line(
                trace_draw,
                seg,
                trace_tint,
                (68 if is_axon else 54) * seg.alpha * memory * (1.0 - 0.20 * defocus),
                seg.width * ((1.62 if is_axon else 1.20) + (1.04 if is_axon else 0.86) * memory),
            )
            if defocus > 0.02:
                draw_soft_line(
                    trace_soft_draw,
                    seg,
                    trace_tint,
                    (42 if is_axon else 34) * seg.alpha * memory * defocus,
                    seg.width * ((2.00 if is_axon else 1.58) + (1.20 if is_axon else 1.02) * memory + 0.24 * defocus),
                )
        active = packet_activity(
            seg.activity_profile,
            loop_ms,
            seg.distance,
            seg.flow_direction,
            segment_phase_ms,
            segment_gain,
        )
        active = min(1.0, active ** (0.54 if is_axon else 0.64) * (1.18 if is_axon else 1.0))
        if active < (0.022 if is_axon else 0.045):
            continue
        bloom_tint = brighten(pulse_tint, 1.38, 5.0)
        core_tint = mix(pulse_tint, brighten(pulse_tint, 1.55, 12.0), min(0.72, 0.24 + 0.42 * active))
        if active > 0.10:
            draw_packet_line(
                bloom_draw,
                seg,
                bloom_tint,
                (96 if is_axon else 76) * active * (1.0 - 0.12 * defocus),
                seg.width * ((14.8 if is_axon else 12.0) + (5.0 if is_axon else 4.2) * active) * (1.0 + 0.16 * defocus),
                active,
                0.32 if is_axon else 0.22,
            )
        draw_packet_line(
            glow_draw,
            seg,
            brighten(pulse_tint, 1.18, 3.0),
            (148 if is_axon else 116) * active * (1.0 - 0.22 * defocus),
            seg.width * ((8.8 if is_axon else 6.8) + (2.8 if is_axon else 2.2) * active) * (1.0 + 0.22 * defocus),
            active,
            0.22 if is_axon else 0.14,
        )
        if defocus > 0.02:
            draw_packet_line(
                active_soft_draw,
                seg,
                core_tint,
                (70 if is_axon else 56) * active * defocus,
                seg.width * ((5.2 if is_axon else 4.2) + (2.0 if is_axon else 1.6) * active + 0.28 * defocus),
                active,
                0.18 if is_axon else 0.10,
            )
        draw_packet_line(
            core_draw,
            seg,
            core_tint,
            (246 if is_axon else 238) * active * (1.0 - 0.34 * defocus),
            seg.width * ((2.55 if is_axon else 1.88) + (1.20 if is_axon else 0.95) * active),
            active,
            0.14 if is_axon else 0.04,
        )
        if active > 0.62:
            highlight = (active - 0.62) / 0.38
            draw_packet_line(
                core_draw,
                seg,
                WHITE,
                (214 if is_axon else 192) * highlight * (1.0 - 0.28 * defocus),
                max(1.0, seg.width * ((0.68 if is_axon else 0.48) + (0.28 if is_axon else 0.24) * highlight)),
                active,
                0.10 if is_axon else 0.0,
            )
        if active > (0.68 if is_axon else 0.80):
            hot_floor = 0.68 if is_axon else 0.80
            hot = (active - hot_floor) / (1.0 - hot_floor)
            draw_packet_line(
                spark_draw,
                seg,
                WHITE,
                (230 if is_axon else 212) * hot * (1.0 - 0.24 * defocus),
                max(1.0, seg.width * ((0.56 if is_axon else 0.34) + (0.28 if is_axon else 0.18) * hot)),
                active,
                0.12 if is_axon else 0.0,
            )
            ellipse(
                spark_draw,
                0.5 * (seg.x0 + seg.x1),
                0.5 * (seg.y0 + seg.y1),
                seg.width * ((0.82 if is_axon else 0.58) + (0.68 if is_axon else 0.54) * hot),
                rgba(WHITE, (170 if is_axon else 150) * hot * (1.0 - 0.22 * defocus)),
            )

    image = Image.alpha_composite(image, trace)
    trace_soft = trace_soft.filter(ImageFilter.GaussianBlur(radius=1.65 * SUPERSAMPLE))
    bloom = bloom.filter(ImageFilter.GaussianBlur(radius=8.3 * SUPERSAMPLE))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=4.2 * SUPERSAMPLE))
    active_soft = active_soft.filter(ImageFilter.GaussianBlur(radius=2.20 * SUPERSAMPLE))
    image = Image.alpha_composite(image, trace_soft)
    image = Image.alpha_composite(image, bloom)
    image = Image.alpha_composite(image, glow)
    image = Image.alpha_composite(image, active_soft)
    image = Image.alpha_composite(image, core)

    for node in scene.nodes:
        defocus = depth_defocus(node.z)
        node_active = packet_activity(
            node.activity_profile,
            loop_ms,
            node.distance,
            node.flow_direction,
            node.branch_phase_ms,
            node.branch_gain,
        )
        node_active = min(1.0, node_active ** 0.66)
        terminal_flash = node.terminal and node_active
        soma_flash = packet_activity(node.activity_profile, loop_ms, 0.0, 1.0, 0.0, 1.0) if node.soma else 0.0
        soma_flash = min(1.0, soma_flash ** 0.66)
        if terminal_flash:
            r = node.radius * (1.5 + 2.6 * terminal_flash)
            ellipse(
                spark_draw,
                node.x,
                node.y,
                r,
                rgba(mix(node.color, np.array([255, 255, 255], dtype=float), 0.15), 158 * terminal_flash * (1.0 - 0.20 * defocus)),
            )
        if node.soma:
            delayed_voltage = min(1.0, delayed_profile_value(node.activity_profile, loop_ms, SOMA_RESPONSE_DELAY_MS) ** 0.92)
            soma_spike = min(1.0, packet_activity(node.activity_profile, loop_ms, 0.0, 1.0, SOMA_RESPONSE_DELAY_MS, 1.0) ** 0.50)
            afterhyper = event_decay_activity(
                node.activity_profile,
                loop_ms,
                delay_ms=SOMA_RESPONSE_DELAY_MS + SOMA_AFTERHYPERPOLARIZATION_DELAY_MS,
                window_ms=SOMA_AFTERHYPERPOLARIZATION_WINDOW_MS,
                tau_ms=SOMA_AFTERHYPERPOLARIZATION_TAU_MS,
                gain=1.0,
            )
            soma_level = np.clip(0.08 + 0.22 * delayed_voltage + 1.02 * soma_spike - 0.24 * afterhyper, 0.0, 1.0)
            rx, ry = soma_body_axes(node)
            voltage_tint = mix(node.color, WHITE, 0.16 + 0.62 * soma_spike)
            body_tint = mix(np.array([15, 20, 23], dtype=float), node.color, 0.34 + 0.22 * delayed_voltage)
            disc_tint = mix(body_tint, voltage_tint, 0.14 + 0.74 * soma_level)
            disc_tint = mix(disc_tint, np.array([7, 9, 10], dtype=float), min(0.34, 0.30 * afterhyper))
            draw_soma_shape(
                soma_glow_draw,
                node,
                rx,
                ry,
                fill=rgba(voltage_tint, (18 + 110 * soma_spike) * (1.0 - 0.20 * defocus)),
                scale=1.18 + 0.34 * soma_spike,
            )
            if defocus > 0.02 and soma_level > 0.02:
                draw_soma_shape(
                    soma_soft_draw,
                    node,
                    rx,
                    ry,
                    fill=rgba(voltage_tint, (36 + 76 * soma_level) * defocus),
                    scale=1.08 + 0.16 * defocus + 0.10 * soma_spike,
                )
            draw_soma_shape(
                soma_core_draw,
                node,
                rx,
                ry,
                fill=rgba(disc_tint, 252 - 40 * defocus),
                scale=0.98 + 0.06 * soma_spike,
            )
            if soma_spike > 0.46:
                hot = (soma_spike - 0.46) / 0.54
                draw_soma_shape(spark_draw, node, rx, ry, fill=rgba(WHITE, 72 * hot * (1.0 - 0.22 * defocus)), scale=0.52)
    soma_glow = soma_glow.filter(ImageFilter.GaussianBlur(radius=4.0 * SUPERSAMPLE))
    soma_soft = soma_soft.filter(ImageFilter.GaussianBlur(radius=2.35 * SUPERSAMPLE))
    image = Image.alpha_composite(image, soma_glow)
    image = Image.alpha_composite(image, soma_soft)
    image = Image.alpha_composite(image, soma_core)
    spark = spark.filter(ImageFilter.GaussianBlur(radius=0.55 * SUPERSAMPLE))
    image = Image.alpha_composite(image, spark)
    return image


def build_scene(variant: str, width: int, height: int) -> SceneCache:
    morphs = {
        "mc_a": load_morphology("mitral-a", "MC", REPO / "prev_ob_models/Birgiolas2020/SWCs/MC/IF04360.CNG.swc"),
        "mc_b": load_morphology("mitral-b", "MC", REPO / "prev_ob_models/Birgiolas2020/SWCs/MC/IF04344.CNG.swc"),
        "tc_a": load_morphology("tufted-a", "TC", REPO / "prev_ob_models/Birgiolas2020/SWCs/TC/IF04355.CNG.swc"),
        "tc_b": load_morphology("tufted-b", "TC", REPO / "prev_ob_models/Birgiolas2020/SWCs/TC/IF04204.CNG.swc"),
        "gc_a": load_morphology("granule-a", "GC", REPO / "prev_ob_models/Birgiolas2020/SWCs/GC/OB_granule_cell7.CNG.swc"),
        "gc_b": load_morphology("granule-b", "GC", REPO / "prev_ob_models/Birgiolas2020/SWCs/GC/OB_granule_cell6.CNG.swc"),
    }
    if variant == "connected_field":
        placed = [
            PlacedMorph(morphs["mc_a"], (width * 0.25, height * 0.56), 0.48, -23, 17, -8, TYPE_COLORS["MC"], 0.90, 1.30, 0.02, -0.04),
            PlacedMorph(morphs["mc_b"], (width * 0.62, height * 0.57), 0.46, 20, 15, 10, TYPE_COLORS["MC"], 0.86, 1.24, 0.19, 0.02),
            PlacedMorph(morphs["tc_a"], (width * 0.44, height * 0.53), 0.44, 23, 10, 5, TYPE_COLORS["TC"], 0.88, 1.18, 0.33, 0.08),
            PlacedMorph(morphs["tc_b"], (width * 0.77, height * 0.52), 0.39, -18, 12, -6, TYPE_COLORS["TC"], 0.82, 1.10, 0.46, 0.16),
            PlacedMorph(morphs["gc_a"], (width * 0.36, height * 0.74), 0.58, -31, 8, -5, TYPE_COLORS["GC"], 0.86, 1.06, 0.65, 0.22),
            PlacedMorph(morphs["gc_b"], (width * 0.58, height * 0.73), 0.54, 28, 6, 7, TYPE_COLORS["GC"], 0.82, 1.02, 0.78, 0.28),
        ]
        bg_style = "luminous"
    elif variant == "layered_exchange":
        placed = [
            PlacedMorph(
                morphs["mc_a"],
                (width * 0.18, height * 0.58),
                0.44,
                -19,
                16,
                -9,
                TYPE_COLORS["MC"],
                0.88,
                1.22,
                0.03,
                -0.06,
                ((0.42, 0.49, 0.25), (0.54, 0.58, 0.13)),
            ),
            PlacedMorph(
                morphs["mc_b"],
                (width * 0.46, height * 0.56),
                0.43,
                18,
                14,
                8,
                TYPE_COLORS["MC"],
                0.86,
                1.20,
                0.23,
                0.02,
                ((0.43, 0.49, 0.18), (0.58, 0.54, 0.20)),
            ),
            PlacedMorph(
                morphs["mc_a"],
                (width * 0.75, height * 0.58),
                0.40,
                -34,
                12,
                -5,
                TYPE_COLORS["MC"],
                0.80,
                1.12,
                0.41,
                0.08,
                ((0.60, 0.52, 0.18), (0.70, 0.61, 0.18)),
            ),
            PlacedMorph(
                morphs["tc_a"],
                (width * 0.34, height * 0.49),
                0.38,
                26,
                9,
                4,
                TYPE_COLORS["TC"],
                0.88,
                1.10,
                0.35,
                0.12,
                ((0.42, 0.49, 0.28), (0.51, 0.58, 0.16)),
            ),
            PlacedMorph(
                morphs["tc_b"],
                (width * 0.61, height * 0.48),
                0.36,
                -16,
                10,
                -7,
                TYPE_COLORS["TC"],
                0.84,
                1.06,
                0.50,
                0.18,
                ((0.58, 0.54, 0.25), (0.68, 0.61, 0.14)),
            ),
            PlacedMorph(
                morphs["gc_a"],
                (width * 0.29, height * 0.74),
                0.52,
                -27,
                7,
                -4,
                TYPE_COLORS["GC"],
                0.83,
                1.00,
                0.66,
                0.24,
                ((0.43, 0.65, 0.25), (0.52, 0.58, 0.17)),
            ),
            PlacedMorph(
                morphs["gc_b"],
                (width * 0.53, height * 0.74),
                0.50,
                29,
                6,
                6,
                TYPE_COLORS["GC"],
                0.80,
                0.98,
                0.77,
                0.29,
                ((0.55, 0.63, 0.28), (0.63, 0.55, 0.13)),
            ),
            PlacedMorph(
                morphs["gc_a"],
                (width * 0.73, height * 0.75),
                0.49,
                -14,
                8,
                3,
                TYPE_COLORS["GC"],
                0.76,
                0.94,
                0.88,
                0.35,
                ((0.69, 0.65, 0.25), (0.59, 0.55, 0.11)),
            ),
        ]
        bg_style = "graphite"
    else:
        placed = [
            PlacedMorph(morphs["mc_a"], (width * 0.22, height * 0.58), 0.42, -22, 14, -8, TYPE_COLORS["MC"], 0.86, 1.16, 0.04, -0.07),
            PlacedMorph(morphs["mc_b"], (width * 0.43, height * 0.56), 0.40, 21, 13, 9, TYPE_COLORS["MC"], 0.82, 1.12, 0.19, 0.00),
            PlacedMorph(morphs["mc_a"], (width * 0.67, height * 0.58), 0.39, -29, 12, -5, TYPE_COLORS["MC"], 0.78, 1.08, 0.31, 0.05),
            PlacedMorph(morphs["tc_a"], (width * 0.33, height * 0.50), 0.36, 24, 8, 4, TYPE_COLORS["TC"], 0.84, 1.05, 0.41, 0.13),
            PlacedMorph(morphs["tc_b"], (width * 0.55, height * 0.49), 0.34, -18, 9, -6, TYPE_COLORS["TC"], 0.80, 1.00, 0.53, 0.19),
            PlacedMorph(morphs["tc_a"], (width * 0.78, height * 0.50), 0.33, 31, 8, 9, TYPE_COLORS["TC"], 0.74, 0.98, 0.64, 0.25),
            PlacedMorph(morphs["gc_a"], (width * 0.27, height * 0.75), 0.50, -29, 7, -4, TYPE_COLORS["GC"], 0.82, 0.96, 0.68, 0.25),
            PlacedMorph(morphs["gc_b"], (width * 0.48, height * 0.75), 0.48, 28, 6, 6, TYPE_COLORS["GC"], 0.78, 0.94, 0.80, 0.31),
            PlacedMorph(morphs["gc_a"], (width * 0.69, height * 0.76), 0.47, -16, 8, 2, TYPE_COLORS["GC"], 0.74, 0.90, 0.91, 0.37),
        ]
        bg_style = "lattice"
    placed = attach_activity_profiles(placed)
    segments, nodes = project_scene(placed, width, height)
    segments, nodes = center_geometry(segments, nodes, width, height)
    segments, nodes = stretch_geometry_y(segments, nodes, height, VERTICAL_FOREGROUND_SCALE)
    segments, nodes = center_geometry(segments, nodes, width, height)
    base = background(width, height, bg_style)
    scene = SceneCache(base=base, segments=segments, nodes=nodes)
    scene.base = render_base(scene, width, height)
    return scene


def build_global_gif_palette(frames: list[Image.Image], colors: int) -> Image.Image:
    colors = max(2, min(256, colors))
    step = max(1, len(frames) // 18)
    sample_frames = frames[::step]
    scale = max(1, math.ceil(sample_frames[0].width / 760))
    sample_width = max(1, sample_frames[0].width // scale)
    sample_height = max(1, sample_frames[0].height // scale)
    anchor_height = max(24, sample_height // 16)
    palette_source = Image.new("RGB", (sample_width, sample_height * len(sample_frames) + anchor_height), BG)
    for idx, frame in enumerate(sample_frames):
        sample = frame.resize((sample_width, sample_height), Image.Resampling.LANCZOS)
        palette_source.paste(sample, (0, idx * sample_height))
    anchors = [
        BG,
        tuple(WHITE.astype(np.uint8)),
        tuple(GOLD.astype(np.uint8)),
        tuple(CYAN.astype(np.uint8)),
        tuple(GREEN.astype(np.uint8)),
        tuple(TEAL.astype(np.uint8)),
        tuple(MAROON.astype(np.uint8)),
        tuple(mix(MAROON, GOLD, 0.32).astype(np.uint8)),
        tuple(mix(TEAL, GREEN, 0.22).astype(np.uint8)),
    ]
    anchor_y0 = sample_height * len(sample_frames)
    stripe_width = max(1, math.ceil(sample_width / len(anchors)))
    draw = ImageDraw.Draw(palette_source)
    for idx, color in enumerate(anchors):
        x0 = idx * stripe_width
        draw.rectangle((x0, anchor_y0, min(sample_width, x0 + stripe_width), anchor_y0 + anchor_height), fill=color)
    return palette_source.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


def quantize_for_gif(frames: list[Image.Image], colors: int) -> list[Image.Image]:
    palette = build_global_gif_palette(frames, colors)
    return [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]


def render_variant(
    variant: str,
    output_dir: Path,
    *,
    width: int,
    height: int,
    frames: int,
    duration_ms: int,
    gif_colors: int,
) -> tuple[Path, Path]:
    work_width = width * SUPERSAMPLE
    work_height = height * SUPERSAMPLE
    scene = build_scene(variant, work_width, work_height)
    rendered: list[Image.Image] = []
    for idx in range(frames):
        phase = idx / frames
        frame = render_frame(scene, phase, width=work_width, height=work_height, mode=variant)
        frame = frame.resize((width, height), Image.Resampling.LANCZOS).convert("RGB")
        edge = 2
        pixels = np.asarray(frame).copy()
        pixels[:edge, :, :] = BG
        pixels[-edge:, :, :] = BG
        pixels[:, :edge, :] = BG
        pixels[:, -edge:, :] = BG
        frame = Image.fromarray(pixels)
        rendered.append(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / f"{variant}.gif"
    gif_frames = quantize_for_gif(rendered, gif_colors)
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    poster_path = output_dir / f"{variant}_poster.png"
    rendered[frames // 3].save(poster_path)
    return gif_path, poster_path


def save_contact_sheet(posters: dict[str, Path], output_path: Path, *, width: int, height: int) -> Path:
    rows = []
    for _, poster in posters.items():
        rows.append(Image.open(poster).convert("RGB"))
    sheet = Image.new("RGB", (width, height * len(rows)), BG)
    for idx, row in enumerate(rows):
        sheet.paste(row, (0, idx * height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def export_all(
    output_dir: Path,
    *,
    variants: list[str] | None,
    width: int,
    height: int,
    frames: int,
    duration_ms: int,
    gif_colors: int,
) -> dict[str, Path]:
    selected = variants or ["layered_exchange"]
    artifacts: dict[str, Path] = {}
    posters: dict[str, Path] = {}
    for variant in selected:
        gif_path, poster_path = render_variant(
            variant,
            output_dir,
            width=width,
            height=height,
            frames=frames,
            duration_ms=duration_ms,
            gif_colors=gif_colors,
        )
        artifacts[variant] = gif_path
        posters[variant] = poster_path
    if posters:
        artifacts["contact_sheet"] = save_contact_sheet(posters, output_dir / "contact_sheet.png", width=width, height=height)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Render BlenderNeuron-style animated morphology headers from SWC files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--duration-ms", type=int, default=60)
    parser.add_argument("--gif-colors", type=int, default=GIF_COLORS)
    args = parser.parse_args()

    artifacts = export_all(
        args.output_dir,
        variants=args.variants,
        width=args.width,
        height=args.height,
        frames=args.frames,
        duration_ms=args.duration_ms,
        gif_colors=args.gif_colors,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
