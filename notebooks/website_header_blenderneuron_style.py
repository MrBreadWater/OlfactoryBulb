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
DEFAULT_OUTPUT_DIR = REPO / "media/website_header_blenderneuron_style_v10"
DEFAULT_ACTIVITY_RUN = REPO / "results/notebook_runs/obgpu_experiment_GammaSignature_fast_20260520_035424"
WIDTH = 2280
HEIGHT = 720
SUPERSAMPLE = 2
GIF_COLORS = 144
VERTICAL_FOREGROUND_SCALE = 1.20
ACTIVITY_PERIOD_MS = 200.0
ACTIVITY_PROFILE_BINS = 400
ACTIVITY_SKIP_MS = 400.0
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
    peak_phase_ms: float
    score: float


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


@dataclass(frozen=True)
class RenderNode:
    x: float
    y: float
    z: float
    radius: float
    distance: float
    color: np.ndarray
    cell_type: str
    terminal: bool
    soma: bool
    activity_profile: ActivityProfile | None


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
    unique_label = f"{label} #{trace_index:03d}"
    return ActivityProfile(unique_label, cell_type_from_label(label), activity, peak_phase_ms, float(score))


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


def circular_phase_distance(a_ms: float, b_ms: float, period_ms: float = ACTIVITY_PERIOD_MS) -> float:
    return abs(((a_ms - b_ms + 0.5 * period_ms) % period_ms) - 0.5 * period_ms)


def select_activity_profiles(counts: dict[str, int]) -> dict[str, list[ActivityProfile]]:
    profiles = load_activity_profiles()
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
                if chosen:
                    nearest_phase = min(circular_phase_distance(profile.peak_phase_ms, other.peak_phase_ms) for other in chosen)
                    phase_bonus = 0.52 + 0.48 * min(1.0, nearest_phase / 42.0)
                    base_label = profile.label.split(" #", 1)[0]
                    label_bonus = 0.88 if any(other.label.split(" #", 1)[0] == base_label for other in chosen) else 1.0
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


def sample_activity(profile: ActivityProfile | None, phase_ms: float) -> float:
    if profile is None:
        return 1.0
    values = profile.values
    if values.size == 0:
        return 1.0
    position = (phase_ms % ACTIVITY_PERIOD_MS) / ACTIVITY_PERIOD_MS * values.size
    lo = int(math.floor(position)) % values.size
    hi = (lo + 1) % values.size
    frac = position - math.floor(position)
    return float((1.0 - frac) * values[lo] + frac * values[hi])


def activity_drive(profile: ActivityProfile | None, phase_ms: float) -> float:
    activity = sample_activity(profile, phase_ms)
    shaped = np.clip(activity, 0.0, 1.0) ** 0.72
    return float(np.clip(0.12 + 0.94 * shaped, 0.0, 1.0))


def activity_phase_shift(profile: ActivityProfile | None) -> float:
    if profile is None:
        return 0.0
    return (profile.peak_phase_ms / ACTIVITY_PERIOD_MS) % 1.0


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


def project_scene(placed: Iterable[PlacedMorph], width: int, height: int) -> tuple[list[RenderSegment], list[RenderNode]]:
    segments: list[RenderSegment] = []
    nodes_out: list[RenderNode] = []
    for item in placed:
        morph = item.morphology
        coords = normalized_coords(morph)
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

        for node_id, node in morph.nodes.items():
            parent_id = node.parent_id
            if parent_id < 0 or parent_id not in morph.nodes:
                continue
            p0 = projected[parent_id]
            p1 = projected[node_id]
            parent_node = morph.nodes[parent_id]
            radius = 0.5 * (node.radius + parent_node.radius)
            dist = 0.5 * (morph.distances[node_id] + morph.distances.get(parent_id, 0.0)) / morph.max_distance
            kind = segment_kind(node, parent_node)
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
                    activity_profile=item.activity_profile,
                )
            )

        for node_id, node in morph.nodes.items():
            p = projected[node_id]
            degree = len(morph.children.get(node_id, [])) + int(node.parent_id in morph.nodes)
            nodes_out.append(
                RenderNode(
                    x=float(p[0]),
                    y=float(p[1]),
                    z=float(p[2]),
                    radius=float(max(1.4, item.width_scale * (1.0 + 0.55 * math.sqrt(node.radius)))),
                    distance=float((morph.distances[node_id] / morph.max_distance + item.distance_offset) % 1.0),
                    color=item.color,
                    cell_type=morph.cell_type,
                    terminal=degree <= 1,
                    soma=node_id == morph.root_id or node.kind == 1,
                    activity_profile=item.activity_profile,
                )
            )
    segments.sort(key=lambda seg: seg.z)
    nodes_out.sort(key=lambda node: node.z)
    return segments, nodes_out


def ellipse(draw: ImageDraw.ImageDraw, x: float, y: float, r: float, fill: tuple[int, int, int, int]) -> None:
    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)


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
    for seg in scene.segments:
        depth = np.clip((seg.z + 0.7) / 1.4, 0.0, 1.0)
        neutral = mix(np.array([174, 188, 197], dtype=float), np.array([24, 38, 47], dtype=float), 0.43 + 0.31 * depth)
        tint = mix(neutral, seg.color, 0.46 + 0.13 * depth)
        if seg.neurite_kind == 2:
            tint = mix(tint, MAROON, 0.24)
        draw_soft_line(shadow_draw, seg, np.array([145, 157, 164], dtype=float), 34 * seg.alpha, seg.width * 3.2)
        draw_soft_line(base_draw, seg, tint, 178 * seg.alpha * (0.68 + 0.32 * depth), seg.width * 1.05)
        draw_soft_line(base_draw, seg, np.array([255, 255, 255], dtype=float), 20 * seg.alpha, max(1.0, seg.width * 0.18))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=2.2 * SUPERSAMPLE))
    image = Image.alpha_composite(image, shadow)
    image = Image.alpha_composite(image, base)
    return image


def pulse_value(distance: float, phase: float, *, count: int, width: float, direction: float = 1.0) -> float:
    centers = [(phase * direction + i / count) % 1.0 for i in range(count)]
    best = 0.0
    for center in centers:
        delta = abs(((distance - center + 0.5) % 1.0) - 0.5)
        best = max(best, math.exp(-(delta * delta) / (2.0 * width * width)))
    return best


def render_frame(
    scene: SceneCache,
    phase: float,
    *,
    width: int,
    height: int,
    mode: str,
) -> Image.Image:
    image = scene.base.copy()
    bloom = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    core = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    spark = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    bloom_draw = ImageDraw.Draw(bloom, "RGBA")
    glow_draw = ImageDraw.Draw(glow, "RGBA")
    core_draw = ImageDraw.Draw(core, "RGBA")
    spark_draw = ImageDraw.Draw(spark, "RGBA")
    if mode == "single_arbor_signal":
        pulse_color = CYAN
        count = 2
        pulse_width = 0.035
    elif mode == "split_wavefront":
        pulse_color = GOLD
        count = 3
        pulse_width = 0.030
    elif mode == "layered_exchange":
        pulse_color = CYAN
        count = 3
        pulse_width = 0.038
    else:
        pulse_color = TEAL
        count = 3
        pulse_width = 0.042
    loop_ms = phase * ACTIVITY_PERIOD_MS

    for seg in scene.segments:
        drive = activity_drive(seg.activity_profile, loop_ms)
        local_phase = (phase - activity_phase_shift(seg.activity_profile)) % 1.0
        p = pulse_value(seg.distance, local_phase, count=count, width=pulse_width, direction=seg.flow_direction)
        cell_shift = 0.11 if seg.cell_type == "TC" else 0.21 if seg.cell_type == "GC" else 0.0
        secondary = pulse_value(
            (seg.distance + cell_shift) % 1.0,
            local_phase + 0.17,
            count=2,
            width=0.055,
            direction=-seg.flow_direction,
        )
        active = max(p, 0.54 * secondary) * drive
        if active < 0.030:
            continue
        pulse_tint = mix(pulse_color, seg.color, 0.45)
        if seg.neurite_kind == 2:
            pulse_tint = mix(MAROON, seg.color, 0.22)
        if mode in ("dense_microcircuit", "layered_exchange") and seg.cell_type == "MC":
            pulse_tint = mix(MAROON, GOLD, 0.32)
        elif mode in ("dense_microcircuit", "layered_exchange") and seg.cell_type == "TC":
            pulse_tint = GOLD
        elif mode in ("dense_microcircuit", "layered_exchange"):
            pulse_tint = mix(TEAL, GREEN, 0.22)
        bloom_tint = brighten(pulse_tint, 1.38, 5.0)
        core_tint = mix(pulse_tint, brighten(pulse_tint, 1.55, 12.0), min(0.72, 0.24 + 0.42 * active))
        if active > 0.10:
            draw_soft_line(bloom_draw, seg, bloom_tint, 76 * active, seg.width * (12.0 + 4.2 * active))
        draw_soft_line(glow_draw, seg, brighten(pulse_tint, 1.18, 3.0), 116 * active, seg.width * (6.8 + 2.2 * active))
        draw_soft_line(core_draw, seg, core_tint, 238 * active, seg.width * (1.88 + 0.95 * active))
        if active > 0.62:
            highlight = (active - 0.62) / 0.38
            draw_soft_line(core_draw, seg, WHITE, 192 * highlight, max(1.0, seg.width * (0.48 + 0.24 * highlight)))
        if active > 0.80:
            hot = (active - 0.80) / 0.20
            draw_soft_line(spark_draw, seg, WHITE, 212 * hot, max(1.0, seg.width * (0.34 + 0.18 * hot)))
            ellipse(
                spark_draw,
                0.5 * (seg.x0 + seg.x1),
                0.5 * (seg.y0 + seg.y1),
                seg.width * (0.58 + 0.54 * hot),
                rgba(WHITE, 150 * hot),
            )

    bloom = bloom.filter(ImageFilter.GaussianBlur(radius=8.3 * SUPERSAMPLE))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=4.2 * SUPERSAMPLE))
    image = Image.alpha_composite(image, bloom)
    image = Image.alpha_composite(image, glow)
    image = Image.alpha_composite(image, core)

    for node in scene.nodes:
        drive = activity_drive(node.activity_profile, loop_ms)
        local_phase = (phase - activity_phase_shift(node.activity_profile)) % 1.0
        terminal_flash = node.terminal and drive * pulse_value(node.distance, local_phase + 0.015, count=count, width=0.025, direction=-1.0)
        soma_flash = node.soma and drive * (0.56 + 0.44 * math.sin(math.tau * local_phase))
        if terminal_flash:
            r = node.radius * (1.5 + 2.6 * terminal_flash)
            ellipse(spark_draw, node.x, node.y, r, rgba(mix(node.color, np.array([255, 255, 255], dtype=float), 0.15), 158 * terminal_flash))
        if node.soma:
            r = node.radius * (2.6 + 0.7 * soma_flash)
            ellipse(spark_draw, node.x, node.y, r, rgba(mix(node.color, np.array([255, 255, 255], dtype=float), 0.05), 124 + 58 * soma_flash))
            ellipse(spark_draw, node.x, node.y, max(1.2, r * 0.36), rgba(np.array([255, 255, 255], dtype=float), 76))
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
