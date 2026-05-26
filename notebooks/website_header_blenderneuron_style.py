from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


REPO = Path("/home/alek/OlfactoryBulb")
DEFAULT_OUTPUT_DIR = REPO / "media/website_header_blenderneuron_style_v1"
WIDTH = 1280
HEIGHT = 360
SUPERSAMPLE = 2
BG = (255, 255, 255)
INK = np.array([24, 35, 42], dtype=float)
CYAN = np.array([42, 177, 213], dtype=float)
MINT = np.array([59, 204, 171], dtype=float)
AMBER = np.array([242, 151, 67], dtype=float)
ROSE = np.array([236, 93, 91], dtype=float)


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


@dataclass
class SceneCache:
    base: Image.Image
    segments: list[RenderSegment]
    nodes: list[RenderNode]


def mix(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


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
            projected[node_id] = np.array([sx, sy, pos[2] + item.z_bias], dtype=float)

        for node_id, node in morph.nodes.items():
            parent_id = node.parent_id
            if parent_id < 0 or parent_id not in morph.nodes:
                continue
            p0 = projected[parent_id]
            p1 = projected[node_id]
            radius = 0.5 * (node.radius + morph.nodes[parent_id].radius)
            dist = 0.5 * (morph.distances[node_id] + morph.distances.get(parent_id, 0.0)) / morph.max_distance
            width_px = max(1.0, item.width_scale * (0.75 + 0.42 * math.sqrt(radius)))
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


def background(width: int, height: int, style: str) -> Image.Image:
    image = Image.new("RGB", (width, height), BG)
    arr = np.asarray(image).astype(float)
    yy, xx = np.mgrid[0:height, 0:width]
    x = xx / max(1, width - 1)
    y = yy / max(1, height - 1)
    if style == "luminous":
        glows = [
            (0.18, 0.56, 0.35, np.array([223, 245, 249]), 0.34),
            (0.75, 0.40, 0.30, np.array([246, 226, 190]), 0.25),
        ]
    elif style == "graphite":
        glows = [
            (0.42, 0.40, 0.40, np.array([230, 242, 244]), 0.30),
            (0.88, 0.58, 0.28, np.array([240, 230, 210]), 0.20),
        ]
    else:
        glows = [
            (0.28, 0.48, 0.40, np.array([226, 244, 240]), 0.28),
            (0.72, 0.42, 0.34, np.array([232, 240, 248]), 0.22),
        ]
    for cx, cy, radius, color, strength in glows:
        field = np.exp(-(((x - cx) / radius) ** 2 + ((y - cy) / radius) ** 2))
        arr = arr * (1.0 - strength * field[..., None]) + color * (strength * field[..., None])
    image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    layer = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    for idx, y_frac in enumerate((0.18, 0.32, 0.52, 0.74)):
        y0 = int(height * y_frac)
        amp = 7 + idx * 2
        points = [
            (int(width * t / 160.0), int(y0 + amp * math.sin((t / 160.0) * math.tau * (1.15 + idx * 0.16))))
            for t in range(161)
        ]
        draw.line(points, fill=(170, 188, 196, 18), width=max(1, int(1.3 * SUPERSAMPLE)))
    return Image.alpha_composite(image.convert("RGBA"), layer)


def render_base(scene: SceneCache, width: int, height: int) -> Image.Image:
    image = scene.base.copy()
    shadow = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    shadow_draw = ImageDraw.Draw(shadow, "RGBA")
    base = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    base_draw = ImageDraw.Draw(base, "RGBA")
    for seg in scene.segments:
        depth = np.clip((seg.z + 0.7) / 1.4, 0.0, 1.0)
        neutral = mix(np.array([181, 194, 202], dtype=float), np.array([30, 45, 54], dtype=float), 0.45 + 0.34 * depth)
        tint = mix(neutral, seg.color, 0.30 + 0.14 * depth)
        draw_soft_line(shadow_draw, seg, np.array([145, 157, 164], dtype=float), 34 * seg.alpha, seg.width * 3.2)
        draw_soft_line(base_draw, seg, tint, 168 * seg.alpha * (0.68 + 0.32 * depth), seg.width * 1.12)
        draw_soft_line(base_draw, seg, np.array([255, 255, 255], dtype=float), 22 * seg.alpha, max(1.0, seg.width * 0.22))
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
    glow = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    core = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    spark = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    glow_draw = ImageDraw.Draw(glow, "RGBA")
    core_draw = ImageDraw.Draw(core, "RGBA")
    spark_draw = ImageDraw.Draw(spark, "RGBA")
    if mode == "single_arbor_signal":
        pulse_color = CYAN
        count = 2
        pulse_width = 0.035
        direction = 1.0
    elif mode == "split_wavefront":
        pulse_color = AMBER
        count = 3
        pulse_width = 0.030
        direction = 1.0
    else:
        pulse_color = MINT
        count = 3
        pulse_width = 0.042
        direction = -1.0

    for seg in scene.segments:
        p = pulse_value(seg.distance, phase, count=count, width=pulse_width, direction=direction)
        cell_shift = 0.11 if seg.cell_type == "TC" else 0.21 if seg.cell_type == "GC" else 0.0
        secondary = pulse_value((seg.distance + cell_shift) % 1.0, phase + 0.17, count=2, width=0.055, direction=1.0)
        active = max(p, 0.54 * secondary)
        if active < 0.030:
            continue
        pulse_tint = mix(pulse_color, seg.color, 0.35)
        if mode == "microcircuit_exchange" and seg.cell_type == "MC":
            pulse_tint = mix(AMBER, CYAN, 0.25)
        elif mode == "microcircuit_exchange" and seg.cell_type == "TC":
            pulse_tint = CYAN
        elif mode == "microcircuit_exchange":
            pulse_tint = MINT
        draw_soft_line(glow_draw, seg, pulse_tint, 92 * active, seg.width * (6.2 + 1.9 * active))
        draw_soft_line(core_draw, seg, mix(pulse_tint, np.array([255, 255, 255], dtype=float), 0.12), 218 * active, seg.width * (1.72 + 0.85 * active))
        if active > 0.62:
            draw_soft_line(core_draw, seg, np.array([255, 255, 255], dtype=float), 56 * active, max(1.0, seg.width * 0.45))

    glow = glow.filter(ImageFilter.GaussianBlur(radius=4.2 * SUPERSAMPLE))
    image = Image.alpha_composite(image, glow)
    image = Image.alpha_composite(image, core)

    for node in scene.nodes:
        terminal_flash = node.terminal and pulse_value(node.distance, phase + 0.015, count=count, width=0.025, direction=direction)
        soma_flash = node.soma and (0.56 + 0.44 * math.sin(math.tau * phase))
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
    mc = load_morphology("mitral", "MC", REPO / "prev_ob_models/Birgiolas2020/SWCs/MC/IF04360.CNG.swc")
    tc = load_morphology("tufted", "TC", REPO / "prev_ob_models/Birgiolas2020/SWCs/TC/IF04355.CNG.swc")
    gc = load_morphology("granule", "GC", REPO / "prev_ob_models/Birgiolas2020/SWCs/GC/OB_granule_cell7.CNG.swc")
    if variant == "single_arbor_signal":
        placed = [
            PlacedMorph(mc, (width * 0.48, height * 0.56), 0.82, -24, 18, -7, CYAN, 1.0, 2.15, 0.0, 0.0),
        ]
        bg_style = "luminous"
    elif variant == "split_wavefront":
        placed = [
            PlacedMorph(mc, (width * 0.32, height * 0.57), 0.62, -20, 16, -10, AMBER, 0.90, 1.75, 0.02, 0.00),
            PlacedMorph(tc, (width * 0.69, height * 0.55), 0.56, 25, 11, 8, CYAN, 0.82, 1.55, 0.34, 0.08),
        ]
        bg_style = "graphite"
    else:
        placed = [
            PlacedMorph(mc, (width * 0.36, height * 0.58), 0.55, -21, 15, -8, AMBER, 0.84, 1.55, 0.08, 0.0),
            PlacedMorph(tc, (width * 0.67, height * 0.54), 0.51, 28, 10, 9, CYAN, 0.82, 1.45, 0.38, 0.10),
            PlacedMorph(gc, (width * 0.50, height * 0.72), 0.72, -34, 7, -4, MINT, 0.88, 1.30, 0.68, 0.18),
        ]
        bg_style = "lattice"
    segments, nodes = project_scene(placed, width, height)
    base = background(width, height, bg_style)
    scene = SceneCache(base=base, segments=segments, nodes=nodes)
    scene.base = render_base(scene, width, height)
    return scene


def render_variant(
    variant: str,
    output_dir: Path,
    *,
    width: int,
    height: int,
    frames: int,
    duration_ms: int,
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
        pixels[:edge, :, :] = 255
        pixels[-edge:, :, :] = 255
        pixels[:, :edge, :] = 255
        pixels[:, -edge:, :] = 255
        frame = Image.fromarray(pixels)
        rendered.append(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / f"{variant}.gif"
    rendered[0].save(
        gif_path,
        save_all=True,
        append_images=rendered[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    poster_path = output_dir / f"{variant}_poster.png"
    rendered[frames // 3].save(poster_path)
    return gif_path, poster_path


def save_contact_sheet(posters: dict[str, Path], output_path: Path) -> Path:
    rows = []
    for _, poster in posters.items():
        rows.append(Image.open(poster).convert("RGB"))
    sheet = Image.new("RGB", (WIDTH, HEIGHT * len(rows)), BG)
    for idx, row in enumerate(rows):
        sheet.paste(row, (0, idx * HEIGHT))
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
) -> dict[str, Path]:
    selected = variants or ["single_arbor_signal", "split_wavefront", "microcircuit_exchange"]
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
        )
        artifacts[variant] = gif_path
        posters[variant] = poster_path
    if posters:
        artifacts["contact_sheet"] = save_contact_sheet(posters, output_dir / "contact_sheet.png")
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Render BlenderNeuron-style animated morphology headers from SWC files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--duration-ms", type=int, default=55)
    args = parser.parse_args()

    artifacts = export_all(
        args.output_dir,
        variants=args.variants,
        width=args.width,
        height=args.height,
        frames=args.frames,
        duration_ms=args.duration_ms,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
