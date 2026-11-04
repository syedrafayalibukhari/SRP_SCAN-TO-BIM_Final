"""Element clean-up: L and T junctions, overlap resolution and measured heights.

Overlaps are resolved by priority: column > wall > glass wall > partition. A parallel overlap thins the
lower-priority element, any other overlap cuts it. Pieces that become too short are dropped.

Walls and columns end at the 99th percentile height of their own points. Window points give the height
of a glazed run: a run that starts more than 0.3 m above the floor is a window band with a measured sill
and head. Nothing is modelled below such a sill, because the scan does not show what is there.
"""
from __future__ import annotations

import collections
from typing import List, Tuple

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ..classes import COLUMN, WALL, WINDOW
from . import junctions
from .context import ModelContext

PRIORITY = {"column": 0, "wall": 1, "glass_wall": 2, "partition": 4}
GAP, MIN_LENGTH, PARALLEL_ANGLE = 0.001, 0.10, 10.0
MIN_PARTITION_LENGTH = 0.30
GLASS_EXTENSION = 1.0                  # m, largest gap between a glass end and the wall or data edge behind it
GLASS_GAP_DENSITY = 500                # points per metre; an emptier gap stays open as a possible doorway
HEIGHT_POINTS_MIN = 200                # fewer own points than this: keep the storey height
GLASS_FLOOR_BAND = 0.3                 # m, window points starting this close to the floor are a full glass wall


def rectangle(x: dict) -> Polygon:
    d, n = x["d"], x["n"]
    t0, t1, o, half = x["t0"], x["t1"], x["o"], x["T"] / 2
    return Polygon([d * t0 + n * (o - half), d * t1 + n * (o - half), d * t1 + n * (o + half), d * t0 + n * (o + half)])


def collect_elements(ctx: ModelContext) -> Tuple[List[dict], float, float]:
    structure, rooms, detail = ctx.load("structure"), ctx.load("rooms"), ctx.load("columns_partitions")
    floor_z = structure["floor_z"]
    height = max(r["z_ceil"] for r in rooms["rooms"]) - floor_z
    elements = []
    for i, slab in enumerate(rooms["slabs"]):
        glass = slab["kind"] == "glass_wall"
        if slab["paired"]:
            source = "glass wall from window points, 6 cm" if glass else "two faces, measured thickness"
        else:
            source = "one face, other side unseen, assumed thickness"
        elements.append(dict(kind=slab["kind"], src=("slab", i), d=np.array(slab["d"]), n=np.array(slab["n"]), o=(slab["o0"] + slab["o1"]) / 2,
                             T=slab["o1"] - slab["o0"], t0=slab["t0"], t1=slab["t1"], H=height, source=source, paired=slab["paired"]))
    for i, panel in enumerate(detail["partitions"]):
        elements.append(dict(kind="partition", src=("partition", i), d=np.array(panel["d"]), n=np.array(panel["n"]), o=panel["o"], T=panel["T"],
                             t0=panel["t0"], t1=panel["t1"], H=panel["H"], source="two faces" if panel["paired"] else "one face, median thickness",
                             paired=panel["paired"]))
    for i, column in enumerate(detail["columns"]):
        d = np.array(column["d"])
        n = np.array([-d[1], d[0]])
        centre = np.array(column["c"])
        elements.append(dict(kind="column", src=("column", i), d=d, n=n, o=float(centre @ n), T=column["b"], t0=float(centre @ d - column["a"] / 2),
                             t1=float(centre @ d + column["a"] / 2), H=height, source="column points, outer box", paired=True))
    return elements, floor_z, height


def extend_glass_walls(ctx: ModelContext, elements: List[dict], floor_z: float) -> None:
    """A glass wall ending within GLASS_EXTENSION of a perpendicular wall or the data edge runs up to it when the
    gap is full of points (junction clutter, mullions); an empty gap stays open."""
    points = ctx.points
    points = points[(points[:, 2] > floor_z + 0.3) & (points[:, 2] < floor_z + 2.0), :2]
    crop = ctx.crop
    walls = [x for x in elements if x["kind"] == "wall"]
    for glass in (x for x in elements if x["kind"] == "glass_wall"):
        for end, sign in (("t0", -1), ("t1", 1)):
            tip = glass[end]
            stops = []
            for wall in walls:
                if abs(wall["d"] @ glass["d"]) >= 0.1 or abs(wall["n"] @ glass["d"]) <= 0.9:
                    continue
                crossing = [glass["o"] - (wall["d"] * t + wall["n"] * wall["o"]) @ glass["n"] for t in (wall["t0"], wall["t1"])]
                if min(crossing) > 0.2 or max(crossing) < -0.2:
                    continue
                for face in (wall["o"] - wall["T"] / 2, wall["o"] + wall["T"] / 2):
                    position = (wall["n"] * face) @ glass["d"]
                    if 0 < sign * (position - tip) <= GLASS_EXTENSION:
                        stops.append(position)
            axis = 0 if abs(glass["d"][0]) > 0.9 else 1
            for limit in (crop[axis], crop[axis + 2]):
                position = limit * np.sign(glass["d"][axis])
                if 0 < sign * (position - tip) <= GLASS_EXTENSION:
                    stops.append(position)
            if not stops:
                continue
            target = min(stops, key=lambda t: abs(t - tip))
            low, high = sorted((tip, target))
            along = points @ glass["d"]
            in_band = (np.abs(points @ glass["n"] - glass["o"]) < 0.15) & (along > low) & (along < high)
            density = in_band.sum() / max(high - low, 1e-3)
            if density >= GLASS_GAP_DENSITY:
                glass[end] = target
                ctx.log(f"glass wall {end} extended by {abs(target - tip):.2f} m ({density:.0f} points/m in the gap)")
            else:
                ctx.log(f"glass wall {end}: gap of {abs(target - tip):.2f} m left open ({density:.0f} points/m)")


def thin_against_parallel(x: dict, accepted: List[dict], outlines: List[Polygon], stats: collections.Counter) -> None:
    outline = rectangle(x)
    low, high = x["o"] - x["T"] / 2, x["o"] + x["T"] / 2
    touched = False
    for other_outline, other in zip(outlines, accepted):
        shared = outline.intersection(other_outline)
        if shared.area < 1e-6 or abs(other["d"] @ x["d"]) < np.cos(np.radians(PARALLEL_ANGLE)):
            continue
        corners = np.vstack([np.array(g.exterior.coords) for g in (shared.geoms if hasattr(shared, "geoms") else [shared]) if g.geom_type == "Polygon"])
        across = corners @ x["n"]
        if across.max() - across.min() < 0.5 * x["T"] - 1e-9:
            touched = True
            if across.mean() > x["o"]:
                high = min(high, across.min() - GAP)
            else:
                low = max(low, across.max() + GAP)
    if touched and high - low >= max(0.02, 0.4 * x["T"]):
        x["o"], x["T"] = (low + high) / 2, high - low
        stats["thinned"] += 1


def cut_pieces(x: dict, outlines: List[Polygon], stats: collections.Counter) -> List[Tuple[float, float]]:
    outline = rectangle(x)
    hits = [p.buffer(GAP, join_style=2) for p in outlines if outline.intersection(p).area > 1e-6]
    pieces = [(x["t0"], x["t1"])]
    if hits:
        shared = outline.intersection(unary_union(hits))
        for part in (shared.geoms if hasattr(shared, "geoms") else [shared]):
            if part.geom_type != "Polygon" or part.area < 1e-7:
                continue
            along = np.array(part.exterior.coords) @ x["d"]
            a, b = along.min(), along.max()
            remaining = []
            for s0, s1 in pieces:
                if b <= s0 or a >= s1:
                    remaining.append((s0, s1))
                else:
                    if a > s0:
                        remaining.append((s0, a))
                    if b < s1:
                        remaining.append((b, s1))
            pieces = remaining
        stats["trimmed"] += 1
    return pieces


def resolve_overlaps(elements: List[dict], log, tag: str) -> List[dict]:
    accepted, outlines, stats = [], [], collections.Counter()
    for x in sorted(elements, key=lambda e: (PRIORITY[e["kind"]], -(e["t1"] - e["t0"]))):
        if x["kind"] == "column":
            accepted.append(dict(x))
            outlines.append(rectangle(x))
            continue
        thin_against_parallel(x, accepted, outlines, stats)
        kept = [(a, b) for a, b in cut_pieces(x, outlines, stats) if b - a >= MIN_LENGTH]
        stats["split"] += len(kept) > 1
        stats[f"removed {x['kind']}"] += not kept
        for a, b in kept:
            accepted.append(dict(x, t0=a, t1=b))
            outlines.append(rectangle(accepted[-1]))
    log(f"overlap resolution ({tag}): {dict(stats)}")
    return accepted


def measure_heights(ctx: ModelContext, elements: List[dict], floor_z: float) -> None:
    points, labels = ctx.points, ctx.labels
    own = {"wall": points[labels == WALL], "column": points[labels == COLUMN], "glass_wall": points[labels == WINDOW]}
    bands = 0
    for x in elements:
        x["base"] = 0.0
        if x["kind"] == "partition":
            continue
        cloud = own[x["kind"]]
        reach = 0.3 if x["kind"] == "glass_wall" else 0.06
        along, across = cloud[:, :2] @ x["d"], cloud[:, :2] @ x["n"]
        z = cloud[(along > x["t0"]) & (along < x["t1"]) & (np.abs(across - x["o"]) < x["T"] / 2 + reach), 2] - floor_z
        if len(z) < HEIGHT_POINTS_MIN:
            continue
        bottom, top = np.percentile(z, [1, 99])
        if x["kind"] != "glass_wall":
            x["H"] = min(x["H"], float(top))
        elif bottom > GLASS_FLOOR_BAND:
            x["base"], x["H"] = float(bottom), float(top - bottom)
            x["source"] = "window band, sill and head from window points"
            bands += 1
    ctx.log(f"heights measured from own points; window bands: {bands}")


def clean_elements(ctx: ModelContext) -> Tuple[List[dict], float, float]:
    elements, floor_z, height = collect_elements(ctx)
    extend_glass_walls(ctx, elements, floor_z)
    ctx.log(f"elements in: {dict(collections.Counter(x['kind'] for x in elements))}")
    cleaned = resolve_overlaps(elements, ctx.log, "pass 1")
    for number in (2, 3):
        cleaned = junctions.remove_stubs(cleaned, ctx.log)
        junctions.join(cleaned, PRIORITY, ctx.log)
        for x in cleaned:
            x["t1"] = max(x["t1"], x["t0"] + 1e-3)
        cleaned = resolve_overlaps(cleaned, ctx.log, f"pass {number}")
    short = [x for x in cleaned if x["kind"] == "partition" and x["t1"] - x["t0"] < MIN_PARTITION_LENGTH]
    cleaned = [x for x in cleaned if not (x["kind"] == "partition" and x["t1"] - x["t0"] < MIN_PARTITION_LENGTH)]
    ctx.log(f"partition pieces below {MIN_PARTITION_LENGTH} m removed: {len(short)}")
    ctx.log(f"elements out: {dict(collections.Counter(x['kind'] for x in cleaned))}")
    measure_heights(ctx, cleaned, floor_z)
    return cleaned, floor_z, height
