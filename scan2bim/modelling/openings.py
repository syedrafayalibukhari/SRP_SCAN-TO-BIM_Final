"""Door leaves from door points and windows in walls.

Doors: door points are grouped in plan (DBSCAN); one group is one door. In each group, straight lines are
fitted one after the other (RANSAC); lines that are parallel and close together are two faces of one
line. A group holds one line (a closed door, or an open leaf) or two (the doorway and the open leaf).
The components stage puts each door into its doorway.

Windows use a per-wall image (Previtali 2018), 5 cm pixels in (t along the wall, z). Each pixel is
    wall      wall points within the slab (+-3 cm of its faces)
    window    window points within 0.15 m of the slab
    empty     no point, but a scanner has a free line of sight to the pixel
    occluded  no point and every scanner ray is blocked: treated as wall
Openings are connected components of (empty | window) after a 2-pixel closing; the rectangle is their
bounding box. A window starts at least 0.3 m above the floor, is 0.5-3.0 m wide, at least 60 % filled
and at least 10 % of its pixels hold window points: a gap seen by rays alone is not a window.
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN

from ..classes import DOOR, WALL, WINDOW
from .context import ModelContext

PIXEL, VOXEL = 0.05, 0.05
MIN_WINDOW_LABEL_SHARE = 0.10
DOOR_GROUP_EPS, LEAF_BAND, LEAF_MIN_WIDTH, LEAF_MAX_WIDTH = 0.12, 0.03, 0.5, 2.6
LEAF_FACE_DISTANCE, LEAF_FACE_ANGLE = 0.20, 10.0
SAME_FLOOR_BAND = 0.2


def fit_lines(plan: np.ndarray, rng: np.random.Generator) -> List[dict]:
    """Fit straight lines to a group of plan points, the strongest first."""
    lines: List[dict] = []
    remaining = plan
    while len(remaining) >= max(200, 0.1 * len(plan)) and len(lines) < 4:
        best = None
        for _ in range(400):
            a, b = remaining[rng.choice(len(remaining), 2, replace=False)]
            distance = np.linalg.norm(b - a)
            if distance < 0.2:
                continue
            d = (b - a) / distance
            n = np.array([-d[1], d[0]])
            inliers = np.abs((remaining - a) @ n) < LEAF_BAND
            if best is None or inliers.sum() > best[0]:
                best = (int(inliers.sum()), a, d, n, inliers)
        if best is None:
            break
        count, a, d, n, inliers = best
        t0, t1 = np.percentile((remaining[inliers] - a) @ d, [1, 99])
        lines.append(dict(point=a + d * (t0 + t1) / 2, d=d, n=n, width=float(t1 - t0), count=count))
        remaining = remaining[np.abs((remaining - a) @ n) > 2 * LEAF_BAND]
    return lines


def merge_faces(lines: List[dict]) -> List[dict]:
    """Keep one line per leaf: a parallel line close to a stronger line is the other face of that leaf."""
    leaves: List[dict] = []
    for line in lines:
        same = next((leaf for leaf in leaves if abs(leaf["d"] @ line["d"]) > np.cos(np.radians(LEAF_FACE_ANGLE))
                     and abs((line["point"] - leaf["point"]) @ leaf["n"]) < LEAF_FACE_DISTANCE), None)
        if same is None:
            leaves.append(dict(line, faces=1))
        else:
            same["point"] = same["point"] + same["n"] * ((line["point"] - same["point"]) @ same["n"]) / (same["faces"] + 1)
            same["width"] = max(same["width"], line["width"])
            same["count"] += line["count"]
            same["faces"] += 1
    return leaves


def detect_door_leaves(points: np.ndarray, labels: np.ndarray, floor_z: float) -> List[dict]:
    door_points = points[labels == DOOR]
    if len(door_points) < 200:
        return []
    usable = door_points[(door_points[:, 2] > floor_z + 0.2) & (door_points[:, 2] < floor_z + 2.3)]
    groups = DBSCAN(eps=DOOR_GROUP_EPS, min_samples=15).fit_predict(usable[:, :2])
    rng = np.random.default_rng(0)
    leaves = []
    for group in sorted(set(groups) - {-1}):
        members = usable[groups == group]
        if len(members) < 200:
            continue
        lines = [leaf for leaf in merge_faces(fit_lines(members[:, :2], rng))
                 if LEAF_MIN_WIDTH <= leaf["width"] <= LEAF_MAX_WIDTH and leaf["count"] >= 0.1 * len(members)]
        if not lines:
            continue
        low, high = np.percentile(door_points[np.linalg.norm(door_points[:, :2] - members[:, :2].mean(0), axis=1) < 1.5, 2] - floor_z, [1, 99])
        bottom = 0.0 if low <= SAME_FLOOR_BAND else float(low)
        leaves.append(dict(lines=[dict(c=line["point"].tolist(), d=line["d"].tolist(), width=line["width"]) for line in lines[:2]],
                           z0=bottom, height=float(high) - bottom))
    return leaves


def sight_lines(voxel_origin: np.ndarray, grid: np.ndarray, stations: list, targets: np.ndarray,
                n: np.ndarray, d: np.ndarray, o0: float, o1: float) -> np.ndarray:
    """True for every (t, z) target with a free ray from at least one scanner, stopping 12 cm before the slab."""
    seen = np.zeros(len(targets), bool)
    for station in stations:
        side = o0 - 0.12 if station[:2] @ n < o0 else o1 + 0.12
        ends = np.c_[targets[:, 0:1] * d + side * n, targets[:, 1]]
        vectors = ends - station
        samples = int(np.linalg.norm(vectors, axis=1).max() / (VOXEL / 2)) + 1
        along = station + vectors[:, None, :] * np.linspace(0, 1, samples)[None, :, None]
        along = along[:, int(0.3 / (VOXEL / 2)):]               # skip the first 30 cm around the scanner
        cells = np.clip(((along - voxel_origin) / VOXEL).astype(int), 0, np.array(grid.shape) - 1)
        seen |= ~grid[cells[..., 0], cells[..., 1], cells[..., 2]].any(1)
    return seen


def detect_on_slab(index: int, slab: dict, points: np.ndarray, labels: np.ndarray, stations: np.ndarray, floor_z: float, top: float,
                   voxel_origin: np.ndarray, grid: np.ndarray, found: List[dict], images: List[tuple]) -> None:
    n, d = np.array(slab["n"]), np.array(slab["d"])
    o0, o1, t0, t1 = slab["o0"], slab["o1"], slab["t0"], slab["t1"]
    columns, rows = int((t1 - t0) / PIXEL) + 1, int((top - floor_z) / PIXEL) + 1
    offset, along = points[:, :2] @ n, points[:, :2] @ d
    near = (offset > o0 - 0.35) & (offset < o1 + 0.35) & (along > t0 - 0.05) & (along < t1 + 0.05) & (points[:, 2] > floor_z - 0.05) & (points[:, 2] < top + 0.05)
    index_near = np.where(near)[0]
    it = np.clip(((along[index_near] - t0) / PIXEL).astype(int), 0, columns - 1)
    iz = np.clip(((points[index_near, 2] - floor_z) / PIXEL).astype(int), 0, rows - 1)
    off = offset[index_near]
    near_labels = labels[index_near]

    wall_px = np.zeros((columns, rows), bool)
    window_px = np.zeros((columns, rows), bool)
    mask = (off > o0 - 0.03) & (off < o1 + 0.03) & (near_labels == WALL)
    wall_px[it[mask], iz[mask]] = True
    mask = (near_labels == WINDOW) & (off > o0 - 0.15) & (off < o1 + 0.15)
    window_px[it[mask], iz[mask]] = True

    grid[:] = False
    keep = np.ones(len(points), bool)
    keep[index_near[(off > o0 - 0.12) & (off < o1 + 0.12)]] = False
    cells = ((points[keep] - voxel_origin) / VOXEL).astype(int)
    grid[cells[:, 0], cells[:, 1], cells[:, 2]] = True

    candidates = np.argwhere(~(wall_px | window_px))
    empty_px = np.zeros((columns, rows), bool)
    occluded_px = np.zeros((columns, rows), bool)
    if len(candidates):
        along_station = stations[:, :2] @ n
        chosen = []
        for side in (along_station < o0, along_station > o1):       # nearest 12 positions per side
            rank = np.abs((stations[side][:, :2] @ d) - (t0 + t1) / 2) + np.abs(along_station[side] - o1)
            chosen.extend(stations[side][np.argsort(rank)[:12]])
        centres = np.c_[t0 + (candidates[:, 0] + 0.5) * PIXEL, floor_z + (candidates[:, 1] + 0.5) * PIXEL]
        for begin in range(0, len(candidates), 4000):
            part = candidates[begin:begin + 4000]
            seen = sight_lines(voxel_origin, grid, chosen, centres[begin:begin + 4000], n, d, o0, o1)
            empty_px[part[seen, 0], part[seen, 1]] = True
            occluded_px[part[~seen, 0], part[~seen, 1]] = True

    opening_mask = ndimage.binary_closing(empty_px | window_px, iterations=2) & ~ndimage.binary_erosion(wall_px, iterations=1)
    regions, count = ndimage.label(opening_mask)
    for k in range(1, count + 1):
        ii, zz = np.where(regions == k)
        a, b = t0 + ii.min() * PIXEL, t0 + (ii.max() + 1) * PIXEL
        bottom, upper = zz.min() * PIXEL, (zz.max() + 1) * PIXEL
        width, height = b - a, upper - bottom
        windows = window_px[ii, zz].sum()
        fill = len(ii) / ((ii.max() - ii.min() + 1) * (zz.max() - zz.min() + 1))
        if bottom < 0.3 or height < 0.4 or upper >= top - floor_z - 0.1 or not 0.5 <= width <= 3.0 or fill < 0.6:
            continue
        if windows < MIN_WINDOW_LABEL_SHARE * len(ii):
            continue
        found.append(dict(slab=index, t0=a, t1=b, z0=bottom, z1=upper, width=width, height=height, fill=fill))
    images.append((index, wall_px, window_px, empty_px, occluded_px, t0))


def plot(ctx: ModelContext, images: List[tuple], found: List[dict]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    long_walls = [image for image in images if image[1].shape[0] * PIXEL > 1]
    if not long_walls:
        return
    fig, axes = plt.subplots(len(long_walls), 1, figsize=(18, 1.6 * len(long_walls)))
    for ax, (index, wall_px, window_px, empty_px, occluded_px, t0) in zip(np.atleast_1d(axes), long_walls):
        picture = np.ones(wall_px.shape + (3,))
        picture[occluded_px] = [0.75, 0.75, 0.75]
        picture[wall_px] = [0.35, 0.35, 0.35]
        picture[empty_px] = [1.0, 0.85, 0.3]
        picture[window_px] = [0.3, 0.6, 1.0]
        ax.imshow(picture.transpose(1, 0, 2), origin="lower", aspect="equal",
                  extent=[t0, t0 + wall_px.shape[0] * PIXEL, 0, wall_px.shape[1] * PIXEL])
        for item in found:
            if item["slab"] == index:
                ax.add_patch(Rectangle((item["t0"], item["z0"]), item["width"], item["height"], fc="none", ec="m", lw=2))
        ax.set_ylabel(f"wall {index}")
        ax.set_yticks([])
    fig.suptitle("Wall images: dark = wall, grey = occluded, yellow = seen empty, blue = window; boxes = windows")
    fig.tight_layout()
    fig.savefig(ctx.output / "wall_images.png", dpi=60)
    plt.close(fig)


def run(ctx: ModelContext) -> None:
    structure, rooms = ctx.load("structure"), ctx.load("rooms")
    points, labels, stations = ctx.points, ctx.labels, ctx.stations
    floor_z = structure["floor_z"]
    top = max(r["z_ceil"] for r in rooms["rooms"])
    voxel_origin = points.min(0) - 0.5
    grid = np.zeros(tuple(((points.max(0) + 0.5 - voxel_origin) / VOXEL).astype(int) + 1), bool)
    found, images = [], []
    for index, slab in enumerate(rooms["slabs"]):
        if slab["kind"] == "wall":
            detect_on_slab(index, slab, points, labels, stations, floor_z, top, voxel_origin, grid, found, images)
    doors = detect_door_leaves(points, labels, floor_z)
    ctx.log(f"doors: {len(doors)}; windows in walls: {len(found)}")
    for door in doors:
        ctx.log(f"  door with {len(door['lines'])} line(s) at ({door['lines'][0]['c'][0]:.2f}, {door['lines'][0]['c'][1]:.2f}), height {door['height']:.2f} m")
    for item in found:
        ctx.log(f"  wall {item['slab']:2d} window {item['width']:.2f} x {item['height']:.2f} m, sill {item['z0']:.2f} m, fill {item['fill']:.2f}")
    ctx.save("openings", dict(doors=doors, windows=found))
    plot(ctx, images, found)
