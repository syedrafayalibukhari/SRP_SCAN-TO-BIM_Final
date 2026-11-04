"""Levels, dominant directions, wall and glass-wall faces, and wall slabs.

Levels (Oesau 2014, Cui 2019): z-histogram of horizontal-normal floor and ceiling points, 1 cm bins, peaks.
Directions (Cui 2019, Wei 2023): histogram of horizontal normals modulo 90 degrees.
Faces: for each direction, 1 cm histogram of the offset along the normal, peaks, inlier band +-3 cm,
       runs along the axis split at gaps. Faces are found separately on wall, curtain wall and window
       points; the label decides what a glazed run becomes.
Slabs: a face with normal -n at p and a face with normal +n at q (q - p in 0.04-0.45 m, overlapping) form
       one wall with measured thickness; a glazed pair may be only a few cm apart. Unpaired faces are
       single-sided walls whose thickness is decided in the rooms stage.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

from ..classes import CEILING, COLUMN, CURTAIN_WALL, FLOOR, PARTITION, WALL, WINDOW
from .context import ModelContext

GAP_SPLIT, BAND, MIN_LEN, MIN_POINTS = 0.5, 0.03, 0.3, 300
RAY_PIXEL = 0.02
WALL_THICKNESS_RANGE = (0.04, 0.45)
GLASS_THICKNESS_RANGE = (-0.03, 0.15)
GLASS_MIN_THICKNESS = 0.06
GLAZED = ("glass_wall", "window_run")


def histogram_peaks(values: np.ndarray, step: float = 0.01, min_share: float = 0.05) -> List[Tuple[float, float]]:
    counts, edges = np.histogram(values, bins=np.arange(values.min() - 0.05, values.max() + 0.05, step))
    smooth = gaussian_filter1d(counts.astype(float), 1.5)
    peaks = np.where((smooth == maximum_filter1d(smooth, 15)) & (smooth > min_share * smooth.max()))[0]
    return [(float(edges[k] + step / 2), float(smooth[k])) for k in peaks]


def dominant_directions(points_normals: np.ndarray) -> List[float]:
    """Direction families: local maxima of the 1-degree histogram of normals modulo 90 degrees that stand
    at least 25 % above the lowest bin within +-10 degrees and hold at least 0.5 % of the points."""
    angles = np.degrees(np.arctan2(points_normals[:, 1], points_normals[:, 0])) % 90
    histogram = np.histogram(angles, bins=90, range=(0, 90))[0] / len(angles)
    families: List[float] = []
    for k in np.argsort(-histogram):
        window = [histogram[(k + j) % 90] for j in range(-8, 9)]
        lowest = min(histogram[(k + j) % 90] for j in range(-10, 11))
        if histogram[k] < 0.005:
            break
        if histogram[k] < max(window) or histogram[k] < 1.25 * lowest:
            continue
        if all(min(abs(k + 0.5 - f), 90 - abs(k + 0.5 - f)) > 10 for f in families):
            centre = k + 0.5
            delta = (angles - centre + 45) % 90 - 45
            families.append(float((centre + delta[np.abs(delta) < 2].mean()) % 90))
    return families


class LineOfSight:
    """2 cm plan raster of wall, window and column points, used to decide from which side a face was seen."""

    def __init__(self, points: np.ndarray, labels: np.ndarray, floor_z: float, viewpoints: np.ndarray) -> None:
        self.viewpoints = viewpoints
        self.x0, self.y0 = points[:, 0].min() - 0.5, points[:, 1].min() - 0.5
        shape = (int((points[:, 0].max() + 0.5 - self.x0) / RAY_PIXEL) + 1, int((points[:, 1].max() + 0.5 - self.y0) / RAY_PIXEL) + 1)
        self.blocked = np.zeros(shape, bool)
        mask = np.isin(labels, [WALL, WINDOW, CURTAIN_WALL, COLUMN]) & (points[:, 2] > floor_z + 0.5) & (points[:, 2] < floor_z + 2.4)
        cells = ((points[mask, :2] - [self.x0, self.y0]) / RAY_PIXEL).astype(int)
        self.blocked[cells[:, 0], cells[:, 1]] = True

    def seen_from(self, point: np.ndarray, side: np.ndarray) -> int:
        """Number of viewpoints on ``side`` with a free 2D line of sight to ``point``."""
        count = 0
        for view in self.viewpoints:
            if (view - point) @ side <= 0:
                continue
            steps = int(np.linalg.norm(view - point) / (RAY_PIXEL / 2))
            fractions = np.linspace(0, 1, steps)[4:]            # skip the first 4 cm next to the point
            cells = ((point + np.outer(fractions, view - point) - [self.x0, self.y0]) / RAY_PIXEL).astype(int)
            cells = cells[(cells[:, 0] >= 0) & (cells[:, 0] < self.blocked.shape[0]) & (cells[:, 1] >= 0) & (cells[:, 1] < self.blocked.shape[1])]
            count += not self.blocked[cells[:, 0], cells[:, 1]].any()
        return count


def detect_faces(points: np.ndarray, normals: np.ndarray, masks: Dict[str, np.ndarray], families: List[float], sight: LineOfSight) -> List[dict]:
    """Faces are found on unsigned offsets; each face's viewing side is decided by line of sight."""
    faces = []
    for kind, mask in masks.items():
        for family in families:
            for quarter in range(2):
                angle = np.radians(family + 90 * quarter)
                n = np.array([np.cos(angle), np.sin(angle)])
                d = np.array([-n[1], n[0]])
                selected = mask & (np.abs(normals[:, :2] @ n) > np.cos(np.radians(15)))
                if selected.sum() < MIN_POINTS:
                    continue
                offsets = points[selected, :2] @ n
                for offset, _ in histogram_peaks(offsets, 0.01, 0.01):
                    inliers = np.where(selected)[0][np.abs(offsets - offset) < BAND]
                    if len(inliers) < MIN_POINTS:
                        continue
                    offset = float(np.median(points[inliers, :2] @ n))
                    along = points[inliers, :2] @ d
                    order = np.argsort(along)
                    cuts = np.where(np.diff(along[order]) > GAP_SPLIT)[0]
                    for run in np.split(order, cuts + 1):
                        t = along[run]
                        if len(run) < MIN_POINTS or t.max() - t.min() < MIN_LEN:
                            continue
                        members = inliers[run]
                        z = points[members, 2]
                        centred = points[members, :2] - points[members, :2].mean(0)
                        principal = np.linalg.svd(centred, full_matrices=False)[2][0]
                        t0, t1 = float(np.percentile(t, 0.2)), float(np.percentile(t, 99.8))
                        probes = [n * offset + d * tq for tq in np.linspace(t0, t1, 7)[1:-1]]
                        visible = {sign: sum(sight.seen_from(c + sign * 0.15 * n, sign * n) > 0 for c in probes) for sign in (1, -1)}
                        sides = [sign for sign in (1, -1) if visible[sign] >= 2] or [max(visible, key=visible.get)]
                        for sign in sides:                  # a face seen from both sides is a thin element: two faces
                            nn = sign * n
                            faces.append(dict(fam=family, n=nn.tolist(), d=[-nn[1], nn[0]], o=float(sign * offset),
                                              t0=t0 if sign > 0 else -t1, t1=t1 if sign > 0 else -t0,
                                              z0=float(np.percentile(z, 1)), z1=float(np.percentile(z, 99)), npts=int(len(run)),
                                              pca=principal.tolist(), kind=kind, seen=dict(plus=visible[1], minus=visible[-1])))
    return faces


def orient_unseen_faces(faces: List[dict]) -> None:
    """A face no scan position can see faces away from a seen parallel face 4-45 cm away."""
    for face in faces:
        if face["seen"]["plus"] + face["seen"]["minus"] > 0:
            continue
        n = np.array(face["n"])
        for other in faces:
            if other is face or other["kind"] != face["kind"] or other["seen"]["plus"] + other["seen"]["minus"] == 0 or abs(np.dot(other["n"], n)) < 0.99:
                continue
            other_offset = other["o"] * np.dot(other["n"], n)
            span = sorted((other["t0"], other["t1"])) if np.dot(other["d"], face["d"]) > 0 else sorted((-other["t1"], -other["t0"]))
            if 0.04 <= abs(face["o"] - other_offset) <= 0.45 and min(span[1], face["t1"]) - max(span[0], face["t0"]) > 0.3:
                if np.sign(face["o"] - other_offset) != 1:   # the other face lies on the + side: this one faces -n
                    face.update(n=(-n).tolist(), d=[n[1], -n[0]], o=-face["o"], t0=-face["t1"], t1=-face["t0"])
                face["seen"]["inferred"] = True
                break


def pair_slabs(faces: List[dict]) -> List[dict]:
    """Pair opposite faces into slabs; keep long unpaired faces as single-sided slabs."""
    candidates = []
    for i, f in enumerate(faces):
        for j, g in enumerate(faces):
            if i == j or f["fam"] != g["fam"] or f["kind"] != g["kind"] or np.dot(f["n"], g["n"]) > -0.99:
                continue
            thickness = g["o"] + f["o"]
            low, high = WALL_THICKNESS_RANGE if g["kind"] == "wall" else GLASS_THICKNESS_RANGE
            if not low <= thickness <= high:
                continue
            span = sorted((-f["t1"], -f["t0"])) if np.dot(f["d"], g["d"]) < 0 else (f["t0"], f["t1"])
            overlap = min(span[1], g["t1"]) - max(span[0], g["t0"])
            if overlap < 0.5 * min(span[1] - span[0], g["t1"] - g["t0"]):
                continue
            candidates.append((thickness, i, j, span))
    used, slabs = set(), []
    for _, i, j, span in sorted(candidates, key=lambda c: c[:3]):
        if i in used or j in used:
            continue
        f, g = faces[i], faces[j]
        used |= {i, j}
        o0, o1 = -f["o"], g["o"]
        if g["kind"] in GLAZED and o1 - o0 < GLASS_MIN_THICKNESS:   # at least the frame depth, centred on the two faces
            centre = (o0 + o1) / 2
            o0, o1 = centre - GLASS_MIN_THICKNESS / 2, centre + GLASS_MIN_THICKNESS / 2
        slabs.append(dict(fam=g["fam"], n=g["n"], d=g["d"], o0=o0, o1=o1, t0=min(span[0], g["t0"]), t1=max(span[1], g["t1"]),
                          faces=[i, j], paired=True, kind=g["kind"]))
    for i, f in enumerate(faces):
        if i not in used and f["t1"] - f["t0"] >= (1.0 if f["kind"] in GLAZED else 0.5):
            slabs.append(dict(fam=f["fam"], n=f["n"], d=f["d"], o0=None, o1=f["o"], t0=f["t0"], t1=f["t1"], faces=[i], paired=False, kind=f["kind"]))

    def on_paired_face(slab: dict) -> bool:
        n = np.array(slab["n"])
        for other in slabs:
            if other["paired"] and abs(np.dot(other["n"], n)) > 0.99:
                sign = np.dot(other["n"], n)
                if min(abs(slab["o1"] - sign * other["o0"]), abs(slab["o1"] - sign * other["o1"])) < 0.02:
                    return True
        return False

    return [s for s in slabs if not (not s["paired"] and on_paired_face(s)) and not (s["kind"] in GLAZED and s["t1"] - s["t0"] < 1.0)]


def plot(ctx: ModelContext, points: np.ndarray, floor_z: float, slabs: List[dict]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 11))
    sample = np.random.default_rng(0).choice(len(points), min(400000, len(points)), replace=False)
    sample = sample[(points[sample, 2] > floor_z + 0.3) & (points[sample, 2] < floor_z + 2.2)]
    ax.scatter(points[sample, 0], points[sample, 1], s=0.03, c="0.75")
    for slab in slabs:
        n, d = np.array(slab["n"]), np.array(slab["d"])
        for offset in ([slab["o0"], slab["o1"]] if slab["paired"] else [slab["o1"]]):
            a, b = n * offset + d * slab["t0"], n * offset + d * slab["t1"]
            ax.plot([a[0], b[0]], [a[1], b[1]], c="#d62728" if slab["paired"] else "#1f77b4", lw=1.5)
    ax.plot([], [], c="#d62728", label="paired faces (measured thickness)")
    ax.plot([], [], c="#1f77b4", label="single face")
    ax.legend()
    ax.set_aspect("equal")
    ax.set_title("Wall and glazing faces")
    fig.tight_layout()
    fig.savefig(ctx.output / "plan_faces.png", dpi=80)
    plt.close(fig)


def run(ctx: ModelContext) -> None:
    points, normals, labels = ctx.points, ctx.normals, ctx.labels
    horizontal = np.abs(normals[:, 2]) > 0.95
    floor_z = max(histogram_peaks(points[horizontal & (labels == FLOOR), 2]), key=lambda peak: peak[1])[0]
    ceilings = sorted(z for z, _ in histogram_peaks(points[horizontal & (labels == CEILING), 2]))
    ctx.log(f"floor level {floor_z:.3f} m; ceiling levels {[round(z, 3) for z in ceilings]} m")

    vertical = np.abs(normals[:, 2]) < 0.2
    masks = {"wall": (labels == WALL) & vertical, "glass_wall": (labels == CURTAIN_WALL) & vertical, "window_run": (labels == WINDOW) & vertical}
    families = dominant_directions(normals[np.isin(labels, [WALL, WINDOW, CURTAIN_WALL, PARTITION]) & vertical])
    ctx.log(f"dominant directions (mod 90): {[round(f, 2) for f in families]} deg")

    sight = LineOfSight(points, labels, floor_z, ctx.stations[:, :2])
    faces = detect_faces(points, normals, masks, families, sight)
    for _ in range(2):      # refine each direction from its face inliers (PCA angle, length-weighted)
        refined = []
        for family in families:
            members = [f for f in faces if f["fam"] == family]
            offsets = [((np.degrees(np.arctan2(f["pca"][1], f["pca"][0])) - family + 45) % 90 - 45) for f in members]
            refined.append(float((family + np.average(offsets, weights=[f["t1"] - f["t0"] for f in members])) % 90))
        families = refined
        faces = detect_faces(points, normals, masks, families, sight)
    ctx.log(f"refined directions (mod 90): {[round(f, 3) for f in families]} deg")
    ctx.log(f"faces: {len(faces)} (band +-{BAND} m, split at gaps > {GAP_SPLIT} m, length >= {MIN_LEN} m)")
    orient_unseen_faces(faces)
    slabs = pair_slabs(faces)
    widths = [s["o1"] - s["o0"] for s in slabs if s["paired"] and s["kind"] == "wall"] or [0.0]
    ctx.log(f"slabs: {sum(s['paired'] for s in slabs)} paired (wall thickness median {np.median(widths):.3f} m, "
            f"range {min(widths):.3f}-{max(widths):.3f}), {sum(not s['paired'] for s in slabs)} single-face")
    ctx.save("structure", dict(floor_z=floor_z, ceilings=ceilings, families=families, faces=faces, slabs=slabs))
    plot(ctx, points, floor_z, slabs)
