"""Columns and cubicle partitions.

Columns: column points at 0.3-2.4 m above the floor, DBSCAN in plan, box aligned to the main direction.
    The sides are the 3rd and 97th percentiles along each axis, which ignores scan noise around the faces.
Partitions: the face accumulation of the structure stage on partition points, band +-2 cm, runs split
    at gaps above 0.15 m. Opposite faces 2-15 cm apart form one panel with measured thickness; single
    faces get the median thickness. The height is the 99th percentile z of the panel's points, snapped to the
    nearest reference height within 5 cm (reference heights are peaks of the run-height histogram, Wei 2023).
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from sklearn.cluster import DBSCAN

from ..classes import COLUMN, PARTITION
from .context import ModelContext


def histogram_peaks(values: np.ndarray, step: float = 0.01, min_share: float = 0.01) -> List[float]:
    counts, edges = np.histogram(values, bins=np.arange(values.min() - 0.05, values.max() + 0.05, step))
    smooth = gaussian_filter1d(counts.astype(float), 1.0)
    peaks = np.where((smooth == maximum_filter1d(smooth, 5)) & (smooth > min_share * smooth.max()))[0]
    return [float(edges[k] + step / 2) for k in peaks]


def detect_columns(points: np.ndarray, labels: np.ndarray, floor_z: float, family: float) -> List[dict]:
    plan = points[(labels == COLUMN) & (points[:, 2] > floor_z + 0.3) & (points[:, 2] < floor_z + 2.4), :2]
    if len(plan) == 0:
        return []
    clusters = DBSCAN(eps=0.15, min_samples=20).fit_predict(plan)
    angle = np.radians(family)
    d = np.array([np.cos(angle), np.sin(angle)])
    n = np.array([-d[1], d[0]])
    columns = []
    for k in sorted(set(clusters) - {-1}):
        members = plan[clusters == k]
        if len(members) < 200:
            continue
        u0, u1 = np.percentile(members @ d, [3, 97])
        v0, v1 = np.percentile(members @ n, [3, 97])
        if max(u1 - u0, v1 - v0) > 2.0:
            continue
        centre = d * (u0 + u1) / 2 + n * (v0 + v1) / 2
        columns.append(dict(c=centre.tolist(), d=d.tolist(), a=float(u1 - u0), b=float(v1 - v0), npts=int(len(members))))
    return columns


def partition_faces(points: np.ndarray, normals: np.ndarray, labels: np.ndarray, floor_z: float, families: List[float]) -> List[dict]:
    mask = (labels == PARTITION) & (np.abs(normals[:, 2]) < 0.3) & (points[:, 2] > floor_z + 0.1)
    faces = []
    for family in families:
        for quarter in range(2):
            angle = np.radians(family + 90 * quarter)
            n = np.array([np.cos(angle), np.sin(angle)])
            d = np.array([-n[1], n[0]])
            selected = np.where(mask & (np.abs(normals[:, :2] @ n) > np.cos(np.radians(20))))[0]
            if len(selected) < 100:
                continue
            offsets = points[selected, :2] @ n
            for offset in histogram_peaks(offsets):
                inliers = selected[np.abs(offsets - offset) < 0.02]
                if len(inliers) < 100:
                    continue
                along = points[inliers, :2] @ d
                order = np.argsort(along)
                for run in np.split(order, np.where(np.diff(along[order]) > 0.15)[0] + 1):
                    t = along[run]
                    if len(run) < 100 or t.max() - t.min() < 0.3:
                        continue
                    faces.append(dict(fam=family, n=n, d=d, o=float(np.median(points[inliers[run], :2] @ n)),
                                      t0=float(np.percentile(t, 0.5)), t1=float(np.percentile(t, 99.5)),
                                      ztop=float(np.percentile(points[inliers[run], 2], 99))))
    return faces


def pair_panels(faces: List[dict]) -> List[dict]:
    candidates = []
    for i, f in enumerate(faces):
        for j, g in enumerate(faces):
            if j <= i or f["fam"] != g["fam"] or f["n"] @ g["n"] < 0.99:
                continue
            thickness = abs(g["o"] - f["o"])
            if not 0.02 <= thickness <= 0.15:
                continue
            overlap = min(f["t1"], g["t1"]) - max(f["t0"], g["t0"])
            if overlap >= 0.5 * min(f["t1"] - f["t0"], g["t1"] - g["t0"]):
                candidates.append((thickness, i, j))
    used, panels = set(), []
    for thickness, i, j in sorted(candidates):
        if i in used or j in used:
            continue
        f, g = faces[i], faces[j]
        used |= {i, j}
        panels.append(dict(n=g["n"], d=g["d"], o=(g["o"] + f["o"]) / 2, T=thickness, t0=min(f["t0"], g["t0"]), t1=max(f["t1"], g["t1"]),
                           ztop=max(f["ztop"], g["ztop"]), paired=True))
    median = float(np.median([p["T"] for p in panels])) if panels else 0.05
    for i, f in enumerate(faces):
        if i not in used:
            panels.append(dict(n=f["n"], d=f["d"], o=f["o"] - median / 2, T=median, t0=f["t0"], t1=f["t1"], ztop=f["ztop"], paired=False))
    return panels


def measure_tops(panels: List[dict], partition_points: np.ndarray) -> None:
    """Panel tops from all partition points: the normals of the face search tilt at the top edge."""
    for panel in panels:
        along = partition_points[:, :2] @ panel["d"]
        across = partition_points[:, :2] @ panel["n"]
        z = partition_points[(along > panel["t0"]) & (along < panel["t1"]) & (np.abs(across - panel["o"]) < panel["T"] / 2 + 0.04), 2]
        if len(z) >= 100:
            panel["ztop"] = float(np.percentile(z, 99))


def snap_heights(panels: List[dict], floor_z: float) -> List[float]:
    heights = np.array([p["ztop"] - floor_z for p in panels])
    lengths = np.array([p["t1"] - p["t0"] for p in panels])
    counts, edges = np.histogram(heights, bins=np.arange(0, 3, 0.02), weights=lengths)
    references = [float(edges[k] + 0.01) for k in np.where((counts == maximum_filter1d(counts, 5)) & (counts > 0.1 * counts.max()))[0]]
    for panel in panels:
        height = panel["ztop"] - floor_z
        nearest = min(references, key=lambda r: abs(r - height))
        panel["H"] = nearest if abs(nearest - height) <= 0.05 else height
    return references


def plot(ctx: ModelContext, points: np.ndarray, labels: np.ndarray, panels: List[dict], columns: List[dict]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as PolygonPatch

    fig, ax = plt.subplots(figsize=(16, 11))
    sample = np.where(labels == PARTITION)[0][::20]
    ax.scatter(points[sample, 0], points[sample, 1], s=0.05, c="0.7")
    for panel in panels:
        n, d = np.array(panel["n"]), np.array(panel["d"])
        corners = [n * (panel["o"] + side * panel["T"] / 2) + d * t for t, side in
                   ((panel["t0"], -1), (panel["t1"], -1), (panel["t1"], 1), (panel["t0"], 1))]
        ax.add_patch(PolygonPatch(np.array(corners), fc="#2ca02c" if panel["paired"] else "#98df8a", ec="k", lw=0.3))
    for column in columns:
        centre, d = np.array(column["c"]), np.array(column["d"])
        n = np.array([-d[1], d[0]])
        ax.add_patch(PolygonPatch(np.array([centre + sx * d * column["a"] / 2 + sy * n * column["b"] / 2 for sx, sy in
                                            ((-1, -1), (1, -1), (1, 1), (-1, 1))]), fc="#d62728"))
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_title(f"{len(panels)} partition panels, {len(columns)} columns")
    fig.tight_layout()
    fig.savefig(ctx.output / "plan_columns_partitions.png", dpi=80)
    plt.close(fig)


def run(ctx: ModelContext) -> None:
    structure = ctx.load("structure")
    points, normals, labels = ctx.points, ctx.normals, ctx.labels
    floor_z, families = structure["floor_z"], structure["families"]

    columns = detect_columns(points, labels, floor_z, families[0])
    ctx.log(f"columns: {len(columns)}; sizes " + ", ".join(f"{c['a']:.2f} x {c['b']:.2f}" for c in columns))

    faces = partition_faces(points, normals, labels, floor_z, families)
    panels = pair_panels(faces)
    measure_tops(panels, points[labels == PARTITION])
    references = snap_heights(panels, floor_z) if panels else []
    paired = [p["T"] for p in panels if p["paired"]]
    ctx.log(f"partition faces {len(faces)} -> panels {len(panels)} ({len(paired)} paired, thickness median "
            f"{np.median(paired) if paired else 0:.3f} m), total length {sum(p['t1'] - p['t0'] for p in panels):.1f} m; "
            f"reference heights {[round(r, 2) for r in references]} m")
    for panel in panels:
        panel["n"], panel["d"] = panel["n"].tolist(), panel["d"].tolist()
    ctx.save("columns_partitions", dict(columns=columns, partitions=panels))
    plot(ctx, points, labels, panels, columns)
