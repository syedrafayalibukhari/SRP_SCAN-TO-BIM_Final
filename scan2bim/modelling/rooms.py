"""Room footprints. They bound the ceilings and the floor, and they tell whether a wall has space on one side.

Visibility (Cui 2019, Previtali 2018): 5 cm plan raster; blockers are wall, column, door and window points
    at 0.3-2.4 m above the floor plus the detected faces; 2880 rays from every scanner position.
Clustering (Cui 2019): overlap ratio of visible areas between positions, Otsu threshold, union-find.
Cell complex (Oesau 2014, Cui 2019): face lines extended across the floor and polygonized; cells between
    the two faces of a paired wall are fixed as solid.
Ceiling occupancy (Previtali 2018): cells less than half covered by ceiling points are outside.
Labelling: multi-label graph cut (alpha-expansion, Potts) with data term = cell area x (1 - share of the
    cell seen by the room's positions), outside cost = cell area x (largest share seen by any room),
    smoothness = LAMBDA x edge length x (1 - face support of the edge) (Oesau 2014).
Room heights: median z of the ceiling points inside each room.
"""
from __future__ import annotations

from typing import List, Tuple

import networkx as nx
import numpy as np
from matplotlib.path import Path as PlotPath
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes
from shapely.geometry import LineString, Point, Polygon, box, mapping, shape
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree
from skimage.filters import threshold_otsu

from ..classes import CEILING, COLUMN, DOOR, WALL, WINDOW
from .context import ModelContext

PIXEL, LAMBDA = 0.05, 0.6
SINGLE_FACE_WALL_THICKNESS, SINGLE_FACE_GLASS_THICKNESS = 0.25, 0.06
LARGE = 1e6


class PlanRaster:
    def __init__(self, points: np.ndarray) -> None:
        self.x0, self.y0 = points[:, 0].min() - 1.0, points[:, 1].min() - 1.0
        self.nx = int((points[:, 0].max() + 1.0 - self.x0) / PIXEL) + 1
        self.ny = int((points[:, 1].max() + 1.0 - self.y0) / PIXEL) + 1

    def cells(self, xy: np.ndarray) -> np.ndarray:
        return np.floor((np.asarray(xy) - [self.x0, self.y0]) / PIXEL).astype(int)


def blocker_map(raster: PlanRaster, points: np.ndarray, labels: np.ndarray, faces: List[dict], floor_z: float) -> np.ndarray:
    blocked = np.zeros((raster.nx, raster.ny), bool)
    mask = np.isin(labels, [WALL, COLUMN, DOOR, WINDOW]) & (points[:, 2] > floor_z + 0.3) & (points[:, 2] < floor_z + 2.4)
    cells = raster.cells(points[mask, :2])
    blocked[cells[:, 0], cells[:, 1]] = True
    for face in faces:
        n, d = np.array(face["n"]), np.array(face["d"])
        cells = raster.cells(n * face["o"] + np.outer(np.arange(face["t0"], face["t1"], PIXEL / 2), d))
        blocked[cells[:, 0], cells[:, 1]] = True
    return binary_dilation(blocked, iterations=1)


def visible_cells(raster: PlanRaster, blocked: np.ndarray, stations: np.ndarray) -> List[np.ndarray]:
    angles = np.linspace(0, 2 * np.pi, 2880, endpoint=False)
    radii = np.arange(0.2, 30, PIXEL / 2)                   # start 20 cm out: a position may stand next to a wall
    grid = np.stack([np.outer(np.cos(angles), radii), np.outer(np.sin(angles), radii)], axis=-1)
    visible = []
    for station in stations:
        cells = raster.cells((station[:2] + grid).reshape(-1, 2)).reshape(len(angles), len(radii), 2)
        inside = (cells[..., 0] >= 0) & (cells[..., 0] < raster.nx) & (cells[..., 1] >= 0) & (cells[..., 1] < raster.ny)
        clipped = np.clip(cells, 0, [raster.nx - 1, raster.ny - 1])
        hit = blocked[clipped[..., 0], clipped[..., 1]] | ~inside
        first = np.where(hit.any(1), hit.argmax(1), len(radii))
        free = np.arange(len(radii))[None, :] < first[:, None]
        visible.append(np.unique(clipped[free][:, 0] * raster.ny + clipped[free][:, 1]))
    return visible


def cluster_stations(visible: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    count = len(visible)
    overlap = np.zeros((count, count))
    for i in range(count):
        for j in range(i + 1, count):
            shared = len(np.intersect1d(visible[i], visible[j], assume_unique=True))
            overlap[i, j] = overlap[j, i] = shared / max(1, min(len(visible[i]), len(visible[j])))
    values = overlap[np.triu_indices(count, 1)]
    threshold = float(threshold_otsu(values[values > 0])) if (values > 0).any() and len(np.unique(values[values > 0])) > 1 else 0.5
    parent = list(range(count))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(count):
        for j in range(i + 1, count):
            if overlap[i, j] > threshold:
                parent[find(i)] = find(j)
    roots = sorted({find(i) for i in range(count)}, key=lambda r: -sum(find(i) == r for i in range(count)))
    return np.array([roots.index(find(i)) for i in range(count)]), threshold


def cell_complex(raster: PlanRaster, faces: List[dict], slabs: List[dict]) -> Tuple[list, list]:
    bounds = box(raster.x0 + 0.5, raster.y0 + 0.5, raster.x0 + raster.nx * PIXEL - 0.5, raster.y0 + raster.ny * PIXEL - 0.5)
    lines = {}
    for face in faces:
        lines[(round(face["fam"], 3), round(face["n"][0], 3), round(face["n"][1], 3), round(face["o"], 2))] = face
    segments = []
    for face in lines.values():
        n, d = np.array(face["n"]), np.array(face["d"])
        segments.append(LineString([n * face["o"] - d * 100, n * face["o"] + d * 100]).intersection(bounds))
    cells = [c for c in polygonize(unary_union(segments + [bounds.exterior])) if c.area > 1e-4]
    solids = []
    for slab in slabs:
        if slab["paired"]:
            n, d = np.array(slab["n"]), np.array(slab["d"])
            solids.append(Polygon([n * slab["o0"] + d * slab["t0"], n * slab["o0"] + d * slab["t1"],
                                   n * slab["o1"] + d * slab["t1"], n * slab["o1"] + d * slab["t0"]]))
    return cells, [any(s.contains(c.representative_point()) for s in solids) for c in cells]


def data_terms(raster: PlanRaster, cells: list, solid: list, visibility: np.ndarray, ceiling_map: np.ndarray) -> np.ndarray:
    rooms = visibility.shape[0]
    terms = np.zeros((len(cells), rooms + 1))
    covered = np.zeros(len(cells))
    cx = raster.x0 + (np.arange(raster.nx) + 0.5) * PIXEL
    cy = raster.y0 + (np.arange(raster.ny) + 0.5) * PIXEL
    for k, cell in enumerate(cells):
        xa, ya, xb, yb = cell.bounds
        i0, i1 = np.searchsorted(cx, [xa, xb])
        j0, j1 = np.searchsorted(cy, [ya, yb])
        if i1 <= i0 or j1 <= j0:
            terms[k, :rooms] = cell.area
            continue
        gx, gy = np.meshgrid(cx[i0:i1], cy[j0:j1], indexing="ij")
        inside = PlotPath(np.array(cell.exterior.coords)).contains_points(np.c_[gx.ravel(), gy.ravel()])
        if not inside.any():
            terms[k, :rooms] = cell.area
            continue
        index = (np.repeat(np.arange(i0, i1), j1 - j0) * raster.ny + np.tile(np.arange(j0, j1), i1 - i0))[inside]
        share = visibility[:, index].mean(1)
        terms[k, :rooms] = cell.area * (1 - share)
        terms[k, rooms] = cell.area * share.max()
        covered[k] = ceiling_map.ravel()[index].mean()
    for k in range(len(cells)):
        if covered[k] < 0.5 or solid[k]:        # no ceiling above (rays escape through glass) or wall interior
            terms[k, :rooms] = LARGE
            terms[k, rooms] = 0
    return terms


def edge_weights(cells: list, faces: List[dict]) -> List[tuple]:
    strips = [LineString([np.array(f["n"]) * f["o"] + np.array(f["d"]) * f["t0"], np.array(f["n"]) * f["o"] + np.array(f["d"]) * f["t1"]]).buffer(0.02) for f in faces]
    support = unary_union(strips)
    tree = STRtree(cells)
    edges = []
    for i, cell in enumerate(cells):
        for j in tree.query(cell):
            j = int(j)
            if j <= i:
                continue
            shared = cell.boundary.intersection(cells[j].boundary)
            if shared.length < 1e-4:
                continue
            edges.append((i, j, LAMBDA * shared.length * (1 - shared.intersection(support).length / shared.length)))
    return edges


def energy(terms: np.ndarray, edges: List[tuple], labelling: np.ndarray) -> float:
    return terms[np.arange(len(labelling)), labelling].sum() + sum(w for i, j, w in edges if labelling[i] != labelling[j])


def expand(terms: np.ndarray, edges: List[tuple], labelling: np.ndarray, alpha: int) -> np.ndarray:
    """One alpha-expansion move solved as a minimum cut."""
    graph = nx.DiGraph()
    for i in range(len(labelling)):
        graph.add_edge("s", i, capacity=float(terms[i, alpha]))
        graph.add_edge(i, "t", capacity=float(terms[i, labelling[i]]) if labelling[i] != alpha else LARGE * 10)
    for i, j, weight in edges:
        if weight <= 0:
            continue
        li, lj = labelling[i], labelling[j]
        if li == lj:
            if li != alpha:
                graph.add_edge(i, j, capacity=weight)
                graph.add_edge(j, i, capacity=weight)
        else:
            auxiliary = ("aux", i, j)
            for node in (i, j):
                graph.add_edge(node, auxiliary, capacity=weight)
                graph.add_edge(auxiliary, node, capacity=weight)
            graph.add_edge(auxiliary, "t", capacity=weight)
    _, (_, sink_side) = nx.minimum_cut(graph, "s", "t")
    result = labelling.copy()
    for i in range(len(labelling)):
        if i in sink_side:
            result[i] = alpha
    return result


def label_cells(terms: np.ndarray, edges: List[tuple]) -> Tuple[np.ndarray, float, float, int]:
    labelling = terms.argmin(1)
    start = energy(terms, edges, labelling)
    for sweep in range(4):
        changed = False
        for alpha in range(terms.shape[1]):
            moved = expand(terms, edges, labelling, alpha)
            if energy(terms, edges, moved) < energy(terms, edges, labelling) - 1e-9:
                labelling, changed = moved, True
        if not changed:
            break
    return labelling, start, energy(terms, edges, labelling), sweep + 1


def plot(ctx: ModelContext, rooms: List[dict]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as PolygonPatch

    fig, ax = plt.subplots(figsize=(14, 9))
    colours = plt.get_cmap("tab20")
    for i, room in enumerate(rooms):
        polygon = shape(room["poly"])
        ax.add_patch(PolygonPatch(np.array(polygon.exterior.coords), fc=colours(i % 20), alpha=0.5, ec="k"))
        q = polygon.representative_point()
        ax.text(q.x, q.y, f"{room['name']}\n{room['area']:.1f} m2", ha="center", fontsize=8)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_title(f"Room footprints: {len(rooms)}")
    fig.tight_layout()
    fig.savefig(ctx.output / "plan_rooms.png", dpi=80)
    plt.close(fig)


def run(ctx: ModelContext) -> None:
    structure = ctx.load("structure")
    points, labels, stations = ctx.points, ctx.labels, ctx.stations
    floor_z = structure["floor_z"]
    slabs = structure["slabs"]
    faces = [structure["faces"][i] for slab in slabs for i in slab["faces"]]
    raster = PlanRaster(points)

    blocked = blocker_map(raster, points, labels, faces, floor_z)
    visible = visible_cells(raster, blocked, stations)
    ctx.log(f"visibility: {len(stations)} positions, median visible area {np.median([len(v) for v in visible]) * PIXEL * PIXEL:.1f} m2")
    clusters, threshold = cluster_stations(visible)
    ctx.log(f"position clusters: Otsu threshold {threshold:.2f} -> {clusters.max() + 1} clusters, sizes {np.bincount(clusters).tolist()}")

    count = clusters.max() + 1
    visibility = np.zeros((count, raster.nx * raster.ny), bool)
    for i, cells_seen in enumerate(visible):
        visibility[clusters[i], cells_seen] = True

    cells, solid = cell_complex(raster, faces, slabs)
    ceiling_points = points[(labels == CEILING) & (points[:, 2] > floor_z + 2.3)]
    ceiling_map = np.zeros((raster.nx, raster.ny), bool)
    occupied = raster.cells(ceiling_points[:, :2])
    ceiling_map[occupied[:, 0], occupied[:, 1]] = True
    ceiling_map = binary_fill_holes(binary_closing(ceiling_map, iterations=6))   # lamps, diffusers, dropouts

    terms = data_terms(raster, cells, solid, visibility, ceiling_map)
    edges = edge_weights(cells, faces)
    labelling, before, after, sweeps = label_cells(terms, edges)
    ctx.log(f"cell complex: {len(cells)} cells ({sum(solid)} inside paired walls), {len(edges)} edges; "
            f"alpha-expansion energy {before:.1f} -> {after:.1f} in {sweeps} sweeps")

    crop = box(*ctx.crop)
    all_ceiling = points[labels == CEILING]
    rooms = []
    for k in range(count):
        members = [cells[i] for i in np.where(labelling == k)[0]]
        if not members:
            continue
        merged = unary_union(members).buffer(1e-6).buffer(-1e-6).intersection(crop)
        for part in (merged.geoms if hasattr(merged, "geoms") else [merged]):
            if part.area < 1.0:
                continue
            part = part.simplify(0.005)
            inside = PlotPath(np.array(part.exterior.coords)).contains_points(all_ceiling[:, :2])
            z_ceiling = float(np.median(all_ceiling[inside, 2])) if inside.sum() > 100 else float(structure["ceilings"][-1])
            rooms.append(dict(poly=mapping(part), area=part.area, z_ceil=z_ceiling))
    rooms.sort(key=lambda r: -r["area"])
    for i, room in enumerate(rooms):
        room["name"] = f"Space {i + 1:02d}"
    ctx.log(f"rooms: {len(rooms)}; " + ", ".join(f"{r['name']} {r['area']:.1f} m2 (h {r['z_ceil'] - floor_z:.2f})" for r in rooms))

    inside_rooms = unary_union([shape(r["poly"]) for r in rooms])
    for slab in slabs:
        if slab["paired"]:
            slab["side"] = "interior"
            continue
        n, d = np.array(slab["n"]), np.array(slab["d"])
        behind = Point(n * (slab["o1"] - 0.3) + d * (slab["t0"] + slab["t1"]) / 2)
        slab["side"] = "exterior" if not inside_rooms.contains(behind) else "interior"
        slab["o0"] = slab["o1"] - (SINGLE_FACE_GLASS_THICKNESS if slab["kind"] == "glass_wall" else SINGLE_FACE_WALL_THICKNESS)
    ctx.log(f"single-face slabs: {sum(not s['paired'] and s['side'] == 'exterior' for s in slabs)} bounded by outside or unseen space "
            f"(thickness assumed), {sum(not s['paired'] and s['side'] == 'interior' for s in slabs)} interior")
    ctx.save("rooms", dict(rooms=rooms, slabs=slabs))
    plot(ctx, rooms)
