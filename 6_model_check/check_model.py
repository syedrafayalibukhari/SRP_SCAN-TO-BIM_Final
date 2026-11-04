"""Step 6: check the IFC model against the IFC schema and against the labelled floor.

Checks: (1) IFC4 schema validation, (2) geometry of every element can be built, (3) relationships: every
element sits in the storey, every opening is cut into a wall and filled, (4) overlaps between walls,
partitions and columns in plan, (5) distance from the labelled points of each class to the elements of
that class, (6) groups of door and column points with and without a modelled element nearby.

The distances compare the model with the labels. They are not a survey of the real building.

Usage:  python check_model.py
        python check_model.py --model model.ifc --input floor_merged.ply --components components.json
Output: model_check_report.txt next to the model. The exit code is 1 when check 1, 2 or 3 fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from scan2bim.setup_env import ensure_packages

if __name__ == "__main__":
    ensure_packages({"numpy": "numpy", "open3d": "open3d", "shapely": "shapely", "sklearn": "scikit-learn", "ifcopenshell": "ifcopenshell"})

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.validate
import numpy as np
import open3d as o3d
from shapely.geometry import LineString, Polygon
from sklearn.cluster import DBSCAN

from scan2bim.classes import CEILING, CLASSES, COLUMN, CURTAIN_WALL, DOOR, FLOOR, PARTITION, WALL, WINDOW
from scan2bim.plyio import read_ply

SAMPLE_PER_CLASS = 300_000
DISTANCES_M = (0.05, 0.10)
GROUP_EPS_M = 0.25
GROUP_MIN_POINTS = 300
NEAR_ELEMENT_M = 0.5
OVERLAP_MIN_M2 = 1e-4


def element_group(product) -> str:
    if product.is_a("IfcSlab"):
        return "floor"
    if product.is_a("IfcCovering"):
        return "ceiling"
    if product.is_a("IfcCurtainWall"):
        return "curtain_wall"
    if product.is_a("IfcWindow"):
        return "window"
    if product.is_a("IfcDoor"):
        return "door"
    if product.is_a("IfcColumn"):
        return "column"
    return "partition" if product.PredefinedType == "PARTITIONING" else "wall"


def schema_errors(model) -> List[str]:
    logger = ifcopenshell.validate.json_logger()
    ifcopenshell.validate.validate(model, logger)
    return [str(entry.get("message", entry)) for entry in logger.statements if entry.get("level") == "error"]


def build_meshes(model) -> Tuple[Dict[str, List[Tuple[np.ndarray, np.ndarray]]], List[str]]:
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    meshes: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    failed = []
    for ifc_class in ("IfcSlab", "IfcCovering", "IfcWall", "IfcCurtainWall", "IfcDoor", "IfcWindow", "IfcColumn"):
        for product in model.by_type(ifc_class):
            try:
                shape = ifcopenshell.geom.create_shape(settings, product)
                vertices = np.array(shape.geometry.verts, float).reshape(-1, 3)
                faces = np.array(shape.geometry.faces, np.int32).reshape(-1, 3)
                if len(faces) == 0:
                    raise RuntimeError("empty geometry")
                meshes.setdefault(element_group(product), []).append((vertices, faces))
            except Exception as error:
                failed.append(f"{product.is_a()} {product.Name}: {error}")
    return meshes, failed


def relationship_errors(model) -> List[str]:
    errors = []
    contained = {e.id() for rel in model.by_type("IfcRelContainedInSpatialStructure") for e in rel.RelatedElements}
    for ifc_class in ("IfcSlab", "IfcCovering", "IfcWall", "IfcCurtainWall", "IfcDoor", "IfcWindow", "IfcColumn"):
        errors += [f"{e.Name} is not in a storey" for e in model.by_type(ifc_class) if e.id() not in contained]
    aggregated = {e.id() for rel in model.by_type("IfcRelAggregates") for e in rel.RelatedObjects}
    errors += [f"{e.Name} is not in a storey" for e in model.by_type("IfcSpace") if e.id() not in aggregated]
    errors += [f"{e.Name} is cut into no wall" for e in model.by_type("IfcOpeningElement") if not e.VoidsElements]
    errors += [f"{e.Name} is not filled" for e in model.by_type("IfcOpeningElement") if not e.HasFillings]
    return errors


def plan_outlines(components: dict) -> List[Tuple[str, Polygon]]:
    types = {t["id"]: t for section in ("WallTypes", "ColumnTypes") for t in components[section]}
    outlines = []
    for wall in components["Walls"]:
        line = LineString([wall["start"], wall["end"]])
        outlines.append((wall["name"], line.buffer(types[wall["typeID"]]["thickness"] / 2, cap_style=2, join_style=2)))
    for column in components["Columns"]:
        kind, angle = types[column["typeID"]], np.radians(column["rotation"])
        d = np.array([np.cos(angle), np.sin(angle)])
        n = np.array([-d[1], d[0]])
        centre = np.array(column["location"])
        outlines.append((column["name"], Polygon([centre + a * d * kind["width"] / 2 + b * n * kind["depth"] / 2 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])))
    return outlines


def overlaps(outlines: List[Tuple[str, Polygon]]) -> List[str]:
    from shapely.strtree import STRtree

    polygons = [p for _, p in outlines]
    tree = STRtree(polygons)
    found = []
    for i, polygon in enumerate(polygons):
        for j in tree.query(polygon):
            j = int(j)
            if j > i:
                area = polygon.intersection(polygons[j]).area
                if area > OVERLAP_MIN_M2:
                    found.append(f"{outlines[i][0]} / {outlines[j][0]}: {area:.4f} m2")
    return found


def distance_scene(parts: List[Tuple[np.ndarray, np.ndarray]]):
    scene = o3d.t.geometry.RaycastingScene()
    for vertices, faces in parts:
        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = o3d.core.Tensor(vertices.astype(np.float32))
        mesh.triangle.indices = o3d.core.Tensor(faces.astype(np.uint32))
        scene.add_triangles(mesh)
    return scene


def distances_to(meshes: Dict[str, list], names: Tuple[str, ...], points: np.ndarray) -> np.ndarray:
    parts = [part for name in names for part in meshes.get(name, [])]
    if not parts:
        return np.full(len(points), np.inf)
    return distance_scene(parts).compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy()


def label_groups(points: np.ndarray, meshes: Dict[str, list], names: Tuple[str, ...]) -> Tuple[int, int, List[List[float]]]:
    """Groups of labelled points in plan, and how many have an element of the class within NEAR_ELEMENT_M."""
    if len(points) == 0:
        return 0, 0, []
    keep = points[np.random.default_rng(0).choice(len(points), min(60_000, len(points)), replace=False)]
    groups = DBSCAN(eps=GROUP_EPS_M, min_samples=10).fit_predict(keep[:, :2])
    scale = len(points) / len(keep)
    total = matched = 0
    missing = []
    for k in sorted(set(groups) - {-1}):
        members = keep[groups == k]
        if len(members) * scale < GROUP_MIN_POINTS:
            continue
        total += 1
        if distances_to(meshes, names, members).min() <= NEAR_ELEMENT_M:
            matched += 1
        else:
            missing.append([round(float(v), 2) for v in members[:, :2].mean(0)])
    return total, matched, missing


def main() -> None:
    default_output = ROOT.parent / "5_modeling" / "output"
    parser = argparse.ArgumentParser(description="Check the IFC model.")
    parser.add_argument("--model", type=Path, default=default_output / "model.ifc", help="IFC model")
    parser.add_argument("--input", type=Path, default=ROOT.parent / "4_merge_tiles" / "merged" / "floor_merged.ply", help="labelled floor")
    parser.add_argument("--components", type=Path, help="component description (default: next to the model)")
    args = parser.parse_args()
    for path in (args.model, args.input):
        if not path.exists():
            raise SystemExit(f"File not found: {path}")
    lines: List[str] = []

    def log(message: str = "") -> None:
        print(message, flush=True)
        lines.append(message)

    model = ifcopenshell.open(str(args.model))
    log(f"MODEL CHECK: {args.model}\n")
    errors = schema_errors(model)
    log(f"1. IFC4 schema validation: {len(errors)} errors")
    for error in errors[:20]:
        log(f"   {error}")

    meshes, failed = build_meshes(model)
    log(f"2. Geometry: {sum(len(v) for v in meshes.values())} elements built, {len(failed)} failed")
    for name in ("floor", "ceiling", "wall", "door", "window", "curtain_wall", "column", "partition"):
        log(f"   {name:13s}{len(meshes.get(name, [])):5d}")
    log(f"   spaces: {len(model.by_type('IfcSpace'))}")
    for item in failed[:20]:
        log(f"   {item}")

    broken = relationship_errors(model)
    log(f"3. Relationships: {len(broken)} problems")
    for item in broken[:20]:
        log(f"   {item}")

    components_path = args.components or args.model.with_name("components.json")
    if components_path.exists():
        found = overlaps(plan_outlines(json.loads(components_path.read_text(encoding="utf-8"))))
        log(f"4. Plan overlaps above {OVERLAP_MIN_M2} m2: {len(found)}")
        for item in found[:20]:
            log(f"   {item}")
    else:
        log("4. Plan overlaps: skipped (components.json not found)")

    data = read_ply(args.input, required=("x", "y", "z", "label"))
    xyz = np.c_[data["x"], data["y"], data["z"]].astype(np.float64)
    labels = data["label"]
    log("5. Labelled points near elements of their class (sample per class)")
    log(f"   {'class':13s}{'points':>12s}{'median cm':>11s}" + "".join(f"{f'<= {d * 100:.0f} cm':>10s}" for d in DISTANCES_M))
    rng = np.random.default_rng(0)
    targets = {FLOOR: ("floor",), CEILING: ("ceiling",), WALL: ("wall",), DOOR: ("door",), WINDOW: ("window",), CURTAIN_WALL: ("curtain_wall",), COLUMN: ("column",), PARTITION: ("partition",)}
    for value, names in targets.items():
        members = xyz[labels == value]
        if len(members) == 0:
            log(f"   {CLASSES[value]:13s}{0:12,}")
            continue
        sample = members[rng.choice(len(members), min(SAMPLE_PER_CLASS, len(members)), replace=False)]
        distance = distances_to(meshes, names, sample)
        shares = "".join(f"{100 * (distance <= d).mean():9.1f}%" for d in DISTANCES_M)
        log(f"   {CLASSES[value]:13s}{len(members):12,}{100 * np.median(distance):11.1f}{shares}")

    log(f"6. Groups of labelled points with a modelled element within {NEAR_ELEMENT_M} m (at least {GROUP_MIN_POINTS} points)")
    for value, names in ((DOOR, ("door",)), (COLUMN, ("column",))):
        total, matched, missing = label_groups(xyz[labels == value], meshes, names)
        log(f"   {CLASSES[value]:10s}{matched} of {total} groups")
        for position in missing[:40]:
            log(f"      no element near x {position[0]}, y {position[1]}")

    report = args.model.with_name("model_check_report.txt")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {report}")
    if errors or failed or broken:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
