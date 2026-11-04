"""Parametric component description of the reconstructed floor (Tang et al. 2022, section 3.4).

Every component is a type (shared parameters) plus occurrences (instance parameters). Types are found by
grouping measured instances. The description is written as components.json and components.gml; the IFC
model is built from it alone, without reading the point cloud.

    Wall types / walls        function (exterior, interior, glass_wall, partition), thickness, centre line, height
    Door and window types     width, height, sill; occurrences are hosted by a wall or glass wall at an offset
                              along it; a door in a gap between elements has its own location and rotation
    Spaces                    room name, boundary polygon, height
    Floor and ceiling types   thickness, material; occurrences have a boundary polygon
    Column types / columns    width, depth; occurrences have a location and a rotation
"""
from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from typing import List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Point, shape
from shapely.ops import unary_union

from .cleanup import clean_elements, rectangle
from .context import ModelContext

FLOOR_THICKNESS, CEILING_THICKNESS = 0.2, 0.02
WALL_MATERIALS = {"exterior": "Exterior wall (material not observable)", "interior": "Interior wall", "glass_wall": "Glass", "partition": "Workstation panel"}
HOST_ANGLE, HOST_DISTANCE = 10.0, 0.12
MIN_SWING_ANGLE, HINGE_DISTANCE = 45.0, 0.25
NAMES = {"exterior": "Wall", "interior": "Wall", "glass_wall": "Glass wall", "partition": "Partition"}


def r3(value: float) -> float:
    return round(float(value), 3)


def group(values: Sequence[Sequence[float]], tolerance: np.ndarray) -> List[Tuple[np.ndarray, List[int]]]:
    """Greedy grouping of parameter vectors; returns (mean vector, member indices)."""
    groups: List[Tuple[List[np.ndarray], List[int]]] = []
    for i in np.argsort([-v[0] for v in values]):
        vector = np.asarray(values[i], float)
        for members, index in groups:
            if np.all(np.abs(np.mean(members, 0) - vector) <= tolerance):
                members.append(vector)
                index.append(int(i))
                break
        else:
            groups.append(([vector], [int(i)]))
    return [(np.mean(members, 0), index) for members, index in groups]


def ring(line) -> List[List[float]]:
    return [[r3(x), r3(y)] for x, y in line.coords]


def describe_walls(elements: List[dict], room_polygons: list) -> Tuple[List[dict], List[dict], List[dict]]:
    walls = [x for x in elements if x["kind"] != "column"]
    for x in walls:
        x["function"] = x["kind"] if x["kind"] in ("partition", "glass_wall") else ("interior" if x["paired"] else "exterior")
    types = []
    for function in WALL_MATERIALS:
        members = [x for x in walls if x["function"] == function]
        if not members:
            continue
        for mean, index in group([[x["T"], x["H"]] for x in members], np.array([0.01, 0.05])):
            type_id = f"WT{len(types) + 1:02d}"
            assumed = any(not members[i]["paired"] for i in index)
            if assumed:
                evidence = "assumed thickness (only one face scanned)"
            else:
                evidence = "glass panel, 6 cm frame depth assumed" if function == "glass_wall" else "measured from two faces"
            types.append(dict(id=type_id, function=function, thickness=r3(mean[0]), height=r3(mean[1]), material=WALL_MATERIALS[function], evidence=evidence))
            for i in index:
                members[i]["typeID"] = type_id
    occurrences, counters = [], {}
    for k, x in enumerate(sorted(walls, key=lambda e: (e["function"], -(e["t1"] - e["t0"])))):
        start = x["d"] * x["t0"] + x["n"] * x["o"]
        end = x["d"] * x["t1"] + x["n"] * x["o"]
        left = np.array([-(end - start)[1], (end - start)[0]]) / max(np.linalg.norm(end - start), 1e-9)
        middle = (start + end) / 2

        def has_space(sign: int) -> bool:
            probe = Point(middle + sign * left * (x["T"] / 2 + 0.3))
            return any(p.contains(probe) for p in room_polygons)

        side = "both" if has_space(1) and has_space(-1) else ("left" if has_space(1) else ("right" if has_space(-1) else "none"))
        label = "Window band" if x["base"] > 0 else NAMES[x["function"]]
        counters[label] = counters.get(label, 0) + 1
        x["id"] = f"W{k + 1:03d}"
        occurrences.append(dict(id=x["id"], name=f"{label} {counters[label]:03d}", typeID=x["typeID"], start=[r3(start[0]), r3(start[1])], end=[r3(end[0]), r3(end[1])],
                                baseOffset=r3(x["base"]), height=r3(x["H"]), spaceSide=side, source=x["source"]))
    return walls, types, occurrences


def in_line_with(line: dict, walls: List[dict]) -> bool:
    """True if the line continues a wall or glass wall."""
    centre, d = np.array(line["c"]), np.array(line["d"])
    return any(x["function"] != "partition" and abs(x["d"] @ d) > np.cos(np.radians(HOST_ANGLE))
               and abs(centre @ x["n"] - x["o"]) <= x["T"] / 2 + HOST_DISTANCE for x in walls)


def line_ends(line: dict) -> Tuple[np.ndarray, np.ndarray]:
    centre, d = np.array(line["c"]), np.array(line["d"])
    return centre - d * line["width"] / 2, centre + d * line["width"] / 2


def doorway_of_two_lines(doorway: dict, leaf: dict) -> dict:
    """The door starts at the hinge, where the lines meet, and has the width of the open leaf."""
    pairs = [(np.linalg.norm(a - b), a, c) for a, c in (line_ends(doorway), line_ends(doorway)[::-1]) for b in line_ends(leaf)]
    _, hinge, far = min(pairs, key=lambda item: item[0])
    direction = (far - hinge) / np.linalg.norm(far - hinge)
    return dict(c=(hinge + direction * leaf["width"] / 2).tolist(), d=direction.tolist(), width=leaf["width"])


def doorway_of_open_leaf(leaf: dict, walls: List[dict]) -> dict:
    """Turn an open leaf about its hinge into the line of the wall it opens from."""
    best = None
    for end in line_ends(leaf):
        for x in walls:
            if x["function"] == "partition" or abs(x["d"] @ np.array(leaf["d"])) > np.cos(np.radians(MIN_SWING_ANGLE)):
                continue
            off_line = abs(end @ x["n"] - x["o"])
            along = end @ x["d"]
            to_wall = max(x["t0"] - along, along - x["t1"], 0.0)
            if off_line <= HINGE_DISTANCE and to_wall <= 1.5 * leaf["width"] and (best is None or off_line + to_wall < best[0]):
                sign = 1.0 if along < x["t0"] else -1.0
                best = (off_line + to_wall, end, x["d"] * sign)
    if best is None:
        return leaf
    _, hinge, direction = best
    return dict(c=(hinge + direction * leaf["width"] / 2).tolist(), d=direction.tolist(), width=leaf["width"])


def doorway(door: dict, walls: List[dict]) -> dict:
    """Closed position of a door from its one or two lines."""
    lines = door["lines"]
    in_plane = [line for line in lines if in_line_with(line, walls)]
    if len(lines) == 2 and len(in_plane) >= 1:
        other = next(line for line in lines if line is not in_plane[0])
        return doorway_of_two_lines(in_plane[0], other)
    if in_plane:
        return in_plane[0]
    return doorway_of_open_leaf(lines[0], walls)


def door_host(leaf: dict, walls: List[dict]) -> Optional[dict]:
    """The wall or glass wall the door is cut into, if any: parallel, in the wall plane, mostly along it."""
    centre, d = np.array(leaf["c"]), np.array(leaf["d"])
    half = leaf["width"] / 2
    for x in walls:
        if x["function"] == "partition" or abs(x["d"] @ d) < np.cos(np.radians(HOST_ANGLE)):
            continue
        along = centre @ x["d"]
        overlap = min(along + half, x["t1"]) - max(along - half, x["t0"])
        if abs(centre @ x["n"] - x["o"]) <= x["T"] / 2 + HOST_DISTANCE and overlap >= 0.5 * leaf["width"]:
            return x
    return None


def describe_openings(openings: dict, slabs: List[dict], walls: List[dict]) -> Tuple[dict, List[dict], List[dict]]:
    items = []
    for opening in openings["windows"]:
        slab = slabs[opening["slab"]]
        d, n = np.array(slab["d"]), np.array(slab["n"])
        centre = d * (opening["t0"] + opening["t1"]) / 2 + n * (slab["o0"] + slab["o1"]) / 2
        hosts = [x for x in walls if x["function"] != "partition" and abs(x["d"] @ d) > 0.99]
        host = min(hosts, key=lambda x: abs(centre @ x["n"] - x["o"]) + max(0, x["t0"] - centre @ x["d"], centre @ x["d"] - x["t1"]))
        items.append(dict(kind="window", wall=host, offset=float(centre @ host["d"] - host["t0"]), W=opening["width"], H=opening["height"], sill=opening["z0"]))
    for door in openings["doors"]:
        leaf = doorway(door, walls)
        host = door_host(leaf, walls)
        item = dict(kind="door", wall=host, W=leaf["width"], H=door["height"], sill=door["z0"])
        if host is not None:
            item["offset"] = float(np.array(leaf["c"]) @ host["d"] - host["t0"])
        else:
            item["location"] = [r3(v) for v in leaf["c"]]
            item["rotation"] = r3(np.degrees(np.arctan2(leaf["d"][1], leaf["d"][0])))
        items.append(item)
    types = {}
    for kind in ("door", "window"):
        members = [item for item in items if item["kind"] == kind]
        types[kind] = []
        if not members:
            continue
        for mean, index in group([[m["W"], m["H"], m["sill"]] for m in members], np.array([0.10, 0.10, 0.10])):
            type_id = f"{'DT' if kind == 'door' else 'NT'}{len(types[kind]) + 1:02d}"
            operation = "SINGLE_PANEL"
            if kind == "door":
                operation = "SINGLE_SWING_LEFT" if mean[0] < 1.3 else "DOUBLE_DOOR_SINGLE_SWING"
            types[kind].append(dict(id=type_id, width=r3(round(mean[0] / 0.05) * 0.05), height=r3(round(mean[1] / 0.05) * 0.05),
                                    sill=r3(round(mean[2] / 0.05) * 0.05), count=len(index), operation=operation))
            for i in index:
                members[i]["typeID"] = type_id
    doors, windows = [], []
    for item in items:
        group_list = doors if item["kind"] == "door" else windows
        record = dict(typeID=item["typeID"], wallID=None, width=r3(item["W"]), height=r3(item["H"]), sill=r3(item["sill"]))
        if item["wall"] is not None:
            record.update(wallID=item["wall"]["id"], offset=r3(item["offset"]), source=f"{item['kind']} points, in the wall")
        else:
            record.update(location=item["location"], rotation=item["rotation"], source="door points, in the gap between walls")
        record["id"] = f"{'D' if item['kind'] == 'door' else 'N'}{len(group_list) + 1:03d}"
        record["name"] = f"{item['kind'].title()} {len(group_list) + 1:03d}"
        group_list.append(record)
    return types, doors, windows


def describe_columns(elements: List[dict]) -> Tuple[List[dict], List[dict]]:
    columns = [x for x in elements if x["kind"] == "column"]
    types = []
    if columns:
        for mean, index in group([sorted([x["t1"] - x["t0"], x["T"]], reverse=True) for x in columns], np.array([0.03, 0.03])):
            type_id = f"KT{len(types) + 1:02d}"
            types.append(dict(id=type_id, shape="rectangular", width=r3(mean[0]), depth=r3(mean[1])))
            for i in index:
                columns[i]["typeID"] = type_id
    occurrences = []
    for i, x in enumerate(columns):
        centre = x["d"] * (x["t0"] + x["t1"]) / 2 + x["n"] * x["o"]
        along_d = x["t1"] - x["t0"] >= x["T"]         # the type width runs along the longer side
        axis = x["d"] if along_d else x["n"]
        occurrences.append(dict(id=f"K{i + 1:02d}", name=f"Column {i + 1:02d}", typeID=x["typeID"], location=[r3(centre[0]), r3(centre[1])],
                                rotation=r3(np.degrees(np.arctan2(axis[1], axis[0]))), height=r3(x["H"]), source=x["source"]))
    return types, occurrences


def describe_surfaces(rooms: List[dict], walls: List[dict], floor_z: float) -> Tuple[List[dict], List[dict], List[dict]]:
    ceilings, spaces = [], []
    for i, room in enumerate(rooms):
        polygon = shape(room["poly"])
        spaces.append(dict(id=f"S{i + 1:02d}", name=room["name"], height=r3(room["z_ceil"] - floor_z), area=round(room["area"], 2),
                           boundary=ring(polygon.exterior), holes=[ring(hole) for hole in polygon.interiors]))
        ceilings.append(dict(id=f"C{i + 1:02d}", name=f"Ceiling {i + 1:02d}", typeID="CT01", height=r3(room["z_ceil"] - floor_z), boundary=ring(polygon.exterior),
                             holes=[ring(hole) for hole in polygon.interiors], source="room footprint, median ceiling height"))
    outline = unary_union([shape(r["poly"]) for r in rooms] + [rectangle(x) for x in walls if x["function"] != "partition"])
    outline = outline.buffer(0.02, join_style=2).buffer(-0.02, join_style=2)
    floors = [dict(id=f"F{k + 1:02d}", name=f"Floor {k + 1:02d}", typeID="SL01", boundary=ring(part.exterior), holes=[ring(hole) for hole in part.interiors],
                   source="room footprints and walls")
              for k, part in enumerate(outline.geoms if hasattr(outline, "geoms") else [outline])]
    return floors, ceilings, spaces


def write_gml(ctx: ModelContext, description: dict) -> None:
    root = ET.Element("IndoorBIM", {"version": "1.0"})
    building = ET.SubElement(root, "Building", {k: str(v) for k, v in description["Building"].items()})
    storey = ET.SubElement(building, "Storey", {k: str(v) for k, v in description["Storey"].items()})
    positions = lambda pts: " ".join(f"{x} {y}" for x, y in pts)
    for section in ("WallTypes", "DoorTypes", "WindowTypes", "FloorTypes", "CeilingTypes", "ColumnTypes", "Doors", "Windows", "Columns"):
        parent = ET.SubElement(storey, section)
        for item in description[section]:
            ET.SubElement(parent, section[:-1], {k: (" ".join(map(str, v)) if isinstance(v, list) else str(v)) for k, v in item.items() if v is not None})
    parent = ET.SubElement(storey, "Walls")
    for wall in description["Walls"]:
        node = ET.SubElement(parent, "Wall", {k: str(v) for k, v in wall.items() if k not in ("start", "end")})
        ET.SubElement(node, "CentreLine").text = positions([wall["start"], wall["end"]])
    for section in ("Floors", "Ceilings", "Spaces"):
        parent = ET.SubElement(storey, section)
        for item in description[section]:
            node = ET.SubElement(parent, section[:-1], {k: str(v) for k, v in item.items() if k not in ("boundary", "holes")})
            ET.SubElement(node, "Boundary").text = positions(item["boundary"])
            for hole in item["holes"]:
                ET.SubElement(node, "Hole").text = positions(hole)
    ET.indent(root)
    ET.ElementTree(root).write(ctx.output / "components.gml", encoding="utf-8", xml_declaration=True)


def write_element_list(ctx: ModelContext, description: dict) -> None:
    types = {t["id"]: t for section in ("WallTypes", "ColumnTypes") for t in description[section]}
    with open(ctx.output / "elements.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "element", "x", "y", "length_or_width_m", "thickness_or_depth_m", "height_m", "source"])
        for wall in description["Walls"]:
            kind = types[wall["typeID"]]["function"]
            start, end = np.array(wall["start"]), np.array(wall["end"])
            writer.writerow([wall["name"], "wall" if kind in ("interior", "exterior") else kind, *np.round((start + end) / 2, 3),
                             r3(np.linalg.norm(end - start)), types[wall["typeID"]]["thickness"], wall["height"], wall["source"]])
        for section, kind in (("Doors", "door"), ("Windows", "window")):
            for item in description[section]:
                writer.writerow([item["name"], kind, *item.get("location", ["", ""]), item["width"], "", item["height"], item["source"]])
        for column in description["Columns"]:
            kind = types[column["typeID"]]
            writer.writerow([column["name"], "column", *column["location"], kind["width"], kind["depth"], column["height"], column["source"]])


def plot(ctx: ModelContext, description: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as PolygonPatch

    fig, ax = plt.subplots(figsize=(18, 12))
    types = {t["id"]: t for t in description["WallTypes"]}
    colours = {"exterior": "#333333", "interior": "#333333", "glass_wall": "#1f9ed1", "partition": "#2ca02c"}
    walls = {w["id"]: w for w in description["Walls"]}
    for wall in description["Walls"]:
        start, end = np.array(wall["start"]), np.array(wall["end"])
        kind = types[wall["typeID"]]
        ax.plot(*np.c_[start, end], c=colours[kind["function"]], lw=max(1.0, kind["thickness"] * 12), solid_capstyle="butt")
    for section, colour in (("Doors", "#ff7f0e"), ("Windows", "#e377c2")):
        for item in description[section]:
            if item["wallID"] is None:
                angle = np.radians(item["rotation"])
                direction, centre = np.array([np.cos(angle), np.sin(angle)]), np.array(item["location"])
            else:
                wall = walls[item["wallID"]]
                start, end = np.array(wall["start"]), np.array(wall["end"])
                direction = (end - start) / np.linalg.norm(end - start)
                centre = start + direction * item["offset"]
            a, b = centre - direction * item["width"] / 2, centre + direction * item["width"] / 2
            ax.plot(*np.c_[a, b], c=colour, lw=5, solid_capstyle="butt")
    column_types = {t["id"]: t for t in description["ColumnTypes"]}
    for column in description["Columns"]:
        kind = column_types[column["typeID"]]
        angle = np.radians(column["rotation"])
        d = np.array([np.cos(angle), np.sin(angle)])
        n = np.array([-d[1], d[0]])
        centre = np.array(column["location"])
        ax.add_patch(PolygonPatch(np.array([centre + sx * d * kind["width"] / 2 + sy * n * kind["depth"] / 2 for sx, sy in
                                            ((-1, -1), (1, -1), (1, 1), (-1, 1))]), fc="#d62728"))
    for name, colour in (("wall", "#333333"), ("glass wall", "#1f9ed1"), ("partition", "#2ca02c"), ("door", "#ff7f0e"), ("window", "#e377c2"), ("column", "#d62728")):
        ax.plot([], [], c=colour, lw=4, label=name)
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_title("Model plan")
    fig.tight_layout()
    fig.savefig(ctx.output / "plan_model.png", dpi=70)
    plt.close(fig)


def run(ctx: ModelContext) -> None:
    elements, floor_z, height = clean_elements(ctx)
    rooms = ctx.load("rooms")
    polygons = [shape(r["poly"]) for r in rooms["rooms"]]
    walls, wall_types, wall_items = describe_walls(elements, polygons)
    opening_types, doors, windows = describe_openings(ctx.load("openings"), rooms["slabs"], walls)
    column_types, columns = describe_columns(elements)
    floors, ceilings, spaces = describe_surfaces(rooms["rooms"], walls, floor_z)
    description = dict(Building=dict(id="B01", name=ctx.project_name),
                       Storey=dict(id="L00", name="Level 0", elevation=r3(floor_z), height=r3(height)),
                       WallTypes=wall_types, Walls=wall_items, DoorTypes=opening_types["door"], Doors=doors,
                       WindowTypes=opening_types["window"], Windows=windows,
                       FloorTypes=[dict(id="SL01", thickness=FLOOR_THICKNESS, material="Floor slab (assumed)")], Floors=floors,
                       CeilingTypes=[dict(id="CT01", thickness=CEILING_THICKNESS, material="Suspended ceiling (assumed)")], Ceilings=ceilings,
                       ColumnTypes=column_types, Columns=columns, Spaces=spaces)
    with open(ctx.output / "components.json", "w", encoding="utf-8") as handle:
        json.dump(description, handle, indent=1)
    write_gml(ctx, description)
    write_element_list(ctx, description)
    plot(ctx, description)
    ctx.log(f"components: {len(wall_items)} walls of {len(wall_types)} types, {len(doors)} doors, {len(windows)} windows, "
            f"{len(columns)} columns, {len(floors)} floors, {len(ceilings)} ceilings, {len(spaces)} spaces")
