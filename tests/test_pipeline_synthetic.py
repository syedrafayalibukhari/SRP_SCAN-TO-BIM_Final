"""End-to-end test of the modelling stages on a small generated floor.

Two rooms at 100 m elevation, separated by a wall with a door, with a column, an L-shaped partition and a
glass wall. The high elevation proves that nothing depends on the floor being near z = 0.
"""
import json
import subprocess
import sys
from pathlib import Path

import ifcopenshell
import numpy as np
import pytest

from scan2bim.classes import CEILING, COLUMN, DOOR, FLOOR, OTHER, PARTITION, WALL, WINDOW
from scan2bim.modelling import columns_partitions, components, ifc_writer, openings, prepare, rooms, structure
from scan2bim.modelling.context import ModelContext
from scan2bim.plyio import write_ply

REPO = Path(__file__).resolve().parents[1]
SPACING = 0.025
Z0, HEIGHT = 100.0, 2.7
DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("label", "u1")])


def grid(a, b):
    return np.arange(a, b, SPACING)


def plane_x(x, y0, y1, z0, z1, label, keep=None):
    y, z = np.meshgrid(grid(y0, y1), grid(z0, z1))
    y, z = y.ravel(), z.ravel()
    mask = np.ones(len(y), bool) if keep is None else keep(y, z)
    return np.c_[np.full(mask.sum(), x), y[mask], z[mask], np.full(mask.sum(), label)]


def plane_y(y, x0, x1, z0, z1, label):
    x, z = np.meshgrid(grid(x0, x1), grid(z0, z1))
    return np.c_[x.ravel(), np.full(x.size, y), z.ravel(), np.full(x.size, label)]


def plane_z(z, x0, x1, y0, y1, label):
    x, y = np.meshgrid(grid(x0, x1), grid(y0, y1))
    return np.c_[x.ravel(), y.ravel(), np.full(x.size, z), np.full(x.size, label)]


def generated_floor() -> np.ndarray:
    top = Z0 + HEIGHT
    outside_door = lambda y, z: ~((y > 3.0) & (y < 4.0) & (z < Z0 + 2.1))
    parts = [plane_z(Z0, 0, 10, 0, 8, FLOOR), plane_z(top, 0, 10, 0, 8, CEILING),
             plane_x(0, 0, 8, Z0, top, WALL), plane_x(10, 0, 8, Z0, top, WALL), plane_y(0, 0, 10, Z0, top, WALL),
             plane_y(8, 0, 5, Z0, top, WALL), plane_y(8, 5, 10, Z0, top, WINDOW),
             plane_x(5.0, 0, 8, Z0, top, WALL, outside_door), plane_x(5.15, 0, 8, Z0, top, WALL, outside_door)]
    for x in (5.03, 5.08):                                             # closed door leaf
        parts.append(plane_x(x, 3.05, 3.95, Z0, Z0 + 2.1, DOOR))
    for x in (2.25, 2.75):                                             # column
        parts.append(plane_x(x, 3.75, 4.25, Z0, top, COLUMN))
    for y in (3.75, 4.25):
        parts.append(plane_y(y, 2.25, 2.75, Z0, top, COLUMN))
    for x in (7.0, 7.05):                                              # L-shaped partition
        parts.append(plane_x(x, 2.0, 5.0, Z0, Z0 + 1.5, PARTITION))
    for y in (5.0, 5.05):
        parts.append(plane_y(y, 7.05, 9.0, Z0, Z0 + 1.5, PARTITION))
    parts.append(plane_z(Z0 + 0.75, 1.0, 1.8, 1.0, 2.5, OTHER))        # a desk top
    table = np.vstack(parts)
    floor = np.empty(len(table), DTYPE)
    floor["x"], floor["y"], floor["z"], floor["label"] = table[:, 0], table[:, 1], table[:, 2], table[:, 3]
    return floor


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    folder = tmp_path_factory.mktemp("synthetic")
    write_ply(folder / "floor.ply", generated_floor())
    stations = [dict(scan=i, xyz=[x, y, Z0 + 1.6]) for i, (x, y) in enumerate(((1.5, 1.5), (3.5, 6.5), (3.5, 1.5), (6.2, 1.5), (8.5, 6.5), (8.5, 1.5), (6.2, 6.5)))]
    (folder / "positions.json").write_text(json.dumps(dict(stations=stations)))
    ctx = ModelContext(folder / "output", "Synthetic floor")
    prepare.run(ctx, folder / "floor.ply", folder / "positions.json")
    for stage in (structure, rooms, openings, columns_partitions, components, ifc_writer):
        stage.run(ctx)
    return ctx, ifcopenshell.open(str(folder / "output" / "model.ifc")), folder


def test_levels(result):
    ctx, _, _ = result
    assert ctx.load("structure")["floor_z"] == pytest.approx(Z0, abs=0.02)


def test_element_types_present(result):
    _, model, _ = result
    for ifc_class in ("IfcSlab", "IfcCovering", "IfcCurtainWall", "IfcDoor", "IfcColumn"):
        assert model.by_type(ifc_class), ifc_class
    predefined = {wall.PredefinedType for wall in model.by_type("IfcWall")}
    assert predefined == {"STANDARD", "PARTITIONING"}


def test_only_the_requested_elements(result):
    _, model, _ = result
    for ifc_class in ("IfcFurniture", "IfcFurnishingElement"):
        assert not model.by_type(ifc_class), ifc_class


def test_spaces_exported(result):
    _, model, _ = result
    spaces = model.by_type("IfcSpace")
    assert len(spaces) == 2
    assert all(space.Decomposes for space in spaces)


def test_closed_door_is_cut_into_its_wall(result):
    _, model, _ = result
    door = model.by_type("IfcDoor")[0]
    assert door.FillsVoids
    assert door.FillsVoids[0].RelatingOpeningElement.VoidsElements[0].RelatingBuildingElement.is_a("IfcWall")


def test_column_size(result):
    ctx, _, _ = result
    columns = ctx.load("columns_partitions")["columns"]
    assert len(columns) == 1
    assert columns[0]["a"] == pytest.approx(0.5, abs=0.05)
    assert columns[0]["c"] == pytest.approx([2.5, 4.0], abs=0.05)


def test_partition_height(result):
    ctx, _, _ = result
    panels = ctx.load("columns_partitions")["partitions"]
    assert panels and all(panel["H"] == pytest.approx(1.5, abs=0.06) for panel in panels)


def test_door_position(result):
    ctx, _, _ = result
    doors = ctx.load("openings")["doors"]
    assert len(doors) == 1
    assert doors[0]["lines"][0]["width"] == pytest.approx(0.9, abs=0.1)
    assert doors[0]["height"] == pytest.approx(2.1, abs=0.15)


def test_two_rooms_and_floor(result):
    ctx, _, _ = result
    areas = sorted(room["area"] for room in ctx.load("rooms")["rooms"])
    assert len(areas) == 2
    assert sum(areas) == pytest.approx(80, rel=0.05)


def test_model_check_passes(result):
    _, _, folder = result
    subprocess.check_call([sys.executable, str(REPO / "6_model_check" / "check_model.py"), "--model", str(folder / "output" / "model.ifc"),
                           "--input", str(folder / "floor.ply")])
