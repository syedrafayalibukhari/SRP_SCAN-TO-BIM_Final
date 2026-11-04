"""Unit tests of the shared library and of the steps that need no large data."""
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scan2bim.classes import CLASSES, COLOURS, OTHER, PARTITION, UNKNOWN, invalid_labels, legacy_lookup
from scan2bim.modelling import components, junctions
from scan2bim.plyio import PlyStreamWriter, read_ply, write_ply

REPO = Path(__file__).resolve().parents[1]


def load_script(relative: str):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_corrector_tool_uses_the_same_classes():
    html = (REPO / "3_label_correction" / "point_label_corrector.html").read_text(encoding="utf-8")
    classes = json.loads(re.search(r"const CLASSES = (\[.*?\]);", html).group(1))
    colours = json.loads(re.search(r"const COLORS = (\[.*?\]);", html).group(1))
    values = json.loads(re.search(r"const LABEL_VALUE = (\[.*?\]);", html).group(1))
    assert tuple(classes[:-1]) == CLASSES
    assert tuple(colours[:-1]) == COLOURS
    assert values == list(range(len(CLASSES))) + [UNKNOWN]


def test_legacy_labels():
    table = legacy_lookup()
    assert table[[0, 1, 2, 3, 4, 5]].tolist() == [0, 1, 2, 3, 4, 5]
    assert table[9] == PARTITION
    assert table[[6, 7, 8, 10, 11]].tolist() == [OTHER] * 5
    assert table[255] == UNKNOWN


def test_invalid_labels():
    assert invalid_labels(np.array([0, 7, 255], np.uint8)).tolist() == []
    assert invalid_labels(np.array([0, 9, 255], np.uint8)).tolist() == [9]


def test_ply_round_trip(tmp_path):
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("label", "u1"), ("idx", "<i4")])
    data = np.zeros(5, dtype)
    data["x"] = np.arange(5)
    data["label"] = 3
    write_ply(tmp_path / "a.ply", data, ["a comment"])
    assert np.array_equal(read_ply(tmp_path / "a.ply"), data)


def test_ply_rejects_incomplete_file(tmp_path):
    data = np.zeros(5, np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")]))
    path = tmp_path / "a.ply"
    write_ply(path, data)
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(ValueError, match="incomplete"):
        read_ply(path)


def test_ply_stream_writer_checks_count(tmp_path):
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    writer = PlyStreamWriter(tmp_path / "a.ply", dtype, 3)
    writer.write(np.zeros(2, dtype))
    with pytest.raises(RuntimeError):
        writer.close()


def test_voxel_grid_matches_one_pass_average():
    step1 = load_script("1_preprocessing/preprocess_pointcloud.py")
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 1, (20000, 3)).astype(np.float32)
    colours = rng.integers(0, 256, (20000, 3)).astype(np.uint8)
    origin = np.zeros(3)
    in_parts = step1.VoxelGrid(origin, np.ones(3), 0.1)
    for chunk in np.array_split(np.arange(20000), 7):
        in_parts.add(points[chunk], colours[chunk])
    at_once = step1.VoxelGrid(origin, np.ones(3), 0.1)
    at_once.add(points, colours)
    assert np.array_equal(in_parts.keys, at_once.keys)
    assert np.allclose(in_parts.sums, at_once.sums)
    a, b = in_parts.to_points(), at_once.to_points()
    assert np.allclose(a["x"], b["x"]) and np.array_equal(a["red"], b["red"])


def test_clean_records_removes_bad_points():
    step1 = load_script("1_preprocessing/preprocess_pointcloud.py")
    points = np.array([[1, 2, 3], [1, 2, 3], [0, 0, 0], [np.nan, 1, 1], [4, 5, 6]], np.float32)
    kept, _, invalid, duplicates = step1.clean_records(points, None)
    assert kept.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert (invalid, duplicates) == (2, 1)


def test_e57_to_preprocessed(tmp_path):
    pye57 = pytest.importorskip("pye57")
    rng = np.random.default_rng(0)
    path = tmp_path / "scan.e57"
    e57 = pye57.E57(str(path), mode="w")
    for shift in (0.0, 1.0):
        count = 20000
        e57.write_scan_raw({"cartesianX": rng.uniform(0, 5, count) + shift, "cartesianY": rng.uniform(0, 5, count), "cartesianZ": rng.uniform(0, 2, count),
                            "colorRed": rng.integers(0, 256, count), "colorGreen": rng.integers(0, 256, count), "colorBlue": rng.integers(0, 256, count)})
    e57.close()
    out = tmp_path / "out"
    subprocess.check_call([sys.executable, str(REPO / "1_preprocessing" / "preprocess_pointcloud.py"), "--input", str(path),
                           "--output-folder", str(out), "--voxel-size", "0.1"])
    cloud = read_ply(out / "preprocessed.ply", required=("x", "y", "z", "red", "green", "blue"))
    assert 0 < len(cloud) <= len(read_ply(out / "original.ply"))
    assert len(json.loads((out / "scan_positions.json").read_text())["stations"]) == 2
    assert "Voxel downsampling" in (out / "preprocessing_report.txt").read_text()


def test_merge_keeps_labels(tmp_path):
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("label", "u1")])
    folder = tmp_path / "corrected"
    folder.mkdir()
    for number, label in enumerate((2, 6)):
        data = np.zeros(10, dtype)
        data["x"] = np.arange(10) + 100 * number
        data["label"] = label
        write_ply(folder / f"tile_{number}_corrected.ply", data)
    subprocess.check_call([sys.executable, str(REPO / "4_merge_tiles" / "merge_tiles.py"), "--input-folder", str(folder), "--output-folder", str(tmp_path / "merged")])
    merged = read_ply(tmp_path / "merged" / "floor_merged.ply")
    assert merged["label"].tolist() == [2] * 10 + [6] * 10


def test_merge_rejects_old_labels(tmp_path):
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("label", "u1")])
    folder = tmp_path / "corrected"
    folder.mkdir()
    data = np.zeros(4, dtype)
    data["label"] = 9
    write_ply(folder / "tile.ply", data)
    result = subprocess.run([sys.executable, str(REPO / "4_merge_tiles" / "merge_tiles.py"), "--input-folder", str(folder), "--output-folder", str(tmp_path / "m")],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "not part of the classes" in result.stderr


def test_open_door_goes_into_its_doorway():
    wall = dict(element("wall", (1, 0), 0.0, 1.0, 5.0), function="interior")
    leaf = dict(c=[0.0, 0.5], d=[0.0, 1.0], width=0.9)                 # hinge at (0, 0.05), standing open
    door = components.doorway(dict(lines=[leaf], z0=0.0, height=2.1), [wall])
    assert door["d"] == pytest.approx([1.0, 0.0])
    assert door["c"][1] == pytest.approx(0.05)


def test_two_lines_are_one_door_with_the_width_of_the_leaf():
    wall = dict(element("wall", (1, 0), 0.0, 1.5, 5.0), function="interior")
    in_plane = dict(c=[0.7, 0.0], d=[1.0, 0.0], width=1.4)
    leaf = dict(c=[0.0, -0.45], d=[0.0, 1.0], width=0.9)
    door = components.doorway(dict(lines=[in_plane, leaf], z0=0.0, height=2.1), [wall])
    assert door["width"] == pytest.approx(0.9)
    assert door["c"] == pytest.approx([0.45, 0.0])


def element(kind, d, o, t0, t1, thickness=0.2):
    d = np.array(d, float)
    return dict(kind=kind, src=(kind, o), d=d, n=np.array([-d[1], d[0]]), o=o, T=thickness, t0=t0, t1=t1)


def test_l_corner_closes_without_overshoot():
    a = element("wall", (1, 0), 0.0, 0.0, 4.9)
    b = element("wall", (0, 1), -5.0, 0.2, 3.0)         # centre line at x = 5
    junctions.join([a, b], {"wall": 1}, lambda message: None)
    assert a["t1"] == pytest.approx(5.1)                # longer wall runs to the outer face
    assert b["t0"] == pytest.approx(0.1)                # shorter wall stops at its inner face


def test_t_junction_stops_at_near_face():
    through = element("wall", (1, 0), 0.0, -3.0, 3.0)
    branch = element("partition", (0, 1), 0.0, 0.2, 2.0, 0.05)
    junctions.join([through, branch], {"wall": 1, "partition": 4}, lambda message: None)
    assert branch["t0"] == pytest.approx(0.1)
