"""Step 1: clean and voxel-downsample a laser scan (E57).

Steps: (a) remove invalid, scanner-origin and duplicate records, (b) remove points far outside the
building, (c) statistical outlier removal, (d) voxel downsampling. Writes original.ply, preprocessed.ply,
scan_positions.json (scanner positions from the E57 header) and a report.

Usage:  python preprocess_pointcloud.py                      (uses VOXEL_SIZE_M below)
        python preprocess_pointcloud.py --voxel-size 0.03    (overrides it for this run)
Put the .e57 file in this folder, or name it with --input.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from scan2bim.setup_env import ensure_packages

if __name__ == "__main__":
    ensure_packages({"numpy": "numpy", "scipy": "scipy", "open3d": "open3d", "pye57": "pye57"})

import numpy as np
import open3d as o3d
import pye57
from scipy.stats import iqr

from scan2bim.plyio import PlyStreamWriter, write_ply

# ============================================================================
# VOXEL SIZE (metres) - SET THIS NUMBER BEFORE RUNNING
#   0.02 = 2 cm cubes (recommended for labelling and modelling)
#   0.03 = 3 cm cubes (smaller files)
# A value given with --voxel-size overrides this one.
VOXEL_SIZE_M = 0.02
# ============================================================================

SAMPLE_PER_SCAN = 5_000
RANGE_IQR_MULTIPLIER = 3.0
SAMPLE_SEED = 20260609
# Indoor setting of Ebrahimi & Czarnuch (2021): 4 neighbours, one standard deviation.
SOR_NEIGHBOURS = 4
SOR_STD_RATIO = 1.0

POINT_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])


@dataclass
class StepStat:
    name: str
    before: int
    after: int
    settings: str

    @property
    def summary(self) -> str:
        removed = self.before - self.after
        share = 100.0 * removed / self.before if self.before else 0.0
        return f"{self.before:,} -> {self.after:,} points, removed {removed:,} ({share:.1f}%)"


class VoxelGrid:
    """Running sum of position and colour per occupied voxel, kept in sorted numpy arrays."""

    def __init__(self, origin: np.ndarray, size: np.ndarray, voxel: float) -> None:
        self.origin, self.voxel = origin, voxel
        dims = np.ceil(size / voxel).astype(np.int64) + 3
        self.stride_z, self.stride_yz = int(dims[2]), int(dims[1] * dims[2])
        self.keys = np.empty(0, np.int64)
        self.sums = np.empty((0, 6), np.float64)
        self.counts = np.empty(0, np.int64)

    def __len__(self) -> int:
        return len(self.keys)

    def add(self, points: np.ndarray, colours: np.ndarray) -> None:
        if len(points) == 0:
            return
        grid = np.floor((points.astype(np.float64) - self.origin) / self.voxel).astype(np.int64)
        keys = grid[:, 0] * self.stride_yz + grid[:, 1] * self.stride_z + grid[:, 2]
        new_keys, inverse, new_counts = np.unique(keys, return_inverse=True, return_counts=True)
        inverse = inverse.ravel()
        values = np.c_[points, colours].astype(np.float64)
        new_sums = np.column_stack([np.bincount(inverse, weights=values[:, c], minlength=len(new_keys)) for c in range(6)])
        merged = np.union1d(self.keys, new_keys)
        sums = np.zeros((len(merged), 6))
        counts = np.zeros(len(merged), np.int64)
        for part_keys, part_sums, part_counts in ((self.keys, self.sums, self.counts), (new_keys, new_sums, new_counts)):
            where = np.searchsorted(merged, part_keys)
            sums[where] += part_sums
            counts[where] += part_counts
        self.keys, self.sums, self.counts = merged, sums, counts

    def to_points(self) -> np.ndarray:
        means = self.sums / self.counts[:, None]
        rgb = np.clip(np.rint(means[:, 3:]), 0, 255).astype(np.uint8)
        return as_records(means[:, :3], rgb)


def as_records(points: np.ndarray, colours: np.ndarray) -> np.ndarray:
    out = np.empty(len(points), POINT_DTYPE)
    out["x"], out["y"], out["z"] = points[:, 0], points[:, 1], points[:, 2]
    out["red"], out["green"], out["blue"] = colours[:, 0], colours[:, 1], colours[:, 2]
    return out


def find_input_e57(given: Optional[Path]) -> Path:
    if given is not None:
        path = given if given.is_absolute() else Path.cwd() / given
        if not path.exists():
            raise SystemExit(f"E57 file not found: {path}")
        return path
    found = sorted(ROOT.glob("*.e57"))
    if len(found) == 1:
        return found[0]
    if found:
        raise SystemExit(f"Several .e57 files in this folder ({', '.join(p.name for p in found)}). "
                         "Choose one with --input <file.e57>.")
    raise SystemExit("No .e57 file found. Put the scan in this folder, or give it with --input <file.e57>.")


def write_scan_positions(e57: "pye57.E57", e57_path: Path, out_path: Path) -> None:
    stations = []
    for index in range(e57.scan_count):
        header = e57.get_header(index)
        stations.append({"scan": index, "name": f"scan{index}", "xyz": [float(v) for v in header.translation],
                         "quat_wxyz": [float(v) for v in header.rotation], "points": int(header.point_count)})
    out_path.write_text(json.dumps({"source": e57_path.name, "frame": "same as original.ply and preprocessed.ply",
                                    "stations": stations}, indent=1), encoding="utf-8")


def read_scan(e57: "pye57.E57", index: int, with_colours: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    data = e57.read_scan(index, colors=with_colours, row_column=False, transform=True, ignore_missing_fields=True)
    points = np.column_stack([data[f"cartesian{axis}"].astype(np.float32, copy=False) for axis in "XYZ"])
    if not with_colours:
        return points, None
    if all(k in data for k in ("colorRed", "colorGreen", "colorBlue")):
        colours = np.column_stack([data[k].astype(np.uint8, copy=False) for k in ("colorRed", "colorGreen", "colorBlue")])
    else:
        colours = np.full((len(points), 3), 255, np.uint8)
    return points, colours


def clean_records(points: np.ndarray, colours: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray], int, int]:
    """Remove non-finite points, points at the origin and exact duplicates within one scan."""
    valid = np.all(np.isfinite(points), axis=1) & ~np.all(points == 0.0, axis=1)
    invalid = int(len(points) - valid.sum())
    points = points[valid]
    colours = None if colours is None else colours[valid]
    if len(points) == 0:
        return points, colours, invalid, 0
    packed = np.ascontiguousarray(points).view(np.dtype((np.void, points.dtype.itemsize * 3))).ravel()
    first = np.sort(np.unique(packed, return_index=True)[1])
    duplicates = len(points) - len(first)
    return points[first], None if colours is None else colours[first], invalid, duplicates


def range_fences(sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Median +/- RANGE_IQR_MULTIPLIER x IQR per axis."""
    sample = sample.astype(np.float64)
    median = np.median(sample, axis=0)
    spread = RANGE_IQR_MULTIPLIER * iqr(sample, axis=0)
    return median - spread, median + spread


def remove_outliers(points: np.ndarray, colours: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points, colours
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    _, kept = cloud.remove_statistical_outlier(nb_neighbors=SOR_NEIGHBOURS, std_ratio=SOR_STD_RATIO)
    kept = np.asarray(kept, np.int64)
    return points[kept], colours[kept]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and voxel-downsample an E57 laser scan.")
    parser.add_argument("--input", type=Path, help="E57 scan; default: the single .e57 file in this folder")
    parser.add_argument("--voxel-size", type=float, default=VOXEL_SIZE_M, help="voxel size in metres (default: VOXEL_SIZE_M)")
    parser.add_argument("--output-folder", type=Path, default=ROOT, help="folder for the results (default: this folder)")
    parser.add_argument("--skip-original", action="store_true", help="do not write original.ply (it is large)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.voxel_size <= 0:
        raise SystemExit("The voxel size must be a positive number of metres, e.g. 0.02.")
    out = args.output_folder if args.output_folder.is_absolute() else Path.cwd() / args.output_folder
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()

    e57_path = find_input_e57(args.input)
    e57 = pye57.E57(str(e57_path))
    scans = e57.scan_count
    raw_records = sum(int(e57.get_header(i).point_count) for i in range(scans))
    print(f"Input: {e57_path}\nScans: {scans}\nRaw E57 records: {raw_records:,}", flush=True)
    write_scan_positions(e57, e57_path, out / "scan_positions.json")

    print("\nPass 1: count clean points and estimate the range fences", flush=True)
    rng = np.random.default_rng(SAMPLE_SEED)
    samples: List[np.ndarray] = []
    clean_count = invalid_count = duplicate_count = 0
    for index in range(scans):
        points, _ = read_scan(e57, index, with_colours=False)
        points, _, invalid, duplicates = clean_records(points, None)
        invalid_count += invalid
        duplicate_count += duplicates
        clean_count += len(points)
        if len(points):
            samples.append(points[rng.choice(len(points), min(SAMPLE_PER_SCAN, len(points)), replace=False)])
        if (index + 1) % 10 == 0 or index + 1 == scans:
            print(f"  scan {index + 1}/{scans}: {clean_count:,} clean points", flush=True)
    low, high = range_fences(np.vstack(samples))

    print("\nPass 2: filter and downsample", flush=True)
    original = None if args.skip_original else PlyStreamWriter(out / "original.ply", POINT_DTYPE, clean_count)
    grid = VoxelGrid(low, high - low, args.voxel_size)
    range_count = sor_count = 0
    for index in range(scans):
        points, colours = read_scan(e57, index, with_colours=True)
        points, colours, _, _ = clean_records(points, colours)
        if original is not None:
            original.write(as_records(points, colours))
        inside = np.all((points >= low) & (points <= high), axis=1)
        points, colours = points[inside], colours[inside]
        range_count += len(points)
        points, colours = remove_outliers(points, colours)
        sor_count += len(points)
        grid.add(points, colours)
        if (index + 1) % 5 == 0 or index + 1 == scans:
            print(f"  scan {index + 1}/{scans}: range {range_count:,}, outlier filter {sor_count:,}, voxels {len(grid):,}", flush=True)
    if original is not None:
        original.close()

    result = grid.to_points()
    write_ply(out / "preprocessed.ply", result, [f"voxel size {args.voxel_size} m"])

    steps = [
        StepStat("a. Invalid record removal", raw_records, clean_count,
                 f"invalid-state records, NaN/Inf and (0,0,0) points ({invalid_count:,} numeric), {duplicate_count:,} exact duplicates per scan"),
        StepStat("b. Range filter", clean_count, range_count,
                 f"median +/- {RANGE_IQR_MULTIPLIER} x IQR per axis; X [{low[0]:.3f}, {high[0]:.3f}], "
                 f"Y [{low[1]:.3f}, {high[1]:.3f}], Z [{low[2]:.3f}, {high[2]:.3f}]"),
        StepStat("c. Statistical outlier removal", range_count, sor_count,
                 f"{SOR_NEIGHBOURS} neighbours, {SOR_STD_RATIO} standard deviation, per scan"),
        StepStat("d. Voxel downsampling", sor_count, len(result), f"{args.voxel_size} m; mean position and colour per voxel"),
    ]
    elapsed = (time.time() - start) / 60
    lines = ["POINT CLOUD PREPROCESSING REPORT", f"Date: {datetime.now():%Y-%m-%d %H:%M}", f"Input: {e57_path}", f"Scans: {scans}",
             f"Elapsed: {elapsed:.1f} min", ""]
    for step in steps:
        print(f"{step.name}: {step.summary}")
        lines += [step.name, f"  {step.summary}", f"  Settings: {step.settings}"]
    (out / "preprocessing_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nResults written to {out} in {elapsed:.1f} min.")


if __name__ == "__main__":
    main()
