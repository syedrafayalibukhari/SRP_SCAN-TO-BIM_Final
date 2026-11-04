"""Split window points into windows and curtain walls without manual labelling.

A first modelling run finds the glazed runs in the window points and measures their height. A run that
reaches the floor is a curtain wall; a run with a sill stays a window. Every window point takes the class
of the nearest run. The input file is only read.

Usage:  python split_glass_labels.py floor_merged.ply components.json floor_split.ply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import numpy as np

from scan2bim.classes import CLASSES, CURTAIN_WALL, WINDOW
from scan2bim.plyio import read_ply, write_ply

MAX_DISTANCE_M = 1.0            # window points further than this from every run stay window


def distance_to_segment(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    axis = end - start
    position = np.clip((points - start) @ axis / (axis @ axis), 0, 1)
    return np.linalg.norm(points - (start + position[:, None] * axis), axis=1)


def split(xy: np.ndarray, runs: list) -> np.ndarray:
    """Return True for the points in ``xy`` that belong to a curtain wall run."""
    distances = np.column_stack([distance_to_segment(xy, np.array(r["start"]), np.array(r["end"])) for r in runs])
    nearest = distances.argmin(1)
    reaches_floor = np.array([r["baseOffset"] == 0 for r in runs])
    return reaches_floor[nearest] & (distances.min(1) <= MAX_DISTANCE_M)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split window points into windows and curtain walls.")
    parser.add_argument("source", type=Path, help="labelled floor")
    parser.add_argument("components", type=Path, help="components.json of a modelling run on that floor")
    parser.add_argument("target", type=Path, help="new PLY file to write")
    args = parser.parse_args()
    if args.source.resolve() == args.target.resolve():
        raise SystemExit("The target must be a new file; the source file is never changed.")
    data = read_ply(args.source, required=("x", "y", "z", "label")).copy()
    description = json.loads(args.components.read_text(encoding="utf-8"))
    functions = {t["id"]: t["function"] for t in description["WallTypes"]}
    runs = [w for w in description["Walls"] if functions[w["typeID"]] == "window_run"]
    if not runs:
        raise SystemExit("No window runs in the component description. Run step 5 on the labelled floor first.")
    window = np.nonzero(data["label"] == WINDOW)[0]
    curtain = split(np.c_[data["x"][window], data["y"][window]].astype(float), runs)
    data["label"][window[curtain]] = CURTAIN_WALL
    write_ply(args.target, data, [f"classes: {' '.join(CLASSES)}", f"glass split from {args.source.name}"])
    print(f"{len(runs)} runs: {sum(r['baseOffset'] == 0 for r in runs)} curtain walls, {sum(r['baseOffset'] > 0 for r in runs)} windows")
    print(f"{int(curtain.sum()):,} points set to curtain wall, {int((~curtain).sum()):,} stay window -> {args.target}")


if __name__ == "__main__":
    main()
