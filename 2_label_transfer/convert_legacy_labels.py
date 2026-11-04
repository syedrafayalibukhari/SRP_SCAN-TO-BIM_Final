"""Convert a floor labelled with the earlier twelve-class scheme to the current classes.

Desk, chair, monitor, cabinet_shelf and clutter points become "other". The input file is only read.

Usage:  python convert_legacy_labels.py floor_labelled.ply floor_labelled_elements.ply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import numpy as np

from scan2bim.classes import CLASSES, UNKNOWN, legacy_lookup
from scan2bim.plyio import read_ply, write_ply


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert twelve-class labels to the current classes.")
    parser.add_argument("source", type=Path, help="labelled PLY with the twelve-class scheme")
    parser.add_argument("target", type=Path, help="new PLY file to write")
    args = parser.parse_args()
    if args.source.resolve() == args.target.resolve():
        raise SystemExit("The target must be a new file; the source file is never changed.")
    data = read_ply(args.source, required=("x", "y", "z", "label"))
    values = np.unique(data["label"])
    if values[values != UNKNOWN].max(initial=0) < 8:
        raise SystemExit(f"{args.source.name} has no label above 7; it does not look like a twelve-class file.")
    data = data.copy()
    data["label"] = legacy_lookup()[data["label"]]
    write_ply(args.target, data, [f"classes: {' '.join(CLASSES)}", f"converted from {args.source.name}"])
    counts = np.bincount(data["label"], minlength=256)
    print(f"{len(data):,} points written to {args.target}")
    for value, name in enumerate(CLASSES):
        print(f"  {name:12s}{counts[value]:>12,}")
    print(f"  {'unknown':12s}{counts[UNKNOWN]:>12,}")


if __name__ == "__main__":
    main()
