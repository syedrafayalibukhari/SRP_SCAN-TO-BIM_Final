"""Step 4: merge manually corrected tiles into one labelled floor.

Joins all files saved by the Point Label Corrector. The corrected files are only read (checksums are
compared before and after) and the labels are copied unchanged.

Usage:  python merge_tiles.py                         (corrected files in the folder 'corrected')
        python merge_tiles.py --input-folder D:/my_tiles
Output: merged/floor_merged.ply  (x y z red green blue label conf idx tile)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from scan2bim.setup_env import ensure_packages

if __name__ == "__main__":
    ensure_packages({"numpy": "numpy"})

import numpy as np

from scan2bim.classes import CLASSES, UNKNOWN, invalid_labels
from scan2bim.plyio import read_ply, write_ply

REQUIRED = ("x", "y", "z", "red", "green", "blue", "label")
OUTPUT_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                         ("label", "u1"), ("conf", "<f4"), ("idx", "<i4"), ("tile", "u1")])


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge manually corrected tiles into one file.")
    parser.add_argument("--input-folder", type=Path, default=ROOT / "corrected", help="folder with the corrected .ply files")
    parser.add_argument("--output-folder", type=Path, default=ROOT / "merged", help="folder for the result")
    args = parser.parse_args()
    if not args.input_folder.is_dir():
        raise SystemExit(f"Folder not found: {args.input_folder}\nCreate it and put the corrected files in it, "
                         "or name the folder with --input-folder.")
    files = sorted(args.input_folder.glob("*.ply"))
    if not files:
        raise SystemExit(f"No .ply files in {args.input_folder}.")
    if len(files) > 255:
        raise SystemExit("More than 255 files; use larger tiles in label_transfer.py.")
    args.output_folder.mkdir(parents=True, exist_ok=True)

    before = {path: md5(path) for path in files}
    parts = []
    for number, path in enumerate(files):
        data = read_ply(path, required=REQUIRED)
        wrong = invalid_labels(data["label"])
        if len(wrong):
            raise SystemExit(f"{path.name}: label values {wrong.tolist()} are not part of the classes {', '.join(CLASSES)}.")
        part = np.zeros(len(data), OUTPUT_DTYPE)
        for name in REQUIRED:
            part[name] = data[name]
        part["conf"] = data["conf"] if "conf" in data.dtype.names else 1.0
        part["idx"] = data["idx"] if "idx" in data.dtype.names else -1
        part["tile"] = number
        parts.append(part)
        print(f"[{number:2d}] {path.name}: {len(data):,} points, unknown {100 * (data['label'] == UNKNOWN).mean():.1f}%")
    merged = np.concatenate(parts)

    positions = np.ascontiguousarray(np.c_[merged["x"], merged["y"], merged["z"]]).view([("", "<f4", 3)]).ravel()
    duplicates = len(merged) - len(np.unique(positions))
    print(f"\npoints that appear more than once (same position): {duplicates:,}"
          + ("" if duplicates == 0 else "  <- check that no tile was copied twice"))

    target = args.output_folder / "floor_merged.ply"
    write_ply(target, merged, ["tiles: " + " ".join(f"{i}={p.stem}" for i, p in enumerate(files)),
                               "labels: " + " ".join(f"{i}={c}" for i, c in enumerate(CLASSES)) + f" {UNKNOWN}=unknown"])

    if before != {path: md5(path) for path in files}:
        raise SystemExit("A corrected file changed while merging.")
    if not np.array_equal(np.concatenate([read_ply(path)["label"] for path in files]), merged["label"]):
        raise SystemExit("The merged labels differ from the corrected files.")
    counts = np.bincount(merged["label"], minlength=256)
    print(f"merged: {len(merged):,} points from {len(files)} files -> {target}")
    print("labels copied unchanged: yes; corrected files unchanged (MD5 before = after): yes")
    print("class shares: " + ", ".join(f"{name} {100 * counts[i] / len(merged):.1f}%" for i, name in enumerate(CLASSES))
          + f", unknown {100 * counts[UNKNOWN] / len(merged):.1f}%")


if __name__ == "__main__":
    main()
