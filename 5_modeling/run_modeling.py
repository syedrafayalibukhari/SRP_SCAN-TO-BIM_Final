"""Step 5: generate the IFC model from a labelled floor and the scanner positions.

Usage:  python run_modeling.py --project-name "Building A, level 3"
        python run_modeling.py --input floor_merged.ply --positions scan_positions.json
        python run_modeling.py --start-at openings              (repeat a run from one stage onwards)

Without arguments the inputs are taken from the output folders of steps 4 and 1.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from scan2bim.setup_env import ensure_packages

ensure_packages({"numpy": "numpy", "scipy": "scipy", "open3d": "open3d", "shapely": "shapely", "networkx": "networkx",
                 "skimage": "scikit-image", "sklearn": "scikit-learn", "matplotlib": "matplotlib", "ifcopenshell": "ifcopenshell"})
os.environ.setdefault("MPLBACKEND", "Agg")

from scan2bim.modelling import columns_partitions, components, ifc_writer, openings, prepare, rooms, structure
from scan2bim.modelling.context import ModelContext

STAGES = {"prepare": "Normals for every point", "structure": "Levels, wall and glass-wall faces", "rooms": "Room footprints",
          "openings": "Doors and windows", "columns_partitions": "Columns and partitions",
          "components": "Element clean-up and parametric description", "ifc": "IFC model"}


def locate(given: Optional[Path], default: Path) -> Path:
    path = given or default
    path = path if path.is_absolute() else Path.cwd() / path
    if not path.exists():
        raise SystemExit(f"File not found: {path}\nRun the earlier steps first or give the file (see --help).")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an IFC model from a labelled point cloud.")
    parser.add_argument("--input", type=Path, help="labelled point cloud (default: ../4_merge_tiles/merged/floor_merged.ply)")
    parser.add_argument("--positions", type=Path, help="scanner positions (default: ../1_preprocessing/scan_positions.json)")
    parser.add_argument("--output-folder", type=Path, default=ROOT / "output", help="folder for the results (default: output)")
    parser.add_argument("--project-name", default="Scanned floor", help="project and building name in the IFC file")
    parser.add_argument("--start-at", choices=list(STAGES), default="prepare", help="first stage to run; earlier results are reused")
    args = parser.parse_args()

    output = args.output_folder if args.output_folder.is_absolute() else Path.cwd() / args.output_folder
    output.mkdir(parents=True, exist_ok=True)
    ctx = ModelContext(output, args.project_name)
    actions = {"structure": structure.run, "rooms": rooms.run, "openings": openings.run, "columns_partitions": columns_partitions.run,
               "components": components.run, "ifc": ifc_writer.run}
    if args.start_at == "prepare":
        cloud = locate(args.input, ROOT.parent / "4_merge_tiles" / "merged" / "floor_merged.ply")
        positions = locate(args.positions, ROOT.parent / "1_preprocessing" / "scan_positions.json")
        ctx.log(f"Input point cloud: {cloud}\nScanner positions: {positions}\nOutput folder: {output}")
        actions["prepare"] = lambda context: prepare.run(context, cloud, positions)

    names = list(STAGES)
    started = time.time()
    for number, name in enumerate(names, 1):
        if number <= names.index(args.start_at):
            continue
        ctx.log(f"\n[{number}/{len(names)}] {STAGES[name]}")
        begin = time.time()
        actions[name](ctx)
        ctx.log(f"    done in {time.time() - begin:.0f} s")
    ctx.log(f"\nFinished in {(time.time() - started) / 60:.1f} min. Model: {output / 'model.ifc'}")


if __name__ == "__main__":
    main()
