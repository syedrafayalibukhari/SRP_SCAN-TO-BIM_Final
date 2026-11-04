"""Step 2: starting labels and tiling for manual point labelling.

Gives every point of a cleaned point cloud a starting label with the already trained model, then writes
the labelled cloud as one file or as square tiles, ready for the Point Label Corrector.

Steps: (a) read the cleaned point cloud, (b) group the points into 5 cm cubes and describe the shape and
colour around each cube, (c) predict a label and a confidence (0-1), (d) give every point the label of
its cube, (e) write one file or tiles, a tile map and a report.

Usage:  python label_transfer.py                    (uses TILE_SIZE_M below)
        python label_transfer.py --tile-size 0      (no tiling: one file for the whole floor)
Input: preprocessed.ply next to this script or in ../1_preprocessing, or a file named with --input.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from scan2bim.setup_env import ensure_packages

if __name__ == "__main__":
    ensure_packages({"numpy": "numpy", "scipy": "scipy", "lightgbm": "lightgbm", "matplotlib": "matplotlib"})

# LightGBM optionally imports dask, which breaks with NumPy 2 in some Anaconda installations; not needed here.
sys.modules.setdefault("dask", None)

import numpy as np

from scan2bim.classes import CEILING, CLASSES, COLOURS, OTHER
from scan2bim.features import FEATURE_VOXEL_M, compute_features, voxel_representatives
from scan2bim.plyio import read_ply, write_ply

MODEL_PATH = ROOT / "model" / "label_model.txt.gz"
MODEL_INFO_PATH = ROOT / "model" / "model_info.json"

# ============================================================================
# TILE SIZE (metres) - SET THIS NUMBER BEFORE RUNNING
#   8   = cut the floor into 8 m x 8 m tiles (one file per tile)
#   0   = no tiling: one labelled file for the whole floor
# A value given with --tile-size overrides this one.
TILE_SIZE_M = 8
# ============================================================================

OUTSIDE_MARGIN_M = 0.3          # points this far below the floor / above the ceiling -> other
LOW_CONFIDENCE = 0.5            # reported share of points below this confidence
MIN_TILE_POINTS = 1000          # tiles with fewer points are joined to a neighbour

OUTPUT_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"),
                         ("blue", "u1"), ("label", "u1"), ("conf", "<f4"), ("idx", "<i4")])


def read_cloud(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read positions and 8-bit colours of a point cloud."""
    data = read_ply(path)
    colour_names = next((c for c in (("red", "green", "blue"), ("r", "g", "b")) if all(k in data.dtype.names for k in c)), None)
    if colour_names is None:
        raise SystemExit(f"{path.name}: the point cloud has no colours (red/green/blue). Colours are needed.")
    xyz = np.c_[data["x"], data["y"], data["z"]].astype(np.float64)
    rgb = np.c_[[data[c] for c in colour_names]].T
    if rgb.dtype.kind == "f":
        rgb = np.clip(rgb * 255.0, 0, 255) if rgb.max() <= 1.0 else rgb
    return xyz, rgb.astype(np.uint8)


def write_labelled(path: Path, xyz: np.ndarray, rgb: np.ndarray, label: np.ndarray, conf: np.ndarray, idx: np.ndarray) -> None:
    out = np.empty(len(idx), OUTPUT_DTYPE)
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out["red"], out["green"], out["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    out["label"], out["conf"], out["idx"] = label, conf, idx
    write_ply(path, out, [f"starting labels; classes: {' '.join(CLASSES)}"])


def load_model():
    import lightgbm as lgb

    if not MODEL_PATH.exists():
        raise SystemExit(f"Trained model not found: {MODEL_PATH}")
    with gzip.open(MODEL_PATH, "rt", encoding="utf-8") as handle:
        booster = lgb.Booster(model_str=handle.read())
    info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8")) if MODEL_INFO_PATH.exists() else {}
    trained = info.get("classes", list(CLASSES))
    if trained != list(CLASSES[:len(trained)]):      # a model may lack the newest classes; they are then labelled by hand
        raise SystemExit("The trained model uses other classes than this workflow. Retrain it with train_label_model.py.")
    return booster, info


def tile_grid(xyz: np.ndarray, size: float) -> Tuple[np.ndarray, np.ndarray, int, int, np.ndarray]:
    """Row and column of every point; stray edge points are clipped into the edge tiles."""
    low = np.percentile(xyz[:, :2], 0.5, axis=0) - 0.5
    high = np.percentile(xyz[:, :2], 99.5, axis=0) + 0.5
    ncol = max(1, int(np.ceil((high[0] - low[0]) / size)))
    nrow = max(1, int(np.ceil((high[1] - low[1]) / size)))
    col = np.clip(((xyz[:, 0] - low[0]) // size).astype(int), 0, ncol - 1)
    row = np.clip(((xyz[:, 1] - low[1]) // size).astype(int), 0, nrow - 1)
    return row, col, nrow, ncol, low


def draw_tile_map(path: Path, xyz: np.ndarray, label: np.ndarray, nrow: int, ncol: int, low: np.ndarray, size: float, kept) -> None:
    """Top view coloured by starting label, with the tile grid; tiles with points are outlined."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch, Rectangle

    show = np.where(label != CEILING)[0]
    show = show[np.argsort(xyz[show, 2])][::4]
    fig, ax = plt.subplots(figsize=(14, 14 * nrow / max(ncol, 1) + 1))
    ax.scatter(xyz[show, 0], xyz[show, 1], c=label[show], s=0.2, cmap=ListedColormap(COLOURS), vmin=0, vmax=len(CLASSES) - 1, linewidths=0)
    for r, c in kept:
        ax.add_patch(Rectangle((low[0] + c * size, low[1] + r * size), size, size, fill=False, ec="k", lw=1))
        ax.text(low[0] + (c + 0.5) * size, low[1] + (r + 0.5) * size, f"r{r:02d}_c{c:02d}", ha="center", va="center", fontsize=9)
    ax.legend(handles=[Patch(color=COLOURS[i], label=c) for i, c in enumerate(CLASSES) if i != CEILING], loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def find_input(given: Optional[str]) -> Path:
    if given:
        path = Path(given)
        path = path if path.is_absolute() else Path.cwd() / path
        if not path.exists():
            raise SystemExit(f"Input file not found: {path}")
        return path
    for default in (ROOT / "preprocessed.ply", ROOT.parent / "1_preprocessing" / "preprocessed.ply"):
        if default.exists():
            return default
    raise SystemExit("No point cloud found. Run step 1 first, or name the file with --input.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Give starting labels to a cleaned point cloud and cut it into tiles.")
    parser.add_argument("--input", help="cleaned point cloud (.ply); default: preprocessed.ply from step 1")
    parser.add_argument("--tile-size", type=float, default=TILE_SIZE_M, help="tile size in metres (0 = no tiling); default: TILE_SIZE_M")
    parser.add_argument("--output-folder", default=str(ROOT / "labelled"), help="folder for the results (default: labelled)")
    args = parser.parse_args()
    if args.tile_size < 0:
        raise SystemExit("The tile size must be 0 (no tiling) or a positive number of metres.")

    source = find_input(args.input)
    out = Path(args.output_folder)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(message: str = "") -> None:
        print(message, flush=True)
        lines.append(message)

    t_all = time.time()
    log(f"Label transfer - {datetime.now():%Y-%m-%d %H:%M}")
    log(f"Input: {source.name}")
    xyz, rgb = read_cloud(source)
    log(f"[Step a] Read point cloud  |  {len(xyz):,} points")

    first, inverse = voxel_representatives(xyz)
    log(f"[Step b] Group into {FEATURE_VOXEL_M * 100:.0f} cm cubes  |  {len(first):,} cubes")
    features, (floor_z, ceiling_z) = compute_features(xyz[first], rgb[first], log=log)
    log(f"    floor level {floor_z:.2f} m, ceiling level {ceiling_z:.2f} m")

    booster, info = load_model()
    probability = booster.predict(features)
    cube_label = probability.argmax(1).astype(np.uint8)
    cube_conf = probability.max(1).astype(np.float32)
    height = xyz[first, 2]
    outside = (height < floor_z - OUTSIDE_MARGIN_M) | (height > ceiling_z + OUTSIDE_MARGIN_M)
    cube_label[outside], cube_conf[outside] = OTHER, 1.0
    log(f"[Step c] Predict starting labels  |  model trained on {info.get('trained_on', MODEL_PATH.name)}")

    label, conf = cube_label[inverse], cube_conf[inverse]
    idx = np.arange(len(xyz), dtype=np.int32)
    log(f"[Step d] Give every point the label of its cube  |  {len(xyz):,} points labelled\n")
    log(f"    {'class':15s}{'share':>8s}{'mean confidence':>18s}")
    counts = np.bincount(label, minlength=len(CLASSES))
    for i, name in enumerate(CLASSES):
        mean_conf = conf[label == i].mean() if counts[i] else float("nan")
        log(f"    {name:15s}{100 * counts[i] / len(label):7.1f}%{mean_conf:18.2f}")
    log(f"    points with confidence below {LOW_CONFIDENCE}: {100 * (conf < LOW_CONFIDENCE).mean():.1f}%\n")

    written = []
    if args.tile_size == 0:
        path = out / "labelled_floor.ply"
        write_labelled(path, xyz, rgb, label, conf, idx)
        written.append((path.name, len(xyz)))
        log(f"[Step e] No tiling  |  wrote {path.name}")
    else:
        row, col, nrow, ncol, low = tile_grid(xyz, args.tile_size)
        tile = row * ncol + col
        counts = np.bincount(tile, minlength=nrow * ncol)
        large = np.nonzero(counts >= MIN_TILE_POINTS)[0]
        for small in np.nonzero((counts > 0) & (counts < MIN_TILE_POINTS))[0]:
            if len(large):
                nearest = np.argmin((large // ncol - small // ncol) ** 2 + (large % ncol - small % ncol) ** 2)
                tile[tile == small] = large[nearest]
        order = np.argsort(tile, kind="stable")
        bounds = np.searchsorted(tile[order], np.arange(nrow * ncol + 1))
        kept = []
        for index in range(nrow * ncol):
            chosen = order[bounds[index]:bounds[index + 1]]
            if len(chosen) == 0:
                continue
            r, c = divmod(index, ncol)
            path = out / f"tile_r{r:02d}_c{c:02d}.ply"
            write_labelled(path, xyz[chosen], rgb[chosen], label[chosen], conf[chosen], idx[chosen])
            written.append((path.name, len(chosen)))
            kept.append((r, c))
        draw_tile_map(out / "tile_map.png", xyz, label, nrow, ncol, low, args.tile_size, kept)
        log(f"[Step e] Cut into {args.tile_size:g} m tiles  |  {nrow} rows x {ncol} columns, {len(written)} tiles with points")
    for name, count in written:
        log(f"    {name:28s}{count:>12,} points")
    log(f"\nResults written to: {out}")
    log(f"Total time: {(time.time() - t_all) / 60:.1f} min")
    (out / "label_transfer_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
