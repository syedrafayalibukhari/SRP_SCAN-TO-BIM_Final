"""Train the starting-label model used by label_transfer.py.

Reads one or more manually labelled point clouds, describes every 5 cm cube exactly as label_transfer.py
does, checks the accuracy by leaving out one part of the floor at a time, then trains on everything and
saves model/label_model.txt.gz. The labelled files are only read.

Usage:  python train_label_model.py floor_merged.ply [more_floors.ply ...]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.modules.setdefault("dask", None)

from scan2bim.setup_env import ensure_packages

ensure_packages({"numpy": "numpy", "scipy": "scipy", "lightgbm": "lightgbm"})

import lightgbm as lgb
import numpy as np

from scan2bim.classes import CLASSES, UNKNOWN, invalid_labels
from scan2bim.features import FEATURE_VOXEL_M, NEIGHBOURS, compute_features, voxel_representatives
from scan2bim.plyio import read_ply

MODEL_PATH = ROOT / "model" / "label_model.txt.gz"
MODEL_INFO_PATH = ROOT / "model" / "model_info.json"

PER_CLASS = 40_000              # training cubes per class and floor (cap)
N_BLOCKS = 4                    # accuracy check: floor split into this many strips, one left out at a time
N_TREES = 400
PARAMS = dict(objective="multiclass", num_class=len(CLASSES), learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              seed=0, num_threads=0, verbose=-1)


def miou(truth: np.ndarray, predicted: np.ndarray):
    per_class = {}
    for value, name in enumerate(CLASSES):
        t, p = truth == value, predicted == value
        if t.sum():
            per_class[name] = float((t & p).sum() / (t | p).sum())
    return float(np.mean(list(per_class.values()))), per_class


def sample(labels: np.ndarray, rng: np.random.Generator, mask=None) -> np.ndarray:
    index = np.arange(len(labels)) if mask is None else np.nonzero(mask)[0]
    index = index[labels[index] != UNKNOWN]
    return np.concatenate([rng.permutation(index[labels[index] == value])[:PER_CLASS] for value in range(len(CLASSES))])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the starting-label model.")
    parser.add_argument("labelled", nargs="+", type=Path, help="manually labelled PLY file(s)")
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    floors = []
    for path in args.labelled:
        start = time.time()
        data = read_ply(path, required=("x", "y", "z", "red", "green", "blue", "label"))
        wrong = invalid_labels(data["label"])
        if len(wrong):
            raise SystemExit(f"{path.name}: label values {wrong.tolist()} are not part of the current classes. "
                             "Convert the file with convert_legacy_labels.py first.")
        xyz = np.c_[data["x"], data["y"], data["z"]].astype(np.float64)
        rgb = np.c_[data["red"], data["green"], data["blue"]].astype(np.uint8)
        first, _ = voxel_representatives(xyz)
        features, _ = compute_features(xyz[first], rgb[first])
        floors.append(dict(name=path.name, X=features, y=data["label"][first], x=xyz[first, 0]))
        print(f"{path.name}: {len(xyz):,} points -> {len(first):,} cubes ({time.time() - start:.0f} s)", flush=True)

    results = []
    for block in range(N_BLOCKS):
        x_train, y_train, x_test, y_test = [], [], [], []
        for floor in floors:
            edges = np.quantile(floor["x"], np.linspace(0, 1, N_BLOCKS + 1))
            part = np.clip(np.searchsorted(edges, floor["x"], side="right") - 1, 0, N_BLOCKS - 1)
            chosen = sample(floor["y"], rng, part != block)
            left_out = (part == block) & (floor["y"] != UNKNOWN)
            x_train.append(floor["X"][chosen])
            y_train.append(floor["y"][chosen])
            x_test.append(floor["X"][left_out])
            y_test.append(floor["y"][left_out])
        booster = lgb.train(PARAMS, lgb.Dataset(np.vstack(x_train), np.concatenate(y_train)), N_TREES)
        value, per_class = miou(np.concatenate(y_test), booster.predict(np.vstack(x_test)).argmax(1))
        results.append((value, per_class))
        print(f"left-out strip {block + 1}/{N_BLOCKS}: mIoU {value:.3f}", flush=True)
    mean_miou = float(np.mean([value for value, _ in results]))
    per_class = {name: float(np.mean([p[name] for _, p in results if name in p])) for name in CLASSES
                 if any(name in p for _, p in results)}
    print(f"mean mIoU over left-out strips: {mean_miou:.3f}")
    print({name: round(value, 3) for name, value in per_class.items()})

    picks = [sample(floor["y"], rng) for floor in floors]
    x_all = np.vstack([floor["X"][chosen] for floor, chosen in zip(floors, picks)])
    y_all = np.concatenate([floor["y"][chosen] for floor, chosen in zip(floors, picks)])
    booster = lgb.train(PARAMS, lgb.Dataset(x_all, y_all), N_TREES)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with gzip.open(MODEL_PATH, "wt", encoding="utf-8") as handle:
        handle.write(booster.model_to_string())
    info = dict(trained_on=" + ".join(floor["name"] for floor in floors), created=f"{datetime.now():%Y-%m-%d}",
                classes=list(CLASSES), feature_voxel_m=FEATURE_VOXEL_M, neighbours=list(NEIGHBOURS),
                training_cubes=int(len(y_all)), trees=N_TREES, lightgbm=lgb.__version__,
                check=dict(method=f"floor split into {N_BLOCKS} strips along x; each strip left out once",
                           mean_miou=round(mean_miou, 3), per_class_iou={k: round(v, 3) for k, v in per_class.items()}))
    MODEL_INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"saved {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
