"""Prepare the labelled floor: normals from the 20 nearest neighbours, oriented towards the nearest scanner."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from ..classes import CEILING, FLOOR, invalid_labels
from ..plyio import read_ply
from .context import FLOOR_DTYPE, ModelContext


def run(ctx: ModelContext, cloud: Path, positions: Path) -> None:
    data = read_ply(cloud, required=("x", "y", "z", "label"))
    labels = data["label"].astype(np.uint8)
    wrong = invalid_labels(labels)
    if len(wrong):
        raise SystemExit(f"{Path(cloud).name}: label values {wrong.tolist()} are not part of the current classes. "
                         "Convert the file with 2_label_transfer/convert_legacy_labels.py first.")
    xyz = np.c_[data["x"], data["y"], data["z"]].astype(float)
    with open(positions, encoding="utf-8") as handle:
        stations = json.load(handle)["stations"]
    viewpoints = np.array([s["xyz"] for s in stations], float)
    ctx.log(f"{len(xyz):,} points, {len(viewpoints)} scanner positions")

    cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    cloud_o3d.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
    normals = np.asarray(cloud_o3d.normals).copy()
    del cloud_o3d
    nearest = viewpoints[cKDTree(viewpoints).query(xyz)[1]]
    normals[np.einsum("ij,ij->i", normals, nearest - xyz) < 0] *= -1
    normals[(labels == FLOOR) & (normals[:, 2] < 0)] *= -1
    normals[(labels == CEILING) & (normals[:, 2] > 0)] *= -1

    floor = np.empty(len(xyz), FLOOR_DTYPE)
    floor["x"], floor["y"], floor["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    floor["nx"], floor["ny"], floor["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    floor["label"] = labels
    crop = [float(xyz[:, 0].min() - 1), float(xyz[:, 1].min() - 1), float(xyz[:, 0].max() + 1), float(xyz[:, 1].max() + 1)]
    ctx.set_floor(floor, [dict(xyz=s["xyz"], scan=s["scan"]) for s in stations], crop)
