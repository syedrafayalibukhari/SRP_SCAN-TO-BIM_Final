"""Point features shared by label transfer and model training."""
from __future__ import annotations

import time
from typing import Callable, Tuple

import numpy as np
from scipy.spatial import cKDTree

FEATURE_VOXEL_M = 0.05          # labels are predicted for 5 cm cubes
NEIGHBOURS = (10, 30, 60)       # neighbourhood sizes used to describe the local shape


def storey_levels(z: np.ndarray) -> Tuple[float, float]:
    """Floor and ceiling height from the two strongest peaks of the height histogram."""
    counts, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.02, 0.02))
    centres = (edges[:-1] + edges[1:]) / 2
    low = centres < np.median(z)
    floor_z = centres[low][np.argmax(counts[low])]
    high = centres > floor_z + 1.5
    if not high.any():
        return float(floor_z), float(z.max())
    return float(floor_z), float(centres[high & (counts > 0.3 * counts[high].max())].max())


def voxel_representatives(xyz: np.ndarray, voxel: float = FEATURE_VOXEL_M) -> Tuple[np.ndarray, np.ndarray]:
    """Index of the first point in every occupied cube, and the cube of every point."""
    cell = np.floor((xyz - xyz.min(0)) / voxel).astype(np.int64)
    key = (cell[:, 0] << 42) | (cell[:, 1] << 21) | cell[:, 2]
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    return first, inverse.ravel()


def compute_features(xyz: np.ndarray, rgb: np.ndarray, chunk: int = 100_000,
                     log: Callable[[str], None] = print) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Per point: standardised colour, shape descriptors at three neighbourhood sizes, height above the
    floor, depth below the ceiling and neighbourhood radius (27 values)."""
    tree = cKDTree(xyz)
    count, largest = len(xyz), max(NEIGHBOURS)
    shape = np.empty((count, 7 * len(NEIGHBOURS)), np.float32)
    radius = np.empty(count, np.float32)
    start = time.time()
    for begin in range(0, count, chunk):
        end = min(count, begin + chunk)
        distance, nearest = tree.query(xyz[begin:end], k=largest, workers=-1)
        radius[begin:end] = distance[:, -1]
        for j, k in enumerate(NEIGHBOURS):
            local = xyz[nearest[:, :k]]
            local = local - local.mean(1, keepdims=True)
            covariance = np.einsum("nki,nkj->nij", local, local) / k
            values, vectors = np.linalg.eigh(covariance)
            values = np.clip(values, 1e-12, None)
            l3, l2, l1 = values[:, 0], values[:, 1], values[:, 2]
            normal = vectors[:, :, 0]
            shape[begin:end, 7 * j:7 * j + 7] = np.c_[(l1 - l2) / l1, (l2 - l3) / l1, l3 / l1,
                                                     np.cbrt(l1 * l2 * l3) / l1, l3 / (l1 + l2 + l3),
                                                     1 - np.abs(normal[:, 2]), np.sqrt(l1)]
        if (begin // chunk) % 5 == 4 or end == count:
            log(f"    described {end:,} of {count:,} cubes ({time.time() - start:.0f} s)")
    floor_z, ceiling_z = storey_levels(xyz[:, 2])
    rgb01 = rgb.astype(np.float32) / 255.0
    rgb_std = (rgb01 - rgb01.mean(0)) / (rgb01.std(0) + 1e-6)
    heights = np.c_[xyz[:, 2] - floor_z, ceiling_z - xyz[:, 2]].astype(np.float32)
    return np.c_[rgb_std, shape, heights, radius], (floor_z, ceiling_z)
