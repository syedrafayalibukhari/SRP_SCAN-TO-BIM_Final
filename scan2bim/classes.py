"""Element classes used by every step of the workflow.

Point labels are stored as integers. The values below are the only definition of the classes in the
Python code; ``3_label_correction/point_label_corrector.html`` repeats them and a test keeps both equal.
"""
from __future__ import annotations

import numpy as np

CLASSES = ("floor", "ceiling", "wall", "door", "window", "column", "partition", "other", "curtain_wall")
FLOOR, CEILING, WALL, DOOR, WINDOW, COLUMN, PARTITION, OTHER, CURTAIN_WALL = range(len(CLASSES))
UNKNOWN = 255

COLOURS = ("#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#9467bd", "#7f7f7f", "#17becf")

# Twelve-class scheme of the first CUB labelling. Furniture, equipment and clutter become "other".
LEGACY_CLASSES = ("floor", "ceiling", "wall", "door", "window", "column", "desk", "chair", "monitor",
                  "partition", "cabinet_shelf", "clutter")


def legacy_lookup() -> np.ndarray:
    """Return a 256-entry table that converts legacy label values to current label values."""
    table = np.full(256, UNKNOWN, dtype=np.uint8)
    for value, name in enumerate(LEGACY_CLASSES):
        table[value] = CLASSES.index(name) if name in CLASSES else OTHER
    return table


def invalid_labels(labels: np.ndarray) -> np.ndarray:
    """Return the label values that are not part of the current scheme."""
    values = np.unique(labels)
    return values[(values >= len(CLASSES)) & (values != UNKNOWN)]
