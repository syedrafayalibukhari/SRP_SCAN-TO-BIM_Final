"""Reading and writing of binary little-endian PLY point clouds."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np

PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "<i2", "int16": "<i2",
             "ushort": "<u2", "uint16": "<u2", "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
             "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8"}
NUMPY_TO_PLY = {"i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort", "i4": "int", "u4": "uint",
                "f4": "float", "f8": "double"}


def _read_header(handle, path: Path) -> Tuple[int, list]:
    count, props, in_vertex, fmt = 0, [], False, ""
    while True:
        raw = handle.readline()
        if not raw:
            raise ValueError(f"{path.name}: not a valid PLY file (no end_header).")
        line = raw.decode("ascii", "ignore").strip()
        if line.startswith("format"):
            fmt = line.split()[1]
        elif line.startswith("element"):
            in_vertex = line.split()[1] == "vertex"
            count = int(line.split()[2]) if in_vertex else count
        elif line.startswith("property") and in_vertex:
            parts = line.split()
            if parts[1] == "list":
                raise ValueError(f"{path.name}: list properties in the vertex element are not supported.")
            props.append((parts[2], PLY_TYPES[parts[1]]))
        elif line == "end_header":
            break
    if fmt != "binary_little_endian":
        raise ValueError(f"{path.name}: only binary PLY files are supported. In CloudCompare, save the "
                         "file as PLY with 'Binary' selected.")
    return count, props


def read_ply(path: Path, required: Iterable[str] = ("x", "y", "z")) -> np.ndarray:
    """Read the vertex element of a PLY file as a structured array."""
    path = Path(path)
    with open(path, "rb") as handle:
        count, props = _read_header(handle, path)
        data = np.fromfile(handle, dtype=np.dtype(props), count=count)
    if len(data) != count:
        raise ValueError(f"{path.name}: the file is incomplete ({len(data):,} of {count:,} points).")
    missing = [name for name in required if name not in data.dtype.names]
    if missing:
        raise ValueError(f"{path.name}: missing properties: {', '.join(missing)}.")
    return data


def _header(dtype: np.dtype, count: int, comments: Sequence[str]) -> bytes:
    lines = ["ply", "format binary_little_endian 1.0"]
    lines += [f"comment {text}" for text in comments]
    lines.append(f"element vertex {count}")
    lines += [f"property {NUMPY_TO_PLY[dtype[name].str[1:]]} {name}" for name in dtype.names]
    lines.append("end_header\n")
    return "\n".join(lines).encode("ascii")


def write_ply(path: Path, data: np.ndarray, comments: Sequence[str] = ()) -> None:
    """Write a structured array as a binary PLY file."""
    with open(path, "wb") as handle:
        handle.write(_header(data.dtype, len(data), comments))
        data.tofile(handle)


class PlyStreamWriter:
    """Write a PLY file piece by piece when the total point count is known in advance."""

    def __init__(self, path: Path, dtype: np.dtype, count: int) -> None:
        self.path, self.dtype, self.count, self.written = Path(path), np.dtype(dtype), count, 0
        self.handle = open(self.path, "wb")
        self.handle.write(_header(self.dtype, count, ()))

    def write(self, data: np.ndarray) -> None:
        data.tofile(self.handle)
        self.written += len(data)

    def close(self) -> None:
        self.handle.close()
        if self.written != self.count:
            raise RuntimeError(f"{self.path}: header says {self.count:,} points, {self.written:,} written.")
