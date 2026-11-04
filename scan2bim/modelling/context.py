"""Shared state of one modelling run: paths, the prepared floor, the log and the stage results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

FLOOR_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"), ("label", "u1")])


class ModelContext:
    """Paths and data shared by the modelling stages. Each stage result is stored as JSON in ``work``."""

    def __init__(self, output: Path, project_name: str) -> None:
        self.output = Path(output)
        self.work = self.output / "intermediate"
        self.work.mkdir(parents=True, exist_ok=True)
        self.project_name = project_name
        self._floor: Optional[np.ndarray] = None
        self._results: Dict[str, Any] = {}
        self._log = open(self.output / "modelling_log.txt", "a", encoding="utf-8")

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        self._log.write(f"{message}\n")
        self._log.flush()

    def set_floor(self, floor: np.ndarray, stations: list, crop: list) -> None:
        self._floor = floor
        np.save(self.work / "floor.npy", floor)
        self.save("site", dict(stations=stations, crop=crop))

    @property
    def floor(self) -> np.ndarray:
        if self._floor is None:
            self._floor = np.load(self.work / "floor.npy")
        return self._floor

    @property
    def points(self) -> np.ndarray:
        floor = self.floor
        return np.c_[floor["x"], floor["y"], floor["z"]].astype(float)

    @property
    def normals(self) -> np.ndarray:
        floor = self.floor
        return np.c_[floor["nx"], floor["ny"], floor["nz"]].astype(float)

    @property
    def labels(self) -> np.ndarray:
        return self.floor["label"]

    @property
    def stations(self) -> np.ndarray:
        return np.array([s["xyz"] for s in self.load("site")["stations"]], float)

    @property
    def crop(self) -> list:
        return self.load("site")["crop"]

    def save(self, name: str, result: Any) -> None:
        self._results[name] = result
        with open(self.work / f"{name}.json", "w", encoding="utf-8") as handle:
            json.dump(result, handle)

    def load(self, name: str) -> Any:
        if name not in self._results:
            with open(self.work / f"{name}.json", encoding="utf-8") as handle:
                self._results[name] = json.load(handle)
        return self._results[name]
