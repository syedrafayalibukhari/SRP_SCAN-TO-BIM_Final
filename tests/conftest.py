import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.modules.setdefault("dask", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
