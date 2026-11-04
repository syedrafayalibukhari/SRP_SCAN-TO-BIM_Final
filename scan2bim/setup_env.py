"""First-run installation of missing Python libraries."""
from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Mapping

REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements.txt"


def ensure_packages(needed: Mapping[str, str]) -> None:
    """Install the requirements file when a module in ``needed`` (module name -> pip name) is missing."""
    missing = [pip_name for module, pip_name in needed.items() if importlib.util.find_spec(module) is None]
    if not missing:
        return
    command = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)]
    print(f"Missing Python libraries: {', '.join(missing)}. Installing them now (one time only)...")
    try:
        subprocess.check_call(command)
    except (subprocess.CalledProcessError, OSError):
        raise SystemExit("Automatic installation failed. Install the libraries manually with:\n"
                         f"    {sys.executable} -m pip install -r {REQUIREMENTS}")
    importlib.invalidate_caches()
