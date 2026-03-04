from __future__ import annotations

__all__ = [
    "find_quarto",
]

import os
import shutil
from pathlib import Path


def find_quarto() -> Path:
    quarto_exe = os.getenv("QUARTO_PATH")
    if quarto_exe is None:
        quarto_exe = shutil.which("quarto")

    if quarto_exe is None:
        raise FileNotFoundError("Quarto executable not found")

    return Path(quarto_exe)
