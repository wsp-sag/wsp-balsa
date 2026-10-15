from __future__ import annotations

__all__ = [
    "find_quarto",
]

import os
import shutil
from pathlib import Path

try:
    import quarto_cli
except ImportError:
    quarto_cli = None


def find_quarto() -> Path:
    """Find the Quarto executable.

    Looks for the Quarto executable in the following order:
    1. The ``QUARTO_PATH`` environment variable.
    2. The system ``PATH``.
    3. The ``quarto_cli`` package's bin directory, if available.

    Returns:
        Path: The path to the Quarto executable.

    Raises:
        FileNotFoundError: If the Quarto executable is not found.
    """
    quarto_exe = os.getenv("QUARTO_PATH")

    if quarto_exe is None:
        quarto_exe = shutil.which("quarto")

    if quarto_exe is None and quarto_cli is not None:
        quarto_exe = shutil.which("quarto", path=Path(quarto_cli.__path__[0]) / "bin")

    if quarto_exe is None:
        raise FileNotFoundError("Quarto executable not found")

    return Path(quarto_exe).resolve(strict=True)
