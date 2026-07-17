"""
I/O Utilities

File system helpers used across the project.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("sbtf")


def ensure_directory(path: Path) -> Path:
    """
    Create directory (and parents) if it does not exist.

    Args:
        path: Directory path to ensure exists.

    Returns:
        The same path, for chaining convenience.

    Raises:
        OSError: If directory creation fails due to permissions or disk issues.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    logger.debug("Ensured directory exists: %s", path)
    return path
