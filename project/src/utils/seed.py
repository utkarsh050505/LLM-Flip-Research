"""
Seed Utilities

Ensures reproducible experiments by seeding all relevant RNGs.
"""

from __future__ import annotations

import random
import logging

logger = logging.getLogger("sbtf")


def set_deterministic_seed(seed: int = 42) -> None:
    """
    Set deterministic seeds for reproducibility.

    Seeds Python's random module, and optionally NumPy and PyTorch
    if they are available (avoids hard dependency).

    Args:
        seed: The random seed value.
    """
    random.seed(seed)
    logger.info("Python random seed set to %d", seed)

    try:
        import numpy as np
        np.random.seed(seed)
        logger.info("NumPy seed set to %d", seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info("PyTorch seed set to %d", seed)
    except ImportError:
        pass
