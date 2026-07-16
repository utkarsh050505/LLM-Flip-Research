import os
import random
import logging
import numpy as np
# pyrefly: ignore [missing-import]
import torch

def set_seed(seed: int):
    """
    Sets deterministic seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logging(log_file: str = None) -> logging.Logger:
    """
    Sets up a logger that outputs to both stdout and a file.
    """
    logger = logging.getLogger("StopBeforeTheFlip")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def get_vram_usage() -> str:
    """
    Returns a formatted string containing current GPU memory consumption.
    """
    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_idx)
        allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated(device_idx) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 3)
        return (
            f"Device: {device_name} | "
            f"Allocated: {allocated:.3f} GB (Max: {max_allocated:.3f} GB) | "
            f"Reserved: {reserved:.3f} GB"
        )
    return "CUDA is not available"
