"""
Adapters package.

Model-specific adapter implementations behind a common interface.
"""

from src.adapters.base_adapter import BaseAdapter
from src.adapters.qwen_adapter import QwenAdapter

__all__ = [
    "BaseAdapter",
    "QwenAdapter",
]
