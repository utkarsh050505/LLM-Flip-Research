"""
Benchmarking — pluggable benchmark registry.

Uses the same decorator-based auto-discovery pattern as
thinking-past-the-answer: a @benchmark decorator registers each class
by its module filename into BENCHMARK_REGISTER.
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path
from typing import Any

from benchmarking.base_benchmark import BaseBenchmark

BENCHMARK_REGISTER: dict[str, type[BaseBenchmark]] = {}


def benchmark(cls: type[BaseBenchmark]) -> type[BaseBenchmark]:
    """Class decorator that registers a benchmark by its module filename."""
    module = sys.modules[cls.__module__]
    filename = Path(module.__file__).stem  # type: ignore[arg-type]
    BENCHMARK_REGISTER[filename] = cls
    return cls


def load_benchmark(name: str, **kwargs: Any) -> BaseBenchmark:
    """Instantiate a registered benchmark by name."""
    if name not in BENCHMARK_REGISTER:
        available = ", ".join(sorted(BENCHMARK_REGISTER.keys()))
        raise ValueError(
            f"Unknown benchmark: {name!r}. Available: {available}"
        )
    return BENCHMARK_REGISTER[name](benchmark_name=name, **kwargs)


def list_benchmarks() -> list[str]:
    """Return sorted list of registered benchmark names."""
    return sorted(BENCHMARK_REGISTER.keys())


# Auto-discover all modules in this package so @benchmark decorators fire
for _, module_name, _ in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{module_name}")
