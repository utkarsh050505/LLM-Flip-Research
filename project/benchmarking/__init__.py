from benchmarking.base_benchmark import BaseBenchmark
import pkgutil
import importlib
from pathlib import Path
import sys


BENCHMARK_REGISTER = {}

def benchmark(cls):
    module = sys.modules[cls.__module__]
    filename = Path(module.__file__).stem # type: ignore
    BENCHMARK_REGISTER[filename] = cls
    return cls

for _, module_name, _ in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{module_name}")


def load_benchmark(benchmark_name: str, **kwargs) -> BaseBenchmark:
    benchmark_class = BENCHMARK_REGISTER.get(benchmark_name, None)
    if benchmark_class is None:
        raise ValueError(f"Benchmark {benchmark_name} not found in BENCHMARK_REGISTER.")
    return benchmark_class(benchmark_name, **kwargs)
