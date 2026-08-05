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


BENCHMARK_ALIASES = {
    "gms8k": "gsm8k",
    "math-500": "math500",
    "math_500": "math500",
}

def load_benchmark(benchmark_name: str, **kwargs) -> BaseBenchmark:
    clean_name = benchmark_name.lower().strip()
    resolved_name = BENCHMARK_ALIASES.get(clean_name, clean_name)
    benchmark_class = BENCHMARK_REGISTER.get(resolved_name, None)
    if benchmark_class is None:
        raise ValueError(f"Benchmark {benchmark_name} (resolved as '{resolved_name}') not found in BENCHMARK_REGISTER. Available: {list(BENCHMARK_REGISTER.keys())}")
    return benchmark_class(resolved_name, **kwargs)

