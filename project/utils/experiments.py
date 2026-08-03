import os
import argparse
import re
from typing import Optional, List, Dict, Any


def is_debug_mode() -> bool:
    return os.environ.get("DEBUG", "0").lower() in ("1", "true", "yes")


def build_run_path_components(
    seed: int = 42,
    answer_extraction_model: Optional[str] = None,
    include_answer_extraction_model: bool = False,
    budget_forcing_prompt: Optional[str] = None,
) -> str:
    parts = [f"seed_{seed}"]
    if budget_forcing_prompt:
        clean_bf = re.sub(r"[^a-zA-Z0-9_-]", "_", budget_forcing_prompt)[:30]
        parts.append(f"budget_prompt_{clean_bf}")
    if include_answer_extraction_model and answer_extraction_model:
        parts.append(f"extractor_{answer_extraction_model}")
    return os.path.join(*parts)


def setup_experiment_folder(
    output: str,
    model_name: str,
    benchmark_name: str,
    experiment_name: str = "main",
    path_components: str = "",
) -> str:
    folder = os.path.join(output, experiment_name, model_name, benchmark_name, path_components)
    os.makedirs(folder, exist_ok=True)
    return folder


def eval_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thinking-Past-The-Answer Evaluation Stage")
    parser.add_argument("--model", type=str, required=False, default="qwen2_5vl", help="Model name registered in modeling")
    parser.add_argument("--benchmark", type=str, required=False, default="gpqa", help="Benchmark name registered in benchmarking")
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "vllm", "sglang", "ddp"], help="Backend inference engine")
    parser.add_argument("--output", type=str, default="results", help="Output root directory")
    parser.add_argument("--experiment_name", type=str, default="main", help="Name of the experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum generated tokens")
    parser.add_argument("--max_num_images", type=int, default=10, help="Maximum images per sample")
    parser.add_argument("--additional_model_args", type=str, default="", help="Additional kwargs for model")
    parser.add_argument("--additional_benchmark_args", type=str, default="", help="Additional kwargs for benchmark")
    parser.add_argument("--override_system_prompt", type=str, default=None, help="Override default system prompt")
    parser.add_argument("--override_reasoning_prompt", type=str, default=None, help="Override reasoning prompt")
    parser.add_argument("--override_prepend_reasoning_prompt", action="store_true", help="Prepend reasoning prompt")
    parser.add_argument("--path_budget_forcing_prompt", type=str, default=None, help="Tag for path component")
    parser.add_argument("--answer_extraction_model", type=str, default="gpt-4o-mini", help="LLM extractor model")
    parser.add_argument("--answer_extraction_with_llm", action="store_true", help="Enable LLM extractor")
    parser.add_argument("--llm_endpoint", type=str, default=None, help="Endpoint for LLM extractor")
    parser.add_argument("--skip_generation", action="store_true", help="Skip generation if output exists")
    parser.add_argument("--show_models_and_benchmarks", action="store_true", help="List registered models and benchmarks")
    parser.add_argument("--limit", "--max_samples", dest="limit", type=int, default=None, help="Maximum number of dataset samples to process")
    parser.add_argument("--use_wandb", action="store_true", help="Log to wandb")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name")
    return parser.parse_args()


def difficulty_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thinking-Past-The-Answer Difficulty Stage")
    parser.add_argument("--model", type=str, required=False, default="qwen2_5vl", help="Model name registered in modeling")
    parser.add_argument("--benchmark", type=str, required=False, default="gpqa", help="Benchmark name registered in benchmarking")
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "vllm", "sglang", "ddp"], help="Backend inference engine")
    parser.add_argument("--output", type=str, default="results", help="Output root directory")
    parser.add_argument("--experiment_name", type=str, default="main", help="Name of the experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum generated tokens")
    parser.add_argument("--max_num_images", type=int, default=10, help="Maximum images per sample")
    parser.add_argument("--additional_model_args", type=str, default="", help="Additional kwargs for model")
    parser.add_argument("--additional_benchmark_args", type=str, default="", help="Additional kwargs for benchmark")
    parser.add_argument("--override_system_prompt", type=str, default=None, help="Override default system prompt")
    parser.add_argument("--override_reasoning_prompt", type=str, default=None, help="Override reasoning prompt")
    parser.add_argument("--override_prepend_reasoning_prompt", action="store_true", help="Prepend reasoning prompt")
    parser.add_argument("--path_budget_forcing_prompt", type=str, default=None, help="Tag for path component")
    parser.add_argument("--budget_forcing_prompt", type=str, default="Therefore, the final answer is:", help="Budget forcing prompt prefix")
    parser.add_argument("--difficulty_level", type=str, default="utterance", choices=["utterance", "token"], help="Splitting level")
    parser.add_argument("--granularity", type=int, default=1, help="Splitting granularity step")
    parser.add_argument("--path_include_answer_extraction_model", action="store_true", help="Include answer extraction model in path")
    parser.add_argument("--answer_extraction_model", type=str, default="gpt-4o-mini", help="LLM extractor model")
    parser.add_argument("--llm_endpoint", type=str, default="unset", help="Endpoint for LLM extractor")
    parser.add_argument("--limit", "--max_samples", dest="limit", type=int, default=None, help="Maximum number of dataset samples to process")
    parser.add_argument("--show_models_and_benchmarks", action="store_true", help="List registered models and benchmarks")
    parser.add_argument("--results_folder", type=str, default="results", help="Results folder")
    parser.add_argument("--use_wandb", action="store_true", help="Log to wandb")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name")
    return parser.parse_args()


def show_available_models_and_benchmarks(results_folder: Optional[str] = None):
    from modeling import MODEL_REGISTER
    from benchmarking import BENCHMARK_REGISTER

    print("=" * 60)
    print("AVAILABLE MODELS:")
    for m in sorted(MODEL_REGISTER.keys()):
        print(f"  - {m}")
    print("\nAVAILABLE BENCHMARKS:")
    for b in sorted(BENCHMARK_REGISTER.keys()):
        print(f"  - {b}")
    print("=" * 60)


def extract_last_match(text: str, pattern: str) -> Optional[str]:
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1) if matches[-1].groups() else matches[-1].group(0)
    return None
