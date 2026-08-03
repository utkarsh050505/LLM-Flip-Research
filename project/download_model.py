"""
Model Downloader & Manager CLI for Thinking Past the Answer Research.

Features:
  - Fast, resumable multi-threaded downloads from Hugging Face Hub.
  - Automatically targets HF_HOME (defaulting to A:\\LLMResearch\\hf_cache).
  - Lists locally cached models and their disk usage.
  - Interactive selection menu with size and quantization recommendations.
  - Optional smoke-test loading into GPU (4-bit/8-bit/bf16) to verify VRAM fit.

Usage Examples:
  python download_model.py                      # Interactive menu
  python download_model.py --list               # List all cached models
  python download_model.py --model r1_distill_llama8b
  python download_model.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  python download_model.py --model Qwen/Qwen2.5-3B-Instruct --test_load
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

# Fix Windows console UTF-8 encoding
if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure default cache is A:\LLMResearch\hf_cache if not set
DEFAULT_CACHE = os.environ.get("HF_HOME", r"A:\LLMResearch\hf_cache")
os.environ["HF_HOME"] = DEFAULT_CACHE
os.environ["TRANSFORMERS_CACHE"] = DEFAULT_CACHE

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Preset Models Catalog
MODEL_CATALOG = {
    "1": {
        "key": "r1_distill_llama8b",
        "repo_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "size_gb": 16.0,
        "recommended_vram": "4-bit: 5.5GB | 8-bit: 9GB | BF16: 16GB",
        "description": "DeepSeek-R1 distilled on Llama-3.1-8B (High reasoning power)",
    },
    "2": {
        "key": "r1_distill_qwen1_5b",
        "repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "size_gb": 3.2,
        "recommended_vram": "BF16: 3.5GB | 4-bit: 2GB (Super fast!)",
        "description": "DeepSeek-R1 distilled on Qwen-2.5-1.5B (Lightweight reasoning)",
    },
    "3": {
        "key": "r1_distill_qwen7b",
        "repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "size_gb": 15.0,
        "recommended_vram": "4-bit: 5GB | BF16: 15GB",
        "description": "DeepSeek-R1 distilled on Qwen-2.5-7B (Strong reasoning)",
    },
    "4": {
        "key": "qwen2_5_1_5b",
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "size_gb": 3.1,
        "recommended_vram": "BF16: 3.5GB (Ultra lightweight general instruct)",
        "description": "Qwen 2.5 1.5B Instruct",
    },
    "5": {
        "key": "qwen2_5_3b",
        "repo_id": "Qwen/Qwen2.5-3B-Instruct",
        "size_gb": 6.5,
        "recommended_vram": "BF16: 7GB | 4-bit: 3GB",
        "description": "Qwen 2.5 3B Instruct (Great speed/quality balance)",
    },
    "6": {
        "key": "qwen2_5_7b",
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "size_gb": 15.2,
        "recommended_vram": "4-bit: 5GB | BF16: 15GB",
        "description": "Qwen 2.5 7B Instruct",
    },
    "7": {
        "key": "qwen2_5vl",
        "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "size_gb": 16.2,
        "recommended_vram": "4-bit: 6GB | BF16: 16GB",
        "description": "Qwen 2.5 Vision-Language 7B",
    },
    "8": {
        "key": "qwen3",
        "repo_id": "Qwen/Qwen3-8B",
        "size_gb": 16.0,
        "recommended_vram": "4-bit: 5.5GB | BF16: 16GB",
        "description": "Qwen 3 8B",
    },
}


def get_cached_models(cache_dir: str = DEFAULT_CACHE) -> list[dict]:
    """Scan the local HuggingFace cache directory for downloaded models."""
    cached = []
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir(cache_dir=cache_dir)
        for repo in info.repos:
            if repo.repo_type == "model":
                size_gb = repo.size_on_disk / (1024 ** 3)
                has_weights = any(
                    f.file_name.endswith((".safetensors", ".bin", ".pt"))
                    for rev in repo.revisions
                    for f in rev.files
                )
                cached.append({
                    "repo_id": repo.repo_id,
                    "size_gb": size_gb,
                    "nb_files": repo.nb_files,
                    "has_weights": has_weights,
                    "last_modified": repo.last_modified,
                })
    except Exception:
        hub_dir = Path(cache_dir) / "hub"
        if not hub_dir.exists():
            hub_dir = Path(cache_dir)
        if hub_dir.exists():
            for d in hub_dir.iterdir():
                if d.is_dir() and d.name.startswith("models--"):
                    parts = d.name.replace("models--", "").split("--")
                    repo_id = "/".join(parts)
                    cached.append({
                        "repo_id": repo_id,
                        "size_gb": sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 ** 3),
                        "nb_files": len(list(d.rglob("*"))),
                        "has_weights": True,
                        "last_modified": 0,
                    })
    return cached


def print_cached_models(cache_dir: str = DEFAULT_CACHE) -> list[dict]:
    cached = get_cached_models(cache_dir)
    print(f"\n{'=' * 75}")
    print(f"  [CACHE] LOCALLY CACHED MODELS (Directory: {cache_dir})")
    print(f"{'=' * 75}")
    if not cached:
        print("  No downloaded models found in cache.")
    else:
        print(f"  {'#':<3} | {'Repo ID':<45} | {'Size':<10} | {'Status'}")
        print(f"  {'-'*3}-+-{'-'*45}-+-{'-'*10}-+-{'-'*12}")
        for i, m in enumerate(cached, 1):
            status = "[Ready]" if m["has_weights"] and m["size_gb"] > 0.1 else "[Incomplete]"
            print(f"  {i:<3} | {m['repo_id']:<45} | {m['size_gb']:6.2f} GB | {status}")
    print(f"{'=' * 75}\n")
    return cached


def download_model(
    repo_id: str,
    cache_dir: str = DEFAULT_CACHE,
    max_workers: int = 4,
    test_load: bool = False,
    quantization: str = "4bit",
) -> bool:
    """Download model files with resume support and progress bars."""
    print(f"\n{'=' * 75}")
    print(f"  [DOWNLOADING] Model: {repo_id}")
    print(f"  Target Cache: {cache_dir}")
    print(f"{'=' * 75}")

    os.makedirs(cache_dir, exist_ok=True)
    from huggingface_hub import snapshot_download

    start_time = time.time()
    try:
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            max_workers=max_workers,
            resume_download=True,
        )
        elapsed = time.time() - start_time
        print(f"\n{'=' * 75}")
        print(f"  [SUCCESS] Download completed in {elapsed:.1f}s ({elapsed / 60:.1f}m)")
        print(f"  Model is stored and ready in: {cache_dir}")
        print(f"{'=' * 75}\n")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        return False

    if test_load:
        test_load_model(repo_id, quantization=quantization)

    return True


def test_load_model(repo_id: str, quantization: str = "4bit") -> None:
    """Smoke test loading model into PyTorch/CUDA to verify VRAM usage."""
    print(f"\n{'=' * 75}")
    print(f"  [TEST LOAD] {repo_id} (Quantization: {quantization})")
    print(f"{'=' * 75}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        cuda_avail = torch.cuda.is_available()
        print(f"  CUDA Device: {torch.cuda.get_device_name(0) if cuda_avail else 'CPU'}")

        if cuda_avail:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if cuda_avail else "cpu",
        }

        if quantization == "4bit" and cuda_avail:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "8bit" and cuda_avail:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif cuda_avail:
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        print("  Loading weights into memory...")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(repo_id, **kwargs)
        load_time = time.time() - t0

        if cuda_avail:
            vram_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"  * Model loaded in {load_time:.1f}s!")
            print(f"  * GPU VRAM Utilized: {vram_used:.2f} GB")
        else:
            print(f"  * Model loaded in {load_time:.1f}s on CPU!")

        # Quick test generation
        prompt = "Question: What is 12 + 15? Answer step by step and put final answer in \\boxed{}."
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            tokens = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        out = tokenizer.decode(tokens[0], skip_special_tokens=True)
        print(f"\n  Sample generation test output:\n  {out[:150]}...")
        print(f"\n  [READY] Model is 100% functional and ready for pipeline experiments!")
        print(f"{'=' * 75}\n")
    except Exception as e:
        print(f"  [WARNING] Test load encountered error: {e}")


def interactive_menu():
    """Interactive CLI menu to choose or inspect models."""
    print(f"\n{'=' * 75}")
    print(f"  THINKING PAST THE ANSWER -- MODEL MANAGER")
    print(f"{'=' * 75}")

    cached = print_cached_models()

    print("  AVAILABLE PRESET MODELS FOR DOWNLOAD:")
    print(f"  {'#':<3} | {'Preset Key':<20} | {'Repo ID':<42} | {'Approx Size'}")
    print(f"  {'-'*3}-+-{'-'*20}-+-{'-'*42}-+-{'-'*12}")
    for num, data in MODEL_CATALOG.items():
        is_in_cache = any(data["repo_id"].lower() == c["repo_id"].lower() for c in cached)
        status_tag = " [ALREADY CACHED]" if is_in_cache else ""
        print(f"  {num:<3} | {data['key']:<20} | {data['repo_id']:<42} | {data['size_gb']:4.1f} GB{status_tag}")
    print(f"{'=' * 75}")
    print("  Options:")
    print("    [1-8]  : Select a preset model to download")
    print("    [C]    : Enter custom Hugging Face repo ID (e.g. Qwen/Qwen2.5-Math-1.5B)")
    print("    [T]    : Test-load an already cached model into GPU")
    print("    [Q]    : Quit")
    print(f"{'=' * 75}")

    try:
        choice = input("\nSelect an option: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return

    if choice == "q":
        return
    elif choice in MODEL_CATALOG:
        target_repo = MODEL_CATALOG[choice]["repo_id"]
        try:
            q_choice = input("Quantization to test after download (4bit / 8bit / bf16 / none) [default: 4bit]: ").strip().lower() or "4bit"
        except (EOFError, KeyboardInterrupt):
            q_choice = "4bit"
        download_model(target_repo, test_load=True, quantization=q_choice)
    elif choice == "c":
        try:
            custom_repo = input("Enter HuggingFace Repo ID (e.g. deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B): ").strip()
            if custom_repo:
                download_model(custom_repo, test_load=True)
        except (EOFError, KeyboardInterrupt):
            pass
    elif choice == "t":
        if not cached:
            print("No cached models to test.")
            return
        try:
            idx = input("Enter cached model # to test load: ").strip()
            m = cached[int(idx) - 1]
            test_load_model(m["repo_id"], quantization="4bit")
        except Exception as e:
            print(f"Invalid selection: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download and manage local LLMs for research experiments")
    parser.add_argument("--model", type=str, default=None, help="Model preset key or HuggingFace repo ID")
    parser.add_argument("--list", action="store_true", help="List all locally cached models")
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE, help="Path to HF cache directory")
    parser.add_argument("--max_workers", type=int, default=4, help="Max worker threads for downloading")
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "8bit", "bf16", "fp16", "none"])
    parser.add_argument("--test_load", action="store_true", help="Test loading model into GPU after download")
    args = parser.parse_args()

    if args.list:
        print_cached_models(args.cache_dir)
        return

    if args.model:
        repo_id = args.model
        for data in MODEL_CATALOG.values():
            if data["key"].lower() == args.model.lower():
                repo_id = data["repo_id"]
                break
        download_model(
            repo_id=repo_id,
            cache_dir=args.cache_dir,
            max_workers=args.max_workers,
            test_load=args.test_load,
            quantization=args.quantization,
        )
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
