# Phase 1 — Data Collection Infrastructure

## Summary

Implemented the complete trace collection pipeline:

```
Prompt → Load Model → Generate Reasoning Trace → Save Trace JSON
```

All files are production-quality with type hints, dataclasses, docstrings, logging, and modular architecture.

---

## Files Created / Modified

### Utility Layer (foundation)

| File | Action | Purpose |
|------|--------|---------|
| [\_\_init\_\_.py](file:///a:/LLMResearch/project/src/utils/__init__.py) | **Created** | Package init with re-exports |
| [logger.py](file:///a:/LLMResearch/project/src/utils/logger.py) | **Rewritten** | Centralized logging factory (`setup_logging`, `get_logger`) |
| [seed.py](file:///a:/LLMResearch/project/src/utils/seed.py) | **Rewritten** | Deterministic seeding (Python, NumPy, PyTorch) |
| [io.py](file:///a:/LLMResearch/project/src/utils/io.py) | **Rewritten** | `ensure_directory()` helper |

### Trace Layer (core data structures)

| File | Action | Purpose |
|------|--------|---------|
| [reasoning\_trace.py](file:///a:/LLMResearch/project/src/trace/reasoning_trace.py) | **Modified** | Added `GenerationTiming` dataclass, `prompt_text` + `timestamp` to `TraceMetadata` |
| [serializer.py](file:///a:/LLMResearch/project/src/trace/serializer.py) | **Implemented** | `save_trace()` and `load_trace()` — JSON round-trip |
| [\_\_init\_\_.py](file:///a:/LLMResearch/project/src/trace/__init__.py) | **Created** | Package init with re-exports |

### Adapter Layer (model abstraction)

| File | Action | Purpose |
|------|--------|---------|
| [base\_adapter.py](file:///a:/LLMResearch/project/src/adapters/base_adapter.py) | **Rewritten** | Added `cache_dir`, `seed` param, `is_loaded()`, full docstrings |
| [qwen\_adapter.py](file:///a:/LLMResearch/project/src/adapters/qwen_adapter.py) | **Rewritten** | Logging, token_ids/tokens population, timing, VRAM reporting |
| [\_\_init\_\_.py](file:///a:/LLMResearch/project/src/adapters/__init__.py) | **Rewritten** | Package init with re-exports |

### Generation Layer (orchestration)

| File | Action | Purpose |
|------|--------|---------|
| [generator.py](file:///a:/LLMResearch/project/src/generation/generator.py) | **Implemented** | `TraceGenerator` — orchestrates adapter + serializer |
| [\_\_init\_\_.py](file:///a:/LLMResearch/project/src/generation/__init__.py) | **Created** | Package init |

### Config Layer

| File | Action | Purpose |
|------|--------|---------|
| [generation\_config.py](file:///a:/LLMResearch/project/configs/generation_config.py) | **Implemented** | Trace output dir, generation defaults |

### Scripts

| File | Action | Purpose |
|------|--------|---------|
| [01\_download\_model.py](file:///a:/LLMResearch/project/scripts/01_download_model.py) | **Rewritten** | Robust download with logging, error handling, cache verification |
| [02\_generate\_traces.py](file:///a:/LLMResearch/project/scripts/02_generate_traces.py) | **Implemented** | End-to-end trace generation and save |

---

## Architecture Decisions

### 1. Three-Layer Separation: Adapter → Generator → Serializer

The pipeline is deliberately split into three independent concerns:

- **Adapter** — model-specific loading and generation (QwenAdapter, future DeepSeekAdapter, LlamaAdapter)
- **Generator** — orchestration (coordinates adapter + serializer, manages filenames)
- **Serializer** — format-specific I/O (JSON today, potentially HDF5 or Parquet later)

No layer knows about the internals of another. The generator never calls `model.generate()` directly.

### 2. `GenerationTiming` as a Separate Dataclass

Rather than adding timing fields directly to `GenerationData`, a dedicated `GenerationTiming` dataclass groups `total_seconds`, `tokens_per_second`, and `num_generated_tokens`. This keeps the timing concern isolated and easy to extend (e.g., adding first-token latency, prefill time).

### 3. `prompt_text` and `timestamp` on `TraceMetadata`

Every trace is self-documenting — you can always trace back to exactly which prompt generated it and when. This is critical for reproducibility and debugging when you have thousands of traces.

### 4. JSON Serializer Strips Empty Placeholders

`_clean_for_json()` removes empty lists and `None` values. A Phase 1 trace won't have entropy, hidden states, or checkpoints, so the JSON output is clean and readable rather than cluttered with empty arrays. When Phase 2 populates these fields, they appear automatically.

### 5. Hidden States Excluded from JSON

Tensor data (hidden states) is explicitly zeroed out during JSON serialization. These will be saved as `.pt` files in Phase 2. JSON stays human-readable and small.

### 6. `model.generate()` for Phase 1

As specified, we use HuggingFace's `model.generate()` rather than a custom decoding loop. This verifies the architecture end-to-end. The custom autoregressive loop will replace the `generate()` call inside `QwenAdapter.generate_trace()` — no other code needs to change.

### 7. Filename Convention: `trace_<timestamp>_<uuid>.json`

UTC timestamp gives chronological ordering; 8-char UUID suffix prevents collisions if multiple traces are generated in the same second.

---

## Assumptions

1. **Qwen2.5-1.5B is already cached** in `A:\LLMResearch\hf_cache`. If not, run `01_download_model.py` first.
2. **fp16 loading** (not 4-bit) — Phase 1 uses `torch.float16` with `device_map="auto"`. The 1.5B model fits in 8GB VRAM with fp16. For larger models, 4-bit loading via bitsandbytes can be added to the adapter.
3. **The conda environment `llmresearch`** has `torch`, `transformers`, and `bitsandbytes` installed.
4. **The `TRANSFORMERS_CACHE` warning** is benign — HuggingFace is deprecating the environment variable name. It still works.

---

## Repository Tree (Changed Files)

```
A:\LLMResearch\
├── datasets/
│   └── traces/                    ← trace JSONs saved here
├── hf_cache/                      ← model cache
├── project/
│   ├── configs/
│   │   ├── __init__.py
│   │   ├── experiment_config.py   (unchanged)
│   │   ├── generation_config.py   ← IMPLEMENTED
│   │   ├── model_config.py        (unchanged)
│   │   └── feature_config.py      (empty, future)
│   ├── scripts/
│   │   ├── 01_download_model.py   ← REWRITTEN
│   │   └── 02_generate_traces.py  ← IMPLEMENTED
│   └── src/
│       ├── adapters/
│       │   ├── __init__.py        ← REWRITTEN
│       │   ├── base_adapter.py    ← REWRITTEN
│       │   ├── qwen_adapter.py    ← REWRITTEN
│       │   ├── deepseek_adapter.py (empty, future)
│       │   └── llama_adapter.py    (empty, future)
│       ├── generation/
│       │   ├── __init__.py        ← CREATED
│       │   ├── generator.py       ← IMPLEMENTED
│       │   └── stream_generation.py (empty, future)
│       ├── trace/
│       │   ├── __init__.py        ← CREATED
│       │   ├── reasoning_trace.py ← MODIFIED
│       │   ├── serializer.py      ← IMPLEMENTED
│       │   └── loader.py          (empty, future)
│       └── utils/
│           ├── __init__.py        ← CREATED
│           ├── logger.py          ← IMPLEMENTED
│           ├── seed.py            ← IMPLEMENTED
│           └── io.py              ← IMPLEMENTED
```

---

## Run Instructions (Windows PowerShell)

### Step 0: Activate Environment

```powershell
conda activate llmresearch
```

### Step 1: Download Model (if not already cached)

```powershell
cd A:\LLMResearch\project
python scripts/01_download_model.py
```

> [!NOTE]
> If you've already downloaded Qwen2.5-1.5B-Instruct previously, this will verify the cache and exit quickly. The model is ~3 GB.

**Expected output:**
```
2026-07-17 22:00:00 | INFO     | sbtf | ============================================================
2026-07-17 22:00:00 | INFO     | sbtf | Model Download Script
2026-07-17 22:00:00 | INFO     | sbtf | ============================================================
2026-07-17 22:00:00 | INFO     | sbtf | Active model key : qwen_1.5b
2026-07-17 22:00:00 | INFO     | sbtf | HuggingFace ID   : Qwen/Qwen2.5-1.5B-Instruct
2026-07-17 22:00:00 | INFO     | sbtf | Cache directory  : A:\LLMResearch\hf_cache
2026-07-17 22:00:00 | INFO     | sbtf | Downloading tokenizer ...
2026-07-17 22:00:02 | INFO     | sbtf | Tokenizer ready. Vocab size: 151665
2026-07-17 22:00:02 | INFO     | sbtf | Downloading model weights (this may take several minutes) ...
2026-07-17 22:01:30 | INFO     | sbtf | Model downloaded. Parameters: 1543.71M
2026-07-17 22:01:30 | INFO     | sbtf | Total cache size: 3021.5 MB
2026-07-17 22:01:30 | INFO     | sbtf | ============================================================
2026-07-17 22:01:30 | INFO     | sbtf | Download complete!
2026-07-17 22:01:30 | INFO     | sbtf | ============================================================
```

### Step 2: Generate a Trace

```powershell
cd A:\LLMResearch\project
python scripts/02_generate_traces.py
```

**Expected output:**
```
2026-07-17 22:02:00 | INFO     | sbtf | ============================================================
2026-07-17 22:02:00 | INFO     | sbtf | Trace Generation Script — Phase 1
2026-07-17 22:02:00 | INFO     | sbtf | ============================================================
2026-07-17 22:02:00 | INFO     | sbtf | Model key   : qwen_1.5b
2026-07-17 22:02:00 | INFO     | sbtf | Model ID    : Qwen/Qwen2.5-1.5B-Instruct
2026-07-17 22:02:00 | INFO     | sbtf | Temperature : 0.60
2026-07-17 22:02:00 | INFO     | sbtf | Max tokens  : 512
2026-07-17 22:02:00 | INFO     | sbtf | Seed        : 42
2026-07-17 22:02:00 | INFO     | sbtf | Output dir  : A:\LLMResearch\datasets\traces
...
2026-07-17 22:02:05 | INFO     | sbtf | Loading model Qwen/Qwen2.5-1.5B-Instruct (fp16, device_map=auto) ...
2026-07-17 22:02:12 | INFO     | sbtf | GPU memory — allocated: 2.85 GB, reserved: 2.92 GB
2026-07-17 22:02:12 | INFO     | sbtf | Model loaded successfully: Qwen/Qwen2.5-1.5B-Instruct
...
2026-07-17 22:02:12 | INFO     | sbtf | Generating trace for problem 'prob_001' ...
2026-07-17 22:02:12 | INFO     | sbtf | Prompt length: 68 tokens
2026-07-17 22:02:25 | INFO     | sbtf | Generated 312 tokens in 12.54s (24.9 tok/s)
2026-07-17 22:02:25 | INFO     | sbtf | Trace saved to ... (8.2 KB)
...
2026-07-17 22:02:25 | INFO     | sbtf | ============================================================
2026-07-17 22:02:25 | INFO     | sbtf | RESULTS
2026-07-17 22:02:25 | INFO     | sbtf | ============================================================
2026-07-17 22:02:25 | INFO     | sbtf | Output file     : A:\LLMResearch\datasets\traces\trace_20260717_163225_a1b2c3d4.json
2026-07-17 22:02:25 | INFO     | sbtf | Tokens generated: 312
2026-07-17 22:02:25 | INFO     | sbtf | Time elapsed    : 12.54 s
2026-07-17 22:02:25 | INFO     | sbtf | Throughput      : 24.9 tok/s
...
2026-07-17 22:02:25 | INFO     | sbtf | Trace generation complete!
```

---

## Expected JSON Output

The saved trace file at `datasets/traces/trace_<timestamp>_<uuid>.json` will look like:

```json
{
  "metadata": {
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "benchmark": "debug",
    "problem_id": "prob_001",
    "prompt_text": "A box contains 3 red balls and 7 blue balls. We draw two balls at random without replacement. What is the probability that we draw one red ball and one blue ball? Explain your reasoning step-by-step.",
    "temperature": 0.6,
    "max_new_tokens": 512,
    "seed": 42,
    "timestamp": "2026-07-17T16:32:25.123456+00:00"
  },
  "generation": {
    "reasoning_text": "To solve this problem, I need to find the probability of drawing one red ball and one blue ball...",
    "token_ids": [1271, 11625, 419, ...],
    "tokens": ["To", " solve", " this", ...],
    "timing": {
      "total_seconds": 12.54,
      "tokens_per_second": 24.88,
      "num_generated_tokens": 312
    }
  }
}
```

> [!NOTE]
> Empty placeholder fields (`entropy`, `checkpoints`, `latent`, `events`, `outcome`) are automatically stripped from the JSON by the serializer. They will appear when populated in later phases.

---

## Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'torch'` | Wrong Python environment | Run `conda activate llmresearch` first |
| `ModuleNotFoundError: No module named 'src'` | Wrong working directory | `cd A:\LLMResearch\project` before running scripts |
| `CUDA out of memory` | Other processes using VRAM | Close other GPU apps, or restart Python |
| `OSError: ... does not appear to have a file named config.json` | Model not downloaded | Run `python scripts/01_download_model.py` first |
| `FutureWarning: Using TRANSFORMERS_CACHE is deprecated` | Benign deprecation warning | Safe to ignore; will work fine |
| `RuntimeError: Model not loaded` | Called generate before setup | Ensure `generator.setup()` is called before `generate_and_save()` |

---

## Verification Results

- ✅ All 15 files compile without syntax errors
- ✅ All imports resolve correctly in the `llmresearch` conda environment
- ✅ Serializer round-trip test passed (`save_trace → load_trace → assert fields match`)
