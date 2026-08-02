import os
from pathlib import Path

# Base paths
PROJECT_DIR = Path(__file__).parent.parent
PROBLEMS_DIR = PROJECT_DIR / "problems"
TRACES_DIR = PROJECT_DIR / "traces"
RESULTS_DIR = PROJECT_DIR / "results"

# Ensure output directories exist
PROBLEMS_DIR.mkdir(parents=True, exist_ok=True)
TRACES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Shared Model Settings
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE = "cuda"

# Stage 1: Natural Traces
N_TRACES_PER_PROBLEM = 8
TRACE_MAX_TOKENS = 8000
TRACE_TEMPERATURE = 0.8
RANDOM_SEED = 42

# Stage 2: Checkpoint Probes
CHECKPOINT_INTERVAL = 400
PROBE_MAX_TOKENS = 30
PROBE_TEMPERATURE = 0.1

# Stage 3: Flip Detection
# (Uses RESULTS_DIR)

# ── Thinking-Past-the-Answer Pipeline Settings ──────────────────────────

# Stage 4: Benchmark Evaluation (eval)
EVAL_MAX_TOKENS = 8000
EVAL_TEMPERATURE = 0.7
EVAL_EXPERIMENT_NAME = "main"
EVAL_SEED = 42
BUDGET_FORCING_PROMPT = "</think>\n\n**Final Answer:**\n\\boxed{"
PATH_BUDGET_FORCING_LABEL = "Final_Answer"

# Stage 5: Difficulty Prefix-Continuation
DIFFICULTY_LEVEL = "utterance"        # "utterance" (split by \\n) or "token"
DIFFICULTY_GRANULARITY = 1            # segments per split unit
DIFFICULTY_MAX_CONTINUATION = 2048    # max tokens per prefix continuation
DIFFICULTY_TEMPERATURE = 0.1          # low temp for deterministic continuation

# Stage 6: Taxonomy Classification
TAXONOMY_MAX_TOKENS = 300
TAXONOMY_TEMPERATURE = 0.1
