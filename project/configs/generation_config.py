# configs/generation_config.py
"""
Generation Configuration

Centralizes all generation-related parameters.
These are defaults — scripts may override per-experiment.
"""

from configs.experiment_config import SEED

# Default generation parameters
TEMPERATURE: float = 0.6
MAX_NEW_TOKENS: int = 512
TOP_P: float = 0.95
DO_SAMPLE: bool = True
DEFAULT_SEED: int = SEED

# Decoder instrumentation (Phase 2)
TOP_K_LOGITS: int = 5                # Number of top logits to record per step
EXTRACT_HIDDEN_STATES: bool = True    # Whether to extract hidden state layers

# Checkpoints (Phase 2.5)
CHECKPOINT_WINDOW: int = 16          # Tokens per reasoning checkpoint

# Trace output directory (outside the repo, per project rules)
TRACE_OUTPUT_DIR: str = "A:\\LLMResearch\\datasets\\traces"

# Default benchmark identifier for development runs
DEFAULT_BENCHMARK: str = "debug"
DEFAULT_PROBLEM_ID: str = "prob_001"
