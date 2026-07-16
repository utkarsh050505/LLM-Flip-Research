# configs/experiment_config.py

# Random seed for reproducibility
SEED = 42

# Default generation parameters
GENERATION_PARAMS = {
    "max_new_tokens": 512,
    "temperature": 0.6,
    "top_p": 0.95,
    "do_sample": True,
}

# The test reasoning prompt used to generate traces
DEFAULT_PROMPT = (
    "A box contains 3 red balls and 7 blue balls. We draw two balls at random "
    "without replacement. What is the probability that we draw one red ball and "
    "one blue ball? Explain your reasoning step-by-step."
)

# Prefix dataset settings
DATASET_NAME = "math_word_problems"
