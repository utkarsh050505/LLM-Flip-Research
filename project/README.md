# Thinking Past the Answer

1. run benchmark generation and evaluation with `eval.py`;
2. extract final answers from generated JSONL files;
3. run difficulty-prefix continuation experiments with `difficulty.py`;
4. extract and re-evaluate difficulty generations;
5. generate the overthinking failure taxonomy with an LLM judge.

## Repository Layout

- `eval.py`: generate benchmark answers and compute benchmark metrics.
- `difficulty.py`: replay prefixes of a generated reasoning trace, append a budget-forcing prompt, and generate continuations for difficulty analysis.
- `evaluate_answers_standalone.py`: evaluate a JSONL generations file after answer extraction.
- `scripts/python/extract_answers_jsonl.py`: run LLM-based answer extraction over a JSONL file.
- `taxonomy/categorize_overthinking_from_last_true.py`: generate the failure-mode taxonomy labels from difficulty outputs.
- `taxonomy/summarize_overthinking_categories.py`: summarize taxonomy JSONL outputs into JSON/CSV aggregates.
- `benchmarking/`: benchmark adapters.
- `modeling/`: model adapters.
- `evaluation/`: answer parsing and answer extraction utilities.
- `utils/`: logging, reproducibility, experiment folder, and DDP utilities.

## Installation

Use Python 3.10+ in a fresh virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some models require recent `transformers`, `vllm`, CUDA, and enough GPU memory. The scripts download model weights and datasets from Hugging Face through the normal `transformers` and `datasets` caches.

## Available Models and Benchmarks

Print the registered models and benchmarks:

```bash
python eval.py --show_models_and_benchmarks --model NONE --benchmark NONE
```

Registered model names include:

- `dualmind_vlm`
- `interns1`
- `mm_eureka`
- `qwen2_5vl`
- `qwen3`
- `qwen3_5`
- `vl_rethinker`
- `r1_vl`
- `thinklite_vl`
- `vl_rethinker`

Registered benchmark names include:

- `ai2d`
- `aime2025`
- `gpqa`
- `mathverse`
- `mathvision`
- `mathvista`
- `mmstar`
- `thinktrain`
- `vmcbench`

## 1. Run Evaluation

This command generates `generations.jsonl`, evaluates it, and writes `results.json` under:

`OUTPUT/EXPERIMENT_NAME/MODEL/BENCHMARK/seed_SEED/budget_prompt_LABEL/`

```bash
python eval.py \
  --model vl_rethinker \
  --benchmark mathvista \
  --backend vllm \
  --output ./output \
  --experiment_name main \
  --seed 42 \
  --max_tokens 8196 \
  --path_budget_forcing_prompt "Oh, I suddenly got the answer to the whole problem.
<answer> ### **Final Answer**: \[ boxed{:"
```

To use LLM-based answer extraction during evaluation, first start a local OpenAI-compatible vLLM server for the extractor model:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host localhost \
  --port 8000 \
  --trust-remote-code
```

Then add answer extraction to the evaluation command:

```bash
python eval.py \
  --model vl_rethinker \
  --benchmark mathvista \
  --backend vllm \
  --output ./output \
  --experiment_name main \
  --seed 42 \
  --max_tokens 8196 \
  --path_budget_forcing_prompt "Oh, I suddenly got the answer to the whole problem.
<answer> ### **Final Answer**: \[ boxed{" \
  --answer_extraction_with_llm \
  --answer_extraction_model Qwen/Qwen3-4B-Instruct-2507 \
  --llm_endpoint localhost:8000
```

## 2. Extract Answers After Generation

If generation was run without LLM extraction, extract answers from an existing `generations.jsonl`:

```bash
python scripts/python/extract_answers_jsonl.py \
  --input_file ./output/main/vl_rethinker/mathvista/seed_42/budget_prompt_Final_Answer/generations.jsonl \
  --output_file ./output/main/vl_rethinker/mathvista/seed_42/budget_prompt_Final_Answer/generations.jsonl \
  --answer_extraction_model Qwen/Qwen3-4B-Instruct-2507 \
  --llm_endpoint localhost:8000 \
  --api_mode chat
```

Then re-evaluate the extracted file:

```bash
python evaluate_answers_standalone.py \
  --benchmark mathvista \
  --input_file ./output/main/vl_rethinker/mathvista/seed_42/budget_prompt_Final_Answer/generations.jsonl \
  --parsed_responses_filename parsed_responses_only.jsonl
```

## 3. Run Difficulty Generation

Run `difficulty.py` after `eval.py` has produced the source `generations.jsonl`. The script currently supports the `vllm` backend.

```bash
python difficulty.py \
  --model vl_rethinker \
  --benchmark mathvista \
  --backend vllm \
  --output ./output \
  --experiment_name main \
  --seed 42 \
  --max_tokens 2048 \
  --difficulty_level utterance \
  --granularity 1 \
  --budget_forcing_prompt "Final Answer: " \
  --path_budget_forcing_prompt "Final Answer: " \
  --llm_endpoint unset
```

This writes `difficulty_generations.jsonl` under the matching run directory. If the input evaluation run path included answer extraction in its folder name, pass `--path_include_answer_extraction_model` and the same `--answer_extraction_model`.

## 4. Extract and Evaluate Difficulty Answers

Extract answers for `difficulty_generations.jsonl`:

```bash
python scripts/python/extract_answers_jsonl.py \
  --input_file ./output/main/vl_rethinker/mathvista/seed_42/budget_prompt_Final_Answer/difficulty_generations.jsonl \
  --output_file ./output/main/vl_rethinker/mathvista/seed_42/budget_prompt_Final_Answer/difficulty_generations.jsonl \
  --answer_extraction_model Qwen/Qwen3-4B-Instruct-2507 \
  --llm_endpoint localhost:8000 \
  --api_mode chat
```

Then evaluate with difficulty metadata:

```bash
python evaluate_answers_standalone.py \
  --benchmark mathvista \
  --input_file ./output/main/vl_rethinker/mathvista/seed_42/budget_prompt_Final_Answer/difficulty_generations.jsonl \
  --parsed_responses_filename parsed_responses_difficulty_utterance_granularity_1.jsonl \
  --include_difficulty
```

The parsed responses file is used by the taxonomy script to find the last correct prefix for each example.

## 5. Generate the Taxonomy

The taxonomy is produced by comparing the last correct difficulty prefix with the final/full difficulty trace. The built-in taxonomy labels are:

- `visual_hallucination_or_perception`
- `calculation_error`
- `logical_error`

Run the taxonomy judge with an OpenAI-compatible endpoint:

```bash
python baseline/categorize_overthinking_from_last_true.py \
  --backend openai \
  --openai_base_url http://localhost:8000/v1 \
  --openai_api_key dummy \
  --input_file ./output/main/vl_rethinker/mathvista/seed_42/budget_prompt_Final_Answer/difficulty_generations.jsonl \
  --parsed_responses_file ./output/main/vl_rethinker/mathvista/seed_42/budget_prompt_Final_Answer/parsed_responses_difficulty_utterance_granularity_1.jsonl \
  --output_file ./output/main/vl_rethinker/mathvista/seed_42/budget_prompt_Final_Answer/failure_taxonomy.jsonl \
  --prompt_variant taxonomy \
  --max_tokens 4096
```

To attach benchmark images to taxonomy judge prompts, add:

```bash
  --benchmark mathvista --use_benchmark_images
```

Summarize taxonomy labels:

```bash
python baseline/summarize_overthinking_categories.py \
  --input_file ./output/main/vl_rethinker/mathvista/seed_42/budget_prompt_Final_Answer/failure_taxonomy.jsonl \
  --output_json ./output/main/vl_rethinker/mathvista/seed_42/budget_prompt_Final_Answer/failure_taxonomy_summary.json \
  --output_csv ./output/main/vl_rethinker/mathvista/seed_42/budget_prompt_Final_Answer/failure_taxonomy_rows.csv
```

## Notes

- `llm_endpoint` in `eval.py`, `difficulty.py`, and `extract_answers_jsonl.py` expects a host and port such as `localhost:8000`; the code constructs `http://HOST:PORT/v1`.
- `baseline/categorize_overthinking_from_last_true.py --openai_base_url` expects the full OpenAI-compatible base URL, such as `http://localhost:8000/v1`.
- Set `CUDA_VISIBLE_DEVICES` before running vLLM if you want to control tensor parallelism.
- Set `DEBUG_MODE=1` to process a small prefix of each benchmark.
