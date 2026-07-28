# PCC Branch Runner Walkthrough

This document explains the capability-aware PCC branch runner added on branch
`feature/pcc-backend-architecture`.

The goal of this change is to separate model access from PCC experiment logic.
The old scripts assumed every local model could expose Hugging Face-style
internals. The new runner makes those assumptions explicit before it tries to
run a same-prefix PCC experiment.

## What Changed

### 1. Backend capability contract

Files:

- `project/backends/base.py`
- `project/backends/transformers_backend.py`
- `project/backends/__init__.py`

`BackendCapabilities` declares whether a backend supports:

- logits
- hidden states
- KV-cache access
- manual token stepping
- chat templates

Full PCC branch experiments require logits, hidden states, KV-cache access, and
manual stepping. Text-only local LLM APIs should not claim full support unless
they expose those internals.

`TransformersBackend` is the first concrete implementation. It wraps:

- `AutoTokenizer`
- `AutoModelForCausalLM`
- optional 4-bit or 8-bit BitsAndBytes loading
- manual prefill and one-token stepping
- cache cloning

### 2. PCC configuration model

Files:

- `project/pcc/config.py`
- `project/configs/pcc_experiment.example.json`

`ExperimentConfig` centralizes the runner settings that were previously spread
across script constants:

- model ID
- backend type
- device map
- dtype
- quantization
- FCS search budget
- branch count
- branch temperature
- budget forcing settings
- reasoning format
- output path

The example config is ready to run with the default DeepSeek R1 distilled Qwen
model, assuming the local environment has the required ML dependencies.

### 3. Answer extraction helpers

File:

- `project/pcc/answer.py`

This module handles provisional answer parsing and normalization:

- extracts the last `\boxed{...}` answer
- supports `FINAL ANSWER: ...`
- strips common LaTeX formatting such as `\text{...}`
- normalizes simple fractions such as `\frac{a}{b}` to `a/b`
- deliberately ignores generic reasoning phrases such as `the answer is ...`
  when assigning branch outcome labels

This is still a heuristic verifier. It is suitable for exploratory PCC runs,
not final mathematical evaluation.

### 4. Reusable PCC branch experiment

File:

- `project/pcc/branching.py`

`PCCBranchExperiment` owns the core experiment flow:

1. Generate from the prompt token by token.
2. Detect the first-correct-solution boundary before reasoning closes.
3. Rebuild the same prefix for each branch.
4. Continue from the identical prefix with budget forcing.
5. Collect entropy, top-2 margin, JSD, and hidden-state movement metrics.
6. Label each branch as `STABLE_CORRECT`, `PCC`, `NO_FINAL_ANSWER`, or
   `DEGENERATE`.

The branch runner intentionally checks backend capabilities before starting.
If a backend cannot provide the required internals, the experiment fails early
with an explicit error instead of silently producing invalid PCC data.

### 5. Persisted command-line runner

File:

- `project/scripts/11_pcc_branch_experiment.py`

This script loads a problem, config, and backend, then persists structured run
artifacts. It is the preferred path for new same-prefix PCC branch experiments.

### 6. README update

File:

- `README.md`

The README now links the new runner, explains why full PCC branching needs a
Transformers-style backend, and describes the output files.

## Install Requirements

The repository still does not include a lock file or `requirements.txt`. Install
the ML stack manually for the target machine.

Typical setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch transformers accelerate bitsandbytes
```

Choose the PyTorch build that matches the machine and CUDA version.

## Step-by-Step Run

Run commands from the repository root.

### Step 1: Check the active branch

```bash
git status --short --branch
```

Expected output:

```text
## feature/pcc-backend-architecture...origin/feature/pcc-backend-architecture
```

The status should not show unexpected local changes before starting a run.

### Step 2: Inspect the default config

```bash
cat project/configs/pcc_experiment.example.json
```

Expected shape:

```json
{
  "backend": "transformers",
  "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
  "device_map": "auto",
  "dtype": "bfloat16",
  "num_branches": 6,
  "output_jsonl": "project/results/pcc_branch_runs.jsonl"
}
```

The real file contains additional budget, sampling, and reasoning-format
settings.

### Step 3: Verify the CLI is available

```bash
python3 project/scripts/11_pcc_branch_experiment.py --help
```

Expected output:

```text
usage: 11_pcc_branch_experiment.py [-h] [--config CONFIG]
                                   [--problem-file PROBLEM_FILE]
                                   [--prompt PROMPT] [--answer ANSWER]
                                   [--seed SEED]

Run a capability-aware same-prefix PCC branch experiment.
```

If this command fails before model loading, fix the Python import path or syntax
issue first.

### Step 4: Run the default problem

```bash
python3 project/scripts/11_pcc_branch_experiment.py \
  --config project/configs/pcc_experiment.example.json \
  --problem-file project/problems/problem_000_aya.json \
  --seed 42
```

Expected early console output:

```text
Loading backend=transformers model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B...
Capabilities: {'supports_logits': True, 'supports_hidden_states': True, 'supports_kv_cache': True, 'supports_manual_step': True, 'supports_chat_template': True}
```

If the model and dependencies are available, the script then searches for the
first-correct-solution boundary and branches from that prefix.

### Step 5: Interpret successful output

If the FCS boundary is found, the end of the run should look like:

```text
Saved run metadata to /absolute/path/project/results/pcc_branch_runs.jsonl
Saved branch artifacts to /absolute/path/project/results/pcc_branch_run_YYYYMMDDTHHMMSSffffffZ
FCS found after 1234 generated tokens; target_budget=3702
Branch labels: {'STABLE_CORRECT': 4, 'PCC': 1, 'NO_FINAL_ANSWER': 1}
```

The exact token counts and label counts are stochastic and depend on the model,
seed, hardware, and sampling settings.

### Step 6: Interpret no-FCS output

If the model does not derive the correct answer before reasoning closes, the
run should end like:

```text
Saved run metadata to /absolute/path/project/results/pcc_branch_runs.jsonl
Saved branch artifacts to /absolute/path/project/results/pcc_branch_run_YYYYMMDDTHHMMSSffffffZ
FCS not found: no correct answer found before reasoning close or token ceiling
```

This is not a crash. It means the selected problem/model pair is not useful for
same-prefix PCC branching under the current settings.

## Output Files

### Run index

Path:

```text
project/results/pcc_branch_runs.jsonl
```

One JSON object is appended per experiment run. Important fields:

- `run_id`
- `model_id`
- `backend`
- `problem`
- `ground_truth`
- `fcs`
- `target_budget`
- `branches`
- `config`
- `artifact_dir`

### Artifact directory

Path pattern:

```text
project/results/pcc_branch_run_YYYYMMDDTHHMMSSffffffZ/
```

Each run gets a timestamped directory with microseconds to avoid same-second
collisions.

### Branch transcripts

Path pattern:

```text
project/results/pcc_branch_run_*/branch_000_transcript.txt
```

Each transcript contains the full prefix plus the generated branch continuation.
Use this file to inspect whether a `PCC` label is semantically plausible.

### Branch metrics

Path pattern:

```text
project/results/pcc_branch_run_*/branch_000_metrics.jsonl
```

Each row represents one generated branch token. Expected fields:

```json
{
  "step": 0,
  "entropy": 1.23,
  "top2_margin": 0.45,
  "jsd_vs_prev": null,
  "l2_early": null,
  "cos_early": null,
  "l2_mid": null,
  "cos_mid": null,
  "l2_late": null,
  "cos_late": null
}
```

After the first token, `jsd_vs_prev`, `l2_*`, and `cos_*` should become numeric
when hidden states are available.

## Branch Labels

`STABLE_CORRECT` means the branch produced a final answer and the normalized
answer matched the ground truth.

`PCC` means the branch started from a prefix where the correct answer had
already appeared, but the branch's final extracted answer did not match the
ground truth.

`NO_FINAL_ANSWER` means no final answer could be extracted from the branch.
For branch labels, extraction requires a declared final answer: either a closed
`\boxed{...}` expression or an explicit `FINAL ANSWER:` line. Generic reasoning
text such as `the answer is ...` is not enough because it often appears before
the branch has actually concluded.

`DEGENERATE` means the branch entered a repeated-answer pattern under budget
forcing and was stopped as an artifact rather than treated as useful PCC data.

## Label Correction Note

Earlier output artifacts used a looser fallback that treated generic reasoning
phrases such as `the answer is 35280 vs 30240, which is conflicting` as final
answers. That created false `PCC` labels when a branch had not actually
concluded.

The stricter rule is:

- use `STABLE_CORRECT` only when a declared final answer exists and matches the
  ground truth;
- use `PCC` only when a declared final answer exists and does not match the
  ground truth;
- use `NO_FINAL_ANSWER` when the branch contains no closed `\boxed{...}` and no
  explicit `FINAL ANSWER:` line, even if unfinished reasoning contains phrases
  like `the answer is ...`.

After applying this rule to `pcc_branch_run_20260725T072315064627Z`, the branch
that was previously labeled `PCC` is correctly categorized as
`NO_FINAL_ANSWER`.

## Capability Rules

Full PCC branching requires:

- next-token logits
- hidden states
- KV-cache access
- cloneable cache state
- manual one-token stepping

Backends such as simple HTTP text-generation servers, Ollama-style APIs, or
other text-only local LLM runtimes should be marked as unsupported for full PCC
branching unless they expose those internals.

Text-only backends can still be useful for separate experiments, but they should
not produce branch-matched PCC metrics.

## Expected Failure Modes

### Missing ML dependencies

Example:

```text
BackendError: TransformersBackend requires torch and transformers to be installed.
```

Fix by installing the ML stack in the active Python environment.

### Unsupported backend

Example:

```text
BackendError: Unsupported backend 'ollama'. Full PCC branching currently requires the transformers backend...
```

This is intentional. Add a backend only after its capability flags accurately
reflect what the runtime exposes.

### Model does not emit visible reasoning markers

The default config uses DeepSeek R1-style `</think>` reasoning closure. If a
model does not emit visible reasoning text, same-prefix FCS detection may never
find a usable prefix.

For such models, set a different `reasoning_format` or treat the run as a
different, lower-confidence experiment.

### No PCC branches appear

This can be a valid result. It may mean:

- the problem is not PCC-prone for the selected model
- the model rarely reaches the correct answer early
- branch temperature is too low
- budget forcing is too weak
- answer extraction missed the final answer format

Inspect the branch transcripts before changing labels or drawing conclusions.

## Verification Commands

These checks were used before pushing the implementation:

```bash
python3 -m py_compile $(rg --files -g '*.py')
python3 project/scripts/11_pcc_branch_experiment.py --help
git diff --check
```

Helper-level checks:

```bash
python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path('project').resolve()))

from dataclasses import asdict
from pcc.answer import answer_appears, extract_final_answer, normalize_answer, evaluate_answer
from pcc.config import ExperimentConfig, load_experiment_config

assert extract_final_answer('x \\\\boxed{204}') == '204'
assert extract_final_answer('FINAL ANSWER: 282240') == '282240'
assert normalize_answer('\\\\text{mph} 45') == '45'
assert evaluate_answer('therefore \\\\boxed{204}', '204').correct is True
assert answer_appears('so the answer is 204', '204') is True

cfg = load_experiment_config('project/configs/pcc_experiment.example.json')
assert cfg.backend == 'transformers'
assert cfg.reasoning_format.think_close == '</think>'
assert asdict(ExperimentConfig())['reasoning_format']['name'] == 'deepseek_r1'
print('pcc helper checks PASS')
PY
```

Expected output:

```text
pcc helper checks PASS
```

Full model-backed generation was not verified in the Codex workspace because
the available Python environment did not have `torch` installed.

## Walkthrough Summary

Use this sequence for a real PCC run:

1. Install `torch`, `transformers`, `accelerate`, and optional `bitsandbytes`.
2. Confirm the branch is `feature/pcc-backend-architecture`.
3. Review `project/configs/pcc_experiment.example.json`.
4. Run `python3 project/scripts/11_pcc_branch_experiment.py --help`.
5. Run `python3 project/scripts/11_pcc_branch_experiment.py --config project/configs/pcc_experiment.example.json --problem-file project/problems/problem_000_aya.json`.
6. Read `project/results/pcc_branch_runs.jsonl`.
7. Inspect each `branch_*_transcript.txt` before trusting `PCC` labels.
8. Use `branch_*_metrics.jsonl` for token-level feature analysis.
