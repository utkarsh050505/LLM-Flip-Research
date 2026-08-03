import argparse
import base64
import json
import multiprocessing as mp
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.parsing import ParsingHelper

DEFAULT_PARSED_RESPONSES_FILENAME = "parsed_responses_difficulty_difficulty_utterance_granularity_1.jsonl"
DEFAULT_REASONING_PROMPT = (
    "Return exactly one compact JSON object. Do not think step by step. "
    "Do not output <think>, analysis, markdown, or prose."
)
PROBE_STUB_PATTERN = re.compile(
    r"Oh,\s+I\s+(?:suddenly|finally)\s+got\s+the\s+answer\s+to\s+the\s+whole\s+problem\."
    r"\s*<answer>\s*"
    r"(?:\\n|\n|\s)*"
    r"###\s*\*\*Final Answer\*\*:\s*"
    r"\\?\[\s*\\?boxed\{",
    flags=re.IGNORECASE,
)

PROBE_SUFFIX_INSTRUCTION = """Important probe artifact:
- Ignore forced final-answer probe suffixes that start like "Oh, I suddenly/finally got the answer..." and lead into "\\boxed{".
- Treat that text as evaluator scaffolding, not as reasoning produced by the model under test.
- Do not classify a sample as a logical error only because this standard probe suffix appears.
- Classify the drift using the substantive reasoning or final-answer change before/around that scaffold."""

COMPACT_OUTPUT_INSTRUCTION = """Output style:
- First character of your response must be "{" and the last character must be "}".
- Return exactly one compact JSON object. No analysis, no markdown, no prose, no preamble.
- Do not discuss contradictions in the prompt. Use the metadata as ground truth for last/final predictions.
- If you need internal reasoning, keep it under 2048 tokens and do not include it in the response.
- Keep went_wrong to one short sentence.
- Use example for one minimal quote/paraphrase from this sample that illustrates the reason."""

TAXONOMY = [
    "visual_hallucination_or_perception",
    "calculation_error",
    "logical_error",
]

TAXONOMY_GUIDE = """Category guide:
- visual_hallucination_or_perception: the new/final trace invents unsupported visual facts or misreads image details, geometry, counts, labels, spatial relations, options, or visible objects.
- calculation_error: the new/final trace makes an arithmetic, algebraic, counting, formula, unit-conversion, or numerical-computation mistake after the last correct prefix.
- logical_error: the new/final trace uses flawed non-numerical reasoning, draws an unsupported conclusion, maps the right reasoning to the wrong option, follows an irrelevant detour, repeats itself into a contradiction, or changes only the final answer/format without a visual or calculation mistake.
"""

PROMPT_VARIANTS = {
    "taxonomy": """Return exactly one compact JSON object and nothing else.
Do not think step by step in the visible response. Do not output analysis, markdown, or prose.
The first character must be "{" and the last character must be "}".

You are analyzing overthinking in a nested difficulty reasoning trace.

The first trace is the LAST prefix where the model's parsed answer was still correct.
The second trace is the FINAL prefix with all retained utterances.

Task:
1. Compare only what changed after the last-correct prefix.
2. Identify the main failure mode introduced by the final/full trace.
3. If an image is provided, use it to decide whether the added suffix hallucinates or misreads visual evidence.
4. Choose the best available category even when the drift is small or ambiguous.
5. Ignore the standard forced final-answer probe suffix; it is instrumentation, not the model's failure reason.

{probe_suffix_instruction}

Allowed categories:
{categories}

{category_guide}

Severity:
- 0 = no real failure or final answer still correct
- 25 = minor drift but failure source is weak/ambiguous
- 50 = clear failure mode
- 75 = strong failure mode that dominates the final trace
- 100 = extreme failure mode, severe hallucination/format collapse/repetition

Return only valid JSON with this exact schema:
{{
  "category": "one_allowed_category",
  "secondary_categories": ["zero_or_more_allowed_categories"],
  "severity": 0_to_100_integer,
  "went_wrong": "short explanation",
  "evidence": "short quote or paraphrase from the added/final trace",
  "example": "minimal quote or paraphrase illustrating the reason", 
  "confidence": 0.0
}}

{compact_output_instruction}

Metadata:
- idx: {idx}
- last_correct_difficulty_idx: {last_true_difficulty_idx}
- final_difficulty_idx: {last_difficulty_idx}
- last_correct_prediction: {last_true_prediction}
- final_prediction: {last_prediction}
- ground_truth: {ground_truth}

Original benchmark question:
```text
{original_question}
```

Last-correct trace:
```text
{last_true_trace}
```

Final/full trace:
```text
{last_trace}
```

New content after last-correct prefix:
```text
{added_suffix}
```""",
    "delta": """Return exactly one compact JSON object and nothing else.
Do not think step by step in the visible response. Do not output analysis, markdown, or prose.
The first character must be "{" and the last character must be "}".

You are a failure-analysis judge. Classify the DELTA from a correct reasoning prefix to the final full trace.

Do not grade the whole answer from scratch. Focus on the new suffix and the change from:
correct prefix prediction = {last_true_prediction}
to final prediction = {last_prediction}

If an image is provided, use it to check whether the suffix invents unsupported visual facts or misreads details.
Ignore the standard forced final-answer probe suffix; it is instrumentation, not the model's failure reason.

{probe_suffix_instruction}

Allowed labels:
{categories}

{category_guide}

Return compact valid JSON:
{{
  "category": "one_allowed_category",
  "secondary_categories": [],
  "severity": 0_to_100_integer,
  "went_wrong": "one sentence",
  "evidence": "brief evidence phrase",
  "example": "minimal quote or paraphrase from this sample",
  "confidence": 0.0
}}

{compact_output_instruction}

Original benchmark question:
```text
{original_question}
```

Correct prefix:
```text
{last_true_trace}
```

Added suffix:
```text
{added_suffix}
```

Final trace:
```text
{last_trace}
```""",
    "minimal": """Return exactly one compact JSON object and nothing else.
Do not think step by step in the visible response. Do not output analysis, markdown, or prose.
The first character must be "{" and the last character must be "}".

Classify why the final reasoning trace is worse than the last correct trace.

Labels: {categories}

{category_guide}

Return only JSON with keys category, secondary_categories and evidence.

If an image is attached, use it to check hallucinated visual evidence and visual misreads.
Ignore the standard forced final-answer probe suffix; it is instrumentation, not the model's failure reason.

{probe_suffix_instruction}

{compact_output_instruction}

original_question: {original_question}

last_correct_prediction: {last_true_prediction}
final_prediction: {last_prediction}
ground_truth: {ground_truth}

LAST CORRECT TRACE:
{last_true_trace}

FINAL TRACE:
{last_trace}

ADDED SUFFIX:
{added_suffix}""",
}


def timestamped_log(message: str):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}|INFO] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Use a vLLM judge to categorize what goes wrong between the last "
            "correct difficulty CoT and the final/all-utterances CoT."
        )
    )
    parser.add_argument("--model", type=str, default="qwen3vl", help="Model name registered in ./modeling.")
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Optional benchmark name registered in ./benchmarking. Required by --use_benchmark_images.",
    )
    parser.add_argument(
        "--additional_benchmark_args",
        type=str,
        default="",
        help="Additional benchmark args, semicolon separated.",
    )
    parser.add_argument(
        "--use_benchmark_images",
        action="store_true",
        help="Load the benchmark and attach the image with the same idx to each judge prompt.",
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to difficulty_generations.jsonl.")
    parser.add_argument(
        "--parsed_responses_file",
        type=str,
        default=None,
        help="Optional path to parsed difficulty responses. Defaults to sibling parsed-responses file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional output JSONL path. Defaults to *_last_true_failure_categories.jsonl.",
    )
    parser.add_argument(
        "--additional_model_args",
        type=str,
        default="",
        help="Additional model args, semicolon separated (e.g. key1=value1;key2=value2).",
    )
    parser.add_argument("--max_tokens", type=int, default=100000, help="Maximum number of generated tokens.")
    parser.add_argument("--max_num_images", type=int, default=1, help="Maximum number of images per prompt.")
    parser.add_argument(
        "--override_reasoning_prompt",
        type=str,
        default=DEFAULT_REASONING_PROMPT,
        help="Override the model reasoning prompt.",
    )
    parser.add_argument("--override_system_prompt", type=str, default=None, help="Optional system prompt override.")
    parser.add_argument(
        "--override_prepend_reasoning_prompt",
        action="store_true",
        help="Whether to prepend the reasoning prompt to the system message.",
    )
    parser.add_argument("--prompt_variant", type=str, default="taxonomy", choices=sorted(PROMPT_VARIANTS))
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=None,
        help="Custom prompt template. Must contain the placeholders used by the built-in templates.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "ddp", "openai"],
        help="Backend to use. Use openai for an OpenAI-compatible remote vLLM server.",
    )
    parser.add_argument(
        "--openai_base_url",
        type=str,
        help="Base URL for --backend openai.",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default="dummy",
        help="API key for --backend openai. vLLM usually ignores this.",
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default=None,
        help="Model id for --backend openai. Defaults to the first model returned by /models.",
    )
    parser.add_argument(
        "--openai_request_timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for OpenAI-compatible HTTP requests.",
    )
    parser.add_argument(
        "--openai_stop",
        type=str,
        default="",
        help="Optional stop strings for --backend openai, separated by semicolons.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--debug", action="store_true", help="Process only a small prefix.")
    parser.add_argument(
        "--include_already_correct",
        action="store_true",
        help="Also categorize samples whose final/all-utterances prediction is correct.",
    )
    parser.add_argument(
        "--print_raw_outputs",
        action="store_true",
        help="Print each raw judge response after every request.",
    )
    return parser.parse_args()


def resolve_paths(args):
    input_path = Path(args.input_file).resolve()
    parsed_path = (
        input_path.parent / DEFAULT_PARSED_RESPONSES_FILENAME
        if args.parsed_responses_file is None
        else Path(args.parsed_responses_file).resolve()
    )
    output_path = (
        parsed_path.with_name(parsed_path.stem + "_last_true_failure_categories" + parsed_path.suffix)
        if args.output_file is None
        else Path(args.output_file).resolve()
    )
    return input_path, parsed_path, output_path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index_by_key(rows: list[dict], source_name: str) -> dict[tuple[int, int], dict]:
    indexed = {}
    for row in rows:
        if "idx" not in row or "difficulty_idx" not in row:
            raise ValueError(f"{source_name} row is missing idx or difficulty_idx: {row}")
        key = (int(row["idx"]), int(row["difficulty_idx"]))
        if key in indexed:
            raise ValueError(f"Duplicate key {key} found in {source_name}.")
        indexed[key] = row
    return indexed


def sanitize_trace(text: str) -> str:
    sanitized = str(text)
    replacements = {
        "<|vision_start|>": "[VISION_START]",
        "<|vision_end|>": "[VISION_END]",
        "<|image_pad|>": "[IMAGE_PAD]",
        "<|video_pad|>": "[VIDEO_PAD]",
        "<|im_start|>": "[CHAT_START]",
        "<|im_end|>": "[CHAT_END]",
    }
    for old, new in replacements.items():
        sanitized = sanitized.replace(old, new)
    sanitized = re.sub(r"<image\d+>", "[IMAGE]", sanitized)
    sanitized = re.sub(r"<video\d+>", "[VIDEO]", sanitized)
    return sanitized


def strip_probe_stub(text: str) -> str:
    return PROBE_STUB_PATTERN.sub("", str(text))


def clean_value(value) -> str:
    if value is None:
        return ""
    return ParsingHelper.clean(str(value))


def parsed_row_is_correct(row: dict) -> bool:
    prediction = clean_value(row.get("prediction", row.get("model_parsed_answer", "")))
    ground_truth = clean_value(row.get("ground_truth", ""))
    if not prediction or not ground_truth:
        return False
    return prediction == ground_truth or prediction in ground_truth


def get_prediction(row: dict) -> str:
    return clean_value(row.get("prediction", row.get("model_parsed_answer", "")))


def extract_added_suffix(prefix_trace: str, final_trace: str) -> str:
    if final_trace.startswith(prefix_trace):
        return final_trace[len(prefix_trace):]

    common_len = 0
    max_common_len = min(len(prefix_trace), len(final_trace))
    while common_len < max_common_len and prefix_trace[common_len] == final_trace[common_len]:
        common_len += 1

    if common_len > 0:
        return final_trace[common_len:]

    return final_trace


def truncate_trace(text: str, max_chars: int = 1200) -> str:
    if not text or len(text) <= max_chars:
        return text
    head_len = 300
    tail_len = max_chars - head_len - 50
    return text[:head_len] + "\n\n... [middle trace omitted for brevity] ...\n\n" + text[-tail_len:]


def build_prompt(args, comparison: dict) -> str:
    template = args.prompt_template or PROMPT_VARIANTS[args.prompt_variant]
    if args.prompt_template is not None:
        required = ["{categories}", "{last_true_trace}", "{last_trace}", "{added_suffix}"]
        for placeholder in required:
            if placeholder not in template:
                raise ValueError(f"Custom prompt template must contain {placeholder}.")

    prompt = template
    values = {
        "{categories}": ", ".join(TAXONOMY),
        "{category_guide}": TAXONOMY_GUIDE,
        "{probe_suffix_instruction}": PROBE_SUFFIX_INSTRUCTION,
        "{compact_output_instruction}": COMPACT_OUTPUT_INSTRUCTION,
        "{idx}": str(comparison["idx"]),
        "{last_true_difficulty_idx}": str(comparison["last_true_difficulty_idx"]),
        "{last_difficulty_idx}": str(comparison["last_difficulty_idx"]),
        "{last_true_prediction}": comparison["last_true_prediction"],
        "{last_prediction}": comparison["last_prediction"],
        "{ground_truth}": comparison["ground_truth"],
        "{original_question}": comparison.get("original_question", ""),
        "{last_true_trace}": truncate_trace(comparison["last_true_trace"]),
        "{last_trace}": truncate_trace(comparison["last_trace"]),
        "{added_suffix}": truncate_trace(comparison["added_suffix"], max_chars=4000),
    }
    for placeholder, value in values.items():
        prompt = prompt.replace(placeholder, value)
    return prompt


def image_to_data_url(image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_openai_messages(args, prompt: str, images: list) -> list[dict]:
    system_prompt = args.override_system_prompt or args.override_reasoning_prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if images:
        content = []
        for image in images[: args.max_num_images]:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(image)},
                }
            )
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})

    return messages


def extract_openai_content(response) -> str:
    if isinstance(response, dict):
        message = response["choices"][0].get("message", {})
        for key in ("content", "reasoning_content", "reasoning"):
            content = message.get(key, "")
            if content:
                return content.strip() if isinstance(content, str) else str(content).strip()
        return ""

    message = response.choices[0].message
    for key in ("content", "reasoning_content", "reasoning"):
        content = getattr(message, key, None)
        if isinstance(content, str) and content:
            return content.strip()
        if isinstance(content, list) and content:
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts).strip()
    return ""


def serialize_openai_response(response) -> dict:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return {"raw_response_repr": repr(response)}


def sanitize_openai_messages_for_output(messages: list[dict]) -> list[dict]:
    sanitized = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, list):
            cleaned_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    cleaned_content.append(
                        {"type": "image_url", "image_url": {"url": "[IMAGE_DATA_URL]"}}
                    )
                else:
                    cleaned_content.append(item)
            copied["content"] = cleaned_content
        sanitized.append(copied)
    return sanitized


class OpenAICompatibleHTTPClient:
    def __init__(self, base_url: str, api_key: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        max_retries = 5
        for attempt in range(max_retries):
            request = urllib.request.Request(
                f"{self.base_url}{path}",
                data=data,
                method=method,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429 and attempt < max_retries - 1:
                    # Extract suggested wait time if present
                    wait_sec = 15.0
                    match = re.search(r"try again in ([0-9.]+)s", body)
                    if match:
                        try:
                            wait_sec = float(match.group(1)) + 1.0
                        except ValueError:
                            pass
                    timestamped_log(f"Rate limited (HTTP 429). Waiting {wait_sec:.1f}s before retrying (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_sec)
                    continue
                raise RuntimeError(f"HTTP {exc.code} from {path}: {body}") from exc

    def list_models(self) -> list[str]:
        response = self._request("GET", "/models")
        return [item["id"] for item in response.get("data", [])]

    def chat_completion(self, **kwargs) -> dict:
        return self._request("POST", "/chat/completions", kwargs)


class OpenAISDKClient:
    def __init__(self, base_url: str, api_key: str, timeout: float):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def list_models(self) -> list[str]:
        return [item.id for item in self.client.models.list().data]

    def chat_completion(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)


def create_openai_compatible_client(base_url: str, api_key: str, timeout: float):
    try:
        return OpenAISDKClient(base_url=base_url, api_key=api_key, timeout=timeout)
    except ModuleNotFoundError:
        timestamped_log("Python package openai is not installed; using urllib HTTP fallback.")
        return OpenAICompatibleHTTPClient(base_url=base_url, api_key=api_key, timeout=timeout)


def generate_openai_judge(client, model_id: str, messages: list[dict], args) -> tuple[str, dict]:
    max_tokens = min(args.max_tokens, 4096) if args.max_tokens else 4096
    request_kwargs = {
        "model": model_id,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    stop_strings = [item for item in args.openai_stop.split(";") if item]
    if stop_strings:
        request_kwargs["stop"] = stop_strings
    response = client.chat_completion(**request_kwargs)
    return extract_openai_content(response), serialize_openai_response(response)


def parse_json_output(raw_output: str) -> dict:
    text = (raw_output or "").strip()
    text = re.sub(r"^<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    for candidate in [text, re.search(r"\{.*\}", text, flags=re.DOTALL).group(0) if re.search(r"\{.*\}", text, flags=re.DOTALL) else None]:
        if not candidate:
            continue
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            try:
                # Sanitize unescaped LaTeX backslashes
                sanitized = re.sub(r'\\(?![/\"\\bfnrtu])', r'\\\\', candidate)
                return json.loads(sanitized, strict=False)
            except Exception:
                pass

    return {
        "category": "other",
        "secondary_categories": [],
        "severity": None,
        "went_wrong": "Could not parse judge JSON.",
        "evidence": "",
        "example": "",
        "confidence": 0.0,
        "parse_error": True,
    }


def normalize_category_result(result: dict) -> dict:
    category = result.get("category", "other")
    if category not in TAXONOMY:
        category = "other"

    secondary = result.get("secondary_categories", [])
    if not isinstance(secondary, list):
        secondary = []
    secondary = [item for item in secondary if item in TAXONOMY and item != category]

    normalized = dict(result)
    normalized["category"] = category
    normalized["secondary_categories"] = secondary

    try:
        normalized["severity"] = max(0, min(100, int(normalized.get("severity", 0))))
    except (TypeError, ValueError):
        normalized["severity"] = None

    try:
        normalized["confidence"] = max(0.0, min(1.0, float(normalized.get("confidence", 0.0))))
    except (TypeError, ValueError):
        normalized["confidence"] = 0.0

    return normalized


def load_benchmark_samples(args) -> dict[int, dict]:
    if not args.use_benchmark_images:
        return {}
    if args.benchmark is None:
        raise ValueError("--benchmark is required when --use_benchmark_images is set.")

    from benchmarking import load_benchmark

    timestamped_log(f"Loading benchmark images from: {args.benchmark}")
    benchmark = load_benchmark(
        args.benchmark,
        additional_args=args.additional_benchmark_args,
    )

    samples_by_idx = {}
    for idx, raw_sample in enumerate(benchmark.dataset):
        processed_sample = benchmark.preprocess_samples([raw_sample])[0]
        samples_by_idx[idx] = processed_sample
    return samples_by_idx


def build_comparisons(
    args,
    difficulty_rows: list[dict],
    parsed_by_key: dict[tuple[int, int], dict],
    benchmark_samples_by_idx: dict[int, dict] | None = None,
) -> list[dict]:
    benchmark_samples_by_idx = benchmark_samples_by_idx or {}
    rows_by_idx = defaultdict(list)
    for row in difficulty_rows:
        rows_by_idx[int(row["idx"])].append(row)

    comparisons = []
    num_final_correct_skipped = 0
    num_never_correct_skipped = 0
    for idx in sorted(rows_by_idx):
        ordered = sorted(rows_by_idx[idx], key=lambda row: int(row["difficulty_idx"]))
        parsed_rows = []
        for difficulty_row in ordered:
            key = (idx, int(difficulty_row["difficulty_idx"]))
            if key not in parsed_by_key:
                raise ValueError(f"Could not find parsed response row for key={key}.")
            parsed_rows.append(parsed_by_key[key])

        correct_positions = [
            pos for pos, parsed_row in enumerate(parsed_rows) if parsed_row_is_correct(parsed_row)
        ]
        if not correct_positions:
            num_never_correct_skipped += 1
            continue

        last_true_pos = correct_positions[-1]
        last_pos = len(ordered) - 1
        last_true_row = ordered[last_true_pos]
        last_row = ordered[last_pos]
        last_true_parsed = parsed_rows[last_true_pos]
        last_parsed = parsed_rows[last_pos]
        final_is_correct = parsed_row_is_correct(last_parsed)

        if final_is_correct and not args.include_already_correct:
            num_final_correct_skipped += 1
            continue

        last_true_trace = strip_probe_stub(sanitize_trace(last_true_row["actual_query"]))
        last_trace = strip_probe_stub(sanitize_trace(last_row["actual_query"]))
        added_suffix = extract_added_suffix(last_true_trace, last_trace)
        benchmark_sample = benchmark_samples_by_idx.get(idx, {})

        comparisons.append(
            {
                "idx": idx,
                "last_true_difficulty_idx": int(last_true_row["difficulty_idx"]),
                "last_difficulty_idx": int(last_row["difficulty_idx"]),
                "last_true_prediction": get_prediction(last_true_parsed),
                "last_prediction": get_prediction(last_parsed),
                "ground_truth": clean_value(last_parsed.get("ground_truth", "")),
                "final_is_correct": final_is_correct,
                "num_difficulty_steps": len(ordered),
                "last_true_trace": last_true_trace,
                "last_trace": last_trace,
                "added_suffix": added_suffix,
                "original_question": benchmark_sample.get(
                    "query",
                    last_row.get("question", last_true_row.get("question", "")),
                ),
                "benchmark_has_image": benchmark_sample.get("decoded_image") is not None,
                "source_question": last_row.get("question"),
                "source_model_output": last_row.get("model_output"),
            }
        )

    timestamped_log(
        "Comparison filtering summary: "
        f"total_idx={len(rows_by_idx)}, "
        f"kept={len(comparisons)}, "
        f"skipped_never_correct={num_never_correct_skipped}, "
        f"skipped_final_still_correct={num_final_correct_skipped}, "
        f"include_already_correct={args.include_already_correct}"
    )

    return comparisons


def categorize_rows(
    args,
    comparisons: list[dict],
    output_path: Path,
    benchmark_samples_by_idx: dict[int, dict] | None = None,
):
    from utils import is_debug_mode, setup_reproducibility_environment
    from utils.ddp import setup_ddp_environment

    if is_debug_mode():
        timestamped_log("Running in DEBUG MODE")

    setup_reproducibility_environment(args.seed)

    if args.backend == "ddp":
        setup_ddp_environment()

    model = None
    openai_client = None
    openai_model_id = None
    if args.backend == "openai":
        openai_client = create_openai_compatible_client(
            base_url=args.openai_base_url,
            api_key=args.openai_api_key,
            timeout=args.openai_request_timeout,
        )
        openai_model_id = args.openai_model
        if openai_model_id is None:
            model_ids = openai_client.list_models()
            if not model_ids:
                raise RuntimeError(f"No models returned by {args.openai_base_url}/models.")
            openai_model_id = model_ids[0]
        timestamped_log(
            f"Using OpenAI-compatible judge model: {openai_model_id} at {args.openai_base_url}"
        )
    else:
        from modeling import load_model

        model = load_model(
            model_name=args.model,
            backend=args.backend,
            max_tokens=args.max_tokens,
            max_num_images=args.max_num_images,
            seed=args.seed,
            additional_args=args.additional_model_args,
            override_system_prompt=args.override_system_prompt,
            override_reasoning_prompt=args.override_reasoning_prompt,
            override_prepend_reasoning_prompt=args.override_prepend_reasoning_prompt,
        )
        model.eval()

    with open(output_path, "w", encoding="utf-8") as f:
        for row_idx, comparison in enumerate(comparisons):
            timestamped_log(
                f"Categorizing {row_idx + 1}/{len(comparisons)} "
                f"(idx={comparison['idx']}, "
                f"last_true={comparison['last_true_difficulty_idx']}, "
                f"last={comparison['last_difficulty_idx']})"
            )
            prompt = build_prompt(args, comparison)
            benchmark_sample = (benchmark_samples_by_idx or {}).get(int(comparison["idx"]), {})
            image = benchmark_sample.get("decoded_image")
            can_use_images = args.backend == "openai" or not getattr(model, "text_only", False)
            images = [image] if args.use_benchmark_images and image is not None and can_use_images else []
            openai_response = None
            if args.backend == "openai":
                messages = build_openai_messages(args, prompt, images)
                timestamped_log(
                    f"Sending OpenAI-compatible judge request "
                    f"(idx={comparison['idx']}, model={openai_model_id}, "
                    f"images={len(images)})"
                )
                raw_output, openai_response = generate_openai_judge(
                    openai_client,
                    openai_model_id,
                    messages,
                    args,
                )
            else:
                output_texts, messages = model.generate_batch(
                    [{"question": prompt, "images": images}],
                    return_inputs=True,
                )
                raw_output = output_texts[0]
            if args.print_raw_outputs:
                print(
                    "\n"
                    f"===== RAW JUDGE OUTPUT idx={comparison['idx']} "
                    f"row={row_idx + 1}/{len(comparisons)} =====\n"
                    f"{raw_output}\n"
                    "===== END RAW JUDGE OUTPUT =====\n",
                    flush=True,
                )
                if args.backend == "openai" and not raw_output:
                    print(
                        "===== FULL OPENAI-COMPATIBLE RESPONSE WITH EMPTY CONTENT =====\n"
                        f"{json.dumps(openai_response, indent=2)}\n"
                        "===== END FULL OPENAI-COMPATIBLE RESPONSE =====\n",
                        flush=True,
                    )
            parsed_result = normalize_category_result(parse_json_output(raw_output))

            output_row = dict(comparison)
            output_row["judge_category"] = parsed_result["category"]
            output_row["judge_secondary_categories"] = parsed_result["secondary_categories"]
            output_row["judge_severity"] = parsed_result["severity"]
            output_row["judge_went_wrong"] = parsed_result.get("went_wrong", "")
            output_row["judge_evidence"] = parsed_result.get("evidence", "")
            output_row["judge_example"] = parsed_result.get("example", "")
            output_row["judge_confidence"] = parsed_result["confidence"]
            output_row["judge_raw_output"] = raw_output
            output_row["judge_parsed_output"] = parsed_result
            output_row["judge_query"] = (
                sanitize_openai_messages_for_output(messages)
                if args.backend == "openai"
                else messages[0]
            )
            output_row["prompt_variant"] = args.prompt_variant
            output_row["judge_used_benchmark_image"] = bool(images)
            output_row["judge_backend"] = args.backend
            if args.backend == "openai":
                output_row["judge_model"] = openai_model_id
                if not raw_output:
                    output_row["judge_raw_response"] = openai_response

            f.write(json.dumps(output_row) + "\n")
            f.flush()

            if args.debug and row_idx >= 30:
                break


def main(args):
    input_path, parsed_path, output_path = resolve_paths(args)
    if not input_path.exists():
        raise FileNotFoundError(f"Input difficulty file not found: {input_path}")
    if not parsed_path.exists():
        raise FileNotFoundError(f"Parsed responses file not found: {parsed_path}")

    timestamped_log(f"Loading difficulty generations from: {input_path}")
    difficulty_rows = load_jsonl(input_path)
    timestamped_log(f"Loading parsed responses from: {parsed_path}")
    parsed_rows = load_jsonl(parsed_path)
    parsed_by_key = index_by_key(parsed_rows, str(parsed_path))
    benchmark_samples_by_idx = load_benchmark_samples(args)

    comparisons = build_comparisons(
        args,
        difficulty_rows,
        parsed_by_key,
        benchmark_samples_by_idx=benchmark_samples_by_idx,
    )
    timestamped_log(f"Built {len(comparisons)} last-true vs final comparisons.")
    timestamped_log(f"Writing categories to: {output_path}")
    categorize_rows(
        args,
        comparisons,
        output_path,
        benchmark_samples_by_idx=benchmark_samples_by_idx,
    )
    timestamped_log("Done.")


if __name__ == "__main__":
    args = parse_args()
    mp.set_start_method("spawn", force=True)
    main(args)
