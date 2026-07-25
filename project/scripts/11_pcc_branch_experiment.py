from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backends import BackendError, TransformersBackend, TransformersBackendConfig
from pcc import ExperimentConfig, PCCBranchExperiment, load_experiment_config


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    problem = load_problem(args.problem_file, args.prompt, args.answer)

    backend = build_backend(config)
    print(f"Loading backend={config.backend} model={config.model_id}...")
    backend.load()
    print(f"Capabilities: {asdict(backend.capabilities)}")

    experiment = PCCBranchExperiment(backend, config)
    result = experiment.run(
        problem["prompt"],
        problem["answer"],
        seed=args.seed,
    )

    output_record = persist_result(result, config)
    print_summary(output_record)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a capability-aware same-prefix PCC branch experiment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON config. Defaults are defined in pcc/config.py.",
    )
    parser.add_argument(
        "--problem-file",
        type=Path,
        default=PROJECT_ROOT / "problems" / "problem_000_aya.json",
        help="Problem JSON with 'problem' and 'answer' fields.",
    )
    parser.add_argument("--prompt", default=None, help="Inline prompt override.")
    parser.add_argument("--answer", default=None, help="Ground-truth answer override.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_backend(config: ExperimentConfig):
    if config.backend != "transformers":
        raise BackendError(
            f"Unsupported backend {config.backend!r}. Full PCC branching currently "
            "requires the transformers backend because it needs logits, hidden states, "
            "manual stepping, and KV-cache cloning."
        )
    return TransformersBackend(
        TransformersBackendConfig(
            model_id=config.model_id,
            device_map=config.device_map,
            dtype=config.dtype,
            quantization=config.quantization,
            attn_implementation=config.attn_implementation,
        )
    )


def load_problem(problem_file: Path, prompt: str | None, answer: str | None) -> dict[str, str]:
    if prompt is not None:
        if answer is None:
            raise ValueError("--answer is required when --prompt is used")
        return {"prompt": prompt, "answer": answer}

    data = json.loads(problem_file.read_text(encoding="utf-8"))
    problem_text = data["problem"]
    return {
        "prompt": _prompt_for_problem(problem_text),
        "answer": answer or str(data["answer"]),
    }


def _prompt_for_problem(problem_text: str) -> str:
    lower = problem_text.lower()
    if "\\boxed" in problem_text or "final answer" in lower:
        return problem_text
    return f"Solve step by step and put your final answer in \\boxed{{}}: {problem_text}"


def persist_result(result, config: ExperimentConfig) -> dict:
    output_jsonl = Path(config.output_jsonl)
    if not output_jsonl.is_absolute():
        output_jsonl = PROJECT_ROOT.parent / output_jsonl
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifact_dir = output_jsonl.parent / f"pcc_branch_run_{run_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    branch_records = []
    for branch in result.branches:
        transcript_path = artifact_dir / f"branch_{branch.branch_id:03d}_transcript.txt"
        metrics_path = artifact_dir / f"branch_{branch.branch_id:03d}_metrics.jsonl"
        transcript_path.write_text(branch.transcript, encoding="utf-8")
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            for row in branch.metrics:
                metrics_file.write(json.dumps(row) + "\n")

        branch_records.append({
            "branch_id": branch.branch_id,
            "label": branch.label,
            "raw_answer": branch.raw_answer,
            "normalized_answer": branch.normalized_answer,
            "correct": branch.correct,
            "tokens_generated": branch.tokens_generated,
            "wait_injections": branch.wait_injections,
            "degenerate": branch.degenerate,
            "distinct_boxed_answers": branch.distinct_boxed_answers,
            "transcript_path": str(transcript_path),
            "metrics_path": str(metrics_path),
        })

    fcs_record = {
        "found": result.fcs.found,
        "attempt": result.fcs.attempt,
        "prompt_len": result.fcs.prompt_len,
        "generated_tokens": result.fcs.generated_tokens,
        "prefix_token_len": (
            int(result.fcs.prefix_ids.shape[1]) if result.fcs.prefix_ids is not None else None
        ),
        "reason": result.fcs.reason,
        "prefix_tail": result.fcs.prefix_tail,
    }
    record = {
        "run_id": run_id,
        "model_id": result.model_id,
        "backend": result.backend,
        "problem": result.problem,
        "ground_truth": result.ground_truth,
        "fcs": fcs_record,
        "target_budget": result.target_budget,
        "branches": branch_records,
        "config": result.config,
        "artifact_dir": str(artifact_dir),
        "output_jsonl": str(output_jsonl),
    }
    with output_jsonl.open("a", encoding="utf-8") as runs_file:
        runs_file.write(json.dumps(record) + "\n")
    return record


def print_summary(record: dict) -> None:
    print(f"\nSaved run metadata to {record['output_jsonl']}")
    print(f"Saved branch artifacts to {record['artifact_dir']}")
    if not record["fcs"]["found"]:
        print(f"FCS not found: {record['fcs']['reason']}")
        return

    counts = {}
    for branch in record["branches"]:
        counts[branch["label"]] = counts.get(branch["label"], 0) + 1
    print(
        "FCS found after "
        f"{record['fcs']['generated_tokens']} generated tokens; "
        f"target_budget={record['target_budget']}"
    )
    print(f"Branch labels: {counts}")


if __name__ == "__main__":
    raise SystemExit(main())
