import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.answer_extractor import AnswerExtractionPipeline


def atomic_replace_jsonl(records, output_file):
    output_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(output_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".extract_answers_", suffix=".jsonl", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        shutil.move(tmp_path, output_file)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main(args):
    pipeline = AnswerExtractionPipeline(
        model_name=args.answer_extraction_model,
        llm_endpoint=args.llm_endpoint,
        api_mode=args.api_mode,
    )

    records = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            record = json.loads(line)
            model_answer = record["model_output"]
            print(f"Extracting answer {idx + 1}", flush=True)
            record["model_parsed_answer"] = pipeline.extract_answer(
                model_answer=model_answer,
            )
            record["answer_extraction_model"] = args.answer_extraction_model
            records.append(record)

            if args.limit is not None and idx + 1 >= args.limit:
                break

    atomic_replace_jsonl(records, args.output_file)
    print(f"Wrote extracted answers to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract answers for a JSONL generations file.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--answer_extraction_model", required=True)
    parser.add_argument("--llm_endpoint", required=True, help="OpenAI-compatible host:port, e.g. localhost:8026")
    parser.add_argument("--api_mode", choices=["chat", "completion"], default="chat")
    parser.add_argument("--limit", type=int, default=None)
    main(parser.parse_args())
