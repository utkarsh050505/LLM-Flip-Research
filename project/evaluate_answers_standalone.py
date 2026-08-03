import copy
import json
import os
import argparse
from benchmarking import load_benchmark


def load_jsonl_outputs(file_path):
    """
    Load generated outputs from a JSONL file.

    Args:
        file_path: Path to the JSONL file containing generated outputs
    Returns:
        List of records with questions, model answers, and metadata
    """
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records



def main(args):

    
    benchmark = load_benchmark(args.benchmark, additional_args="")
    metrics = benchmark.evaluate(
        model=None,
        output_file=args.input_file,
        parsed_responses_filename=args.parsed_responses_filename,
        include_difficulty=args.include_difficulty,
        show_progress=True,
    )
    print(json.dumps(metrics, indent=2))



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Evaluate model answers on a benchmark.")
    argparser.add_argument("--benchmark", type=str, required=True, help="Benchmark name or path.")
    argparser.add_argument("--input_file", type=str, required=True, help="Path to the input generations file (JSONL format).")
    argparser.add_argument("--parsed_responses_filename", type=str, default="parsed_responses_only.jsonl")
    argparser.add_argument("--include_difficulty", action="store_true")
    
    args = argparser.parse_args()
    main(args)
