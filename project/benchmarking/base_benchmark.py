import json
import os
import re
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
from utils.custom_logging import Logger
from utils import is_debug_mode
from evaluation.parsing import ParsingHelper
try:
    from rich.progress import track
except ImportError:
    try:
        from tqdm import tqdm as track
    except ImportError:
        def track(iterable, description=""):
            return iterable

if TYPE_CHECKING:
    from modeling.base_model import BaseModel

class BaseBenchmark(ABC):
    def __init__(self, 
                 benchmark_name: str,
                 additional_args: Optional[str] = None
                 ):
        self.benchmark_name = benchmark_name
        
        if additional_args:
            self.parse_additional_args(additional_args)
        self.load_benchmark()
    
    @classmethod                      
    def get_url(cls) -> str:
        if getattr(cls, "get_hf_name", None) is not None:
            return cls.get_hf_name()
        else:
            return "NotApplicable"
    
    @abstractmethod
    def preprocess_samples(self, samples: list[dict]) -> list[dict]:
        """Convert a benchmark document to text input for the model.
        
        Here the sample has to contain at least the following fields:
        - "query": The main question or prompt to be answered.
        - "decoded_image": The image data associated with the query.
        - "pid": The unique identifier for the sample.
        - "answer": The ground truth answer for evaluation.
        - "choices": (Optional) A list of multiple-choice options if applicable.
        """
        pass

    def parse_additional_args(self, additional_args: str):
        """Parse additional arguments specific to the benchmark."""
        parsed_args = {v[0]: v[1] for v in (arg.split('=') for arg in additional_args.split(';') if additional_args != "")}
        
        for key, value in parsed_args.items():
            setattr(self, key, value)

    def load_benchmark(self):
        """Load the benchmark dataset here."""
        pass
        
    def generate_outputs(self, model: "BaseModel", output_file: Optional[str], limit: Optional[int] = None) -> list[dict[str, str]]:
        total_len = len(self.dataset)
        max_items = total_len
        if limit is not None and limit > 0:
            max_items = min(limit, total_len)
        elif is_debug_mode():
            max_items = min(30, total_len)

        with open(output_file, "w", encoding="utf-8") as f:
            for i in range(max_items):
                sample = self.dataset[i]
                Logger.info(f"Processing example {i+1}/{max_items}")

                processed_sample = self.preprocess_samples([sample])[0]
                images = [] if processed_sample["decoded_image"] is None else [processed_sample["decoded_image"]]

                # TODO: Batch not supported yet
                output_texts, messages = model.generate_batch([{"question": processed_sample["query"], 
                                                                "images": images}], 
                                                                return_inputs=True)

                # ---- SAVE OUTPUT ----
                record = {
                    "idx": i,
                    "pid": processed_sample["pid"],
                    "question": processed_sample["query"],
                    "choices": processed_sample.get("choices", []),
                    "actual_query": messages[0],
                    "model_output": output_texts[0],
                    "ground_truth": processed_sample["answer"],
                }

                f.write(json.dumps(record) + "\n")
                f.flush()
    
    def evaluate(self, model: "BaseModel" = None, output_file: str = None, parsed_responses_filename: str = "parsed_responses_only.jsonl", include_difficulty: bool = False, show_progress: bool = False) -> dict[str, float]:
        """Evaluate the model on the benchmark and return metrics."""
        correct = 0
        total = 0

        choices_map = ["a", "b", "c", "d", "e", "f", "g", "h", "i", 
                        "j", "k", "l", "m", "n", "o", "p", "q", "r", 
                        "s", "t", "u", "v", "w", "x", "y", "z"]

        response_list = []

        with open(output_file, "r") as f:

            if show_progress:
                total_lines = sum(1 for _ in f)
                f.seek(0)
                records_iter = track(f, total=total_lines)
            else:
                records_iter = f
            
            for line in records_iter:

                record = json.loads(line)

                out = record["model_output"]
                pred = record["model_parsed_answer"] if "model_parsed_answer" in record else None
                gt = record["ground_truth"]

                if pred is None:
                    if model is not None:
                        pred = model.normalize_answer(out, clean_only=False)
                        parsed_gt = model.normalize_answer(gt, clean_only=True)
                    else:
                        pred = ParsingHelper.extract_last_boxed(out)
                        if not pred:
                            # fallback: match the last number or token
                            match = re.findall(r"[-+]?\d*\.\d+|\d+", out)
                            pred = match[-1] if match else ""
                        parsed_gt = gt

                pred = ParsingHelper.clean(pred)
                parsed_gt = ParsingHelper.clean(parsed_gt if 'parsed_gt' in locals() else gt)

                if record.get("choices") and pred != parsed_gt:
                    choices = record["choices"]
                    ground_truth = ParsingHelper.clean(record["ground_truth"])
                    choices = [ParsingHelper.clean(choice) for choice in choices]
                    if ground_truth in choices:
                        correct_idx = choices.index(ground_truth)
                        correct_choice = choices_map[correct_idx]
                        parsed_gt = ParsingHelper.clean(correct_choice)

                if pred == parsed_gt or (parsed_gt and parsed_gt in pred): 
                    correct += 1

                total += 1

                if include_difficulty:
                    diff_idx = record.get("difficulty_idx")
                    gran = record.get("granularity")
                    budget = (diff_idx * gran) if (diff_idx is not None and gran is not None) else None
                    response_list.append({
                        "idx": record["idx"],
                        "prediction": pred,
                        "ground_truth": parsed_gt,
                        "difficulty_idx": diff_idx,
                        "granularity": gran,
                        "budget": budget,
                    })
                else:
                    response_list.append({
                        "idx": record["idx"],
                        "prediction": pred,
                        "ground_truth": parsed_gt,
                    })

                

        accuracy = correct / total * 100

        parsed_responses_path = os.path.join(os.path.dirname(output_file), parsed_responses_filename)
        with open(parsed_responses_path, "w", encoding="utf-8") as f:
            for response in response_list:
                f.write(json.dumps(response) + "\n")

        return {"accuracy": accuracy}
