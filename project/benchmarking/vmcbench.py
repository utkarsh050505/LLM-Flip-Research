from datasets import load_dataset
from benchmarking import BaseBenchmark, benchmark
from typing import Optional
from utils import Logger
import json


@benchmark
class VMCBench(BaseBenchmark):

    hf_name = "suyc21/VMCBench"  # Update this with the correct HuggingFace dataset name
    
    # Category mapping for scoring
    datasets_category_map = {
        "SEEDBench": "general",
        "MMStar": "general",
        "A-OKVQA": "general",
        "VizWiz": "general",
        "MMVet": "general",
        "VQAv2": "general",
        "OKVQA": "general",
        "MMMU": "reason",
        "MathVista": "reason",
        "ScienceQA": "reason",
        "RealWorldQA": "reason",
        "GQA": "reason",
        "MathVision": "reason",
        "TextVQA": "ocr",
        "OCRVQA": "ocr",
        "AI2D": "doc",
        "ChartQA": "doc",
        "DocVQA": "doc",
        "InfoVQA": "doc",
        "TableVQABench": "doc",
    }

    def load_benchmark(self):
        self.dataset = load_dataset(self.get_hf_name(), split="dev")  # Update split if needed

    @classmethod
    def get_hf_name(cls) -> str:
        return cls.hf_name

    def preprocess_samples(self, samples: list[dict]) -> list[dict]:
        processed_samples = []

        for sample in samples:
            # Extract question and options (A, B, C, D)
            question = sample["question"]
            
            # Build options prompt
            options = {cand: sample[cand] for cand in "ABCD"}
            options_prompt = "Options:\n"
            for key, item in options.items():
                options_prompt += f"{key}. {item}\n"
            
            # Construct query
            query = f"Question: {question}\n{options_prompt}"

            choices = {"a": sample["A"], "b": sample["B"], "c": sample["C"], "d": sample["D"]}
            choices_v_k = {v: k.upper() for k, v in choices.items()}
            
            processed_sample = {
                "query": query,
                "decoded_image": sample["image"].convert("RGB") if hasattr(sample["image"], "convert") else sample["image"],
                "pid": sample["index"],  # Using index as pid
                "answer": choices[sample["answer"].lower()], 
                "choices": [sample["A"], sample["B"], sample["C"], sample["D"]],
                "category": sample.get("category", "unknown"),
            }
            processed_samples.append(processed_sample)
            
        return processed_samples
