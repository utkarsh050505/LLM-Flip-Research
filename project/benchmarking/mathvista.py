from datasets import load_dataset
from benchmarking import BaseBenchmark, benchmark
from typing import Optional
from utils import Logger
import json


@benchmark
class MathVista(BaseBenchmark):

    hf_name = "AI4Math/MathVista"

    def load_benchmark(self):
        self.dataset = load_dataset(self.get_hf_name(), split="testmini")

    @classmethod
    def get_hf_name(cls) -> str:
        return cls.hf_name
    
    def preprocess_samples(self, samples: list[dict]) -> list[dict]:
        processed_samples = []
        for sample in samples:
            processed_sample = {
                "query": sample["query"],
                "decoded_image": sample["decoded_image"],
                "pid": sample["pid"],
                "answer": sample["answer"],
                "choices": sample.get("choices", [])
            }
            processed_samples.append(processed_sample)
        return processed_samples
        
                    
