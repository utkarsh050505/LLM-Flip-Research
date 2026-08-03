from datasets import load_dataset
from benchmarking import BaseBenchmark, benchmark
import hashlib
import random
import re


@benchmark
class GPQA(BaseBenchmark):

    hf_name = "simplescaling/aime25_nofigures"
    split_name = "train"

    def load_benchmark(self):
        self.dataset = load_dataset(self.get_hf_name(), split="train")
        
    @classmethod
    def get_hf_name(cls) -> str:
        return cls.hf_name


    def preprocess_samples(self, samples: list[dict]) -> list[dict]:
        processed_samples = []

        for idx, sample in enumerate(samples):
            processed_sample = {
                "idx": idx,
                "query": sample["problem"],
                "decoded_image": None,
                "pid": sample.get("id", idx),
                "answer": sample["answer"],
                "choices": []
            }
            processed_samples.append(processed_sample)

        return processed_samples
