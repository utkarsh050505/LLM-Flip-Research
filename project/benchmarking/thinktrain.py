from datasets import load_dataset
from benchmarking import BaseBenchmark, benchmark
import hashlib
import random
import re
from PIL import Image
import io


@benchmark
class ThinkTrain(BaseBenchmark):

    hf_name = "russwang/ThinkLite-VL-hard-11k"
    split_name = "train"

    def load_benchmark(self):
        dataset = load_dataset(self.get_hf_name(), split="train")
        dataset = dataset.shuffle(seed=42)
        half_size = len(dataset) // 2
        self.dataset = dataset.select(range(half_size))
        
    @classmethod
    def get_hf_name(cls) -> str:
        return cls.hf_name

    def preprocess_samples(self, samples: list[dict]) -> list[dict]:
        processed_samples = []
        for sample in samples:
            processed_sample = {
                "query": sample["problem"],
                "decoded_image": Image.open(io.BytesIO(sample['image'])),
                "pid": sample["id"],
                "answer": sample["ground_truth"],
                "choices": sample.get("choices", [])
            }
            processed_samples.append(processed_sample)
        return processed_samples