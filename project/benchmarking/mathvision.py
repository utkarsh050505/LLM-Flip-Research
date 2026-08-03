from datasets import load_dataset
from benchmarking import BaseBenchmark, benchmark
from typing import Optional
from utils import Logger
import json


@benchmark
class MathVision(BaseBenchmark):

    hf_name = "MathLLMs/MathVision"
    mc_prompt = "Answer the question with the option's letter from the given choices directly."
    short_answer_prompt = "Answer the question with a number directly."

    def load_benchmark(self):
        self.dataset = load_dataset(self.get_hf_name(), split="testmini")

    @classmethod
    def get_hf_name(cls) -> str:
        return cls.hf_name

    def preprocess_samples(self, samples: list[dict]) -> list[dict]:
        processed_samples = []

        for idx, sample in enumerate(samples):
            # Extract question and options
            question = sample["question"]
            choices = sample.get("options", [])
            
            if choices:
                # Multiple choice question
                len_choices = len(choices)
                options = [chr(ord("A") + i) for i in range(len_choices)]
                choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
                query = f"{question}\nChoices: {choices_str}\n{self.mc_prompt}"
            else:
                # Short answer question
                query = f"{question}"
            
            processed_sample = {
                "idx": idx,
                "query": query,
                "decoded_image": sample["decoded_image"],
                "pid": idx+1,
                "answer": choices[options.index(sample["answer"].upper())] if choices else sample["answer"],
                "choices": choices
            }
            processed_samples.append(processed_sample)
            
        return processed_samples
