from datasets import load_dataset
from benchmarking import BaseBenchmark, benchmark
import hashlib
import random
import re


@benchmark
class GPQA(BaseBenchmark):

    hf_name = "Idavidrein/gpqa"
    subset_name = "gpqa_diamond"
    split_name = "train"
    mc_prompt = "Answer the question with the option's letter from the given choices directly."

    def load_benchmark(self):
        self.dataset = load_dataset(self.get_hf_name(), self.subset_name, split="train")
        
    @classmethod
    def get_hf_name(cls) -> str:
        return cls.hf_name

    @staticmethod
    def preprocess_text(text):
        if text is None:
            return " "
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub(r"\[.*?\]", "", text)
        text = text.replace("  ", " ")
        return text

    @classmethod
    def build_processed_doc(cls, sample: dict) -> dict:
        choices = [
            cls.preprocess_text(sample["Incorrect Answer 1"]),
            cls.preprocess_text(sample["Incorrect Answer 2"]),
            cls.preprocess_text(sample["Incorrect Answer 3"]),
            cls.preprocess_text(sample["Correct Answer"]),
        ]

        seed_source = str(sample.get("Record ID", sample["Question"]))
        seed = int(hashlib.md5(seed_source.encode("utf-8")).hexdigest(), 16)
        rng = random.Random(seed)
        rng.shuffle(choices)

        correct_answer = cls.preprocess_text(sample["Correct Answer"])
        correct_answer_index = choices.index(correct_answer)
        options = [chr(ord("A") + i) for i in range(len(choices))]
        choices_str = " ".join([f"({option}) {choice}" for option, choice in zip(options, choices)])
        query = f"Question: {sample['Question']}\nChoices:\n{choices_str}\n{cls.mc_prompt}"

        return {
            "query": query,
            "choices": choices,
            "answer": correct_answer,
            "answer_letter": f"({options[correct_answer_index]})",
        }

    def preprocess_samples(self, samples: list[dict]) -> list[dict]:
        processed_samples = []

        for idx, sample in enumerate(samples):
            processed_doc = self.build_processed_doc(sample)
            processed_sample = {
                "idx": idx,
                "query": processed_doc["query"],
                "decoded_image": None,
                "pid": sample.get("Record ID", idx),
                "answer": processed_doc["answer"],
                "choices": processed_doc["choices"],
                "answer_letter": processed_doc["answer_letter"],
                "domain": sample.get("High-level domain"),
                "subdomain": sample.get("Subdomain"),
            }
            processed_samples.append(processed_sample)

        return processed_samples
