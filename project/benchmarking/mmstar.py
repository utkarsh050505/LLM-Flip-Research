from datasets import load_dataset
from benchmarking import BaseBenchmark, benchmark


@benchmark
class MMStar(BaseBenchmark):

    hf_name = "Lin-Chen/MMStar"
    mc_prompt = "Answer the question with the option's letter directly."

    def load_benchmark(self):
        self.dataset = load_dataset(self.get_hf_name(), split="val")

    @classmethod
    def get_hf_name(cls) -> str:
        return cls.hf_name

    def preprocess_samples(self, samples: list[dict]) -> list[dict]:
        processed_samples = []

        for idx, sample in enumerate(samples):
            image = sample["image"]
            query = f"{sample['question']}\n{self.mc_prompt}"

            processed_sample = {
                "idx": idx,
                "query": query,
                "decoded_image": image.convert("RGB") if hasattr(image, "convert") else image,
                "pid": sample.get("index", idx),
                "answer": sample["answer"],
                "choices": [],
                "category": sample.get("category"),
                "l2_category": sample.get("l2_category"),
                "meta_info": sample.get("meta_info", {}),
            }
            processed_samples.append(processed_sample)

        return processed_samples
