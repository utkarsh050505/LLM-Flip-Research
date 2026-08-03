from datasets import load_dataset
from benchmarking import BaseBenchmark, benchmark


@benchmark
class MathVerse(BaseBenchmark):

    hf_name = "CaraJ/MathVerse-lmmseval"
    query_type = "query_cot"

    def load_benchmark(self):
        self.dataset = load_dataset(self.get_hf_name(), "testmini", split="testmini")

    @classmethod
    def get_hf_name(cls) -> str:
        return cls.hf_name

    def preprocess_samples(self, samples: list[dict]) -> list[dict]:
        processed_samples = []

        for idx, sample in enumerate(samples):
            image = sample.get("image")
            if image is not None and str(image).strip() == "":
                image = None
            decoded_image = image.convert("RGB") if hasattr(image, "convert") else image

            processed_sample = {
                "idx": idx,
                "query": sample[self.query_type],
                "decoded_image": decoded_image,
                "pid": sample.get("sample_index", idx),
                "answer": sample["answer"],
                "choices": [],
                "question_type": sample.get("question_type"),
                "problem_version": sample.get("problem_version"),
            }
            processed_samples.append(processed_sample)

        return processed_samples
