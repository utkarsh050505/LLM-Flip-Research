from datasets import load_dataset
from benchmarking import BaseBenchmark, benchmark


@benchmark
class AI2D(BaseBenchmark):

    hf_name = "lmms-lab/ai2d"
    mc_prompt = "Answer the question with the option's letter from the given choices directly."

    def load_benchmark(self):
        self.dataset = load_dataset(self.get_hf_name(), split="test")

    @classmethod
    def get_hf_name(cls) -> str:
        return cls.hf_name

    def preprocess_samples(self, samples: list[dict]) -> list[dict]:
        processed_samples = []

        for idx, sample in enumerate(samples):
            question = sample["question"]
            choices = sample["options"]
            options = [chr(ord("A") + i) for i in range(len(choices))]
            choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
            query = f"{question}\nChoices: {choices_str}\n{self.mc_prompt}"

            pid = sample.get("pid", idx)

            processed_sample = {
                "idx": idx,
                "query": query,
                "decoded_image": sample["image"].convert("RGB") if hasattr(sample["image"], "convert") else sample["image"],
                "pid": pid,
                "answer": choices[int(sample["answer"])],
                "choices": choices,
            }
            processed_samples.append(processed_sample)

        return processed_samples
