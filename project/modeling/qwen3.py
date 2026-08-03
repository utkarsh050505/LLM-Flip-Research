from modeling.hf_model import HFModel
from modeling import model

from evaluation import Strategies


@model
class Qwen3(HFModel):
    system_prompt = ""
    reasoning_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    hf_name = "Qwen/Qwen3-8B"
    base_model = "Qwen3-8B"
    sampling_params = {"top_k": 20, "top_p": 0.95, "repetition_penalty": 1.0, "temperature": 0.6}
    prepend_reasoning_prompt = False
    parser = staticmethod(Strategies.boxed)
    text_only = True

    @classmethod
    def get_hf_name(cls) -> str:
        return cls.hf_name

    @classmethod
    def get_base_model_name(cls) -> str:
        return cls.base_model
