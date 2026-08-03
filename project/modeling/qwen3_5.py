from modeling.hf_model import HFModel
from modeling import model

from evaluation import Strategies


@model
class Qwen3_5(HFModel):
    system_prompt = ""
    reasoning_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    hf_name = "Qwen/Qwen3.5-9B"
    base_model = "Qwen3.5-9B"
    sampling_params = {
        "top_k": 20,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "temperature": 1.0,
    }
    prepend_reasoning_prompt = False
    parser = staticmethod(Strategies.boxed)

    @classmethod
    def get_hf_name(cls) -> str:
        return cls.hf_name

    @classmethod
    def get_base_model_name(cls) -> str:
        return cls.base_model
