from modeling.hf_model import HFModel
from modeling import model

from evaluation import Strategies


@model
class InternS1(HFModel):
    system_prompt = ""
    reasoning_prompt = "You are an expert reasoner with extensive experience in all areas. You approach problems through systematic thinking and rigorous reasoning. Your response should reflect deep understanding and precise logical thinking, making your solution path and reasoning clear to others. Please put your thinking process within <think>...</think> tags. Plese put your final answer within \\boxed{}."
    hf_name = "internlm/Intern-S1-mini"
    base_model = "Qwen3-8B"
    sampling_params = {"top_k": 50, "top_p": 1.0, "repetition_penalty": 1.0, "temperature": 0.8}
    prepend_reasoning_prompt = False
    parser = staticmethod(Strategies.boxed)
    text_only = True

    @classmethod
    def get_hf_name(cls) -> str:
        return cls.hf_name

    @classmethod
    def get_base_model_name(cls) -> str:
        return cls.base_model
