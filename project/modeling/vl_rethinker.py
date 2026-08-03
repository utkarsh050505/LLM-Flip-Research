from modeling.hf_model import HFModel
from modeling import model
import re

from evaluation import Strategies

@model
class VLRethinker(HFModel):
    system_prompt = "" # R1-VL uses the qwen2vl default system prompt
    reasoning_prompt = "Please reason step by step, and put your final answer within \\boxed{}." # R1-VL do not use special reasoning prompt
    hf_name = "TIGER-Lab/VL-Rethinker-7B"
    base_model = "Qwen2.5-VL-7B"
    sampling_params = {"repetition_penalty": 1.05, "temperature": 0.1, "top_k": 1, "top_p": 0.001}
    prepend_reasoning_prompt = False
    parser = staticmethod(Strategies.boxed)

    @classmethod
    def get_hf_name(cls) -> str:
        return cls.hf_name
    
    @classmethod
    def get_base_model_name(cls) -> str:
        return cls.base_model