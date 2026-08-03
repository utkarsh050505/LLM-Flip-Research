from modeling.hf_model import HFModel
from modeling import model
import re

from evaluation import Strategies

@model
class Qwen2_5VL(HFModel):
    system_prompt = ""
    reasoning_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    hf_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    base_model = "Qwen2.5-VL"
    sampling_params = {"repetition_penalty": 1.05, "temperature": 0.000001}
    prepend_reasoning_prompt = False
    parser = staticmethod(Strategies.boxed)

    @classmethod
    def get_hf_name(cls) -> str:
        return cls.hf_name
    
    @classmethod
    def get_base_model_name(cls) -> str:
        return cls.base_model