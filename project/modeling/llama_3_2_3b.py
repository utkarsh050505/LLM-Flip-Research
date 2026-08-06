from modeling.hf_model import HFModel
from modeling import model
from evaluation import Strategies


@model
class Llama3_2_3B_Instruct(HFModel):
    system_prompt = ""
    reasoning_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    hf_name = "meta-llama/Llama-3.2-3B-Instruct"
    base_model = "Llama-3.2-3B-Instruct"
    sampling_params = {"temperature": 0.6, "top_p": 0.95}
    prepend_reasoning_prompt = False
    parser = staticmethod(Strategies.boxed)
    text_only = True

    @classmethod
    def get_hf_name(cls) -> str:
        return cls.hf_name

    @classmethod
    def get_base_model_name(cls) -> str:
        return cls.base_model
