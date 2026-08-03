from modeling.hf_model import HFModel
from modeling import model
from evaluation import Strategies

@model
class ThinkliteVL(HFModel):
    """
    Potentially the system prompt may be the reasoning prompt
    """
    system_prompt = "" # R1-VL uses the qwen2vl default system prompt
    reasoning_prompt = "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
    hf_name = "russwang/ThinkLite-VL-7B"
    base_model = "Qwen2.5-VL-7B"
    sampling_params = {"temperature": 0.0}
    prepend_reasoning_prompt = False
    parser = staticmethod(Strategies.boxed)

    @classmethod
    def get_hf_name(cls) -> str:
        return cls.hf_name
    
    @classmethod
    def get_base_model_name(cls) -> str:
        return cls.base_model