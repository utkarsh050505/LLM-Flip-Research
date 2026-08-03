from modeling.hf_model import HFModel
from modeling import model
from evaluation import Strategies

@model
class R1VL(HFModel):
    """
    Here we are using the same reasoning prompt of VL-Rethinker as R1-VL does not have a
    specific reasoning prompt defined.
    """
    system_prompt = "" # R1-VL uses the qwen2vl default system prompt
    reasoning_prompt = "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}." # R1-VL do not use special reasoning prompt
    hf_name = "jingyiZ00/R1-VL-7B"
    base_model = "Qwen2-VL-7B"
    sampling_params = {"top_p": 0.001, "top_k": 1, "temperature": 0.01, "repetition_penalty": 1.0}
    prepend_reasoning_prompt = False
    parser = staticmethod(Strategies.boxed)

    @classmethod
    def get_hf_name(cls) -> str:
        return cls.hf_name
    
    @classmethod
    def get_base_model_name(cls) -> str:
        return cls.base_model