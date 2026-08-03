from modeling.hf_model import HFModel
from modeling import model
import re
from evaluation import Strategies

@model
class MMEureka(HFModel):
    system_prompt = "Solve the question. The user asks a question, and you solve it. You first think about the reasoning process in the mind and then provide the user with the answer. The answer is in latex format and wrapped in $...$. The final answer must be wrapped using the \\boxed{} command. The answer should be enclosed within<answer></answer>tags, i.e., Since $1+1=2$, so the answer is $2$. <answer>. The answer is $\boxed{2}$ </answer>, which means the final answer assistant’s output should start with <answer>and end with </answer>."
    reasoning_prompt = ""
    hf_name = "FanqingM/MM-Eureka-Qwen-7B"
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