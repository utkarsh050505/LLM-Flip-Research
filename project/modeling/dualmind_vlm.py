from modeling.hf_model import HFModel
from modeling import model
import re

from evaluation import Strategies

@model
class DualMindVLM(HFModel):
    system_prompt = """You are a Vision-Language Model answering questions about images. 
Follow these rules strictly:  
1. Judge the length of reasoning needed.
- Short: start with "Short Thinking:".
- Long: start with "Long Thinking:".
2. Short Thinking: give a concise thinking process which is sufficient to answer the question, then provide the final answer.
3. Long Thinking: give a structured reasoning process of the question and the image, including question analysis, visual details description, self-verification and then provide the final answer.
4. The final answer MUST BE put in \\boxed{}."""

    reasoning_prompt = ""
    hf_name = "maifoundations/DualMindVLM"
    base_model = "Qwen2.5-VL-7B"
    sampling_params = {"temperature": 1e-06, "top_p": 0.95, "repetition_penalty": 1.05}
    prepend_reasoning_prompt = False
    parser = staticmethod(Strategies.boxed)

    @classmethod
    def get_hf_name(cls) -> str:
        return cls.hf_name
    
    @classmethod
    def get_base_model_name(cls) -> str:
        return cls.base_model