from modeling.hf_model import HFModel
from modeling import model
from evaluation import Strategies


@model
class DeepSeekR1DistillQwen1_5B(HFModel):
    system_prompt = ""
    reasoning_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    hf_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    base_model = "DeepSeek-R1-Distill-Qwen-1.5B"
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


@model
class R1DistillQwen1_5B(DeepSeekR1DistillQwen1_5B):
    pass


@model
class DeepSeekR1DistillQwen7B(HFModel):
    system_prompt = ""
    reasoning_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    hf_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    base_model = "DeepSeek-R1-Distill-Qwen-7B"
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


@model
class R1DistillQwen7B(DeepSeekR1DistillQwen7B):
    pass
