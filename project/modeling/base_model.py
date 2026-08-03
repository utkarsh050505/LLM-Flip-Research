from abc import ABC, abstractmethod
from PIL import Image
from typing import List, Union, Optional, Any

from evaluation import ParsingHelper, Strategies



class BaseModel(ABC):
    def __init__(self, 
                 model_name: str,
                 backend: str,
                 max_tokens: int,
                 max_num_images: int,
                 seed: int,
                 additional_args: str,
                 override_system_prompt: Optional[str] = None,
                 override_reasoning_prompt: Optional[str] = None,
                 override_prepend_reasoning_prompt: bool = False):
        self.model_name = model_name
        self.seed = seed
        self.backend = backend
        self.max_num_images = max_num_images
        self.max_tokens = max_tokens
        self.parse_additional_args(additional_args)
        self.apply_prompt_overrides(
            override_system_prompt=override_system_prompt,
            override_reasoning_prompt=override_reasoning_prompt,
            override_prepend_reasoning_prompt=override_prepend_reasoning_prompt,
        )
        self.load_model()

    @abstractmethod
    def eval(self):
        """Set the model to evaluation mode if applicable."""
        pass

    def parse_additional_args(self, additional_args: str):
        """Parse additional arguments specific to the benchmark."""
        parsed_args = {v[0]: v[1] for v in (arg.split('=') for arg in additional_args.split(';') if additional_args != "")}
        
        for key, value in parsed_args.items():
            setattr(self, key, value)

    def apply_prompt_overrides(
        self,
        override_system_prompt: Optional[str] = None,
        override_reasoning_prompt: Optional[str] = None,
        override_prepend_reasoning_prompt: bool = False,
    ):
        """Apply runtime prompt overrides.

        `None` keeps the model default. An empty system prompt string delegates
        back to the chat template/model default by omitting the system message.
        """
        if override_system_prompt is not None:
            self.system_prompt = override_system_prompt

        if override_reasoning_prompt is not None:
            self.reasoning_prompt = override_reasoning_prompt

        if override_prepend_reasoning_prompt:
            self.prepend_reasoning_prompt = True
    
    @classmethod                          
    def get_url(cls) -> str:
        if getattr(cls, "get_hf_name", None) is not None:
            return cls.get_hf_name()
        else:
            return "NotApplicable"
        
    def normalize_answer(self, ans: str):
        """Normalize the answer string for comparison."""
        pass

    
    @classmethod
    def get_base_model_name(cls) -> str:
        if getattr(cls, "get_base_model_name", None) is not None:
            return cls.get_base_model_name()
        else:
            return "NotApplicable"

    @abstractmethod
    def load_model(self):
        """Load the model and processor/tokenizer here."""
        pass
    
    @abstractmethod
    def preprocess_inputs(self, samples: List[dict[str, Any]]):
        """Preprocess input samples if needed."""
        pass

    @abstractmethod
    def generate_batch(self, samples_batch: List[dict[str, Any]], return_inputs: bool) -> Union[List[str], tuple[List[str], List[dict[str, Any]]]]:
        pass

    def normalize_answer(self, ans: str, clean_only: bool = False) -> str:

        if not clean_only:
            # 1. Execute the strictly assigned strategy
            raw_ans = self.parser(ans)
            
            # 2. If the model failed to output the format (e.g. no box found),
            #    we return empty string (or handle it how you prefer).
            if not raw_ans:
                print(f"Warning: Model {self.model_name} did not follow the {self.parser.__name__} format.")
                return ""
            
        else:
            raw_ans = ans
            # 3. Clean the result

        return ParsingHelper.clean(raw_ans)
