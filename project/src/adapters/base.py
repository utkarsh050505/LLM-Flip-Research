from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

class BaseAdapter(ABC):
    """
    Abstract Base Class for model adapters. All specific model adapters (e.g. QwenAdapter, LlamaAdapter)
    must inherit from this class to ensure interface compatibility across the pipeline.
    """
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        load_in_4bit: bool = True,
        **kwargs
    ):
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.kwargs = kwargs
        
        self.model = None
        self.tokenizer = None
        self._is_loaded = False

    @abstractmethod
    def load_model(self) -> Tuple[Any, Any]:
        """
        Loads the model and tokenizer onto the specified device (using quantization if requested).
        Sets and returns (model, tokenizer).
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        do_sample: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates text given a prompt.
        Must capture and return:
          - 'generated_text': The final generated string.
          - 'tokens': List of strings or IDs of generated tokens.
          - 'logits': Tuple or list of logit tensors for each step.
          - 'hidden_states': Tuple or list of hidden states for each step.
          - 'token_probs': Probabilities of chosen tokens.
        """
        pass
