import os
import json
# pyrefly: ignore [missing-import]
import torch
from typing import List, Dict, Any, Union

class ReasoningTrace:
    """
    Container class representing a generated reasoning trace along with its associated
    metadata and heavy tensors (logits and hidden states) extracted step-by-step.
    """
    def __init__(
        self,
        prompt: str,
        generated_text: str,
        tokens: List[str],
        token_ids: List[int],
        token_probs: List[float],
        logits: List[torch.Tensor],
        hidden_states: List[torch.Tensor],
        prompt_length: int
    ):
        self.prompt = prompt
        self.generated_text = generated_text
        self.tokens = tokens
        self.token_ids = token_ids
        self.token_probs = token_probs
        self.logits = logits  # List of Tensors
        self.hidden_states = hidden_states  # List of Tensors
        self.prompt_length = prompt_length

    def to_dict(self, include_tensors: bool = True) -> Dict[str, Any]:
        """
        Converts the trace representation to a dictionary.
        """
        data = {
            "prompt": self.prompt,
            "generated_text": self.generated_text,
            "tokens": self.tokens,
            "token_ids": self.token_ids,
            "token_probs": self.token_probs,
            "prompt_length": self.prompt_length
        }
        if include_tensors:
            data["logits"] = self.logits
            data["hidden_states"] = self.hidden_states
        return data

    def save(self, filepath_base: str):
        """
        Saves the trace. Generates two files:
        1. <filepath_base>.json (lightweight metadata for quick inspection)
        2. <filepath_base>.pt (heavy PyTorch binary dictionary containing all logits and hidden states)
        """
        # Ensure directories exist
        os.makedirs(os.path.dirname(filepath_base), exist_ok=True)

        # 1. Save metadata JSON
        meta_path = f"{filepath_base}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(include_tensors=False), f, indent=4, ensure_ascii=False)

        # 2. Save heavy tensor data
        # Pack everything into a torch dictionary. Tensors are stacked if possible for easy indexing.
        pt_path = f"{filepath_base}.pt"
        
        # Stack logits and hidden states list to single tensor if not empty
        stacked_logits = torch.stack(self.logits) if len(self.logits) > 0 else torch.tensor([])
        stacked_hidden = torch.stack(self.hidden_states) if len(self.hidden_states) > 0 else torch.tensor([])
        
        tensor_data = {
            "prompt": self.prompt,
            "generated_text": self.generated_text,
            "tokens": self.tokens,
            "token_ids": torch.tensor(self.token_ids, dtype=torch.long),
            "token_probs": torch.tensor(self.token_probs, dtype=torch.float32),
            "logits": stacked_logits,          # Shape: (seq_len, vocab_size)
            "hidden_states": stacked_hidden,    # Shape: (seq_len, num_layers + 1, hidden_dim)
            "prompt_length": self.prompt_length
        }
        torch.save(tensor_data, pt_path)

    @classmethod
    def load(cls, filepath_base: str) -> "ReasoningTrace":
        """
        Loads the trace from the saved files.
        """
        pt_path = f"{filepath_base}.pt"
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"PyTorch trace file not found at {pt_path}")
            
        tensor_data = torch.load(pt_path, map_location="cpu")
        
        # Convert back to lists of tensors
        logits_list = [t for t in tensor_data["logits"]] if tensor_data["logits"].ndim > 0 else []
        hidden_list = [t for t in tensor_data["hidden_states"]] if tensor_data["hidden_states"].ndim > 0 else []
        
        return cls(
            prompt=tensor_data["prompt"],
            generated_text=tensor_data["generated_text"],
            tokens=tensor_data["tokens"],
            token_ids=tensor_data["token_ids"].tolist(),
            token_probs=tensor_data["token_probs"].tolist(),
            logits=logits_list,
            hidden_states=hidden_list,
            prompt_length=int(tensor_data["prompt_length"])
        )
