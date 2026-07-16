# pyrefly: ignore [missing-import]
import torch
from typing import Dict, Any, Tuple
# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from src.adapters.base import BaseAdapter

class QwenAdapter(BaseAdapter):
    """
    Adapter implementation for Qwen models (specifically tested on Qwen2.5 family).
    Handles 4-bit quantization, device mapping, VRAM optimization, and trace extraction.
    """
    
    def load_model(self) -> Tuple[Any, Any]:
        """
        Loads the Qwen model and tokenizer.
        """
        if self._is_loaded:
            return self.model, self.tokenizer

        cache_dir = self.kwargs.get("cache_dir", "A:\\LLMResearch\\hf_cache")
        
        # Configure tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Configure BitsAndBytes for 4-bit quantization if requested
        if self.load_in_4bit and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            # 4-bit quantized model must be loaded with device_map="auto" or a specific device dict
            device_map = "auto"
        else:
            quantization_config = None
            device_map = None

        # Determine torch dtype
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Load the model
        model_kwargs = {
            "cache_dir": cache_dir,
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = device_map
        else:
            model_kwargs["device_map"] = "auto" if self.device == "cuda" else None

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        if self.device == "cuda" and quantization_config is None:
            self.model = self.model.to(self.device)

        self._is_loaded = True
        return self.model, self.tokenizer

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        do_sample: bool = True,
        apply_template: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates response while capturing logits, hidden states, and token probabilities.
        """
        if not self._is_loaded:
            self.load_model()

        # Apply chat template if requested
        if apply_template:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        # Tokenize prompt
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        prompt_len = input_ids.shape[1]

        # Generate tokens and collect outputs
        # Note: output_hidden_states=True, output_scores=True return logits and representations
        generation_outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs
        )

        sequences = generation_outputs.sequences
        generated_sequences = sequences[0, prompt_len:]
        generated_text = self.tokenizer.decode(generated_sequences, skip_special_tokens=True)

        # 1. Capture generated tokens
        tokens = [self.tokenizer.decode([tok_id]) for tok_id in generated_sequences]

        # 2. Extract logits
        # scores is a tuple of shape (batch_size, vocab_size) for each step
        logits_list = [score[0].detach().cpu() for score in generation_outputs.scores]

        # 3. Extract hidden states
        # hidden_states is a tuple of length num_generated_tokens
        # hidden_states[t] is a tuple of length num_layers + 1 (including embedding)
        # hidden_states[t][layer] has shape (batch_size, seq_len_step, hidden_dim)
        # For t=0, seq_len_step is prompt_len. For t>0, seq_len_step is 1.
        # We want to extract the final token's hidden state at each layer for each step.
        # For t=0, the hidden state of the generated token is at index -1 of seq_len_step.
        # For t>0, the hidden state is at index 0 of seq_len_step.
        extracted_hidden_states = []
        for t, step_hidden in enumerate(generation_outputs.hidden_states):
            # step_hidden is a tuple of layer tensors
            step_layers = []
            for layer_idx, layer_tensor in enumerate(step_hidden):
                # layer_tensor has shape (1, seq_len_step, hidden_dim)
                # Extract the last token's representation
                if t == 0:
                    token_rep = layer_tensor[0, -1, :].detach().cpu()
                else:
                    token_rep = layer_tensor[0, 0, :].detach().cpu()
                step_layers.append(token_rep)
            # stack layers: shape (num_layers + 1, hidden_dim)
            extracted_hidden_states.append(torch.stack(step_layers))

        # 4. Extract token probabilities of the chosen tokens
        token_probs = []
        for t, logits in enumerate(logits_list):
            probs = torch.softmax(logits, dim=-1)
            chosen_tok_id = generated_sequences[t].item()
            chosen_prob = probs[chosen_tok_id].item()
            token_probs.append(chosen_prob)

        return {
            "generated_text": generated_text,
            "tokens": tokens,
            "token_ids": generated_sequences.tolist(),
            "logits": logits_list,  # List of CPU tensors of shape (vocab_size,)
            "hidden_states": extracted_hidden_states,  # List of CPU tensors of shape (num_layers + 1, hidden_dim)
            "token_probs": token_probs,  # List of floats
            "prompt_length": prompt_len
        }
