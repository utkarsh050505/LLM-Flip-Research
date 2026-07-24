"""
Step 2 — Inspect generation outputs.
 
Goal: understand the shapes of `scores` and `hidden_states` returned by
generate() before you compute a single metric from them. Every bug in
Step 3+ traces back to a wrong assumption about these shapes, so look
first.
 
Also fixes the attention_mask warning from Step 1 — matters a lot once
you're doing branched generation from a cloned KV cache in Step 4.
"""
 
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from transformers import AutoModelForCausalLM, AutoTokenizer
 
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-llama-8B"
MAX_NEW_TOKENS = 20  # small on purpose — we're inspecting, not generating
 
 
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Loading {MODEL_NAME}...")
    print("NOTE: 8B models require ~16GB of free RAM/VRAM. If the script crashes silently here, your system ran out of memory.")
    try:
        # pyrefly: ignore [missing-import]
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb_config, device_map="auto"
        )
        print("Successfully loaded in 4-bit mode.")
    except ImportError:
        print("bitsandbytes not found, falling back to standard bfloat16 loading...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
        )
    model.eval()
 
    prompt = "Solve step by step: What is 17 * 24 - 15?"
    messages = [{"role": "user", "content": prompt}]
 
    # return_dict=True gives you attention_mask alongside input_ids —
    # this is the fix for the Step 1 warning.
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
 
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        )
 
    num_layers_plus_embed = model.config.num_hidden_layers + 1
    prompt_len = inputs["input_ids"].shape[1]
 
    print(f"Model: {MODEL_NAME}")
    print(f"num_hidden_layers: {model.config.num_hidden_layers} "
          f"(+1 for the embedding layer = {num_layers_plus_embed} entries per step)")
    print(f"hidden_size: {model.config.hidden_size}")
    print(f"prompt length (tokens): {prompt_len}")
    print(f"requested new tokens: {MAX_NEW_TOKENS}")
 
    print("\n--- outputs.scores ---")
    print(f"len(outputs.scores) = {len(outputs.scores)}  "
          f"(one entry per generated token)")
    print(f"outputs.scores[0].shape = {outputs.scores[0].shape}  "
          f"(batch, vocab_size) — the logits for the FIRST generated token")
 
    print("\n--- outputs.hidden_states ---")
    print(f"len(outputs.hidden_states) = {len(outputs.hidden_states)}  "
          f"(one tuple per generation step)")
    print(f"len(outputs.hidden_states[0]) = {len(outputs.hidden_states[0])}  "
          f"(one tensor per layer, incl. embedding layer)")
 
    step0_layer_shape = outputs.hidden_states[0][0].shape
    step1_layer_shape = outputs.hidden_states[1][0].shape
    print(f"\noutputs.hidden_states[0][0].shape = {step0_layer_shape}  "
          f"<- step 0 (prefill): seq_len = full prompt ({prompt_len})")
    print(f"outputs.hidden_states[1][0].shape = {step1_layer_shape}  "
          f"<- step 1 onward: seq_len = 1 (just the new token, thanks to KV cache)")
 
    print("\nTake-away: when you loop over generation steps to compute "
          "per-token metrics, step 0 needs different indexing than every "
          "step after it. Handle it as a special case, don't assume "
          "seq_len == 1 everywhere.")
 
    print("\nDecoded output (sanity check):")
    new_tokens = outputs.sequences[0][prompt_len:]
    print(tokenizer.decode(new_tokens, skip_special_tokens=True))
 
 
if __name__ == "__main__":
    main()