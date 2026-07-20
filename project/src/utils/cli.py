"""
CLI Prompt Utilities

Provides interactive command-line interface prompts for model and precision selection.
"""

from typing import Dict, Tuple, Any


def prompt_model_and_precision(model_registry: Dict[str, Any]) -> Tuple[str, Dict[str, Any], str]:
    """
    Prompt the user to choose a model from the registry and select a precision setting.

    Args:
        model_registry: The MODEL_REGISTRY dictionary from configs.model_config.

    Returns:
        A tuple of (model_key, model_config_dict, precision_mode)
        where precision_mode is one of: "bf16", "8bit", "4bit"
    """
    print("\n" + "=" * 60)
    print("  MODEL SELECTION")
    print("=" * 60)
    
    # List models from the registry
    model_keys = list(model_registry.keys())
    for idx, key in enumerate(model_keys, start=1):
        hf_path = model_registry[key]["hf_path"]
        family = model_registry[key]["family"]
        default_quant = "4-bit" if model_registry[key].get("load_in_4bit") else "BF16"
        print(f"  [{idx}] {key:<20} | HF: {hf_path:<45} | default: {default_quant}")

    print("=" * 60)
    
    while True:
        try:
            choice = input(f"Select a model [1-{len(model_keys)}]: ").strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(model_keys):
                selected_key = model_keys[choice_idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(model_keys)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    print("\n" + "=" * 60)
    print("  PRECISION / QUANTIZATION SELECTION")
    print("=" * 60)
    print("  [1] Native BF16/FP16 (Requires more VRAM, best quality)")
    print("  [2] 8-Bit Quantization (BitsAndBytes)")
    print("  [3] 4-Bit Quantization (BitsAndBytes NF4, lowest VRAM)")
    print("=" * 60)

    precision_mapping = {
        "1": "bf16",
        "2": "8bit",
        "3": "4bit"
    }

    while True:
        prec_choice = input("Select precision [1-3]: ").strip()
        if prec_choice in precision_mapping:
            selected_precision = precision_mapping[prec_choice]
            break
        print("Invalid choice. Please select 1, 2, or 3.")

    selected_config = model_registry[selected_key]
    return selected_key, selected_config, selected_precision
