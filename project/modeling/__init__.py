from modeling.base_model import BaseModel
from modeling.hf_model import HFModel

import pkgutil
import importlib
from pathlib import Path
import sys
import re

from modeling.prompt_builder import build_prompt


MODEL_REGISTER = {}


def _to_snake(name: str) -> str:
    s = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    return s.replace('__', '_')


def model(cls):
    module = sys.modules[cls.__module__]
    filename = Path(module.__file__).stem  # type: ignore
    
    # Register filename
    MODEL_REGISTER[filename] = cls
    MODEL_REGISTER[filename.lower()] = cls
    
    # Register class name & variations
    cls_name = cls.__name__
    MODEL_REGISTER[cls_name] = cls
    MODEL_REGISTER[cls_name.lower()] = cls
    MODEL_REGISTER[_to_snake(cls_name)] = cls
    
    # Strip dots/underscores for flexible lookup
    norm_name = re.sub(r'[^a-zA-Z0-9]', '', cls_name).lower()
    MODEL_REGISTER[norm_name] = cls

    # Register by HF repo name if available
    hf_name = getattr(cls, "hf_name", None)
    if hf_name:
        MODEL_REGISTER[hf_name] = cls
        MODEL_REGISTER[hf_name.lower()] = cls

    return cls


# Automatically load all modules in modeling package
for _, module_name, _ in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{module_name}")


def load_model(model_name: str, **kwargs) -> BaseModel:
    # Direct match
    model_class = MODEL_REGISTER.get(model_name)
    
    # Case-insensitive match
    if model_class is None:
        model_class = MODEL_REGISTER.get(model_name.lower())
    
    # Normalized alphanumeric match
    if model_class is None:
        norm_query = re.sub(r'[^a-zA-Z0-9]', '', model_name).lower()
        for k, v in MODEL_REGISTER.items():
            if re.sub(r'[^a-zA-Z0-9]', '', k).lower() == norm_query:
                model_class = v
                break

    if model_class is None:
        available = sorted(set(MODEL_REGISTER.keys()))
        raise ValueError(
            f"Model '{model_name}' not found in MODEL_REGISTER.\n"
            f"Available models: {available}"
        )
    return model_class(model_name, **kwargs)
