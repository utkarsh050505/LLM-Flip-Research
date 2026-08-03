from typing import Any, List, Union, Optional
from abc import abstractmethod
import os

try:
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
    try:
        from transformers import AutoModelForVision2Seq
    except ImportError:
        from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
except ImportError:
    AutoModelForCausalLM = AutoProcessor = AutoTokenizer = AutoModelForVision2Seq = BitsAndBytesConfig = None

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = SamplingParams = None

try:
    from PIL import Image
except ImportError:
    Image = None

from modeling.prompt_builder import build_prompt
from utils.custom_logging import Logger

try:
    import torch
except ImportError:
    torch = None

# Windows safetensors mmap fix for OSError 1455 (paging file limit)
def _patch_windows_safetensors_loader():
    try:
        import transformers.modeling_utils as mu
        import safetensors.torch
        orig_load = getattr(mu, "load_state_dict", None)
        if orig_load is not None and not getattr(mu, "_windows_patch_applied", False):
            def robust_load_state_dict(checkpoint_file, is_quantized=False, map_location="cpu", weights_only=True):
                try:
                    return orig_load(
                        checkpoint_file=checkpoint_file,
                        is_quantized=is_quantized,
                        map_location=map_location,
                        weights_only=weights_only
                    )
                except (OSError, Exception) as e:
                    if ("1455" in str(e) or "paging file" in str(e).lower() or "access violation" in str(e).lower()) and str(checkpoint_file).endswith(".safetensors"):
                        with open(checkpoint_file, "rb") as f:
                            data = f.read()
                        tensors = safetensors.torch.load(data)
                        if map_location == "meta":
                            return {k: torch.empty(size=v.shape, dtype=v.dtype, device="meta") for k, v in tensors.items()}
                        elif map_location == "cpu" or map_location is None:
                            return tensors
                        else:
                            return {k: v.to(map_location) for k, v in tensors.items()}
                    raise e
            mu.load_state_dict = robust_load_state_dict
            mu._windows_patch_applied = True
    except Exception:
        pass

_patch_windows_safetensors_loader()

from modeling.base_model import BaseModel

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28


class HFModel(BaseModel):
    text_only = False

    def eval(self):
        if self.backend != "vllm" and hasattr(self, "model") and self.model is not None:
            self.model.eval()

    def load_model(self):
        self.hf_name = self.get_hf_name()

        print(f"Loading HF model {self.hf_name} (backend={self.backend})...")

        if self.backend == "vllm":
            if torch and torch.are_deterministic_algorithms_enabled():
                Logger.info(
                    "Disabling torch deterministic algorithms before vLLM "
                    "initialization to avoid Inductor compile-time benchmarking "
                    "failures. Generation seeding remains enabled."
                )
                torch.use_deterministic_algorithms(False)

            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible:
                num_gpus = len(cuda_visible.split(","))
            else:
                num_gpus = 1 if (torch and torch.cuda.is_available()) else 4

            Logger.info(f"Using tensor_parallel={num_gpus}")

            if "qwen3" in self.hf_name.lower():
                gpu_uti = 0.85
            else:
                gpu_uti = 0.90
            gpu_uti = float(getattr(self, "gpu_memory_utilization", gpu_uti))

            llm_kwargs = {
                "model": self.hf_name,
                "trust_remote_code": True,
                "dtype": "bfloat16",
                "tensor_parallel_size": num_gpus,
                "gpu_memory_utilization": gpu_uti,
                "seed": self.seed,
            }
            if "qwen3-vl" in self.hf_name.lower():
                qwen3vl_cudagraph_mode = getattr(
                    self, "qwen3vl_cudagraph_mode", "FULL_DECODE_ONLY"
                ).upper()
                Logger.info(
                    f"Using Qwen3-VL cudagraph_mode={qwen3vl_cudagraph_mode}."
                )
                llm_kwargs["compilation_config"] = {
                    "cudagraph_mode": qwen3vl_cudagraph_mode
                }

            if not self.text_only:
                llm_kwargs.update(
                    {
                        "max_model_len": 16392,
                        "limit_mm_per_prompt": {"image": 1, "video": 0},
                        "mm_processor_cache_gb": 0,
                        "mm_encoder_tp_mode": "data",
                    }
                )

            self.model = LLM(**llm_kwargs)

            self.sampling_params = SamplingParams(
                **self.sampling_params,
                max_tokens=self.max_tokens,
                seed=self.seed
            )

        elif self.backend == "ddp":
            raise NotImplementedError("DDP backend is not implemented yet.")

        elif self.backend == "hf":
            # Determine quantization
            quant = getattr(self, "quantization", None)
            load_4bit = getattr(self, "load_in_4bit", False) or (quant == "4bit")
            load_8bit = getattr(self, "load_in_8bit", False) or (quant == "8bit")

            model_kwargs = {
                "trust_remote_code": True,
            }

            if load_4bit and BitsAndBytesConfig is not None:
                print("Enabling 4-bit NormalFloat4 (NF4) quantization via bitsandbytes...")
                compute_dtype = torch.bfloat16 if (torch and torch.cuda.is_bf16_supported()) else torch.float16
                model_kwargs["device_map"] = "auto" if (torch and torch.cuda.is_available()) else "cpu"
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )
            elif load_8bit and BitsAndBytesConfig is not None:
                print("Enabling 8-bit quantization via bitsandbytes...")
                model_kwargs["device_map"] = "auto" if (torch and torch.cuda.is_available()) else "cpu"
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            elif torch and torch.cuda.is_available():
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                model_kwargs["torch_dtype"] = dtype
                model_kwargs["device_map"] = "cuda:0"
            else:
                model_kwargs["device_map"] = "cpu"

            if self.text_only:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.hf_name,
                    **model_kwargs
                )
            else:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.hf_name,
                    **model_kwargs
                )

        else:
            raise NotImplementedError(f"Backend {self.backend} is not supported.")

        # Load Tokenizer / Processor
        if self.text_only:
            self.processor = AutoTokenizer.from_pretrained(
                self.hf_name,
                trust_remote_code=True
            )
            if self.processor.pad_token_id is None:
                self.processor.pad_token_id = self.processor.eos_token_id
        elif "qwen2" in self.hf_name.lower():
            self.processor = AutoProcessor.from_pretrained(
                self.hf_name,
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS,
                trust_remote_code=True
            )
        else:
            self.processor = AutoProcessor.from_pretrained(
                self.hf_name,
                trust_remote_code=True
            )

    def preprocess_inputs(self, samples: List[dict[str, Any]]) -> List[str]:
        """Preprocess input samples and build prompts."""
        prompts = []
        for sample in samples:
            msg = build_prompt(
                sample=sample,
                system_prompt=self.system_prompt,
                reasoning_prompt=self.reasoning_prompt,
                prepend_reasoning_prompt=self.prepend_reasoning_prompt,
            )
            if hasattr(self.processor, "apply_chat_template"):
                formatted = self.processor.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                formatted = msg[0]["content"] if isinstance(msg, list) else str(msg)
            prompts.append(formatted)
        return prompts

    def hf_generate(self, samples: List[dict[str, Any]], return_inputs: bool = False) -> Union[List[str], tuple[List[str], List[str]]]:
        """Generate outputs using standard HuggingFace backend."""
        messages = self.preprocess_inputs(samples)
        results = []

        for msg, sample in zip(messages, samples):
            images = sample.get("images", [])
            max_new = self.max_tokens if (hasattr(self, "max_tokens") and self.max_tokens) else 2048
            temp = getattr(self, "sampling_params", {}).get("temperature", 0.0) if hasattr(self, "sampling_params") else 0.0
            top_p = getattr(self, "sampling_params", {}).get("top_p", 1.0) if hasattr(self, "sampling_params") else 1.0

            if self.text_only or not images:
                inputs = self.processor(msg, return_tensors="pt").to(self.model.device)
                prompt_len = inputs.input_ids.shape[1]
                with torch.no_grad():
                    gen_tokens = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        do_sample=(temp > 0.01),
                        temperature=max(temp, 0.001) if temp > 0.01 else 1.0,
                        top_p=top_p,
                        pad_token_id=self.processor.pad_token_id or self.processor.eos_token_id,
                    )
                generated_text = self.processor.decode(gen_tokens[0][prompt_len:], skip_special_tokens=True).strip()
            else:
                inputs = self.processor(text=[msg], images=images, return_tensors="pt").to(self.model.device)
                prompt_len = inputs.input_ids.shape[1]
                with torch.no_grad():
                    gen_tokens = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        do_sample=(temp > 0.01),
                        temperature=max(temp, 0.001) if temp > 0.01 else 1.0,
                        top_p=top_p,
                    )
                generated_text = self.processor.decode(gen_tokens[0][prompt_len:], skip_special_tokens=True).strip()

            results.append(generated_text)

        if return_inputs:
            return results, messages
        return results

    def vllm_generate(self, samples: List[dict[str, Any]], return_inputs: bool = False) -> Union[List[str], tuple[List[str], List[str]]]:
        """Generate outputs using vLLM backend."""
        messages = self.preprocess_inputs(samples)
        vllm_inputs = []
        for message, sample in zip(messages, samples):
            vllm_input = {"prompt": message}
            if sample.get("images"):
                vllm_input["multi_modal_data"] = {"image": sample["images"]}
            vllm_inputs.append(vllm_input)

        outputs = self.model.generate(vllm_inputs, sampling_params=self.sampling_params, use_tqdm=False)

        if return_inputs:
            return [output.outputs[0].text.strip() for output in outputs], messages
        else:
            return [output.outputs[0].text.strip() for output in outputs]

    def generate_batch(self, samples: List[dict[str, Any]], return_inputs: bool = False) -> Union[List[str], tuple[List[str], List[dict[str, Any]]]]:
        """Generate outputs based on the active backend."""
        if self.backend == "vllm":
            return self.vllm_generate(samples=samples, return_inputs=return_inputs)
        elif self.backend == "hf":
            return self.hf_generate(samples=samples, return_inputs=return_inputs)
        elif self.backend == "ddp":
            raise NotImplementedError("DDP backend is not implemented yet.")
        else:
            raise NotImplementedError(f"{self.backend} generate_batch method is not implemented yet.")
