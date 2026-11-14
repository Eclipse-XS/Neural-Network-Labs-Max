import time
import gc
import torch
from PIL import Image
from typing import Tuple, Optional, Any
from ..utils.device import get_device, clean_gpu_memory, ModelLifecycleManager, serialized_inference

MODEL_ID = "llava-hf/llava-interleave-qwen-0.5b-hf"

class LlavaService:
    """Inference engine for multimodal image captioning and Visual Question Answering (VQA)."""

    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self.device = get_device()
        self.processor: Optional[Any] = None
        self.model: Optional[Any] = None
        
        # Register with coordinated lifecycle manager
        self.manager = ModelLifecycleManager.get_instance()
        self.manager.register("llava", self)

    def _ensure_loaded(self):
        if self.model is None or self.processor is None:
            try:
                from transformers import AutoProcessor, LlavaForConditionalGeneration
            except ImportError as e:
                raise ImportError(
                    f"The 'transformers' library is required to run LlavaService. "
                    f"Please install transformers: {e}"
                )
            # Unload any conflicting heavy models (e.g. Stable Diffusion) before loading LLaVA
            self.manager.prepare_for_model("llava")
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.processor = AutoProcessor.from_pretrained(self.model_id, revision="1090956dd1c79bc93ae98dcf395590369435ec91")
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                revision="1090956dd1c79bc93ae98dcf395590369435ec91",
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.model.eval()

    @serialized_inference
    def generate_caption(
        self,
        image: Image.Image,
        style: str = "descriptive",
        temperature: float = 0.2,
        max_new_tokens: int = 128
    ) -> Tuple[str, float]:
        """Generate caption for an input image given a style prompt."""
        self._ensure_loaded()
        
        style_prompts = {
            "concise": "Provide a concise, single-sentence factual caption for this image.",
            "descriptive": "Describe this image in clear detail, mentioning primary subjects, setting, and composition.",
            "detailed": "Provide an exhaustive, high-detail description of all visible objects, spatial relationships, colors, and lighting."
        }
        instruction = style_prompts.get(style, style_prompts["descriptive"])
        prompt = f"<|im_start|>user\n<image>\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        
        rgb_image = image.convert("RGB")
        try:
            inputs = self.processor(text=prompt, images=rgb_image, return_tensors="pt").to(self.device, dtype=self.model.dtype)
            
            t0 = time.time()
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0.0),
                    temperature=temperature if temperature > 0.0 else None,
                    top_p=0.9 if temperature > 0.0 else None
                )
            elapsed = time.time() - t0
            
            input_len = inputs.input_ids.shape[1]
            caption = self.processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
            return caption, round(elapsed, 3)
        except Exception:
            clean_gpu_memory()
            raise

    @serialized_inference
    def answer_question(
        self,
        image: Image.Image,
        question: str,
        temperature: float = 0.2,
        max_new_tokens: int = 128
    ) -> Tuple[str, float]:
        """Perform Visual Question Answering (VQA) based on image visual evidence."""
        self._ensure_loaded()
        
        instruction = (
            f"Question: {question}\n"
            "Answer the question accurately based strictly on the image content. "
            "If the visual evidence is insufficient or uncertain, clearly state that."
        )
        prompt = f"<|im_start|>user\n<image>\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        
        rgb_image = image.convert("RGB")
        try:
            inputs = self.processor(text=prompt, images=rgb_image, return_tensors="pt").to(self.device, dtype=self.model.dtype)
            
            t0 = time.time()
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0.0),
                    temperature=temperature if temperature > 0.0 else None,
                    top_p=0.9 if temperature > 0.0 else None
                )
            elapsed = time.time() - t0
            
            input_len = inputs.input_ids.shape[1]
            answer = self.processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
            return answer, round(elapsed, 3)
        except Exception:
            clean_gpu_memory()
            raise

    def unload(self):
        """Free GPU VRAM allocated to LLaVA model by explicitly breaking references."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        clean_gpu_memory()
