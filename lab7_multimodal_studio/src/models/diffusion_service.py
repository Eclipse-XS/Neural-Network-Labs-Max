import time
import gc
import torch
from PIL import Image
from typing import Tuple, Optional, Any
from ..utils.device import get_device, clean_gpu_memory, ModelLifecycleManager, serialized_inference

DIFFUSION_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

class DiffusionService:
    """Inference engine for Stable Diffusion 1.5 Image-to-Image transformation (Lab 4 integration)."""

    def __init__(self, model_id: str = DIFFUSION_MODEL_ID):
        self.model_id = model_id
        self.device = get_device()
        self.pipeline: Optional[Any] = None
        
        # Register with coordinated lifecycle manager
        self.manager = ModelLifecycleManager.get_instance()
        self.manager.register("diffusion", self)

    def _ensure_loaded(self):
        if self.pipeline is None:
            try:
                from diffusers import StableDiffusionImg2ImgPipeline
            except ImportError as e:
                raise ImportError(
                    f"The 'diffusers' library is required to run DiffusionService. "
                    f"Please install diffusers: {e}"
                )
            # Unload any conflicting heavy models (e.g. LLaVA) before loading Diffusion
            self.manager.prepare_for_model("diffusion")
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id,
                revision="451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
                torch_dtype=dtype,
                variant="fp16" if self.device.type == "cuda" else None,
                use_safetensors=True,
                safety_checker=None
            )
            if self.device.type == "cuda":
                # Enable CPU offload to fit safely in 4 GB VRAM alongside other services
                self.pipeline.enable_model_cpu_offload()
                self.pipeline.enable_attention_slicing()
                self.pipeline.vae.enable_slicing()
            else:
                self.pipeline.to(self.device)

    @serialized_inference
    def transform_image(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, deformed, artifacts",
        strength: float = 0.6,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: int = 42
    ) -> Tuple[Image.Image, float]:
        """Perform Image-to-Image diffusion transformation on input image."""
        if not 0 < strength <= 1 or num_inference_steps < 1 or int(strength * num_inference_steps) < 1:
            raise ValueError("strength * num_inference_steps must be at least 1")
        self._ensure_loaded()
        
        rgb_image = image.convert("RGB").resize((512, 512))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        t0 = time.time()
        try:
            output = self.pipeline(
                prompt=prompt,
                image=rgb_image,
                negative_prompt=negative_prompt,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator
            )
            elapsed = time.time() - t0
            result_image = output.images[0]
            return result_image, round(elapsed, 3)
        except Exception:
            clean_gpu_memory()
            raise

    def unload(self):
        """Free GPU VRAM allocated to diffusion pipeline by explicitly dropping references."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        clean_gpu_memory()
