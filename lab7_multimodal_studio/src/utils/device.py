import gc
import torch
from typing import Dict, Any, Optional

def get_device() -> torch.device:
    """Return available torch device, preferring CUDA if available."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

def clean_gpu_memory() -> None:
    """Force aggressive garbage collection and clear PyTorch CUDA caching allocators."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def get_gpu_memory_info() -> Dict[str, Any]:
    """Return current GPU memory statistics in megabytes."""
    if not torch.cuda.is_available():
        return {"device": "cpu", "allocated_mb": 0.0, "reserved_mb": 0.0, "total_mb": 0.0, "free_mb": 0.0}
    
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    allocated = torch.cuda.memory_allocated(dev) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(dev) / (1024 ** 2)
    total = props.total_memory / (1024 ** 2)
    
    return {
        "device": props.name,
        "allocated_mb": round(allocated, 2),
        "reserved_mb": round(reserved, 2),
        "total_mb": round(total, 2),
        "free_mb": round(total - allocated, 2)
    }

class ModelLifecycleManager:
    """
    Coordinated model lifecycle manager for constrained GPU memory (e.g. 4 GB VRAM).
    Ensures mutually exclusive residency on the GPU: loading LLaVA automatically releases
    Stable Diffusion and vice versa, guaranteeing that dead Python references are dropped
    before invoking PyTorch CUDA cache evacuation.
    """
    _instance: Optional["ModelLifecycleManager"] = None

    def __init__(self):
        self._registered_services: Dict[str, Any] = {}
        self._active_service: Optional[str] = None

    @classmethod
    def get_instance(cls) -> "ModelLifecycleManager":
        if cls._instance is None:
            cls._instance = ModelLifecycleManager()
        return cls._instance

    def register(self, name: str, service: Any) -> None:
        """Register a model service instance for lifecycle coordination."""
        self._registered_services[name] = service

    def prepare_for_model(self, target_service_name: str) -> None:
        """
        Ensure all conflicting heavy services are unloaded and their GPU memory reclaimed
        prior to loading the target model.
        """
        for name, service in list(self._registered_services.items()):
            if name != target_service_name:
                if hasattr(service, "unload") and callable(service.unload):
                    service.unload()
        clean_gpu_memory()
        self._active_service = target_service_name

    def release_all(self) -> None:
        """Unload all registered models and flush CUDA memory."""
        for name, service in list(self._registered_services.items()):
            if hasattr(service, "unload") and callable(service.unload):
                service.unload()
        clean_gpu_memory()
        self._active_service = None
