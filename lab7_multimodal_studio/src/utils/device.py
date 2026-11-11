import gc
import torch
from typing import Dict, Any

def get_device() -> torch.device:
    """Return available torch device, preferring CUDA if available."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

def clean_gpu_memory() -> None:
    """Force garbage collection and clear PyTorch CUDA caching allocators."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def get_gpu_memory_info() -> Dict[str, Any]:
    """Return current GPU memory statistics in megabytes."""
    if not torch.cuda.is_available():
        return {"device": "cpu", "allocated_mb": 0.0, "reserved_mb": 0.0, "total_mb": 0.0}
    
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
