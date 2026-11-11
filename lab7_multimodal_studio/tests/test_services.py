import os
import json
from pathlib import Path
import pytest
from PIL import Image

LAB7_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = LAB7_DIR / "src"
ASSETS_DIR = LAB7_DIR / "assets"
OUTPUTS_DIR = LAB7_DIR / "outputs"

import sys
if str(LAB7_DIR) not in sys.path:
    sys.path.insert(0, str(LAB7_DIR))

from src.utils.device import get_gpu_memory_info, clean_gpu_memory
from src.storage.gallery_service import GalleryService

def test_device_info():
    info = get_gpu_memory_info()
    assert "device" in info
    assert "allocated_mb" in info
    clean_gpu_memory()

def test_assets_exist():
    images = list(ASSETS_DIR.glob("*.png"))
    assert len(images) >= 4, f"Expected at least 4 demo assets, found {len(images)}"

def test_gallery_service_crud():
    gallery = GalleryService()
    test_img = Image.new("RGB", (64, 64), color=(200, 50, 50))
    rec_id = gallery.add_record(
        task_type="unit_test",
        prompt="test prompt",
        parameters={"test": True, "value": 42},
        output_text="Test output text",
        input_image=test_img,
        latency_sec=0.01
    )
    assert rec_id.startswith("unit_test_")
    
    rec = gallery.get_record(rec_id)
    assert rec is not None
    assert rec["task_type"] == "unit_test"
    assert rec["output_text"] == "Test output text"
    assert rec["parameters"]["value"] == 42
    
    all_recs = gallery.get_all_records()
    assert any(r["id"] == rec_id for r in all_recs)

def test_outputs_exist():
    traces = list((OUTPUTS_DIR / "traces").glob("*.json"))
    assert len(traces) >= 1
    
    benchmarks = OUTPUTS_DIR / "tables" / "studio_model_benchmarks.csv"
    assert benchmarks.exists()
    
    figures = list((OUTPUTS_DIR / "figures").glob("*.png"))
    assert len(figures) >= 1
