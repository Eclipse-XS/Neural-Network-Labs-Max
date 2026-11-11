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

from src.utils.device import get_gpu_memory_info, clean_gpu_memory, ModelLifecycleManager
from src.utils.security import validate_safe_image_path, to_portable_path
from src.storage.gallery_service import GalleryService

def test_device_info():
    info = get_gpu_memory_info()
    assert "device" in info
    assert "allocated_mb" in info
    clean_gpu_memory()

def test_model_lifecycle_manager():
    manager = ModelLifecycleManager.get_instance()
    
    class DummyService:
        def __init__(self):
            self.unloaded = False
        def unload(self):
            self.unloaded = True

    s1 = DummyService()
    s2 = DummyService()
    manager.register("dummy1", s1)
    manager.register("dummy2", s2)
    
    manager.prepare_for_model("dummy1")
    assert s2.unloaded is True
    assert s1.unloaded is False

def test_assets_exist():
    images = list(ASSETS_DIR.glob("*.png"))
    assert len(images) >= 4, f"Expected at least 4 demo assets, found {len(images)}"

def test_security_path_validation_and_rejection(tmp_path):
    # Valid asset path
    valid_p = ASSETS_DIR / "01_coastal_lighthouse.png"
    safe_p = validate_safe_image_path(valid_p, studio_root=LAB7_DIR)
    assert safe_p == valid_p.resolve()

    # Relative path resolution
    rel_safe = validate_safe_image_path("assets/01_coastal_lighthouse.png", studio_root=LAB7_DIR)
    assert rel_safe.exists()

    # Portable path conversion
    port = to_portable_path(valid_p, studio_root=LAB7_DIR)
    assert port == "assets/01_coastal_lighthouse.png"

    # Path traversal attack rejection
    outside_file = tmp_path / "secret.png"
    Image.new("RGB", (32, 32)).save(outside_file)
    with pytest.raises(PermissionError):
        validate_safe_image_path(outside_file, studio_root=LAB7_DIR)

    # Non-existent file rejection
    with pytest.raises(FileNotFoundError):
        validate_safe_image_path("assets/non_existent_image.png", studio_root=LAB7_DIR)

    # Disallowed extension rejection
    bad_ext = ASSETS_DIR / "evil.exe"
    bad_ext.write_text("not an image")
    try:
        with pytest.raises(ValueError):
            validate_safe_image_path(bad_ext, studio_root=LAB7_DIR)
    finally:
        bad_ext.unlink(missing_ok=True)

def test_gallery_service_crud_isolated(tmp_path):
    # Isolated test using temporary directory - does NOT touch outputs/gallery/
    gallery = GalleryService(storage_dir=tmp_path, db_name="test_gallery.db")
    test_img = Image.new("RGB", (64, 64), color=(200, 50, 50))
    rec_id = gallery.add_record(
        task_type="unit_test",
        prompt="test prompt",
        parameters={"test": True, "value": 42},
        output_text="Test output text",
        input_image=test_img,
        latency_sec=0.01,
        status="success"
    )
    assert rec_id.startswith("unit_test_")
    
    rec = gallery.get_record(rec_id)
    assert rec is not None
    assert rec["task_type"] == "unit_test"
    assert rec["output_text"] == "Test output text"
    assert rec["status"] == "success"
    assert rec["parameters"]["value"] == 42
    
    all_recs = gallery.get_all_records()
    assert any(r["id"] == rec_id for r in all_recs)

def test_gallery_service_failure_logging(tmp_path):
    gallery = GalleryService(storage_dir=tmp_path, db_name="test_fail_gallery.db")
    rec_id = gallery.add_record(
        task_type="vqa",
        prompt="question",
        parameters={"image": "missing.png"},
        output_text="",
        latency_sec=0.0,
        status="failed",
        error_message="FileNotFoundError: Image not found"
    )
    rec = gallery.get_record(rec_id)
    assert rec["status"] == "failed"
    assert "FileNotFoundError" in rec["error_message"]
    assert rec["output_image_path"] is None

def test_outputs_exist():
    traces = list((OUTPUTS_DIR / "traces").glob("*.json"))
    assert len(traces) >= 1
    
    benchmarks = OUTPUTS_DIR / "tables" / "studio_model_benchmarks.csv"
    assert benchmarks.exists()
    
    figures = list((OUTPUTS_DIR / "figures").glob("*.png"))
    assert len(figures) >= 1
