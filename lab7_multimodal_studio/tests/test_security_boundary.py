import os
import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from PIL import Image

LAB7_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = LAB7_DIR / "src"
ASSETS_DIR = LAB7_DIR / "assets"

import sys
if str(LAB7_DIR) not in sys.path:
    sys.path.insert(0, str(LAB7_DIR))

from src.utils.security import (
    validate_safe_image_path,
    to_portable_path,
    sanitize_path_string,
    SecurityValidationError,
    ALLOWED_IMAGE_EXTENSIONS,
    MAX_IMAGE_DIMENSION
)
from src.storage.gallery_service import GalleryService

try:
    import llama_index
    HAS_LLAMA_INDEX = True
except ImportError:
    HAS_LLAMA_INDEX = False


def test_allowed_asset_accepted():
    valid_p = ASSETS_DIR / "01_coastal_lighthouse.png"
    assert valid_p.exists()
    safe_p = validate_safe_image_path(valid_p, studio_root=LAB7_DIR)
    assert safe_p == valid_p.resolve()


def test_relative_allowed_path_accepted():
    rel_p = "assets/01_coastal_lighthouse.png"
    safe_p = validate_safe_image_path(rel_p, studio_root=LAB7_DIR)
    assert safe_p.exists()
    assert (LAB7_DIR / "assets") in safe_p.parents or safe_p.parent == (LAB7_DIR / "assets")


def test_harmless_dotdot_inside_sandbox_accepted():
    dotdot_p = "assets/../assets/01_coastal_lighthouse.png"
    safe_p = validate_safe_image_path(dotdot_p, studio_root=LAB7_DIR)
    assert safe_p == (ASSETS_DIR / "01_coastal_lighthouse.png").resolve()


def test_existing_external_image_rejected(tmp_path):
    outside_file = tmp_path / "external_photo.png"
    Image.new("RGB", (32, 32)).save(outside_file)
    assert outside_file.exists()

    with pytest.raises(SecurityValidationError) as exc_info:
        validate_safe_image_path(outside_file, studio_root=LAB7_DIR)

    msg = str(exc_info.value)
    assert "Access denied" in msg
    assert "outside allowed studio image directories" in msg
    # Assert no host absolute directory was exposed in the message
    assert str(tmp_path.resolve()) not in msg
    assert "C:\\Users" not in msg
    assert "/home/" not in msg
    assert "/Users/" not in msg


def test_traversal_dotdot_outside_sandbox_rejected():
    traversal_path = "../lab4_img2img/inputs/01_coastal_lighthouse.png"
    with pytest.raises(SecurityValidationError) as exc_info:
        validate_safe_image_path(traversal_path, studio_root=LAB7_DIR)

    msg = str(exc_info.value)
    assert "Access denied" in msg
    assert "01_coastal_lighthouse.png" in msg
    assert "C:\\Users" not in msg
    assert "PycharmProjects" not in msg


def test_missing_file_raises_filenotfound_not_security():
    missing_path = "assets/completely_missing_photo_123.png"
    with pytest.raises(FileNotFoundError) as exc_info:
        validate_safe_image_path(missing_path, studio_root=LAB7_DIR)

    msg = str(exc_info.value)
    assert "Image file not found" in msg
    assert "completely_missing_photo_123.png" in msg
    assert "C:\\Users" not in msg


def test_unsupported_extension_rejected(tmp_path):
    bad_file = tmp_path / "script.sh"
    bad_file.write_text("echo evil")

    with pytest.raises(ValueError) as exc_info:
        validate_safe_image_path(bad_file, studio_root=LAB7_DIR)

    msg = str(exc_info.value)
    assert "Unsupported image extension" in msg


def test_invalid_corrupted_image_content(tmp_path):
    fake_img = LAB7_DIR / "inputs" / "corrupted_test.png"
    fake_img.parent.mkdir(exist_ok=True)
    fake_img.write_text("This is purely plain text, not a valid PNG binary structure")

    try:
        with pytest.raises(ValueError) as exc_info:
            validate_safe_image_path(fake_img, studio_root=LAB7_DIR)
        msg = str(exc_info.value)
        assert "Corrupted or invalid image file" in msg
    finally:
        fake_img.unlink(missing_ok=True)


def test_excessive_image_dimensions(tmp_path):
    huge_img_path = LAB7_DIR / "inputs" / "huge_test.png"
    huge_img_path.parent.mkdir(exist_ok=True)
    huge_img = Image.new("RGB", (MAX_IMAGE_DIMENSION + 10, 100))
    huge_img.save(huge_img_path)

    try:
        with pytest.raises(ValueError) as exc_info:
            validate_safe_image_path(huge_img_path, studio_root=LAB7_DIR)
        msg = str(exc_info.value)
        assert "exceed maximum allowed" in msg
    finally:
        huge_img_path.unlink(missing_ok=True)


def test_symlink_traversal_rejected_if_supported(tmp_path):
    outside_file = tmp_path / "real_outside.png"
    Image.new("RGB", (32, 32)).save(outside_file)
    symlink_path = LAB7_DIR / "inputs" / "symlink_leak.png"

    try:
        os.symlink(outside_file, symlink_path)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation not permitted or supported on this host")

    try:
        with pytest.raises(SecurityValidationError) as exc_info:
            validate_safe_image_path(symlink_path, studio_root=LAB7_DIR)
        msg = str(exc_info.value)
        assert "Access denied" in msg
        assert str(outside_file.resolve()) not in msg
    finally:
        symlink_path.unlink(missing_ok=True)


def test_sanitizer_removes_host_paths():
    sample_text = f"Failed to open C:\\Users\\Eclipse\\PycharmProjects\\secret\\file.png in {LAB7_DIR.resolve()}"
    sanitized = sanitize_path_string(sample_text, studio_root=LAB7_DIR)
    assert "Users\\Eclipse" not in sanitized
    assert "C:\\Users" not in sanitized
    assert str(LAB7_DIR.resolve()) not in sanitized
    assert "<studio_root>" in sanitized or "<user_home>" in sanitized


def test_tool_rejection_no_inference(tmp_path):
    mock_llava = MagicMock()
    mock_diffusion = MagicMock()
    isolated_gallery = GalleryService(storage_dir=tmp_path, db_name="test_sec.db")

    outside_img = tmp_path / "leak.png"
    Image.new("RGB", (32, 32)).save(outside_img)

    if not HAS_LLAMA_INDEX:
        # Direct tool logic verification in CI environment without heavy llama-index dependency
        try:
            safe_p = validate_safe_image_path(outside_img, studio_root=LAB7_DIR)
            mock_llava.generate_caption(Image.open(safe_p))
        except Exception as e:
            portable_p = to_portable_path(outside_img, studio_root=LAB7_DIR)
            clean_err = sanitize_path_string(str(e), studio_root=LAB7_DIR)
            rec_id = isolated_gallery.add_record(
                task_type="caption",
                prompt="concise",
                parameters={"style": "concise", "image": portable_p},
                output_text="",
                latency_sec=0.0,
                status="failed",
                error_message=clean_err
            )
            raw_result = json.dumps({"error": clean_err, "status": "failed", "record_id": rec_id})
    else:
        from src.agent.studio_agent import StudioAgent
        agent = StudioAgent(
            llava_service=mock_llava,
            diffusion_service=mock_diffusion,
            gallery_service=isolated_gallery,
            studio_root=LAB7_DIR
        )
        root_agent = agent.workflow.agents[agent.workflow.root_agent]
        tools_dict = {t.metadata.name: t for t in root_agent.tools}
        caption_tool = tools_dict["caption_image"]
        raw_result = caption_tool.fn(image_path=str(outside_img), style="concise")

    parsed = json.loads(raw_result)

    assert parsed["status"] == "failed"
    assert "Access denied" in parsed["error"]
    assert str(tmp_path.resolve()) not in parsed["error"]
    assert "C:\\Users" not in parsed["error"]

    # Verify model was NEVER called for inference
    mock_llava.generate_caption.assert_not_called()
    mock_llava.answer_question.assert_not_called()

    # Verify gallery logged failure safely
    rec = isolated_gallery.get_record(parsed["record_id"])
    assert rec["status"] == "failed"
    assert "Access denied" in rec["error_message"]
    assert str(tmp_path.resolve()) not in rec["error_message"]
