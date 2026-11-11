import os
import ast
from pathlib import Path
import pytest

LAB7_DIR = Path(__file__).resolve().parent.parent
APP_PATH = LAB7_DIR / "app.py"

def test_app_file_exists():
    assert APP_PATH.exists(), f"Missing app.py in {LAB7_DIR}"

def test_app_syntax_valid():
    with open(APP_PATH, "r", encoding="utf-8") as f:
        code = f.read()
    # Parse AST to ensure completely valid Python syntax
    tree = ast.parse(code, filename="app.py")
    assert tree is not None

def test_app_contains_key_workflows():
    with open(APP_PATH, "r", encoding="utf-8") as f:
        code = f.read()
    assert "Image Captioning" in code
    assert "Visual QA" in code or "Visual Question Answering" in code
    assert "Image Transformation" in code
    assert "Agent Orchestration" in code
    assert "Studio Gallery" in code
    assert "GalleryService" in code
