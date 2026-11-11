import os
import ast
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

LAB7_DIR = Path(__file__).resolve().parent.parent
APP_PATH = LAB7_DIR / "app.py"

def test_app_file_exists():
    assert APP_PATH.exists(), f"Missing app.py in {LAB7_DIR}"

def test_app_syntax_valid():
    with open(APP_PATH, "r", encoding="utf-8") as f:
        code = f.read()
    tree = ast.parse(code, filename="app.py")
    assert tree is not None

def test_app_contains_key_workflows():
    with open(APP_PATH, "r", encoding="utf-8") as f:
        code = f.read()
    assert "Image Captioning" in code
    assert "Visual QA" in code
    assert "Image Transformation" in code
    assert "Agent Orchestration" in code
    assert "Studio Gallery" in code
    assert "GalleryService" in code

def test_app_smoke_execution():
    """Execute Streamlit AppTest to ensure the page boots without runtime exceptions."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    assert not at.exception, f"AppTest raised an unhandled exception: {at.exception}"
    assert at.title[0].value == "🎨 Multimodal Visual Content Studio"
    assert len(at.tabs) == 5
    assert len(at.sidebar.metric) >= 3
