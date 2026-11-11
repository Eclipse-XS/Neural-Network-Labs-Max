import json
from pathlib import Path
import pytest
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CAPTIONS_DIR = OUTPUTS_DIR / "captions"
METADATA_DIR = OUTPUTS_DIR / "metadata"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"

def test_01_input_images_count_and_properties():
    """Перевірка наявності та властивостей 4 різнорідних вхідних зображень."""
    assert INPUTS_DIR.exists(), f"Каталог {INPUTS_DIR} не існує"
    expected_files = [
        "01_coastal_lighthouse.png",
        "02_cafe_bicycle.png",
        "03_library_interior.png",
        "04_modern_robotics_lab.png"
    ]
    for fname in expected_files:
        fpath = INPUTS_DIR / fname
        assert fpath.exists(), f"Вхідне зображення {fname} відсутнє"
        with Image.open(fpath) as img:
            assert img.size == (512, 512), f"{fname} має розмір {img.size}, очікувалось (512, 512)"
            assert img.mode == "RGB", f"{fname} має режим {img.mode}, очікувалось RGB"

def test_02_caption_experiments_registry():
    """Перевірка зведеного реєстру експериментів (20 запусків)."""
    csv_path = TABLES_DIR / "caption_experiments.csv"
    json_path = TABLES_DIR / "caption_experiments.json"
    assert csv_path.exists(), f"{csv_path} не знайдено"
    assert json_path.exists(), f"{json_path} не знайдено"
    
    df = pd.read_csv(csv_path)
    assert len(df) == 20, f"Очікувалось 20 запусків, знайдено {len(df)}"
    
    expected_cols = {
        "run_id", "image_id", "series", "prompt_style", "temperature",
        "max_new_tokens", "generated_tokens", "char_count", "latency_sec",
        "tokens_per_sec", "caption", "model_id", "device"
    }
    assert expected_cols.issubset(set(df.columns))

def test_03_captions_files_completeness():
    """Перевірка наявності 20 непустих текстових файлів згенерованих описів."""
    assert CAPTIONS_DIR.exists()
    cap_files = list(CAPTIONS_DIR.glob("*.txt"))
    assert len(cap_files) == 20, f"Очікувалось 20 файлів описів, знайдено {len(cap_files)}"
    for f in cap_files:
        text = f.read_text(encoding="utf-8").strip()
        assert len(text) > 10, f"Файл опису {f.name} занадто короткий або порожній"

def test_04_metadata_json_completeness():
    """Перевірка повноти та валідності 20 індивідуальних JSON-файлів метаданих."""
    assert METADATA_DIR.exists()
    meta_files = list(METADATA_DIR.glob("*.json"))
    assert len(meta_files) == 20, f"Очікувалось 20 JSON-файлів метаданих, знайдено {len(meta_files)}"
    for f in meta_files:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        assert "run_id" in data
        assert "latency_sec" in data
        assert data["latency_sec"] > 0
        assert "generated_tokens" in data
        assert data["generated_tokens"] > 0
        assert "caption" in data

def test_05_series_a_prompt_wording_comparison():
    """Перевірка повноти серії А (12 запусків: 4 зображення x 3 стилі промптів)."""
    df = pd.read_csv(TABLES_DIR / "caption_experiments.csv")
    df_a = df[df["series"] == "prompt_wording"]
    assert len(df_a) == 12
    styles = set(df_a["prompt_style"].unique())
    assert styles == {"concise", "descriptive", "detailed"}
    images = set(df_a["image_id"].unique())
    assert len(images) == 4

def test_06_series_b_temperature_sweep():
    """Перевірка серії B (4 температури: 0.0, 0.2, 0.7, 1.0)."""
    df = pd.read_csv(TABLES_DIR / "caption_experiments.csv")
    df_b = df[df["series"] == "temperature"]
    assert len(df_b) == 4
    temps = set(df_b["temperature"].unique())
    assert temps == {0.0, 0.2, 0.7, 1.0}

def test_07_series_c_token_budgets():
    """Перевірка серії C (4 бюджети токенів: 32, 64, 128, 256)."""
    df = pd.read_csv(TABLES_DIR / "caption_experiments.csv")
    df_c = df[df["series"] == "token_budget"]
    assert len(df_c) == 4
    budgets = set(df_c["max_new_tokens"].unique())
    assert budgets == {32, 64, 128, 256}

def test_08_figures_generated_and_valid():
    """Перевірка наявності всіх 4 аналітичних графіків у форматі PNG."""
    expected_figures = [
        "input_images_and_concise_captions.png",
        "latency_vs_token_budget.png",
        "prompt_wording_comparison.png",
        "temperature_lexical_diversity.png"
    ]
    for fig_name in expected_figures:
        fig_path = FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Графік {fig_name} не знайдено"
        assert fig_path.stat().st_size > 5000, f"Графік {fig_name} пошкоджений або порожній"

def test_09_quality_evaluation_table():
    """Перевірка експертної таблиці оцінювання якості описів."""
    eval_csv = TABLES_DIR / "caption_quality_evaluation.csv"
    assert eval_csv.exists()
    df = pd.read_csv(eval_csv)
    assert len(df) == 4, f"Очікувалось 4 рядки оцінки для 4 зображень, знайдено {len(df)}"
    assert "object_coverage_pct" in df.columns
    assert "hallucination_detected" in df.columns
    assert (df["object_coverage_pct"] >= 80.0).all()
