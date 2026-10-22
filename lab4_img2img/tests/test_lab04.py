import os
import json
import hashlib
from pathlib import Path
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
IMAGES_DIR = OUTPUTS_DIR / "images"
META_DIR = OUTPUTS_DIR / "metadata"
TABLES_DIR = OUTPUTS_DIR / "tables"

def test_01_input_images_integrity():
    """Перевірка наявності 3 оригінальних вхідних зображень розміром 512x512."""
    assert INPUTS_DIR.exists(), f"Директорія {INPUTS_DIR} не знайдена"
    input_files = list(INPUTS_DIR.glob("*.png"))
    assert len(input_files) == 3, f"Очікувалось 3 вхідних зображення, знайдено {len(input_files)}"
    
    for p in input_files:
        img = Image.open(p)
        assert img.size == (512, 512), f"Зображення {p.name} має розмір {img.size}, очікувалось (512, 512)"
        assert img.mode == "RGB", f"Зображення {p.name} має режим {img.mode}, очікувалось RGB"

def test_02_baseline_outputs_and_metadata():
    """Перевірка наявності базових зображень для всіх трьох сюжетів."""
    baseline_dir = IMAGES_DIR / "baseline"
    assert baseline_dir.exists(), "Директорія baseline зображень не існує"
    baseline_images = list(baseline_dir.glob("*.png"))
    assert len(baseline_images) == 3, f"Очікувалось 3 baseline зображення, знайдено {len(baseline_images)}"
    
    for img_path in baseline_images:
        img = Image.open(img_path)
        assert img.size == (512, 512)

def test_03_series_a_strength_outputs():
    """Перевірка серії Strength (0.3 / 0.6 / 0.9)."""
    strength_dir = IMAGES_DIR / "strength"
    assert strength_dir.exists(), "Директорія strength не існує"
    
    expected_files = [
        "lighthouse_strength0p3_cfg7p5_steps30_seed42.png",
        "lighthouse_strength0p6_cfg7p5_steps30_seed42.png",
        "lighthouse_strength0p9_cfg7p5_steps30_seed42.png"
    ]
    for ef in expected_files:
        fpath = strength_dir / ef
        assert fpath.exists(), f"Файл {ef} відсутній у серії strength"
        img = Image.open(fpath)
        assert img.size == (512, 512)

def test_04_series_b_cfg_outputs():
    """Перевірка серії Guidance Scale (5.0 / 7.5 / 10.0)."""
    cfg_dir = IMAGES_DIR / "cfg"
    assert cfg_dir.exists(), "Директорія cfg не існує"
    
    expected_files = [
        "lighthouse_strength0p6_cfg5p0_steps30_seed42.png",
        "lighthouse_strength0p6_cfg7p5_steps30_seed42.png",
        "lighthouse_strength0p6_cfg10p0_steps30_seed42.png"
    ]
    for ef in expected_files:
        fpath = cfg_dir / ef
        assert fpath.exists(), f"Файл {ef} відсутній у серії CFG"
        img = Image.open(fpath)
        assert img.size == (512, 512)

def test_05_series_c_steps_outputs():
    """Перевірка серії Steps (20 / 30 / 50)."""
    steps_dir = IMAGES_DIR / "steps"
    assert steps_dir.exists(), "Директорія steps не існує"
    
    expected_files = [
        "lighthouse_strength0p6_cfg7p5_steps20_seed42.png",
        "lighthouse_strength0p6_cfg7p5_steps30_seed42.png",
        "lighthouse_strength0p6_cfg7p5_steps50_seed42.png"
    ]
    for ef in expected_files:
        fpath = steps_dir / ef
        assert fpath.exists(), f"Файл {ef} відсутній у серії steps"
        img = Image.open(fpath)
        assert img.size == (512, 512)

def test_06_series_d_negative_prompt_outputs():
    """Перевірка порівняння з та без Negative Prompt."""
    neg_dir = IMAGES_DIR / "negative_prompt"
    assert neg_dir.exists(), "Директорія negative_prompt не існує"
    
    f_off = neg_dir / "lighthouse_strength0p6_cfg7p5_steps30_noneg_seed42.png"
    f_on = neg_dir / "lighthouse_strength0p6_cfg7p5_steps30_withneg_seed42.png"
    assert f_off.exists(), "Зображення без negative prompt не знайдено"
    assert f_on.exists(), "Зображення з negative prompt не знайдено"
    
    # Файли повинні мати різні SHA-256 хеші
    h_off = hashlib.sha256(f_off.read_bytes()).hexdigest()
    h_on = hashlib.sha256(f_on.read_bytes()).hexdigest()
    assert h_off != h_on, "Зображення з та без negative prompt не повинні бути побітово ідентичними!"

def test_07_metadata_json_completeness():
    """Перевірка повноти збережених JSON метаданих для всіх генерацій."""
    meta_files = list(META_DIR.glob("*.json"))
    assert len(meta_files) >= 10, f"Очікувалось не менше 10 метаданих, знайдено {len(meta_files)}"
    
    required_fields = {
        "input_image", "output_image", "prompt", "strength", "guidance_scale",
        "num_inference_steps", "actual_denoising_steps", "seed", "resolution",
        "model_id", "generation_time_seconds", "input_sha256", "output_sha256"
    }
    
    for mf in meta_files:
        with open(mf, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert required_fields.issubset(data.keys()), f"У файлі {mf.name} відсутні обов'язкові поля"
        assert data["resolution"] == [512, 512]
        assert data["actual_denoising_steps"] == int(data["num_inference_steps"] * data["strength"])

def test_08_generations_registry_csv():
    """Перевірка існування та структури зведеного реєстру генерацій."""
    reg_csv = TABLES_DIR / "generations_registry.csv"
    assert reg_csv.exists(), "Реєстр generations_registry.csv не знайдено"
    
    with open(reg_csv, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) >= 11, f"Очікувалось заголовок + 10 рядків генерацій, знайдено {len(lines)}"

def test_09_reproducibility_fixed_seed_hash_match():
    """Перевірка відтворюваності: генерації з однаковими параметрами дають ідентичний SHA-256."""
    meta_files = list(META_DIR.glob("*.json"))
    records = []
    for mf in meta_files:
        with open(mf, "r", encoding="utf-8") as f:
            records.append(json.load(f))
            
    # Фільтруємо генерації lighthouse при strength=0.6, cfg=7.5, steps=30, seed=42, negative=None
    matching = [r for r in records if r.get("input_image") == "01_coastal_lighthouse.png" and
                r.get("strength") == 0.6 and r.get("guidance_scale") == 7.5 and
                r.get("num_inference_steps") == 30 and r.get("seed") == 42 and r.get("negative_prompt") is None]
    
    assert len(matching) >= 2, "Не знайдено повторюваних конфігурацій для перевірки відтворюваності"
    hashes = set(r["output_sha256"] for r in matching)
    assert len(hashes) == 1, f"Порушення відтворюваності: однакова конфігурація має різні хеші: {hashes}"

def test_10_figures_generated_and_valid():
    """Перевірка наявності всіх аналітичних фігур."""
    fig_dir = OUTPUTS_DIR / "figures"
    expected_figures = [
        "input_images_overview.png",
        "original_vs_baseline_comparison.png",
        "analysis_strength_comparison.png",
        "analysis_cfg_comparison.png",
        "analysis_steps_comparison.png",
        "analysis_negative_prompt_comparison.png",
        "analysis_inference_performance.png"
    ]
    for ef in expected_figures:
        p = fig_dir / ef
        assert p.exists(), f"Фігура {ef} не знайдена"
        assert p.stat().st_size > 5000, f"Фігура {ef} має підозріло малий розмір"
