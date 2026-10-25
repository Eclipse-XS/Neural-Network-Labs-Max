import json
import importlib.util
from pathlib import Path
import pytest
import nbformat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
BUILDERS_DIR = PROJECT_ROOT

CONTROL_CHARS = set([chr(i) for i in range(32) if i not in (9, 10, 13)]) # exclude \t, \n, \r

def test_lab04_notebooks_nbformat_and_control_chars():
    """Перевірка валідності JSON/nbformat та відсутності ASCII-керуючих символів у ЛР4."""
    notebook_files = [
        NOTEBOOKS_DIR / "01_input_images_and_pipeline.ipynb",
        NOTEBOOKS_DIR / "02_parameter_experiments.ipynb",
        NOTEBOOKS_DIR / "03_results_analysis.ipynb",
    ]
    for nb_path in notebook_files:
        assert nb_path.exists(), f"Ноутбук {nb_path.name} не знайдено"
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        nbformat.validate(nb)
        
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type == "markdown":
                found_ctrl = [repr(ch) for ch in cell.source if ch in CONTROL_CHARS]
                assert len(found_ctrl) == 0, (
                    f"Ноутбук {nb_path.name}, комірка {idx}: виявлено керуючі символи: {found_ctrl}"
                )
                for line in cell.source.splitlines():
                    if "\t" in line and ("$" in line or "\\" in line):
                        pytest.fail(
                            f"Ноутбук {nb_path.name}, комірка {idx}: виявлено табуляцію всередині формули: {repr(line)}"
                        )

def test_lab04_notebook01_formulas_integrity():
    """Перевірка цілісності формул Img2Img (VAE, timesteps, noise schedule, CFG)."""
    nb_path = NOTEBOOKS_DIR / "01_input_images_and_pipeline.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    
    sec1_text = nb.cells[1].source
    # Перевірка на відсутність артефактів \x0c, \x08, \x07
    for ch in ["\x0c", "\x08", "\x07"]:
        assert ch not in sec1_text
    
    # Перевірка ключових LaTeX елементів
    assert r"\frac{H}{8}" in sec1_text
    assert r"\frac{W}{8}" in sec1_text
    assert r"\bar{\alpha}_{t_{\text{latent}}}" in sec1_text
    assert r"\hat{\epsilon}" in sec1_text
    assert r"\epsilon_\theta(z_t, c)" in sec1_text
    assert r"\text{init\_timestep}" in sec1_text

def test_lab04_notebook03_lora_formulas_integrity():
    """Перевірка цілісності формул LoRA та висновків у 03_results_analysis.ipynb."""
    nb_path = NOTEBOOKS_DIR / "03_results_analysis.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    
    lora_sec = nb.cells[17].source
    for ch in ["\x0c", "\x08", "\x07"]:
        assert ch not in lora_sec
        
    assert r"\frac{\alpha}{r}" in lora_sec
    assert r"$\alpha$" in lora_sec
    assert r"1024 \times 1024" in lora_sec
    assert r"1024 \times 8 + 8 \times 1024" in lora_sec

def test_lab04_builders_generate_clean_markdown():
    """Перевірка того, що всі скрипти-білдери ЛР4 створюють чисті комірки без керуючих символів."""
    builder_specs = [
        (BUILDERS_DIR / "build_notebook_01.py", "build_lab04_notebook_01"),
        (BUILDERS_DIR / "build_notebook_02.py", "build_lab04_notebook_02"),
        (BUILDERS_DIR / "build_notebook_03.py", "build_lab04_notebook_03"),
    ]
    for b_path, fn_name in builder_specs:
        assert b_path.exists(), f"Білдер {b_path.name} не існує"
        spec = importlib.util.spec_from_file_location("mod_l4_" + b_path.stem, b_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, fn_name)
        nb = fn()
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type == "markdown":
                found = [repr(ch) for ch in cell.source if ch in CONTROL_CHARS]
                assert len(found) == 0, f"{b_path.name} комірка {idx} містить керуючі символи: {found}"
                for line in cell.source.splitlines():
                    if "\t" in line and ("$" in line or "\\" in line):
                        pytest.fail(f"{b_path.name} комірка {idx} містить табуляцію у формулі: {repr(line)}")
