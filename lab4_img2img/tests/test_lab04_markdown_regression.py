import json
from pathlib import Path
import pytest
import nbformat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

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

def test_lab04_notebooks_structure_and_dependencies():
    """Перевірка структури, порядку виконання та наявності змістовного виводу в блокнотах ЛР4."""
    nb1_path = NOTEBOOKS_DIR / "01_input_images_and_pipeline.ipynb"
    nb2_path = NOTEBOOKS_DIR / "02_parameter_experiments.ipynb"
    nb3_path = NOTEBOOKS_DIR / "03_results_analysis.ipynb"
    
    for path in [nb1_path, nb2_path, nb3_path]:
        assert path.exists(), f"Блокнот {path.name} обов'язковий"
        with open(path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        assert len(nb.cells) >= 12, f"Блокнот {path.name} повинен бути змістовним (знайдено лише {len(nb.cells)} комірок)"
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        assert len(code_cells) >= 4, f"Блокнот {path.name} повинен містити безпосередній виконуваний код"
        outputs_count = sum(len(c.get("outputs", [])) for c in code_cells)
        assert outputs_count > 0, f"Блокнот {path.name} повинен містити верифіковані виходи"
