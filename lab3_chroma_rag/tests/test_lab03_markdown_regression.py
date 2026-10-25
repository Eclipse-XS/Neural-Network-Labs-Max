import json
import importlib.util
from pathlib import Path
import pytest
import nbformat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
BUILDERS_DIR = PROJECT_ROOT

CONTROL_CHARS = set([chr(i) for i in range(32) if i not in (9, 10, 13)]) # exclude \t, \n, \r

def test_lab03_notebooks_nbformat_and_control_chars():
    """Перевірка валідності JSON/nbformat та відсутності ASCII-керуючих символів у ЛР3."""
    notebook_files = [
        NOTEBOOKS_DIR / "01_corpus_preparation.ipynb",
        NOTEBOOKS_DIR / "02_chroma_retrieval.ipynb",
        NOTEBOOKS_DIR / "03_rag_evaluation.ipynb",
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

def test_lab03_notebook02_l2_and_cosine_formula_integrity():
    """Перевірка коректності запису формул L2 відстані та косинусної подібності."""
    nb_path = NOTEBOOKS_DIR / "02_chroma_retrieval.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    
    sec1_text = nb.cells[1].source
    # Перевірка назви моделі без пошкодження бекслешем
    assert "all-MiniLM-L6-v2" in sec1_text
    assert "\x07ll-MiniLM-L6-v2" not in sec1_text
    assert "\\all-MiniLM-L6-v2" not in sec1_text
    
    # Перевірка L2 формули з коректними математичними делімітерами
    assert r"\|q - d_i\|_2^2" in sec1_text
    assert r"2 - 2\cos(q, d_i)" in sec1_text
    assert "$$" in sec1_text
    assert "$L_2$" in sec1_text
    assert "$L_2^2$" in sec1_text

def test_lab03_builders_generate_clean_markdown():
    """Перевірка того, що всі скрипти-білдери ЛР3 створюють чисті комірки без керуючих символів."""
    builder_specs = [
        (BUILDERS_DIR / "build_notebook_01.py", "build_notebook_01"),
        (BUILDERS_DIR / "build_notebook_02.py", "build_notebook_02"),
        (BUILDERS_DIR / "build_notebook_03.py", "build_lab03_notebook_03"),
    ]
    for b_path, fn_name in builder_specs:
        assert b_path.exists(), f"Білдер {b_path.name} не існує"
        spec = importlib.util.spec_from_file_location("mod_l3_" + b_path.stem, b_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, fn_name)
        nb = fn()
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type == "markdown":
                found = [repr(ch) for ch in cell.source if ch in CONTROL_CHARS]
                assert len(found) == 0, f"{b_path.name} комірка {idx} містить керуючі символи: {found}"
