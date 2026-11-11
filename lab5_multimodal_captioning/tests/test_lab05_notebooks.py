from pathlib import Path
import pytest
import nbformat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

CONTROL_CHARS = set([chr(i) for i in range(32) if i not in (9, 10, 13)]) # exclude \t, \n, \r

def test_lab05_notebooks_exist_and_valid_nbformat():
    """Перевірка наявності та JSON-валідності всіх трьох блокнотів ЛР5."""
    notebook_files = [
        NOTEBOOKS_DIR / "01_image_audit_and_model_setup.ipynb",
        NOTEBOOKS_DIR / "02_captioning_experiments.ipynb",
        NOTEBOOKS_DIR / "03_caption_quality_analysis.ipynb",
    ]
    for nb_path in notebook_files:
        assert nb_path.exists(), f"Блокнот {nb_path.name} не знайдено"
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        nbformat.validate(nb)
        
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type == "markdown":
                found_ctrl = [repr(ch) for ch in cell.source if ch in CONTROL_CHARS]
                assert len(found_ctrl) == 0, (
                    f"Блокнот {nb_path.name}, комірка {idx}: знайдено керуючі символи: {found_ctrl}"
                )
                for line in cell.source.splitlines():
                    if "\t" in line and ("$" in line or "\\" in line):
                        pytest.fail(f"Блокнот {nb_path.name}, комірка {idx}: табуляція у формулі: {repr(line)}")

def test_lab05_notebooks_substance_and_outputs():
    """Перевірка змістовності блокнотів ЛР5, наявності коду та збережених вихідних результатів."""
    for fname in ["01_image_audit_and_model_setup.ipynb", "02_captioning_experiments.ipynb", "03_caption_quality_analysis.ipynb"]:
        nb_path = NOTEBOOKS_DIR / fname
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        assert len(nb.cells) >= 8, f"Блокнот {fname} має містити щонайменше 8 комірок"
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        assert len(code_cells) >= 3, f"Блокнот {fname} має містити щонайменше 3 кодові комірки"
        outputs_count = sum(len(c.get("outputs", [])) for c in code_cells)
        assert outputs_count > 0, f"Блокнот {fname} має містити виконані результати (знайдено 0)"
