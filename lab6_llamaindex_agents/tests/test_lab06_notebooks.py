import os
from pathlib import Path
import nbformat
import pytest

LAB6_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = LAB6_DIR / "notebooks"

EXPECTED_NOTEBOOKS = [
    "01_agent_tools_and_memory.ipynb",
    "02_multistep_agent_workflows.ipynb",
    "03_agent_evaluation.ipynb"
]

@pytest.mark.parametrize("nb_name", EXPECTED_NOTEBOOKS)
def test_notebook_exists_and_valid(nb_name):
    nb_path = NOTEBOOKS_DIR / nb_name
    assert nb_path.exists(), f"Notebook {nb_name} does not exist in {NOTEBOOKS_DIR}"
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    nbformat.validate(nb)
    assert len(nb.cells) >= 8, f"Notebook {nb_name} must contain at least 8 cells, got {len(nb.cells)}"

@pytest.mark.parametrize("nb_name", EXPECTED_NOTEBOOKS)
def test_notebook_has_executed_outputs(nb_name):
    nb_path = NOTEBOOKS_DIR / nb_name
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) >= 4, f"Notebook {nb_name} must contain at least 4 code cells"
    
    cells_with_output = [c for c in code_cells if len(c.get("outputs", [])) > 0]
    assert len(cells_with_output) == len(code_cells), (
        f"All code cells in {nb_name} must have non-empty outputs. "
        f"Found {len(cells_with_output)}/{len(code_cells)}"
    )

@pytest.mark.parametrize("nb_name", EXPECTED_NOTEBOOKS)
def test_notebook_metadata_and_titles(nb_name):
    nb_path = NOTEBOOKS_DIR / nb_name
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    
    first_cell = nb.cells[0]
    assert first_cell.cell_type == "markdown"
    assert "Лабораторна робота №6" in first_cell.source
    assert "Варіант 2" in first_cell.source
    assert "Львівська політехніка" in first_cell.source
