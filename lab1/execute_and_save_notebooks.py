import sys
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

NOTEBOOKS_DIR = Path(__file__).parent / "notebooks"
NOTEBOOKS = [
    "01_enron_eda_preprocessing.ipynb",
    "02_enron_transformer_classifier.ipynb",
    "03_wikitext_eda_preprocessing.ipynb",
    "04_wikitext_transformer_decoder.ipynb",
]

def execute_and_save(nb_name: str):
    nb_path = NOTEBOOKS_DIR / nb_name
    print(f"\n" + "=" * 65, flush=True)
    print(f"Виконання та збереження виходів: {nb_name}", flush=True)
    print("=" * 65, flush=True)
    start_t = time.time()

    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOKS_DIR)}},
    )

    try:
        client.execute()
        elapsed = time.time() - start_t
        # Зберігаємо виконаний ноутбук разом із збереженими outputs (графіки, таблиці)
        nbformat.write(nb, nb_path)
        print(f"[OK] {nb_name} успішно виконано та збережено за {elapsed:.1f}s.", flush=True)
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - start_t
        print(f"[ERROR] Помилка у {nb_name} після {elapsed:.1f}s:", flush=True)
        print(e, flush=True)
        return False, elapsed

def main():
    target_notebooks = sys.argv[1:] if len(sys.argv) > 1 else NOTEBOOKS
    results = {}
    for nb_name in target_notebooks:
        success, elapsed = execute_and_save(nb_name)
        results[nb_name] = (success, elapsed)
        if not success:
            print(f"\n[ЗБІЙ] Зупинка через помилку у {nb_name}!")
            sys.exit(1)

    print("\n" + "=" * 65)
    print("ПІДСУМОК НАВІЗНОГО ВИКОНАННЯ ТА ЗБЕРЕЖЕННЯ OUTPUTS:")
    print("=" * 65)
    for k, (succ, el) in results.items():
        st = "PASSED & SAVED" if succ else "FAILED"
        print(f"  {k:45s} : {st} ({el:.1f}s)")
    print("Усі 4 ноутбуки містять актуальні виконані виходи, графіки та таблиці.")

if __name__ == "__main__":
    main()
