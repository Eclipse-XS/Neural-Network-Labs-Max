# Лабораторна робота №3. Локальна RAG-система на базі ChromaDB

## Reproducible execution (Windows 11 / Python 3.13)

Run from the repository root in PowerShell. Each lab has an isolated `.venv` and a complete `requirements.lock.txt`.

```powershell
.\setup.ps1 -Lab 3
.\run.ps1 -Lab 3 -Smoke
.\run.ps1 -Lab 3
```

Ollama: http://127.0.0.1:11434. Setup reconstructs the CPU-only qwen3:4b-studio from the committed Modelfile.

`-Cuda` selects PyTorch 2.14.1+cu130 from the official CUDA 13.0 wheel index. The NVIDIA driver is required; these wheels do not require a separate CUDA toolkit. Omit this switch for CPU PyTorch. Vision smoke tests use the original model weights, batch size 1, 512x512 images and short real inference. They do not establish image quality or repeat the full experimental matrix.

Model revision SHAs and weight checksums are in the root `models.lock.json`. Distributed input checksums are in `inputs.sha256.json` (text is normalized to LF). Setup downloads required models and NNTS datasets. Smoke tests use real services; unavailable services cause a failure. See the root `REPRODUCIBILITY.md` for sources, service preparation and verification limits.

Побудова локального конвеєра Retrieval-Augmented Generation (RAG) над корпусом технічних документів з MLOps у форматах Markdown і TXT із використанням ChromaDB, `sentence-transformers/all-MiniLM-L6-v2`, LangChain та локальної LLM (`qwen3:4b-instruct` / `Qwen2.5-1.5B-Instruct-AWQ`).

---

## Структура каталогу

```text
lab3_chroma_rag/
├── README.md
├── data/
│   └── knowledge_base/                  # 8 документів MLOps (6 .md, 2 .txt)
├── notebooks/
│   ├── 01_corpus_preparation.ipynb      # Завантаження, перевірка дублікатів, чанкінг
│   ├── 02_chroma_retrieval.ipynb        # Індексація ChromaDB, Similarity vs MMR, фільтри
│   └── 03_rag_evaluation.ipynb          # RAG-ланцюг із локальною LLM та оцінка
└── outputs/
    ├── tables/                          # corpus_audit_report.csv, chunking_experiments.csv, chunks_summary.csv, chunks_prepared.json, mmr_hyperparameter_tuning.csv, retrieval_comparison_similarity_vs_mmr.csv, retrieval_queries_details.json, rag_evaluation_summary.csv
    ├── figures/                         # corpus_document_lengths.png, chunk_size_distribution.png, mmr_hyperparameter_sensitivity.png, retrieval_similarity_vs_mmr.png, rag_grounding_and_metrics.png
    └── answers/                         # Повні відповіді Similarity та MMR для всіх запитів (q1_answers.txt–q9_answers.txt)
```

---

## Послідовність виконання

1. **`01_corpus_preparation.ipynb`** — завантажує 8 файлів із `data/knowledge_base/`, розбиває їх на 55 чанків через `RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)` та зберігає результат у `outputs/tables/chunks_prepared.json`.
2. **`02_chroma_retrieval.ipynb`** — ініціалізує локальне векторне сховище в `outputs/chroma_db/` з ембедингами `all-MiniLM-L6-v2` (384-d), порівнює Similarity і MMR (`k` $\in \{3, 4, 5\}$, $\lambda \in \{0.2, 0.5, 0.8\}$, `fetch_k` $\in \{8, 12, 16\}$) та тестує фільтрацію за метаданими `type` і `source`.
3. **`03_rag_evaluation.ipynb`** — за потреби автоматично відновлює `outputs/chroma_db/` з `outputs/tables/chunks_prepared.json`, підключається через [`VLLMClient`](../lab2_vllm/cli.py) до локального сервера vLLM (`:8000`) або Ollama (`:11434`) і виконує 9 контрольних запитів (`Q1`–`Q9`) для обох ретриверів.

> Бінарна директорія `outputs/chroma_db/` виключена з Git і відтворюється детерміновано під час запуску `02` або `03`.

---

## Спостережувані результати

- На 9 контрольних запитах (`outputs/tables/rag_evaluation_summary.csv`, артефакти `outputs/answers/q1_answers.txt`–`q9_answers.txt`) Similarity RAG забезпечив валідне цитування файлів джерел для всіх доменних запитів (`Q1`–`Q8`: `8/8`, сумарно `8/9` або `88.9%`), тоді як MMR RAG — для `7/8` доменних запитів і `Q9` (`8/9` або `88.9%`; на `Q4` MMR процитував `chunk_id` замість імені файла, через що автоматична перевірка зафіксувала `Citation Validity = INVALID`). Фактичне узгодження тверджень (`Factual Grounding`) залишається з чесним статусом `NOT_EVALUATED`.
- MMR (`k=4`, `fetch_k=12`, $\lambda=0.5$) залучав ширший набір документів на багатоаспектних запитах (`Q1`, `Q4`, `Q5`, середнє різноманіття `0.56` проти `0.50` у Similarity).
- Фільтри метаданих (`Q6`: `type="md"`, `Q7`–`Q8`: `source=...`) строго ізолювали простір пошуку (`Filter Compliance = 100%`), а позадоменний запит `Q9` коректно завершився відмовою в обох режимах (`OOD Abstention = PASS`, `outputs/answers/q9_answers.txt`).

Обмеження: виконання генерації у `03_rag_evaluation.ipynb` вимагає активного локального сервера Ollama або vLLM; базова конфігурація розрахована на невеликий доменний корпус (55 чанків) без зовнішнього переранжування (reranker).
