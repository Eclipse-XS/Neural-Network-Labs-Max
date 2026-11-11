# Лабораторна робота №3. Локальна RAG-система на базі ChromaDB

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
    ├── tables/                          # chunks_prepared.json, retrieval_queries_details.json, rag_evaluation_summary.csv
    ├── figures/                         # retrieval_similarity_vs_mmr.png, rag_grounding_and_metrics.png
    └── answers/                         # Репрезентативні відповіді (Q1, Q7, Q9)
```

---

## Послідовність виконання

1. **`01_corpus_preparation.ipynb`** — завантажує 8 файлів із `data/knowledge_base/`, розбиває їх на 55 чанків через `RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)` та зберігає результат у `outputs/tables/chunks_prepared.json`.
2. **`02_chroma_retrieval.ipynb`** — ініціалізує локальне векторне сховище в `outputs/chroma_db/` з ембедингами `all-MiniLM-L6-v2` (384-d), порівнює Similarity і MMR (`k` $\in \{3, 4, 5\}$, $\lambda \in \{0.2, 0.5, 0.8\}$, `fetch_k` $\in \{8, 12, 16\}$) та тестує фільтрацію за метаданими `type` і `source`.
3. **`03_rag_evaluation.ipynb`** — за потреби автоматично відновлює `outputs/chroma_db/` з `outputs/tables/chunks_prepared.json`, підключається через [`VLLMClient`](../lab2_vllm/cli.py) до локального сервера vLLM (`:8000`) або Ollama (`:11434`) і виконує 9 контрольних запитів (`Q1`–`Q9`) для обох ретриверів.

> Бінарна директорія `outputs/chroma_db/` виключена з Git і відтворюється детерміновано під час запуску `02` або `03`.

---

## Спостережувані результати

- На 9 контрольних запитах (`outputs/tables/rag_evaluation_summary.csv`) обидва ретривери забезпечили релевантний контекст із явним цитуванням джерел (`Q1`–`Q8`).
- MMR (`k=4`, `fetch_k=12`, $\lambda=0.5$) залучав ширший набір документів на багатоаспектних запитах (`Q1`, `Q4`, `Q5`).
- Фільтри метаданих (`Q6`: `type="md"`, `Q7`–`Q8`: `source=...`) строго ізолювали простір пошуку, а позадоменний запит `Q9` коректно завершився відмовою без галюцинацій (`outputs/answers/q9_answers.txt`).

Обмеження: виконання генерації у `03_rag_evaluation.ipynb` вимагає активного локального сервера Ollama або vLLM; базова конфігурація розрахована на невеликий доменний корпус (55 чанків) без зовнішнього переранжування (reranker).
