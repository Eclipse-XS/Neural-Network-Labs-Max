# Лабораторна робота №3: Реалізація RAG-системи для витягування знань
## Варіант №2: Chroma + Markdown / Текст

Цей проєкт містить повну, незалежну та експериментально підтверджену реалізацію варіанта №2 лабораторної роботи №3 з дисципліни «Нейромережеві технології і системи».

---

## 1. Склад проєкту

```text
lab03_chroma_rag/
├── README.md                              # Опис структури та інструкції запуску
├── data/
│   └── knowledge_base/                    # 8 тематичних документів (6 .md, 2 .txt)
│       ├── 01_data_quality.md
│       ├── 02_feature_engineering.md
│       ├── 03_model_training.md
│       ├── 04_validation_and_metrics.md
│       ├── 05_deployment_architecture.md
│       ├── 06_model_monitoring.md
│       ├── 07_incident_response.txt
│       └── 08_security_and_reproducibility.txt
├── notebooks/
│   ├── 01_corpus_preparation.ipynb        # Завантаження, аудит та чанкінг корпусу
│   ├── 02_chroma_retrieval.ipynb          # ChromaDB, Similarity vs MMR, фільтри метаданих
│   └── 03_rag_evaluation.ipynb            # Наскрізний RAG із локальною LLM (qwen3:4b)
├── outputs/
│   ├── chroma_db/                         # Персистентна база векторів ChromaDB
│   ├── figures/                           # Згенеровані графіки та діаграми
│   ├── tables/                            # Звіти аудиту, реєстри чанків, порівняльні таблиці
│   └── answers/                           # Фактичні згенеровані відповіді LLM
├── tests/
│   └── test_lab03.py                      # 10 автоматичних тестів pytest
└── report/
    ├── report.md                          # Повний академічний звіт (14 розділів)
    ├── defense_notes.md                   # Відповіді на 10 контрольних запитань
    └── methodology_audit.md               # Матриця відповідності всім вимогам методички
```

---

## 2. Швидкий запуск

### Вимоги:
- Python 3.12+ у віртуальному оточенні (`NNTS_VARIANT_2/.venv`);
- Локальний OpenAI-сумісний сервер (наприклад, Ollama з моделлю `qwen3:4b-instruct` або vLLM за адресою `http://127.0.0.1:11434/v1`).

### Запуск тестів:
```powershell
.\NNTS_VARIANT_2\.venv\Scripts\python.exe -m pytest NNTS_VARIANT_2/lab03_chroma_rag/tests/test_lab03.py -v
```

### Запуск ноутбуків у Jupyter:
```powershell
.\NNTS_VARIANT_2\.venv\Scripts\jupyter.exe lab
```

---

## 3. Ключові результати
1. **База знань:** 8 документів, 23 484 символи, 0 порожніх файлів, 0 дублікатів.
2. **Чанкінг:** конфігурація `chunk_size=700`, `chunk_overlap=100` утворила 55 фрагментів із збереженням метаданих `source`, `type`, `chunk_id`, `document_id`.
3. **ChromaDB:** персистентне сховище на ембеддингах `sentence-transformers/all-MiniLM-L6-v2` ($d=384$).
4. **Similarity vs MMR:** MMR суттєво підвищує різноманітність джерел (до 75% унікальних документів проти 50% у Similarity) на багатоаспектних запитах, водночас для вузьких фактів видачі збігаються.
5. **Фільтри метаданих:** підтверджено 100% точність фільтрації за `type="md"` та за конкретним файлом джерела.
6. **Локальна LLM:** згенеровано 18 відповідей (9 Similarity, 9 MMR). Підтверджено 100% заземленість і абсолютну стійкість до галюцинацій (коректна відмова на запиті про квантові обчислення).
