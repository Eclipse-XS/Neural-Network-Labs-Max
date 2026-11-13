# Lab 3: ChromaDB RAG

Мета: дослідити повний RAG workflow для Markdown/TXT корпусу. Варіант 2: ChromaDB, `all-MiniLM-L6-v2`, similarity/MMR, metadata filters і локальна LLM.

## Технології та входи

- 8 UTF-8 документів у `data/knowledge_base/`;
- LangChain text splitters, Sentence Transformers, ChromaDB;
- Ollama/OpenAI-сумісний endpoint, за замовчуванням `http://127.0.0.1:11434/v1`;
- модель генерації `qwen3:4b-instruct`.

Встановлення виконується з кореневого `requirements.txt`. Модель Ollama готується окремо: `ollama pull qwen3:4b-instruct`.

## Порядок виконання

1. `notebooks/01_corpus_preparation.ipynb` — discovery, UTF-8 validation, статистика, chunking і metadata.
2. `notebooks/02_chroma_retrieval.ipynb` — embeddings, Chroma persistence, similarity/MMR, filters і retrieval evaluation.
3. `notebooks/03_rag_evaluation.ipynb` — endpoint check, реальна генерація, latency, grounding review та unsupported questions.

Запускайте Jupyter з `lab3_chroma_rag/notebooks`, або задайте цю директорію як working directory. Notebook 02 потребує artifact Notebook 01, Notebook 03 — індекс Notebook 02. Це послідовний лабораторний workflow, а не незалежні експерименти.

## Очікувані outputs

`outputs/tables`, `outputs/figures`, `outputs/chroma_db`, `outputs/answers`. Наявні у Git файли є історичними результатами й не доводять поточний успішний запуск. Поточний запуск підтверджують execution counts без помилок, актуальні timestamps і журнал acceptance.

## Обмеження

Embedding model завантажується з Hugging Face. LLM evaluation на малому наборі не доводить загальну стійкість до hallucinations. Chroma distance не слід називати probability. Оригінальної методички в репозиторії немає, тому формальна відповідність варіанту не підтверджена незалежно.

Fast tests: `pytest lab3_chroma_rag/tests -m "not integration" -v`. Integration test запускається лише за наявності endpoint і моделі; skip не є успішним inference.

