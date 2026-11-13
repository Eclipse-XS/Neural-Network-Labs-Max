# Neural Network Labs Max

Репозиторій містить лабораторні роботи з нейромережевих технологій. Поточний notebook-first рефакторинг охоплює лише Labs 3–4; Labs 1–2 не змінено.

| Лабораторна | Варіант | Основний workflow |
|---|---|---|
| [Lab 3](lab3_chroma_rag/README.md) | 2: ChromaDB + Markdown/TXT | corpus → chunks → embeddings → Chroma → retrieval → RAG |
| [Lab 4](lab4_img2img/README.md) | 2: Stable Diffusion Img2Img | inputs → pipeline → controlled experiments → analysis |

## Встановлення

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Перед повним виконанням потрібні мережевий доступ для моделей, Ollama з `qwen3:4b-instruct` для Lab 3 та CUDA GPU для практичного виконання Lab 4. CPU-запуск diffusion технічно можливий, але не є прийнятним коротким smoke test.

## Порядок запуску

Запускати notebooks у числовому порядку з їхньої директорії. Fast tests:

```powershell
.\.venv\Scripts\python.exe -m pytest lab3_chroma_rag/tests lab4_img2img/tests -m "not integration" -v
```

Integration tests не маскують відсутність моделей або сервісів як успішний inference. Статус перевірки наведено у [FINAL_ACCEPTANCE.md](FINAL_ACCEPTANCE.md), вихідні обмеження — у [REFACTOR_AUDIT.md](REFACTOR_AUDIT.md).

