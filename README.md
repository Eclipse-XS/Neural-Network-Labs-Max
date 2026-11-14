# Нейромережеві технології і системи — Лабораторні роботи 1–7

Start with [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for isolated Windows environments, pinned dependencies, model/data reconstruction and live smoke tests for Labs 1-7.

Реалізації лабораторних робіт із курсу «Технології та системи нейронних мереж» (Національний університет «Львівська політехніка», кафедра систем штучного інтелекту, спеціальність 122 «Комп'ютерні науки»). Проєкт охоплює трансформерні моделі на PyTorch, сервінг локальних LLM через vLLM, RAG-конвеєр на ChromaDB, Image-to-Image генерацію на Stable Diffusion 1.5, мультимодальний аналіз зображень із LLaVA, агентні системи LlamaIndex та інтегрований веб-додаток Streamlit.

---

## Лабораторні роботи

| № | Тема | Ключові моделі та стек | Каталог |
|:---:|:---|:---|:---:|
| **1** | Трансформерні архітектури: класифікація та авторегресивна генерація | PyTorch, Transformer Encoder/Decoder, Enron Spam, WikiText-2 | [lab1/](./lab1/) |
| **2** | Сервінг та клієнтська інтеграція локальних LLM | vLLM, `Qwen/Qwen2.5-1.5B-Instruct-AWQ`, OpenAI Python client | [lab2_vllm/](./lab2_vllm/) |
| **3** | Локальна RAG-система на базі векторного сховища | ChromaDB, LangChain, `all-MiniLM-L6-v2`, Ollama (`qwen3:4b-instruct`) / vLLM | [lab3_chroma_rag/](./lab3_chroma_rag/) |
| **4** | Контрольована генерація зображень через дифузійні моделі | Diffusers, `stable-diffusion-v1-5`, Img2Img FP16 + CPU offload | [lab4_img2img/](./lab4_img2img/) |
| **5** | Мультимодальний опис і аналіз зображень | `llava-hf/llava-interleave-qwen-0.5b-hf`, PyTorch FP16 | [lab5_multimodal_captioning/](./lab5_multimodal_captioning/) |
| **6** | Інтелектуальні агенти з пам'яттю та викликом інструментів | LlamaIndex `AgentWorkflow`, `ChatMemoryBuffer`, Ollama | [lab6_llamaindex_agents/](./lab6_llamaindex_agents/) |
| **7** | Мультимодальна студія візуального контенту | Streamlit, LLaVA, SD 1.5 Img2Img, LlamaIndex, SQLite | [lab7_multimodal_studio/](./lab7_multimodal_studio/) |

---

## Структура репозиторію

```text
Neural-Network-Labs-Max/
├── lab1/                         # PyTorch Transformer Encoder & Causal Decoder
│   ├── notebooks/                # 4 блокноти (EDA та моделі для Enron і WikiText-2)
│   ├── data/raw/                 # Вхідні вибірки Enron та WikiText-2
│   ├── outputs/                  # Збережені чекпоінти, метрики та ключові графіки
│   ├── utils.py                  # Токенізація, словник, Positional Encoding, маски
│   └── requirements.txt
├── lab2_vllm/                    # Консольний клієнт для vLLM / OpenAI-сумісного API
│   ├── cli.py                    # AppConfig та VLLMClient з експортом діалогів
│   ├── outputs/dialogs/          # Репрезентативні Markdown-транскрипти експериментів
│   └── requirements.txt
├── lab3_chroma_rag/              # RAG-конвеєр над MLOps-документацією
│   ├── data/knowledge_base/      # 8 документів (.md, .txt)
│   ├── notebooks/                # 3 блокноти (чанки, ChromaDB retrieval, RAG)
│   └── outputs/                  # Чанки, таблиця оцінки, приклади відповідей і графіки
├── lab4_img2img/                 # Stable Diffusion 1.5 Image-to-Image
│   ├── inputs/                   # 3 вхідні зображення 512x512 RGB
│   ├── notebooks/                # 3 блокноти (пайплайн, параметричні серії, аналіз)
│   └── outputs/                  # Унікальні PNG-генерації, JSON-метадані, реєстр, графіки
├── lab5_multimodal_captioning/   # Генерація описів зображень із LLaVA-0.5B
│   ├── inputs/                   # 4 тестові сцени 512x512 RGB
│   ├── notebooks/                # 3 блокноти (аудит, серії експериментів, оцінка)
│   ├── outputs/                  # Приклади описів, зведені таблиці, графіки
│   └── requirements.txt
├── lab6_llamaindex_agents/       # Інженерний агент LlamaIndex із 4 інструментами
│   ├── data/                     # Реєстр моделей (JSON) і регламент інцидентів (TXT)
│   ├── notebooks/                # 3 блокноти (інструменти, робочі процеси, оцінка)
│   ├── outputs/                  # Матриця оцінки, графік затримок, JSON-трейси
│   └── requirements.txt
└── lab7_multimodal_studio/       # Інтегрована студія Streamlit
    ├── app.py                    # Веб-інтерфейс із 5 вкладками та телеметрією VRAM
    ├── src/                      # Сервіси моделей, SQLite-галерея, агент, безпека шляхів
    ├── assets/                   # 4 демонстраційні зображення 512x512 RGB
    ├── notebooks/                # 3 блокноти (компоненти, траси агента, оцінка)
    ├── outputs/                  # SQLite gallery.db, зображення галереї, трейси, таблиці
    └── requirements.txt
```

---

## Вимоги до середовища та запуск

Перевірено Windows 11 x64, Python 3.13, NVIDIA RTX 3050 4 GB. Кожна робота має окрему `.venv` і повний `requirements.lock.txt`.

```powershell
.\setup.ps1 -Lab 1
.\run.ps1 -Lab 1 -Smoke
.\run.ps1 -Lab 1
```

Для робіт 4, 5, 7 додайте `-Cuda` до setup. Роботи 2, 3, 6, 7 використовують Ollama на localhost:11434; setup відтворює CPU-модель `qwen3:4b-studio` з публічних вагів і збереженого Modelfile. Попередній локальний alias `qwen3:4b-instruct` не потрібний. Сервер vLLM залишається альтернативним endpoint, але цей сценарій WSL2 не перевірявся. Моделі, контрольні суми, сервіси та межі тестування описані в [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Історичні експериментальні результати

- **ЛР 1:** Transformer Encoder на Enron Spam (`d_model=128`, 2 голови) досяг Test Accuracy `97.00%` і F1 `96.68%`; Causal Decoder на WikiText-2 (`d_model=128`, 4 голови) досяг Test Perplexity `148.48` (BLEU `0.0226`, ROUGE-L `0.0493`).
- **ЛР 2:** CLI-клієнт забезпечує багатоходовий діалог із vLLM (`Qwen2.5-1.5B-Instruct-AWQ`), керування `temperature`, `max_tokens` та системними ролями з автоматичним збереженням сесій у Markdown.
- **ЛР 3:** 8 документів MLOps розбито на 55 чанків (`chunk_size=700`, `overlap=100`); досліджено Similarity та MMR-пошук (`k=4`, `fetch_k=12`, $\lambda=0.5$), фільтрацію за метаданими й відмову моделі на позадоменному запиті `Q9`.
- **ЛР 4:** На RTX 3050 (4 ГБ VRAM) у FP16 із CPU offload виконано 10 унікальних конфігурацій Stable Diffusion 1.5 Img2Img (`4.92–10.04` с/зображення), що продемонстрували компроміс між `strength`, `guidance_scale`, `steps` та вплив `negative_prompt`.
- **ЛР 5:** LLaVA-0.5B у FP16 обробила 4 сцени у 20 конфігураціях (стиль промпту, температура, бюджет токенів); середня точність за рубрикатором склала `0.90–1.00` із середнім покриттям ключових об'єктів `88.8%` без зафіксованих галюцинацій.
- **ЛР 6:** Агент LlamaIndex з 4 інструментами виконав 9 MLOps-сценаріїв із середньою затримкою `18.6` с, точністю вибору інструментів `8/9` (`88.9%`) та коректною відмовою на небезпечному запиті `scenario_09`.
- **ЛР 7:** Об'єднано LLaVA-0.5B, SD 1.5 Img2Img, агента LlamaIndex, валідацію шляхів (`validate_safe_image_path`) та SQLite-сховище (`gallery.db`, 12 успішних експериментальних записів) у єдиний Streamlit-додаток.

---

## Відомі обмеження

1. **Бюджет VRAM 4 ГБ:** Локальний запуск SD 1.5, LLaVA та AWQ-моделей вимагає FP16, квантування та послідовного вивантаження вагів із відеопам'яті.
2. **Обмеження ОС для vLLM:** Сервер vLLM підтримується на Linux або через WSL2 на Windows.
3. **Компактні локальні LLM (0.5B–4B):** Якість маршрутизації інструментів і дотримання обмежень контексту залежать від чіткості системних промптів і схем аргументів.
4. **Офлайн-режим аналізу:** Блокноти аналізу підтримують перегляд збережених артефактів без повторного GPU-інференсу чи активного LLM-сервера; для живої генерації потрібні локальні ваги й запущений бекенд.
