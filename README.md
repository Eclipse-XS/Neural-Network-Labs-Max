# Нейромережеві технології і системи — Лабораторний практикум

**Національний університет «Львівська політехніка»**  
**Кафедра систем штучного інтелекту**  
**Спеціальність:** 122 «Комп’ютерні науки»  
**Дисципліна:** Технології та системи нейронних мереж  

---

## 1. Загальний огляд репозиторію

Репозиторій містить повний комплекс програмних реалізацій, виконаних інтерактивних Jupyter-блокнотів, наборів даних, реальних інференс-артефактів, тестів та автономних додатків для лабораторних робіт 1–7.

Усі лабораторні роботи починаючи з 3-ї реалізовані у повній відповідності до **Варіанту 2** з дотриманням парадигми **Notebook-First** (ЛР 3–6) та гібридної архітектури **Hybrid Application-First** (ЛР 7):

| Лабораторна | Тема | Варіант | Стек технологій | Архітектура | Статус |
|:---:|:---|:---:|:---|:---:|:---:|
| **ЛР 1** | Трансформерні архітектури (Класифікація та Генерація) | Базовий | PyTorch, Hugging Face Transformers | Notebook-First | ✅ 9/9 PASS |
| **ЛР 2** | Високопродуктивний сервінг LLM | Базовий | vLLM, OpenAI API, Python CLI | Script / Client | ✅ 25/25 PASS |
| **ЛР 3** | RAG-система витягування знань | **Варіант 2** (ChromaDB + Markdown/Текст) | LangChain, ChromaDB, all-MiniLM-L6-v2, Ollama (`qwen3:4b-instruct`) | Notebook-First | ✅ 13/13 PASS |
| **ЛР 4** | Генерація зображень через Diffusion | **Варіант 2** (Stable Diffusion Img2Img) | Diffusers, SD 1.5 FP16, CPU Offload | Notebook-First | ✅ 14/14 PASS |
| **ЛР 5** | Мультимодальний зір: Captioning | **Варіант 2** (LLaVA Captioning & Settings) | `LLaVA-Interleave-Qwen-0.5B`, PyTorch FP16 | Notebook-First | ✅ 11/11 PASS |
| **ЛР 6** | Інтелектуальні агенти з пам'яттю | **Варіант 2** (LlamaIndex Agent + Memory) | LlamaIndex `AgentWorkflow`, ChatMemoryBuffer, Ollama | Notebook-First | ✅ 15/15 PASS |
| **ЛР 7** | Мультимодальна студія візуального контенту | **Варіант 2** (Studio App + Multi-Model) | Streamlit, LLaVA, SD 1.5, LlamaIndex, SQLite | **Application-First** + 3 Notebooks | ✅ 16/16 PASS |

---

## 2. Структурний зміст лабораторних робіт

### [lab3_chroma_rag/](./lab3_chroma_rag/) — Лабораторна робота №3 (Варіант 2)
- **Тема:** Побудова локальної RAG-системи на основі векторної бази даних ChromaDB та Markdown/Text корпусу знань.
- **Корпус знань:** 8 технічних документів з практик MLOps (`data/knowledge_base/`).
- **Блокноти:**
  - `01_corpus_preparation.ipynb`: завантаження через `DirectoryLoader`, порівняння чанкінгу (`RecursiveCharacterTextSplitter`), аудит метаданих (`source`, `type`).
  - `02_chroma_retrieval.ipynb`: створення та збереження персистентної колекції ChromaDB (55 векторів), дослідження підбору гіперпараметрів MMR, фільтрація за метаданими.
  - `03_rag_evaluation.ipynb`: інтеграція з локальною моделлю `qwen3:4b-instruct`, експериментальне порівняння Similarity vs MMR, аналіз заземленості на контексті.
- **Тести:** `pytest lab3_chroma_rag/tests -v` (13 перевірок пройдено).

### [lab4_img2img/](./lab4_img2img/) — Лабораторна робота №4 (Варіант 2)
- **Тема:** Контрольоване генерування та стилізація зображень за допомогою `Stable Diffusion 1.5 Image-to-Image`.
- **Вхідні зображення:** 3 сюжети $512 \times 512$ RGB (`01_lighthouse.png`, `02_cafe_bicycle.png`, `03_library.png`).
- **Блокноти:**
  - `01_input_images_and_pipeline.ipynb`: аудит CUDA/VRAM, ініціалізація `StableDiffusionImg2ImgPipeline` FP16 + CPU offload, базові стилізації.
  - `02_parameter_experiments.ipynb`: параметричні серії гіперпараметрів `strength` (0.3 / 0.6 / 0.9), `guidance_scale` (5 / 7.5 / 10), `steps` (20 / 30 / 50), абляція negative prompt при фіксованому seed.
  - `03_results_analysis.ipynb`: візуальні сітки порівнянь, замір затримок GPU (RTX 3050), аналіз збереження геометрії та текстури.
- **Тести:** `pytest lab4_img2img/tests -v` (14 перевірок пройдено).

### [lab5_multimodal_captioning/](./lab5_multimodal_captioning/) — Лабораторна робота №5 (Варіант 2)
- **Тема:** Мультимодальний аналіз зображень та генерація описових підписів за допомогою сімейства LLaVA.
- **Модель:** `llava-hf/llava-interleave-qwen-0.5b-hf` (FP16 на локальному GPU).
- **Вхідні зображення:** 4 візуально різноманітні сцени (морський маяк, міське кафе, класична бібліотека, сучасна лабораторія робототехніки).
- **Блокноти:**
  - `01_image_audit_and_model_setup.ipynb`: аудит вхідних зображень, завантаження процесора та ваг LLaVA, базовий інференс.
  - `02_captioning_experiments.ipynb`: 20 реальних GPU-інференсів: порівняння промптів (concise, descriptive, detailed), вплив температури ($0.0 - 1.0$) та бюджету токенів ($32 - 256$).
  - `03_caption_quality_analysis.ipynb`: оцінювання повноти охоплення об'єктів, аналіз зорових галюцинацій та підсумкові матриці.
- **Тести:** `pytest lab5_multimodal_captioning/tests -v` (11 перевірок пройдено).

### [lab6_llamaindex_agents/](./lab6_llamaindex_agents/) — Лабораторна робота №6 (Варіант 2)
- **Тема:** Інтелектуальні агенти у фреймворку LlamaIndex: інструменти, пам'ять та багатокрокове планування.
- **Ендпоінт:** Локальний OpenAI-сумісний сервер Ollama (`qwen3:4b-instruct`) з підтримкою Function Calling.
- **Спеціалізовані інструменти:** `query_model_registry` (структурований JSON), `list_degraded_models`, `consult_incident_runbook` (пошук у регламенті ліквідації аварій), `calculate_slo_margin` (чисельний калькулятор).
- **Блокноти:**
  - `01_agent_tools_and_memory.ipynb`: визначення інструментів, юніт-тести функцій, ініціалізація `AgentWorkflow`, верифікація контекстної пам'яті `ChatMemoryBuffer` між репліками.
  - `02_multistep_agent_workflows.ipynb`: автономне багатокрокове планування (детекція збою $\rightarrow$ розрахунок запасу SLO $\rightarrow$ отримання регламенту відновлення), перехоплення стрімінгових подій (`stream_events`).
  - `03_agent_evaluation.ipynb`: систематичне тестування на матриці з 9 сценаріїв (Direct QA, Single Tool, Multi-Step Chaining, Boundary/Refusal), аналіз точності вибору інструментів (100%) та затримок.
- **Тести:** `pytest lab6_llamaindex_agents/tests -v` (15 перевірок пройдено).

### [lab7_multimodal_studio/](./lab7_multimodal_studio/) — Лабораторна робота №7 (Варіант 2)
- **Тема:** Фінальний проєкт — Автономна Мультимодальна Студія візуального контенту (Multimodal Visual Content Studio).
- **Архітектурний підхід:** Повноцінний автономний веб-додаток на базі `Streamlit` (`app.py`), підтримуваний модульним ядром у `src/` та трьома дослідницькими блокнотами в `notebooks/`.
- **Інтегровані компоненти:**
  - `LLaVA Captioning & VQA`: текстовий опис та відповіді на запитання щодо візуальних артефактів.
  - `Stable Diffusion 1.5 Img2Img`: художня трансформація вхідних зображень за стилістичними промптами.
  - `LlamaIndex Agent Orchestrator`: автономний диспетчер, що обирає необхідні моделі та інструменти студії.
  - `Persistent Gallery`: постійна база даних на базі SQLite (`gallery.db`) та файлове сховище згенерованого контенту, що переживає перезапуск додатку.
- **Блокноти:**
  - `01_multimodal_models.ipynb`: ізольоване дослідження інференсу LLaVA та SD 1.5, профілювання та керування пам'яттю VRAM.
  - `02_integrated_workflows.ipynb`: наскрізні агентні пайплайни (Vision $\rightarrow$ Reasoning $\rightarrow$ Transformation), аудит подій та збереження в SQLite.
  - `03_evaluation_and_analysis.ipynb`: порівняльний аналіз пропускної здатності, матриці затримок та підсумкові висновки.
- **Тести:** `pytest lab7_multimodal_studio/tests -v` (16 перевірок пройдено).

---

## 3. Інструкція із запуску та перевірки

### 1. Активація віртуального оточення
```bash
# Використання підготовленого віртуального середовища Python 3.13 / PyTorch CUDA
..\.venv-lab04\Scripts\activate
```

### 2. Запуск фінального веб-додатку Лабораторної №7
```bash
cd lab7_multimodal_studio
streamlit run app.py
```

### 3. Запуск повного набору тестів репозиторію
```bash
# Лабораторна робота 1
pytest lab1/tests -v

# Лабораторна робота 2
pytest lab2_vllm/tests -v

# Лабораторна робота 3 (Варіант 2)
pytest lab3_chroma_rag/tests -v

# Лабораторна робота 4 (Варіант 2)
pytest lab4_img2img/tests -v

# Лабораторна робота 5 (Варіант 2)
pytest lab5_multimodal_captioning/tests -v

# Лабораторна робота 6 (Варіант 2)
pytest lab6_llamaindex_agents/tests -v

# Лабораторна робота 7 (Варіант 2)
pytest lab7_multimodal_studio/tests -v
```
