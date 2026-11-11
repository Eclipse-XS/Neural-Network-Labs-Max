# Лабораторна робота №7. Мультимодальна студія візуального контенту

Інтегрований веб-додаток Streamlit ([`app.py`](./app.py)), що поєднує аналіз зображень (`LLaVA-Interleave-Qwen-0.5B`), генерацію Image-to-Image (`Stable Diffusion 1.5` у FP16), автономного агента `LlamaIndex AgentWorkflow` (`qwen3:4b-instruct`), контрольовану валідацію файлових шляхів і персистентне сховище результатів у SQLite.

---

## Архітектура та модулі

- [`app.py`](./app.py) — інтерфейс Streamlit із бічною панеллю телеметрії VRAM і 5 вкладками (`Image Captioning`, `Visual QA`, `Image Transformation`, `Agent Orchestration`, `Studio Gallery`).
- [`src/models/llava_service.py`](./src/models/llava_service.py) — сервіс підписів і Visual QA на `llava-hf/llava-interleave-qwen-0.5b-hf`.
- [`src/models/diffusion_service.py`](./src/models/diffusion_service.py) — сервіс Img2Img на `stable-diffusion-v1-5/stable-diffusion-v1-5` з FP16 та CPU offload.
- [`src/agent/studio_agent.py`](./src/agent/studio_agent.py) — агент `LlamaIndex` із 4 інструментами (`caption_image`, `answer_visual_question`, `transform_image_style`, `inspect_studio_gallery`).
- [`src/storage/gallery_service.py`](./src/storage/gallery_service.py) — реляційне сховище історії операцій `outputs/gallery/gallery.db` (таблиця `studio_records`).
- [`src/utils/security.py`](./src/utils/security.py) — пісочниця шляхів `validate_safe_image_path`, що блокує доступ поза дозволеними каталогами студії (`assets/`, `outputs/`).
- [`src/utils/device.py`](./src/utils/device.py) — моніторинг та звільнення відеопам'яті (`ModelLifecycleManager`).

---

## Структура каталогу

```text
lab7_multimodal_studio/
├── README.md
├── requirements.txt
├── app.py                                       # Головний Streamlit-додаток
├── src/                                         # Модулі моделей, агента, галереї, безпеки та GPU
├── assets/                                      # 4 демонстраційні зображення 512x512 RGB
├── notebooks/
│   ├── 01_multimodal_models.ipynb               # Ініціалізація моделей та аудит галереї
│   ├── 02_integrated_workflows.ipynb            # Офлайн-аналіз 3 JSON-трейсів, безпеки та gallery.db
│   └── 03_evaluation_and_analysis.ipynb         # Порівняння затримок, аудит артефактів і VRAM
└── outputs/
    ├── gallery/                                 # gallery.db (13 записів: 12 успішних + 1 тест безпеки) і 15 збережених PNG-зображень
    ├── traces/                                  # 3 JSON-трейси (caption/gallery, VQA->Img2Img, path rejection)
    ├── tables/                                  # studio_model_benchmarks.csv
    └── figures/                                 # multimodal_latency_breakdown.png
```

---

## Встановлення та запуск

```powershell
pip install -r lab7_multimodal_studio/requirements.txt
streamlit run lab7_multimodal_studio/app.py
```

Для повного локального режиму роботи всіх вкладок потрібні CUDA GPU (від 4 ГБ VRAM) та запущений сервер Ollama (`qwen3:4b-instruct` на порту `11434`). Блокноти `02` і `03` відтворюють аналіз збережених трейсів і бази `gallery.db` без звернення до GPU чи Ollama.

---

## Зафіксовані показники (`outputs/tables/studio_model_benchmarks.csv`)

| Завдання | Об'єкт / Конфігурація | Затримка (с) | Згенеровано токенів |
|:---|:---|---:|---:|
| LLaVA Caption (Concise) | Lighthouse | `3.101` | `10` |
| LLaVA VQA (Equipment) | Robotics Lab | `0.387` | `2` |
| SD 1.5 Img2Img (Van Gogh) | Lighthouse | `6.130` | `—` |
| SD 1.5 Img2Img (Cyberpunk) | Cafe Bicycle | `4.933` | `—` |
| LlamaIndex Agent Multi-Tool | Robotics + Gallery | `52.675` | `146` |

Безпека та персистентність: `validate_safe_image_path` відхиляє спроби обходу шляхів (зокрема `../lab4_img2img/inputs/...`, `../../etc/shadow`, `C:/Windows/System32/cmd.exe`), а `gallery.db` містить 13 записів (12 успішних операцій, що посилаються на 15 наявних PNG-файлів у `outputs/gallery/`, та 1 відхилену спробу path traversal зі статусом `failed`).

Обмеження: одночасне завантаження LLaVA, SD 1.5 та Ollama на GPU з 4 ГБ VRAM вимагає почергового звільнення пам'яті через `ModelLifecycleManager`; багатоінструментні запити агента мають підвищену затримку (~53 с).
