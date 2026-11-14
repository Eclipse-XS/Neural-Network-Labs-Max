# Лабораторна робота №7. Мультимодальна студія візуального контенту

## Reproducible execution (Windows 11 / Python 3.13)

Run from the repository root in PowerShell. Each lab has an isolated `.venv` and a complete `requirements.lock.txt`.

```powershell
.\setup.ps1 -Lab 7 -Cuda
.\run.ps1 -Lab 7 -Smoke
.\run.ps1 -Lab 7
```

Ollama: http://127.0.0.1:11434. Setup reconstructs the CPU-only qwen3:4b-studio from the committed Modelfile.

`-Cuda` selects PyTorch 2.14.1+cu130 from the official CUDA 13.0 wheel index. The NVIDIA driver is required; these wheels do not require a separate CUDA toolkit. Omit this switch for CPU PyTorch. Vision smoke tests use the original model weights, batch size 1, 512x512 images and short real inference. They do not establish image quality or repeat the full experimental matrix.

Model revision SHAs and weight checksums are in the root `models.lock.json`. Distributed input checksums are in `inputs.sha256.json` (text is normalized to LF). Setup downloads required models and NNTS datasets. Smoke tests use real services; unavailable services cause a failure. See the root `REPRODUCIBILITY.md` for sources, service preparation and verification limits.

Інтегрований веб-додаток Streamlit ([`app.py`](./app.py)), що поєднує аналіз зображень (`LLaVA-Interleave-Qwen-0.5B`), генерацію Image-to-Image (`Stable Diffusion 1.5` у FP16), автономного агента `LlamaIndex AgentWorkflow` (`qwen3:4b-studio`, CPU), контрольовану валідацію файлових шляхів і персистентне сховище результатів у SQLite.

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

## Standalone setup and run

Install Python 3.13, a compatible NVIDIA driver, and Ollama. From this directory:

```powershell
.\setup.ps1
.\run.ps1 -Smoke
.\run.ps1
```

Setup selects PyTorch 2.14.1+cu130, downloads/verifies the pinned LLaVA and SD 1.5 weights, and reconstructs the Ollama CPU model from `Modelfile`. Use `-Cpu` for CPU PyTorch. `-SkipModels` / `-SkipServices` deliberately omit provisioning and do not establish a complete installation. The UI listens on http://localhost:8502.

`OLLAMA_MODEL` and `OLLAMA_BASE_URL` override the generated `qwen3:4b-studio` model and OpenAI-compatible URL. The default CPU-only model avoids competing with LLaVA/SD for the same 4 GB VRAM. Selecting a GPU-resident Ollama model can exhaust this GPU. Vision inference is serialized across Streamlit sessions; switching models unloads the other vision model before loading. Clear GPU Memory drops model references and clears the allocator cache.

Gallery images and SQLite records live under this installation's `outputs/gallery`, independent of the terminal directory. UUID record IDs avoid same-millisecond collisions. Committed gallery records and figures are historical experimental evidence; new runs append records. The timings below are historical and are not results of the current short smoke test.

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
