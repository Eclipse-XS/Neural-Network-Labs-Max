# Лабораторна робота №5. Мультимодальний опис зображень (LLaVA)

## Reproducible execution (Windows 11 / Python 3.13)

Run from the repository root in PowerShell. Each lab has an isolated `.venv` and a complete `requirements.lock.txt`.

```powershell
.\setup.ps1 -Lab 5 -Cuda
.\run.ps1 -Lab 5 -Smoke
.\run.ps1 -Lab 5
```

No separate inference server is required.

`-Cuda` selects PyTorch 2.14.1+cu130 from the official CUDA 13.0 wheel index. The NVIDIA driver is required; these wheels do not require a separate CUDA toolkit. Omit this switch for CPU PyTorch. Vision smoke tests use the original model weights, batch size 1, 512x512 images and short real inference. They do not establish image quality or repeat the full experimental matrix.

Model revision SHAs and weight checksums are in the root `models.lock.json`. Distributed input checksums are in `inputs.sha256.json` (text is normalized to LF). Setup downloads required models and NNTS datasets. Smoke tests use real services; unavailable services cause a failure. See the root `REPRODUCIBILITY.md` for sources, service preparation and verification limits.

Дослідження генерації текстових описів зображень (Image Captioning) моделлю `llava-hf/llava-interleave-qwen-0.5b-hf` у точці FP16 на чотирьох тестових сценах $512 \times 512$ RGB.

---

## Структура каталогу

```text
lab5_multimodal_captioning/
├── README.md
├── requirements.txt
├── inputs/                                      # 4 тестові сцени (512x512 RGB)
├── notebooks/
│   ├── 01_image_audit_and_model_setup.ipynb     # Аудит зображень та діагностика пам'яті моделі
│   ├── 02_captioning_experiments.ipynb          # 20 експериментів (режими ANALYSIS_ONLY і LIVE)
│   └── 03_caption_quality_analysis.ipynb        # Агрегація метрик та оцінка якості описів
└── outputs/
    ├── captions/                                # 5 репрезентативних .txt прикладів описів
    ├── metadata/                                # 5 відповідних .json метаданих запусків
    ├── tables/                                  # caption_experiments.{csv,json}, caption_quality_evaluation.csv
    └── figures/                                 # input_images_and_concise_captions.png, prompt_wording_comparison.png
```

---

## Встановлення та виконання

```powershell
```

- `01_image_audit_and_model_setup.ipynb` — перевіряє роздільну здатність і формати файлів у `inputs/`, конфігурує модель у FP16 (`< 2.0` ГБ VRAM на RTX 3050).
- `02_captioning_experiments.ipynb` — охоплює 20 прогонів (3 стилі промпту для 4 сцен, 4 значення температури `[0.0, 0.2, 0.7, 1.0]`, 4 бюджети `max_new_tokens` `[32, 64, 128, 256]`). За замовчуванням `MODE = 'ANALYSIS_ONLY'` завантажує збережені записи або зведений файл `outputs/tables/caption_experiments.json`; значення `MODE = 'LIVE'` запускає локальний GPU-інференс.
- `03_caption_quality_analysis.ipynb` — аналізує довжину, затримку, лексичну різноманітність і повноту покриття об'єктів за `outputs/tables/caption_quality_evaluation.csv`.

---

## Спостережувані результати

- Середня точність за рубрикатором (`caption_quality_evaluation.csv`): `0.975` (`concise`), `0.925` (`descriptive`), `0.925` (`detailed`); середнє покриття ключових об'єктів `88.8%` при відсутності зафіксованих галюцинацій (`hallucination_detected = False`).
- Збільшення `max_new_tokens` лінійно підвищує затримку до природної зупинки моделі (на `256` токенах генерація завершилася на `208` токенах).
- Підвищення температури (`0.7–1.0`) збільшує варіативність формулювань, але підвищує ризик неточних деталей порівняно з жадібним декодуванням (`temperature=0.0`).

Обмеження: модель розміром 0.5B параметрів схильна до стислого або шаблонного опису складних лабораторних деталей за високого бюджету токенів; режим `LIVE` потребує завантаження вагів з Hugging Face і CUDA GPU.
