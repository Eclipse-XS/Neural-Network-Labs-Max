# Лабораторна робота №5. Мультимодальний опис зображень (LLaVA)

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
pip install -r lab5_multimodal_captioning/requirements.txt
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
