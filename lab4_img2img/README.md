# Лабораторна робота №4. Генерування зображень Image-to-Image (Stable Diffusion 1.5)

## Reproducible execution (Windows 11 / Python 3.13)

Run from the repository root in PowerShell. Each lab has an isolated `.venv` and a complete `requirements.lock.txt`.

```powershell
.\setup.ps1 -Lab 4 -Cuda
.\run.ps1 -Lab 4 -Smoke
.\run.ps1 -Lab 4
```

No separate inference server is required.

`-Cuda` selects PyTorch 2.14.1+cu130 from the official CUDA 13.0 wheel index. The NVIDIA driver is required; these wheels do not require a separate CUDA toolkit. Omit this switch for CPU PyTorch. Vision smoke tests use the original model weights, batch size 1, 512x512 images and short real inference. They do not establish image quality or repeat the full experimental matrix.

Model revision SHAs and weight checksums are in the root `models.lock.json`. Distributed input checksums are in `inputs.sha256.json` (text is normalized to LF). Setup downloads required models and NNTS datasets. Smoke tests use real services; unavailable services cause a failure. See the root `REPRODUCIBILITY.md` for sources, service preparation and verification limits.

Параметричне дослідження конвеєра `StableDiffusionImg2ImgPipeline` (Diffusers, `stable-diffusion-v1-5/stable-diffusion-v1-5` у FP16 із `enable_model_cpu_offload()`) на трьох вхідних зображеннях $512 \times 512$ RGB.

---

## Структура каталогу

```text
lab4_img2img/
├── README.md
├── inputs/                                      # 3 вхідні зображення 512x512 RGB
├── notebooks/
│   ├── 01_input_images_and_pipeline.ipynb       # Аудит входів, CUDA-діагностика, 3 baseline-генерації
│   ├── 02_parameter_experiments.ipynb           # Експерименти strength, CFG, steps, negative prompt
│   └── 03_results_analysis.ipynb                # Зіставні сітки, графік затримок, нотатки про LoRA
└── outputs/
    ├── images/                                  # 10 унікальних PNG-результатів за серіями
    │   ├── baseline/                            # 3 базові сцени (strength=0.6, CFG=7.5, steps=30)
    │   ├── strength/                            # Варіації 0.3 та 0.9
    │   ├── cfg/                                 # Варіації 5.0 та 10.0
    │   ├── steps/                               # Варіації 20 та 50
    │   └── negative_prompt/                     # Генерація з negative prompt
    ├── metadata/                                # 10 відповідних JSON-файлів конфігурації
    ├── tables/                                  # generations_registry.csv (реєстр 13 запусків)
    └── figures/                                 # analysis_strength_comparison.png, analysis_negative_prompt_comparison.png
```

---

## Виконання блокнотів

1. `01_input_images_and_pipeline.ipynb` — перевірка формату зображень у `inputs/`, ініціалізація SD 1.5 у FP16 для GPU з 4 ГБ VRAM і базові генерації (`seed=42`).
2. `02_parameter_experiments.ipynb` — контрольовані серії змін `strength` (`0.3 / 0.6 / 0.9`), `guidance_scale` (`5.0 / 7.5 / 10.0`), `num_inference_steps` (`20 / 30 / 50`) та `negative_prompt` для `01_coastal_lighthouse.png` із фіксацією таймінгів у `outputs/tables/generations_registry.csv`.
3. `03_results_analysis.ipynb` — завантаження збережених результатів (із перевикористанням канонічного файлу `baseline_lighthouse_...` для спільних конфігурацій за замовчуванням), побудова порівняльних сіток та аналіз часу інференсу.

---

## Спостережувані результати (RTX 3050 Laptop GPU, 4 ГБ VRAM)

- **Strength:** діапазон `0.5–0.7` зберігає глобальну композицію вхідного зображення та додає стилізацію; `0.9` майже повністю перебудовує сцену.
- **Guidance Scale (CFG):** значення `7.5` формує збалансований контраст, тоді як `10.0` підсилює насиченість і перетемнює тіні.
- **Steps та продуктивність:** час генерації зріс лінійно з кількістю фактичних кроків денойзингу (`steps=20`: `4.92` с, `steps=30`: `6.14` с, `steps=50`: `8.90` с); після 30 кроків візуальний приріст сповільнюється.
- **Negative Prompt:** пригнічує дрібні артефакти контурів і шум у градієнтах неба.

Обмеження: виконання `01` та `02` потребує CUDA GPU (від 4 ГБ VRAM) і завантаження вагів `stable-diffusion-v1-5`; `03_results_analysis.ipynb` працює повністю офлайн поверх збережених зображень і реєстру.
