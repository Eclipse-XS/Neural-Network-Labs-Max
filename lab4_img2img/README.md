# Lab 4: Stable Diffusion Image-to-Image

Мета: дослідити кероване перетворення реальних зображень Stable Diffusion. Варіант 2: Img2Img з контрольованою зміною `strength`, CFG, steps і negative prompt.

## Технології та входи

- Diffusers `StableDiffusionImg2ImgPipeline`;
- `stable-diffusion-v1-5/stable-diffusion-v1-5`;
- PyTorch FP16, CUDA і model CPU offload;
- 3 RGB зображення 512×512 у `inputs/`.

Встановлення виконується з кореневого `requirements.txt`. Для практичного запуску рекомендовано CUDA GPU з 4+ GiB VRAM, достатньо RAM і доступ до Hugging Face. Model checkpoints у Git не зберігаються.

## Порядок виконання

1. `notebooks/01_img2img_pipeline.ipynb` — validation inputs, environment, pipeline і baseline inference.
2. `notebooks/02_parameter_experiments.ipynb` — one-factor-at-a-time серії з фіксованим seed.
3. `notebooks/03_results_analysis.ipynb` — traceability, grids, latency та обмеження метрик.

Запускайте Jupyter з `lab4_img2img/notebooks`. Кожен inference має записувати prompt, negative prompt, model id, seed, strength, CFG, steps, latency, device та hashes input/output. Збіг hashes уже збережених файлів не є repeatability experiment; для такого висновку потрібні два фактичні запуски однакової конфігурації.

## Outputs і відомі проблеми

`outputs/images`, `outputs/metadata`, `outputs/tables`, `outputs/figures`. Наявні outputs — історичні артефакти. Якість не визначається лише SSIM або pixel similarity. «Оптимальна» конфігурація потребує наперед визначеного критерію. Результати можуть відрізнятися між GPU, драйверами та версіями Diffusers/PyTorch.

Fast tests: `pytest lab4_img2img/tests -m "not integration" -v`. GPU smoke test є окремим integration test; skip означає «не виконано».
