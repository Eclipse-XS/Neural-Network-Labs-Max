# Лабораторна робота №2. Сервінг та інтеграція локальних LLM через vLLM

Консольний клієнт мовою Python ([`cli.py`](./cli.py)) для роботи з локальним OpenAI-сумісним API сервера `vLLM` на моделі `Qwen/Qwen2.5-1.5B-Instruct-AWQ`.

---

## Функціональність

- Підключення до локального ендпоінта через офіційний `openai.OpenAI` клієнт (`AppConfig`, `VLLMClient`).
- Налаштування параметрів інференсу через аргументи CLI: `--system-prompt`, `--temperature`, `--max-tokens`, `--base-url`, `--model`.
- Підтримка багатоходової історії розмови в межах сесії та експорт кожного діалогу у Markdown (`outputs/dialogs/`).
- Клієнт використовується також у [ЛР 3](../lab3_chroma_rag/) для RAG-генерації поверх векторного пошуку.

---

## Структура каталогу

```text
lab2_vllm/
├── README.md
├── requirements.txt
├── cli.py                   # Консольний клієнт та експорт транскриптів
└── outputs/
    └── dialogs/             # 4 репрезентативні діалоги (A: low temp, B: high temp, C: expert, D: context)
```

---

## Встановлення та запуск

```powershell
pip install -r lab2_vllm/requirements.txt
```

### 1. Запуск сервера vLLM (WSL2 Ubuntu 24.04 / Linux)

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 1 \
  --enforce-eager
```

### 2. Інтерактивний запуск клієнта (Windows PowerShell)

```powershell
python lab2_vllm/cli.py `
  --base-url http://127.0.0.1:8000/v1 `
  --model Qwen/Qwen2.5-1.5B-Instruct-AWQ `
  --system-prompt "You are a concise ML systems assistant." `
  --temperature 0.2 `
  --max-tokens 200
```

---

## Приклади збережених діалогів (`outputs/dialogs/`)

- `dialog_20261009_231649_662422_A_low_temp_1.md` — низька температура (`0.0–0.2`), стабільні детерміновані відповіді.
- `dialog_20261009_231657_377212_B_high_temp_1.md` — висока температура (`0.9–1.0`), підвищена варіативність лексики.
- `dialog_20261009_231717_491199_C_expert.md` — зміна системної ролі (`expert`) та технічної глибини відповідей.
- `dialog_20261009_231728_084331_D_context.md` — збереження контексту в багатоходовому діалозі.

Обмеження: двигун vLLM потребує Linux або WSL2 з підтримкою CUDA GPU; параметри `--max-model-len 2048` і `--max-num-seqs 1` підібрані спеціально під відеокарту з 4 ГБ VRAM.
