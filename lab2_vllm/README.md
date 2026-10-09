# Лабораторна робота №2. Інтеграція LLM через API та open-source фреймворки

## Варіант 2 — vLLM

## 1. Тема і мета

Тема роботи — інтеграція локальної instruction-моделі з Python-програмою через OpenAI-compatible API сервера vLLM.

Мета — запустити модель через vLLM, перевіряти точний model ID, передавати системну інструкцію та повну історію діалогу, керувати `temperature` і `max_tokens`, обробляти помилки та автоматично зберігати успішні діалоги у Markdown.

## 2. Перевірене середовище Windows

- Windows 11, WSL 2.6.3.0;
- Ubuntu 24.04.5 LTS у WSL2;
- NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB VRAM, Compute Capability 8.6;
- NVIDIA driver 610.88;
- vLLM 0.31.0;
- PyTorch 2.13.0+cu132;
- модель `Qwen/Qwen2.5-1.5B-Instruct-AWQ`;
- сервер `http://127.0.0.1:8000/v1`;
- Python-клієнт перевірено з Python 3.13.5 та `openai` 3.17.0.

Системний CUDA Toolkit не встановлювався. vLLM і CUDA runtime-пакети ізольовані в `/home/vllm-lab/lab2/.venv` усередині Ubuntu.

## 3. Архітектура

```text
Користувач → ConsoleApp → ChatSession → VLLMClient
                                      ↓
                               OpenAI Python SDK
                                      ↓
                    GET /v1/models і POST /v1/chat/completions
                                      ↓
                  vLLM → Qwen/Qwen2.5-1.5B-Instruct-AWQ
```

Пакет `openai` використовується як клієнт OpenAI-compatible API. Запити не надсилаються до OpenAI cloud: `base_url` вказує на локальний vLLM.

## 4. Структура

```text
lab2_vllm/
├── cli.py
├── requirements.txt
├── README.md
├── tests/
│   └── test_cli.py
└── outputs/
    └── dialogs/
        ├── .gitkeep
        └── dialog_*.md
```

Уся прикладна реалізація міститься в одному `cli.py`. `AppConfig` нормалізує і перевіряє параметри, `VLLMClient` ізолює OpenAI SDK, `ChatSession` керує історією та transcript, а `ConsoleApp` відповідає за консольний цикл і повідомлення про помилки. Додатковий клієнтський модуль, notebook, Docker-конфігурація або API-сервер не створювалися.

## 5. Запуск уже встановленого vLLM

### Варіант 1: видимий процес у PowerShell

Виконувати в PowerShell:

```powershell
wsl -d Ubuntu-24.04 -u vllm-lab -- bash /mnt/c/Users/Eclipse/PycharmProjects/neural-network-technologies-and-systems/vllm-lab/start-server.sh
```

Ця команда займає поточне вікно до зупинки сервера через `Ctrl+C`.

### Варіант 2: команда безпосередньо в Ubuntu WSL2

Виконувати в Ubuntu 24.04 WSL2:

```bash
cd /home/vllm-lab/lab2
.venv/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 1 \
  --enforce-eager
```

За перевіреним запуском модель займала приблизно 1.1 GiB, а весь процес після запиту — близько 3.56 GiB VRAM. Зменшувати або збільшувати memory limit без фактичного вимірювання не слід.

## 6. Перевірка API і моделі

Виконувати в PowerShell:

```powershell
curl.exe http://127.0.0.1:8000/v1/models
```

У полі `data[].id` має бути точне значення:

```text
Qwen/Qwen2.5-1.5B-Instruct-AWQ
```

CLI не підміняє відсутню модель першою моделлю зі списку. Невідповідність model ID завершує початкову перевірку з повідомленням про доступні ID.

## 7. Встановлення Python-залежностей

Клієнт запускається тим самим ізольованим середовищем WSL2, де встановлено `openai`. Вихідний код залишається у Windows-каталозі проєкту. Виконувати в Ubuntu WSL2:

```bash
cd /mnt/c/Users/Eclipse/PycharmProjects/neural-network-technologies-and-systems/max/lab2_vllm
/home/vllm-lab/lab2/.venv/bin/python -m pip install -r requirements.txt
```

Ця команда не змінює глобальний Python і не встановлює vLLM безпосередньо у Windows.

## 8. Запуск CLI

Стандартний запуск в Ubuntu WSL2:

```bash
/home/vllm-lab/lab2/.venv/bin/python cli.py
```

Власний system prompt:

```bash
/home/vllm-lab/lab2/.venv/bin/python cli.py \
  --system-prompt "You are a machine learning teacher. Explain concepts in simple English." \
  --temperature 0.2 \
  --max-tokens 200
```

Низька і висока temperature:

```bash
/home/vllm-lab/lab2/.venv/bin/python cli.py --temperature 0.2 --max-tokens 200
/home/vllm-lab/lab2/.venv/bin/python cli.py --temperature 1.2 --max-tokens 200
```

Порівняння обмеження довжини:

```bash
/home/vllm-lab/lab2/.venv/bin/python cli.py --temperature 0.2 --max-tokens 30
/home/vllm-lab/lab2/.venv/bin/python cli.py --temperature 0.2 --max-tokens 200
```

Явні адреса і model ID:

```bash
/home/vllm-lab/lab2/.venv/bin/python cli.py \
  --base-url "http://127.0.0.1:8000/v1" \
  --model "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
```

Команди `exit` і `quit` завершують цикл. `Ctrl+C` та кінець вводу також обробляються. Transcript записується лише після хоча б одного успішного обміну.

## 9. Приклад діалогу

```text
vLLM API: http://127.0.0.1:8000/v1
Model: Qwen/Qwen2.5-1.5B-Instruct-AWQ
Temperature: 0.2
Max tokens: 200

You: What is machine learning? Explain in two sentences.
Assistant: Machine learning is a type of artificial intelligence that enables computers to learn from data, without being explicitly programmed. It involves using algorithms to analyze patterns in the data and make predictions or decisions based on those patterns.

You: Give me one real-world example of it.
Assistant: One real-world example of machine learning is recommendation systems used by online retailers like Amazon and Netflix. These systems use algorithms to analyze user behavior and preferences, such as what items they have bought or watched before, and then recommend similar items to them. This helps users find new products they might be interested in, which can increase their satisfaction with the retailer's services.
```

Це фактичний двокроковий діалог, а не відредагований приклад. Другий запит перевіряє, що CLI повторно передає попередню історію повідомлень.

## 10. Історія та API request

Історія передається як послідовність:

```text
system → user → assistant → user → assistant
```

Поточне питання спочатку додається лише до тимчасового request. До постійної історії воно потрапляє разом із відповіддю тільки після успішного API-виклику. Це не залишає в transcript питання без відповіді після мережевої помилки.

`temperature` впливає на вибір наступних токенів, але один запуск не доводить стабільного ефекту. `max_tokens` задає верхню межу нових токенів і не є лімітом усієї історії.

## 11. Unit-тести

Виконувати в Ubuntu WSL2 з каталогу проєкту:

```bash
cd /mnt/c/Users/Eclipse/PycharmProjects/neural-network-technologies-and-systems/max/lab2_vllm
/home/vllm-lab/lab2/.venv/bin/python -m pytest -q
```

Фактичний результат 9 жовтня 2026 року:

```text
25 passed in 1.48s
```

Перевірено argparse, створення та нормалізацію `AppConfig`, межі `temperature`, тип і додатність `max_tokens`, точний model ID, склад повного API request, порожні choices і content, атомарне оновлення історії, недоступний сервер, вихід без запиту та UTF-8 transcript.

Після рефакторингу виконано реальний CLI smoke test. Модель повернула `vLLM integration works.`, endpoint і model ID пройшли перевірку, transcript створено в `outputs/dialogs/dialog_20261009_232642_206135.md`.

## 12. Матриця реальних експериментів

Усі узгоджені запити виконано 9 жовтня 2026 року через запущений vLLM з checkpoint `Qwen/Qwen2.5-1.5B-Instruct-AWQ`. У таблиці наведено лише спостереження з фактичних відповідей; тексти відповідей не реконструювалися вручну.

| Експеримент | Temperature | Max tokens | Запит | Фактичне спостереження |
|---|---:|---:|---|---|
| Контекст, turn 1 | 0.2 | 200 | `What is machine learning? Explain in two sentences.` | Модель дала визначення machine learning у двох реченнях |
| Контекст, turn 2 | 0.2 | 200 | `Give me one real-world example of it.` | Модель використала займенник `it` у контексті попереднього turn і навела recommendation systems |
| System prompt | 0.2 | 200 | `What is overfitting?` | Відповідь подано простою англійською з аналогією, відповідно до ролі викладача |
| Temperature low | 0.2 | 200 | `Suggest three names for an AI assistant for students.` | `StudyMate`, `EduBot`, `SmartScholar` |
| Temperature high | 1.2 | 200 | той самий запит | `KnowledgeMaster`, `AcademicAI`, `StudentSage` |
| Max tokens short | 0.2 | 30 | `Explain five advantages of machine learning.` | Відповідь обірвалася після вступу та початку першого пункту |
| Max tokens long | 0.2 | 200 | той самий запит | Модель сформувала п'ять пунктів; останній пункт обірвався на межі 200 токенів |

Два temperature-запуски показали різні набори назв, але один запуск для кожного значення не є статистичним доказом впливу temperature. Експеримент з `max_tokens` прямо показав різницю довжини. Другий turn контекстного діалогу був переданий разом з попередньою історією і коректно інтерпретований моделлю.

Фактичні повні відповіді збережено в `outputs/dialogs/`:

- `dialog_20261009_205104_887966.md` — двокрокова перевірка контексту;
- `dialog_20261009_205124_396229.md` — заданий system prompt;
- `dialog_20261009_205128_351014.md` — `temperature=0.2`;
- `dialog_20261009_205128_730659.md` — `temperature=1.2`;
- `dialog_20261009_205132_491080.md` — `max_tokens=30`;
- `dialog_20261009_205144_794357.md` — `max_tokens=200`.

Попередні результати експериментів залишено без змін у тому самому каталозі.

Окремий smoke test повного CLI збережено як `dialog_20261009_231744_862952.md`.

## 13. Обробка помилок

- порожній `base_url` і model ID відхиляються;
- `temperature` дозволена лише від 0 до 2 включно;
- `max_tokens` має бути додатним цілим числом;
- до діалогу перевіряється `/v1/models` і точний model ID;
- окремо обробляються помилки підключення й API;
- порожня відповідь або відсутність choices не записуються в історію;
- `Ctrl+C`, `EOF`, `exit` і `quit` завершують сесію без втрати успішних turns.

## 14. Перенесення на MacBook Pro M3 Max, 48 GB

Цей розділ підготовлено за офіційною документацією, але на MacBook команди ще не виконувалися.

Вимоги vLLM-Metal: Apple Silicon і macOS 15 або новіша. Стабільний варіант встановлення — Homebrew. Виконувати в macOS Terminal:

```bash
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
brew install vllm-project/vllm-metal/vllm-metal
```

Починаючи з `vllm-metal 0.2.0`, офіційний реліз містить підтримку HF AWQ для dense-моделей. Поточна stable-версія `0.30.0` підтримує Qwen2.5 і завантажує сумісні HF AWQ checkpoints через MLX repack. Checkpoint `Qwen/Qwen2.5-1.5B-Instruct-AWQ` був окремо використаний у перевірці paged runner та AWQ. Це буквально той самий Hugging Face checkpoint, що й на Windows, а не інша MLX-конвертація базової моделі.

Запуск того самого checkpoint через Metal backend:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct-AWQ \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.50
```

Перевірка API в macOS Terminal:

```bash
curl http://127.0.0.1:8000/v1/models
```

Створення окремого середовища лише для CLI, не для vLLM-Metal:

```bash
cd /path/to/neural-network-technologies-and-systems
python3 -m venv .venv-client
./.venv-client/bin/python -m pip install -r lab2_vllm/requirements.txt
./.venv-client/bin/python lab2_vllm/cli.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-1.5B-Instruct-AWQ
```

Шлях `/path/to/...` неможливо конкретизувати до перенесення репозиторію на Mac. CUDA `.venv` і NVIDIA wheels з Windows переносити не можна. AWQ checkpoint переносити вручну не потрібно: vLLM-Metal завантажить той самий Hugging Face repo ID. На Mac необхідно фактично підтвердити встановлену версію, model ID, генерацію, memory utilization і всі інтеграційні експерименти.

Офіційні джерела: [vLLM-Metal installation](https://docs.vllm.ai/projects/vllm-metal/en/stable/installation/), [supported models](https://github.com/vllm-project/vllm-metal/blob/main/docs/supported_models.md), [vLLM-Metal project](https://github.com/vllm-project/vllm-metal).

## 15. Перелік скриншотів для звіту

1. `nvidia-smi` всередині Ubuntu WSL2 із RTX 3050 та 4096 MiB.
2. Консоль успішного старту vLLM із model ID і `max_model_len=2048`.
3. `curl.exe http://127.0.0.1:8000/v1/models`.
4. `25 passed` у pytest.
5. Стандартний запуск `cli.py` з показаними параметрами.
6. Двокроковий діалог про machine learning із перевіркою контексту.
7. Відповідь про overfitting із заданим system prompt.
8. Порівняння назв AI assistant при `temperature=0.2` і `temperature=1.2`.
9. Порівняння пояснення переваг machine learning при `max_tokens=30` і `max_tokens=200`.
11. Фрагмент `request_chat_completion` з `model`, `messages`, `temperature`, `max_tokens`.
12. Markdown transcript із timestamp і UTF-8 українським текстом.

Скріншоти мають показувати фактичний запуск. Результати macOS не слід включати як виконані до реальної перевірки на MacBook.

## 16. Основні функції для захисту

- `AppConfig` є незмінним dataclass: очищає рядки та перевіряє URL, model ID, temperature і max tokens.
- `VLLMClient.verify_model()` викликає `/v1/models`, а `complete()` передає історію в `/v1/chat/completions`.
- `ChatSession.ask()` додає user/assistant turns лише після успішної відповіді; `save()` створює UTF-8 Markdown.
- `ConsoleApp` перевіряє сервер, читає команди, показує відповіді й локалізує обробку помилок інтерфейсу.
- `parse_config()` є єдиною межею між argparse і прикладною конфігурацією.

vLLM виконує inference і надає OpenAI-compatible HTTP API. OpenAI SDK тут не запускає модель, а лише формує клієнтські запити. AWQ зменшує обсяг ваг, але квантизація та мала кількість параметрів можуть погіршувати якість. `temperature` змінює стохастичність вибірки; `max_tokens` обмежує тільки нову відповідь. Повна історія передається повторно, оскільки HTTP API сам по собі не зберігає контекст клієнтського діалогу.

## 17. Висновок

CLI, unit-тести, реальний model check, chat completions, multi-turn history і UTF-8 transcripts працюють на Windows 11 + WSL2 + vLLM. Ліміти генерації передаються коректно. Якість відповідей обраної 1.5B AWQ-моделі недостатня для безумовної довіри: технічну інтеграцію підтверджено, фактичну правильність кожної генерації потрібно перевіряти. Перенесення на M3 Max документовано, але ще не протестовано.
