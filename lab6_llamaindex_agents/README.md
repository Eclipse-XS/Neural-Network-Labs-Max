# Лабораторна робота №6. Інтелектуальні агенти з пам'яттю та інструментами (LlamaIndex)

Автономний MLOps-агент на базі `LlamaIndex` (`AgentWorkflow` + `ChatMemoryBuffer`) із локальною моделлю `qwen3:4b-instruct` (через OpenAI-сумісний API Ollama), чотирма спеціалізованими інструментами та журналюванням трас у JSON.

---

## Інструменти агента та дані

- `data/models_registry.json` — локальний реєстр моделей (статуси, метрики затримки, показники якості).
- `data/mlops_runbook.txt` — регламент реагування на інциденти та перевищення SLO.
- Чотири функціональні інструменти в блокнотах `01` і `02`:
  - `query_model_registry` — пошук параметрів і метрик моделі за іменем;
  - `list_degraded_models` — відбір моделей зі зниженою якістю чи деградацією SLO;
  - `consult_incident_runbook` — витяг процедур із регламенту інцидентів;
  - `calculate_slo_margin` — арифметичний розрахунок запасу до порогу SLO.

---

## Структура каталогу

```text
lab6_llamaindex_agents/
├── README.md
├── requirements.txt
├── data/
│   ├── models_registry.json
│   └── mlops_runbook.txt
├── notebooks/
│   ├── 01_agent_tools_and_memory.ipynb          # Інструменти, схема аргументів і ChatMemoryBuffer
│   ├── 02_multistep_agent_workflows.ipynb       # Багатокрокові робочі процеси та збереження трейсів
│   └── 03_agent_evaluation.ipynb                # Аналіз 9 тестових сценаріїв
└── outputs/
    ├── tables/
    │   └── agent_evaluation_results.csv         # Зведена матриця за 9 сценаріями
    ├── figures/
    │   └── scenario_latency_breakdown.png       # Графік затримок за сценаріями
    └── traces/                                  # 4 репрезентативні JSON-трейси (scenario_06..09)
```

---

## Встановлення та запуск

```powershell
pip install -r lab6_llamaindex_agents/requirements.txt
```

Для запуску живих агентних сценаріїв у `01` та `02` потрібен активний локальний сервер Ollama на порту `11434` із моделлю `qwen3:4b-instruct`:

```powershell
ollama run qwen3:4b-instruct
```

Блокнот `03_agent_evaluation.ipynb` працює офлайн поверх `outputs/tables/agent_evaluation_results.csv` та збережених JSON-трейсів (`scenario_08_unknown_model.json`, `scenario_09_safety_refusal.json`).

---

## Спостережувані результати (`outputs/tables/agent_evaluation_results.csv`)

- **Кількість оцінених сценаріїв:** 9 (Direct QA, Single Tool, Multi-Step Chaining, Memory Multi-Turn, Boundary, Safety Refusal).
- **Точність маршрутизації інструментів:** `8/9` (`88.9%`); у `scenario_07_memory_continuation` агент відповів із `ChatMemoryBuffer` без повторного виклику інструментів.
- **Успішність виконання завдань:** `8/9` (`88.9%`); виправлено за первинним JSON-трейсом `scenario_06_multistep_router.json` (`"task_success": false`: попри коректний виклик 4 інструментів, відповідь помилково трактувала `320 RPS` як «нижче порогу 300 RPS» і не ініціювала масштабування). Для `scenario_01..05` збережено початкові мітки CSV без окремих JSON-трейсів; сумісний показник (`selection_correct` та `task_success`) — `7/9` (`77.8%`).
- **Середня затримка відповіді:** `18.58` с (`2.95` с для відмови до `50.74` с для 4-крокової маршрутизації).

Обмеження: компактна модель `qwen3:4b-instruct` демонструє підвищену затримку при послідовному виклику кількох інструментів (38–51 с) і може спиратися на контекст пам'яті замість повторного запиту до інструмента.
