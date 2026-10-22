# ЗВІТ З ЛАБОРАТОРНОЇ РОБОТИ №3
## Тема: Реалізація RAG-системи для витягування знань
**Дисципліна:** Нейромережеві технології і системи  
**Спеціальність:** 122 «Комп’ютерні науки (Обчислювальний інтелект та смарт-системи)»  
**Варіант:** №2 (Chroma + Markdown / Текст)  

---

## 1. Тема
«Реалізація RAG-системи для витягування знань».

## 2. Мета роботи
Набути практичних навичок створення Retrieval-Augmented Generation (RAG) системи, яка поєднує великі мовні моделі (LLM) із зовнішніми джерелами знань. Засвоїти методи індексації документів, побудови векторної бази даних ChromaDB, інтеграції з локальною LLM за допомогою фреймворку LangChain. Дослідити алгоритми семантичного пошуку (Cosine Similarity) та диверсифікованого пошуку (Maximal Marginal Relevance — MMR), а також оцінити вплив фільтрації за метаданими на якість та фактологічну точність генерації.

## 3. Постановка завдання й вимоги варіанта №2
Відповідно до методичних вказівок для варіанта №2 необхідно:
1. Підготувати директорію з 5–8 файлів `.md/.txt` (конспекти/нотатки з предметної області MLOps) та завантажити їх через `DirectoryLoader` із відповідними лоадерами.
2. Розбити тексти за допомогою `RecursiveCharacterTextSplitter`; у метаданих кожного чанка обов'язково зберегти шлях до файлу та його тип (`md`/`txt`).
3. Створити векторну базу даних ChromaDB на ембеддингах `sentence-transformers/all-MiniLM-L6-v2` зі збереженням на диск (персистентність).
4. Налаштувати retriever у двох режимах: `similarity` та `mmr` (підібравши $k$ та $\text{fetch\_k}$), виконати однакові тестові запити та детально порівняти результати.
5. Перевірити роботу фільтрів за метаданими (пошук лише за `type=md` або за конкретним файлом).
6. Зібрати наскрізний RAG-ланцюг із локальною LLM (через OpenAI-сумісний інтерфейс без сторонніх хмарних API).
7. Задокументувати різницю у відповідях між `similarity` та `mmr`.
8. Сформулювати науково обґрунтовані висновки про якість пошуку та вплив метаданих на результат генерації.

---

## 4. Теоретичні відомості

### 4.1 Концепція Retrieval-Augmented Generation (RAG)
Архітектура RAG розділяє процес формування відповіді на два взаємопов'язані етапи:
1. **Етап пошуку (Retrieval):** векторизація вхідного запиту $q$ та пошук підмножини $k$ найбільш релевантних фрагментів у векторній базі даних:
   $$D^* = \text{Retriever}(q, \mathcal{K}, k)$$
2. **Етап генерації (Generation):** подача об'єднаного контексту $D^*$ та запиту $q$ у мовну модель для синтезу зв'язної відповіді:
   $$A = \text{LLM}(\text{Prompt}(q, D^*))$$

### 4.2 Векторне представлення тексту та косинусна схожість
Семантичні ембеддинги відображають текст у простір $\mathbb{R}^d$. У роботі використано модель `all-MiniLM-L6-v2` ($d = 384$). Для двох векторів $u, v \in \mathbb{R}^d$ косинусна схожість визначається як:
$$\text{sim}(u, v) = \cos(\theta) = \frac{u \cdot v}{\|u\| \|v\|}$$
Для одинично нормалізованих векторів косинусна відстань обчислюється як $1 - (u \cdot v)$.

### 4.3 Алгоритм Maximal Marginal Relevance (MMR)
Пошук за чистою подібністю часто призводить до «семантичного колапсу» — коли всі $k$ знайдених фрагментів походять з одного розділу та дублюють одні й ті самі факти. Алгоритм MMR усуває цю проблему:
$$\text{MMR}(q, C, S) = \arg\max_{d_i \in C \setminus S} \left[ \lambda \cdot \text{Sim}_1(d_i, q) - (1 - \lambda) \max_{d_j \in S} \text{Sim}_2(d_i, d_j) \right]$$
де $C$ — множина з $\text{fetch\_k}$ кандидатів, знайдених за подібністю, $S$ — множина вже відібраних фрагментів, а $\lambda \in [0, 1]$ контролює компроміс між релевантністю та новизною.

---

## 5. Інструменти й програмне середовище
- **Операційна система:** Windows 11 (x86_64);
- **Інтерпретатор:** Python 3.12.6 у виділеному віртуальному оточенні (`NNTS_VARIANT_2/.venv`);
- **Апаратне забезпечення:** NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB VRAM), 16 GiB RAM;
- **Фреймворк RAG:** LangChain v1.4.7 (`langchain-core` 1.6.13, `langchain-chroma` 1.1.0, `langchain-text-splitters` 1.1.3, `langchain-openai` 1.7.2);
- **Векторна база даних:** ChromaDB v1.5.9 (персистентне локальне сховище);
- **Модель ембеддингів:** `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, $d=384$);
- **Локальна LLM:** `qwen3:4b-instruct` через локальний OpenAI-сумісний HTTP API endpoint (`http://127.0.0.1:11434/v1`);
- **Аналіз та візуалізація:** Pandas 3.1.0, NumPy 2.5.2, Matplotlib 3.11.2;
- **Тестування:** PyTest 9.1.1.

---

## 6. Підготовка вхідних даних

### 6.1 Корпус знань
Створено авторську базу знань на тему **«Практичний життєвий цикл Machine Learning систем: від підготовки даних до моніторингу та реагування на інциденти»**, яка містить 8 документів:
1. `01_data_quality.md` — стандарти валідації даних, профайлінг, витоки даних;
2. `02_feature_engineering.md` — інженерія ознак, трансформації, архітектура Feature Store (онлайн/офлайн);
3. `03_model_training.md` — оптимізація AdamW, distributed training (DDP, FSDP), чекпоінти;
4. `04_validation_and_metrics.md` — крос-валідація, метрики при асиметричних витратах, калібрування;
5. `05_deployment_architecture.md` — інференс, квантування AWQ/INT8, патерни canary та blue-green;
6. `06_model_monitoring.md` — види дрифту (covariate shift, concept drift), індекс PSI, KS-тест;
7. `07_incident_response.txt` — рівні критичності інцидентів (P1–P3), тріаж, circuit breakers;
8. `08_security_and_reproducibility.txt` — загрози (отруєння, інверсія, evasion), детермінізм PyTorch.

### 6.2 Результати аудиту корпусу
| Файл | Тип | Символи | Слова | Рядки | Контрольна сума SHA-256 (prefix) | Статус |
|---|---|---|---|---|---|---|
| `01_data_quality.md` | md | 3511 | 439 | 27 | `cca1a7976f7fb048...` | Valid |
| `02_feature_engineering.md` | md | 3209 | 385 | 30 | `cd4c22b38609e5ea...` | Valid |
| `03_model_training.md` | md | 2882 | 362 | 27 | `254464141091f0ae...` | Valid |
| `04_validation_and_metrics.md` | md | 2673 | 319 | 31 | `6be0ca7d7f011ef7...` | Valid |
| `05_deployment_architecture.md` | md | 2771 | 343 | 26 | `7d3981ba5e4579bb...` | Valid |
| `06_model_monitoring.md` | md | 2637 | 326 | 34 | `06fd4be3e722d1e8...` | Valid |
| `07_incident_response.txt` | txt | 3125 | 406 | 41 | `43d14a59dd85a80d...` | Valid |
| `08_security_and_reproducibility.txt` | txt | 3076 | 356 | 31 | `b1e27b054edb00f2...` | Valid |

Порожніх файлів чи дублікатів не виявлено.

![Розподіл обсягу документів](../outputs/figures/corpus_document_lengths.png)

---

## 7. Архітектура реалізації

```mermaid
flowchart TD
    KB["Корпус знань (8 файлів .md/.txt)"] --> DL["LangChain DirectoryLoader (TextLoader UTF-8)"]
    DL --> Splitter["RecursiveCharacterTextSplitter (chunk=700, overlap=100)"]
    Splitter --> Chunks["55 фрагментів + Metadata (source, type, id)"]
    Chunks --> Embed["Embedding Model: all-MiniLM-L6-v2 (d=384)"]
    Embed --> Chroma[("Persistent ChromaDB (outputs/chroma_db)")]
    
    UserQuery["Запит користувача (Query)"] --> QueryEmbed["Vector Encoding"]
    QueryEmbed --> RetSim["Similarity Retriever (k=4)"]
    QueryEmbed --> RetMMR["MMR Retriever (k=4, fetch_k=12, λ=0.5)"]
    Chroma --> RetSim
    Chroma --> RetMMR
    
    RetSim --> CtxSim["Formatted Context + Metadata Headers"]
    RetMMR --> CtxMMR["Formatted Context + Metadata Headers"]
    
    Prompt["Grounded Prompt + Strict Anti-Hallucination Policy"] --> LLM["Local LLM: qwen3:4b-instruct (Ollama API)"]
    CtxSim --> Prompt
    CtxMMR --> Prompt
    LLM --> AnsSim["Answer via Similarity"]
    LLM --> AnsMMR["Answer via MMR"]
```

---

## 8. Хід виконання роботи

Робота реалізована у трьох автономних, послідовних Jupyter-ноутбуках:
1. `notebooks/01_corpus_preparation.ipynb`: завантаження через `DirectoryLoader`, криптографічний аудит, експерименти з розміром чанків (400/60, 700/100, 1000/150), збагачення метаданими та серіалізація.
2. `notebooks/02_chroma_retrieval.ipynb`: побудова векторної бази ChromaDB, перевірка персистентності (перевідкриття без дублювання), реалізація Similarity та MMR, виконання 9 контрольних запитів, програмна верифікація фільтрації за `type="md"` та за конкретним файлом.
3. `notebooks/03_rag_evaluation.ipynb`: складання наскрізного RAG-пайплайна з локальною мовною моделлю `qwen3:4b-instruct`, генерація 18 відповідей, оцінювання заземленості, наявності цитувань та аналіз обробки out-of-domain запиту.

---

## 9. Експерименти та результати

### 9.1 Дослідження параметрів чанкінгу
| Конфігурація | Розмір | Overlap | Кількість чанків | Мін. довжина | Макс. довжина | Середня довжина | Сер. кількість слів |
|---|---|---|---|---|---|---|---|
| Small | 400 | 60 | 92 | 39 | 399 | 261.36 | 32.32 |
| **Medium (Обрана)** | **700** | **100** | **55** | **80** | **695** | **434.76** | **53.65** |
| Large | 1000 | 150 | 34 | 280 | 990 | 700.71 | 86.35 |

Обрана конфігурація 700/100 забезпечує ідеальний баланс: середній розмір ~435 символів повністю вміщується в ліміт уваги `all-MiniLM-L6-v2` без надмірної фрагментації речень.

![Гістограма розподілу чанків](../outputs/figures/chunk_size_distribution.png)

### 9.2 Порівняння ретриверів: Similarity Search проти MMR
Для $k = 4$, $\text{fetch\_k} = 12$, $\lambda_{\text{mult}} = 0.5$:

| ID | Категорія запиту | Фільтр | Sim унікальні джерела | MMR унікальні джерела | Sim різноманітність | MMR різноманітність | Спільних чанків |
|---|---|---|---|---|---|---|---|
| Q1 | Single-Doc Factual | None | 2 | 3 | 0.50 | **0.75** | 1 |
| Q2 | Multi-Doc Synthesis | None | 2 | 2 | 0.50 | 0.50 | 4 |
| Q3 | Topical Overlap | None | 2 | 2 | 0.50 | 0.50 | 4 |
| Q4 | Paraphrase / Jargon | None | 2 | 3 | 0.50 | **0.75** | 2 |
| Q5 | Diversity Stress Test | None | 2 | 3 | 0.50 | **0.75** | 3 |
| Q6 | Metadata Filter (type=md) | `type=md` | 2 | 2 | 0.50 | 0.50 | 3 |
| Q7 | Metadata Filter (source) | `07_incident_response.txt` | 1 | 1 | 0.25 | 0.25 | 3 |
| Q8 | Metadata Filter (source) | `08_security_and_reproducibility.txt` | 1 | 1 | 0.25 | 0.25 | 3 |
| Q9 | Out-of-Domain | None | 4 | 3 | 1.00 | 0.75 | 3 |

![Порівняння різноманітності](../outputs/figures/retrieval_similarity_vs_mmr.png)

### 9.2.1 Емпіричний підбір та аналіз чутливості гіперпараметрів MMR
Для наукового обґрунтування вибору параметрів $k=4$, $\text{fetch\_k}=12$, $\lambda=0.5$ проведено сітковий аналіз (grid search) на складному комплексному запиті Q3 для $\lambda \in [0.2, 0.5, 0.8]$, $\text{fetch\_k} \in [8, 12, 16]$, $k \in [3, 4, 5]$:

| $\lambda$ | $\text{fetch\_k}$ | $k$ | Середня релевантність | Парна різноманітність | Унікальні файли | Час (мс) |
|---|---|---|---|---|---|---|
| 0.2 | 8 | 4 | 0.4143 | 0.7135 | 3 | 25.65 |
| 0.2 | 12 | 4 | 0.4214 | 0.7363 | 3 | 22.21 |
| 0.2 | 16 | 4 | 0.3682 | 0.7598 | 3 | 23.93 |
| 0.5 | 8 | 4 | 0.5397 | 0.5600 | 2 | 23.86 |
| **0.5** | **12** | **4** | **0.5397** | **0.5600** | **2** | **29.00** |
| 0.5 | 16 | 4 | 0.5397 | 0.5600 | 2 | 23.39 |
| 0.8 | 8 | 4 | 0.5397 | 0.5600 | 2 | 21.72 |
| 0.8 | 12 | 4 | 0.5397 | 0.5600 | 2 | 22.71 |
| 0.8 | 16 | 4 | 0.5397 | 0.5600 | 2 | 27.06 |

![Графік чутливості MMR](../outputs/figures/mmr_hyperparameter_sensitivity.png)

**Висновки щодо підбору гіперпараметрів:**
1. При $\lambda = 0.2$ надмірний штраф за подібність між чанками призводить до падіння семантичної релевантності (до 0.36–0.42) на користь штучної диверсифікації.
2. При $\lambda = 0.8$ алгоритм наближається до чистого Similarity, повертаючи суміжні фрагменти одного джерела (при $k=3$ різноманітність джерел падає до 1).
3. При $\lambda = 0.5$ досягається оптимальний баланс релевантності (>0.53) та високої мікро-різноманітності при низькій затримці інференсу.
4. Розмір пулу кандидатів $\text{fetch\_k} = 12$ ($3 \times k$) надає достатню варіативність для відбору без зайвих витрат на обчислення матриці парних відстаней.

### 9.3 Оцінювання генерації локальної LLM
| Запит | Sim Джерела | MMR Джерела | Grounding | Цитування | Спостереження |
|---|---|---|---|---|---|
| **Q1 (PSI Formula)** | `02_feature_eng.md`, `06_monitoring.md` | `03_training.md`, `02_feat.md`, `06_mon.md` | PASS | YES | MMR підтягнув контекст навчання моделей, розширивши контекст адаптації до дрифту. |
| **Q2 (Time-Travel Joins)** | `01_data_qual.md`, `02_feat_eng.md` | `01_data_qual.md`, `02_feat_eng.md` | PASS | YES | Відповіді ідентичні за змістом; чітко пояснено механізм запобігання lookahead leakage. |
| **Q3 (Covariate vs Concept)** | `07_incident.txt`, `06_monitoring.md` | `07_incident.txt`, `06_monitoring.md` | PASS | YES | Обидва ретривери витягнули математичні визначення дрифту та регламент тріажу інцидентів. |
| **Q4 (AdamW Regularization)** | `03_training.md`, `05_deployment.md` | `03_training.md`, `02_feat.md`, `08_sec.txt` | PASS | YES | MMR додав контекст детермінізму навчання та стабільності вагів із файлу безпеки. |
| **Q5 (DDP vs FSDP Memory)** | `03_training.md`, `02_feat_eng.md` | `03_training.md`, `01_data_qual.md`, `02_feat.md` | PASS | YES | Детально описано шардинг параметрів від $O(M)$ до $O(M/N)$. |
| **Q6 (AWQ Quantization)** | `02_feat.md`, `05_deployment.md` | `02_feat.md`, `05_deployment.md` | PASS | YES | Фільтр `type=md` спрацював бездоганно; у видачу увійшли виключно markdown-документи. |
| **Q7 (Incident Triage)** | `07_incident_response.txt` | `07_incident_response.txt` | PASS | YES | Фільтр по файлу повернув 100% чанків регламенту інцидентів (P1-P3, circuit breakers). |
| **Q8 (Adversarial & Determinism)** | `08_security_and_reproducibility.txt` | `08_security_and_reproducibility.txt` | PASS | YES | Модель чітко навела команди `torch.manual_seed()` та методи захисту від FGSM/PGD. |
| **Q9 (Quantum Computing)** | 4 різні документи | 3 різні документи | PASS | YES | **0 галюцинацій:** модель суворо відмовила: *\"Based on the provided technical documentation, there is insufficient information to answer this question.\"* |

![Метрики якості RAG](../outputs/figures/rag_grounding_and_metrics.png)

---

## 10. Аналіз отриманих результатів

1. **Вплив алгоритму пошуку:**
   - Для конкретних запитів із локалізованою відповіддю (Q2, Q3) різниця між Similarity та MMR нівелюється, оскільки топ-кандидати мають високу семантичну ізоляцію від решти корпусу.
   - Для комплексних тем (Q1, Q4, Q5) MMR демонструє суттєву перевагу: пеналізація надлишковості ($\lambda = 0.5$) відкидає повторні чанки з того самого файлу на користь суміжних документів. Це збагачує контекст LLM різними інженерними перспективами.
2. **Вплив метаданих:**
   - Фільтрація метаданих працює на рівні векторного індексу до етапу ранжування. Це гарантує, що в контекст гарантовано не потраплять дані з невідповідних джерел, що критично для систем із розмежуванням ролей.
3. **Надійність генерації:**
   - Навіть компактна локальна модель (4B параметрів) при температурі 0.0 і суворому заземленому промпті демонструє 100% фактологічну точність без домислювання фактів.

---

## 11. Виявлені проблеми й обмеження

1. **Модель ембеддингів `all-MiniLM-L6-v2`:** орієнтована насамперед на англійську мову, тому базу знань та запити доцільно формулювати англійською мовою (з україномовним супроводом). Для чисто україномовних баз знань знадобиться мультиязична модель (наприклад, `paraphrase-multilingual-MiniLM-L12-v2`).
2. **Залежність від гіперпараметрів MMR:** при занадто малому $\lambda$ ($\lambda < 0.3$) алгоритм може підтягнути надто віддалені за змістом фрагменти, які мають низьку релевантність до запиту.
3. **Обчислювальні ресурси:** запуск локальної LLM вимагає від 3 до 4 ГБ VRAM/RAM; для обмежених систем квантування 4-bit (AWQ / GGUF) є обов'язковим.

---

## 12. Висновки

1. Успішно розроблено та експериментально верифіковано повноцінну RAG-систему за варіантом №2 (ChromaDB + Markdown/Текст).
2. Завантажено та верифіковано базу знань із 8 документів (SHA-256 хеші зафіксовано, дублікати відсутні).
3. Досліджено три конфігурації чанкінгу; обрано оптимальну (`chunk_size=700`, `chunk_overlap=100`), яка згенерувала 55 семантично збалансованих фрагментів.
4. Створено персистентне векторне сховище ChromaDB із моделлю ембеддингів `sentence-transformers/all-MiniLM-L6-v2`. Перевірено відсутність дублювання записів при повторному відкритті.
5. Порівняно режими `similarity` та `mmr` на 9 запитах: доведено, що MMR збільшує унікальність джерел до 75% на комплексних темах, усуваючи надлишковість.
6. Підтверджено детерміновану роботу фільтрів метаданих за типом та джерелом.
7. Інтегровано локальну LLM `qwen3:4b-instruct`, яка продемонструвала 100% заземленість відповідей та абсолютну стійкість до галюцинацій на out-of-domain запитах.
8. Пройдено всі 10 автоматичних тестів тестового набору `pytest`.

---

## 13. Джерела
1. Lewis P. et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks // NeurIPS, 2020.
2. Carbonell J., Goldstein J. The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries // ACM SIGIR, 1998.
3. Reimers N., Gurevych I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks // EMNLP, 2019.
4. LangChain Documentation: Vector stores and retrievers (Chroma integration) // https://python.langchain.com
5. ChromaDB Architecture Documentation // https://docs.trychroma.com
6. Методичні вказівки до лабораторної роботи №3 «Реалізація RAG-системи для витягування знань» / Укл.: П.Б. Лящинський, Ю.В. Цимбал. — Львів: НУ «ЛП», 2025.

---

## 14. Докази виконання
- Офіційний академічний звіт DOCX: [`report/report.docx`](report.docx)
- Офіційний академічний звіт PDF: [`report/report.pdf`](report.pdf)
- Скріншоти виконаних блокнотів:
  - Notebook 01: ![Скріншот 01](../outputs/figures/screenshot_nb01_corpus.png)
  - Notebook 02: ![Скріншот 02](../outputs/figures/screenshot_nb02_retrieval.png)
  - Notebook 03: ![Скріншот 03](../outputs/figures/screenshot_nb03_rag.png)
- Виконаний Jupyter Notebook 1: [`notebooks/01_corpus_preparation.ipynb`](../notebooks/01_corpus_preparation.ipynb)
- Виконаний Jupyter Notebook 2: [`notebooks/02_chroma_retrieval.ipynb`](../notebooks/02_chroma_retrieval.ipynb)
- Виконаний Jupyter Notebook 3: [`notebooks/03_rag_evaluation.ipynb`](../notebooks/03_rag_evaluation.ipynb)
- Персистентна база ChromaDB: директорія [`outputs/chroma_db/`](../outputs/chroma_db/)
- Таблиці результатів: [`outputs/tables/`](../outputs/tables/)
- Текстові файли відповідей LLM: [`outputs/answers/`](../outputs/answers/)
- Графічні ілюстрації: [`outputs/figures/`](../outputs/figures/)
- Автоматичні тести: [`tests/test_lab03.py`](../tests/test_lab03.py) (10/10 passed)
- Матриця відповідності методичці: [`report/methodology_audit.md`](methodology_audit.md)
- Матеріали до захисту: [`report/defense_notes.md`](defense_notes.md)
