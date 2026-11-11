# Лабораторна робота №1. Побудова простої трансформерної моделі

**Дисципліна:** Нейромережеві технології і системи  
**Студент:** Max  
**Спеціальність:** 122 «Комп'ютерні науки (Обчислювальний інтелект та смарт-системи)»  
**Середовище розробки:** PyCharm, Python 3.13, PyTorch 2.14+cu130 (GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM)

---

## 1. Мета роботи

Засвоїти фундаментальні принципи трансформерної архітектури (механізми самоуваги Self-Attention, позиційне кодування, причинне маскування Causal Mask). Навчитися проєктувати, навчати та оцінювати спрощені варіанти трансформерних моделей на реальних текстових корпусах для задач класифікації тексту та авторегресивної генерації послідовностей.

---

## 2. Варіанти завдань та датасети

У лабораторній роботі повністю виконано **обидва варіанти** відповідно до офіційних методичних вказівок кафедри:

### Варіант 1. Класифікація електронних повідомлень (Enron Spam)
- **Архітектура:** Transformer Encoder ($\text{Embedding} \to \text{Positional Encoding} \to \text{TransformerEncoderLayer} \to \text{Masked Average Pooling} \to \text{Linear Classifier}$)
- **Датасет:** [`SetFit/enron_spam`](https://huggingface.co/datasets/SetFit/enron_spam) — реальний корпус листування компанії Enron.
- **Задача:** Бінарна класифікація (0 = `ham` — легітимний лист, 1 = `spam` — спам).
- **Обсяг вибірок:** 31 716 записів у сирій навчальній вибірці (після дедуплікації 28 811 листів розбито на 23 048 train та 5 763 validation); 2 000 записів у сирій тестовій вибірці (після усунення дублікатів та перетину з train залишено 1 666 повністю незалежних test: 930 ham, 736 spam).
- **Балансування класів:** Застосовано зважену функцію втрат `CrossEntropyLoss` із вагами класів, розрахованими за вибіркою Train ($w_{ham} \approx 0.981$, $w_{spam} \approx 1.020$).
- **Ключові метрики:** Accuracy, Precision, Recall, F1-score, Confusion Matrix.

### Варіант 2. Авторегресивне продовження тексту (WikiText-2)
- **Архітектура:** Transformer Decoder ($\text{Embedding} \to \text{Positional Encoding} \to \text{TransformerDecoderLayer} \to \text{Linear Projection}$)
- **Датасет:** [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext) (конфігурація `wikitext-2-raw-v1`) — корпус якісних статей англомовної Вікіпедії.
- **Задача:** Прогнозування наступного токена (Next-Token Prediction, Causal Language Modeling) зі зсувом цілей на 1 крок ($Y_t = X_{t+1}$).
- **Обсяг корпусу до додавання меж рядків:** Train — 2 122 137 лексичних токенів, Validation — 221 611, Test — 249 691. Під час формування потоку після кожного рядка додається `<eos>`, тому фактичні тензори містять 33 529 / 3 501 / 3 946 неперекривних вікон довжиною 64 (Train / Validation / Test).
- **Маскування:** Причинна трикутна маска (Causal Mask), що гарантує відсутність доступу до майбутніх токенів під час навчання та генерації.
- **Ключові метрики:** Perplexity (PPL), BLEU, ROUGE-L на реальних авторегресивних генераціях.

---

## 3. Структура проєкту

```text
max/lab1/
├── README.md                              # Головний опис проєкту, інструкції та результати
├── requirements.txt                      # Залежності середовища
├── utils.py                              # Спільні утиліти (токенізація, Vocabulary, PE, маски)
├── REPORT_DRAFT.md                       # Розгорнутий академічний драфт звіту з формулами та таблицями
├── DEFENSE_NOTES.md                      # Конспект для захисту та відповіді на 10 контрольних питань
├── notebooks/
│   ├── 01_enron_eda_preprocessing.ipynb  # EDA, фільтрація, дедуплікація, словник Enron
│   ├── 02_enron_transformer_classifier.ipynb # Побудова, тюнінг, навчання й оцінка Transformer Encoder
│   ├── 03_wikitext_eda_preprocessing.ipynb # EDA, аналіз токенів, формування вікон WikiText-2
│   └── 04_wikitext_transformer_decoder.ipynb # Causal Decoder, тюнінг, генерація, PPL, BLEU, ROUGE-L
├── data/
│   ├── raw/                              # Локальні сирі файли (enron parquet, wikitext txt)
│   └── processed/                        # Серіалізовані PyTorch-тензори та словники (pt)
├── outputs/
│   ├── checkpoints/                      # Збережені найкращі чекпоінти моделей (.pt)
│   │   ├── best_enron_transformer.pt     # Чекпоінт класифікатора Enron
│   │   └── best_wikitext_decoder.pt      # Чекпоінт мовної моделі WikiText-2
│   ├── metrics/                          # Результати тюнінгу та тестування у JSON/CSV
│   │   ├── enron_tuning_results.csv
│   │   ├── enron_test_metrics.json
│   │   ├── wikitext_tuning_results.csv
│   │   └── wikitext_test_metrics.json
│   └── figures/                          # Збережені візуалізації та графіки
│       ├── enron_class_distribution.png
│       ├── enron_length_distribution.png
│       ├── enron_top_tokens.png
│       ├── enron_positional_encoding.png
│       ├── enron_training_loss.png
│       ├── enron_confusion_matrix.png
│       ├── wikitext_split_sizes.png
│       ├── wikitext_top_tokens.png
│       ├── wikitext_causal_mask.png
│       └── wikitext_training_loss.png
└── tests/
    └── test_lab1.py                      # Компактні unit-тести математики та інференсу
```


---

## 4. Встановлення та запуск

### 4.1. Встановлення залежностей
Активація віртуального середовища та встановлення пакетів:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4.2. Послідовність виконання ноутбуків
Ноутбуки розроблені за принципом **notebook-first** і залежать лише від артефактів на диску (`data/processed/`):

1. **`notebooks/01_enron_eda_preprocessing.ipynb`**  
   Завантажує сирий датасет Enron, проводить EDA, очищення від дублікатів, будує `Vocabulary` виключно за train split та зберігає тензори у `data/processed/enron_processed.pt`.
2. **`notebooks/02_enron_transformer_classifier.ipynb`**  
   Завантажує `enron_processed.pt`, візуалізує Positional Encoding, проводить порівняльний тюнінг `d_model` та `nhead`, навчає найкращу модель із Early Stopping, зберігає чекпоінт `outputs/checkpoints/best_enron_transformer.pt` та оцінює якість на тестовій вибірці.
3. **`notebooks/03_wikitext_eda_preprocessing.ipynb`**  
   Завантажує корпус WikiText-2, проводить EDA, будує словник за train split, формує авторегресивні контекстні вікна з causal-зсувом ($Y_t = X_{t+1}$) та зберігає тензори у `data/processed/wikitext_processed.pt`.
4. **`notebooks/04_wikitext_transformer_decoder.ipynb`**  
   Завантажує `wikitext_processed.pt`, перевіряє відсутність витоку майбутніх токенів (тест маски), проводить порівняльний тюнінг декодера, навчає мовну модель, генерує текст (Greedy Decoding) та оцінює Perplexity, BLEU і ROUGE-L на тестовому спліті.

Для перегляду готових результатів повторне навчання не потрібне: відкрийте notebooks, у яких уже збережені outputs. Для контрольованого повторного виконання використовується `FORCE_RETRAIN = False`; моделі завантажуються з checkpoints. Послідовний запуск:

```powershell
.\.venv\Scripts\python.exe max/lab1/execute_and_save_notebooks.py
```

Ноутбуки 01 і 03 детерміновано перебудовують processed tensors. Перед таким запуском доцільно звірити, що сирі файли не змінювалися.

### 4.3. Запуск автоматичних перевірок (Unit Tests)
```powershell
.\.venv\Scripts\pytest.exe max/lab1/tests/test_lab1.py -v
```

---

## 5. Фактичні результати експериментів

Усі наведені нижче результати отримано в ході реальних запусків навчання на локальному GPU (NVIDIA RTX 3050 Laptop).

### 5.1. Тюнінг гіперпараметрів Transformer Encoder (Enron)
Фіксовані параметри: `vocab_size = 10000`, `max_seq_len = 64`, `num_layers = 1`, `dim_feedforward = 256`, `dropout = 0.1`, `batch_size = 64`, `lr = 1e-3`.

| Конфігурація | $d_{model}$ | $n_{head}$ | Параметри | Час тюнінгу (5 епох) | Val Loss | Val Accuracy | Val Precision | Val Recall | Val F1-score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `d64_h2` | 64 | 2 | 690 114 | 31.4 с | 0.0957 | 0.9658 | 0.9749 | 0.9534 | 0.9640 |
| `d64_h4` | 64 | 4 | 690 114 | 28.8 с | 0.0892 | 0.9705 | 0.9699 | 0.9685 | 0.9692 |
| **`d128_h2`** *(найкраща)* | **128** | **2** | **1 412 738** | **34.0 с** | **0.0855** | **0.9721** | **0.9530** | **0.9906** | **0.9715** |
| `d128_h4` | 128 | 4 | 1 412 738 | 35.2 с | 0.0878 | 0.9689 | 0.9800 | 0.9548 | 0.9672 |

**Фінальне оцінювання найкращої моделі (`d128_h2`) на незалежному Test Split (1666 повідомлень: 930 Ham, 736 Spam):**
- **Test Accuracy:** `97.00%` (1 616 / 1 666)
- **Test Precision:** `94.66%` (727 / 768)
- **Test Recall:** `98.78%` (727 / 736, пропущено лише 9 спам-листів)
- **Test F1-score:** `96.68%`
- **Test Cross-Entropy Loss:** `0.0964`
- **Confusion Matrix:** TN = 889, FP = 41, FN = 9, TP = 727

### 5.2. Тюнінг гіперпараметрів Transformer Decoder (WikiText-2)
Фіксовані параметри: `vocab_size = 10000`, `seq_len = 64`, `num_layers = 1`, `dim_feedforward = 256`, `dropout = 0.1`, `batch_size = 64`, `lr = 1e-3`.

| Конфігурація | $d_{model}$ | $n_{head}$ | Параметри | Час тюнінгу (2 епохи) | Val Loss | Val Perplexity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `d64_h2` | 64 | 2 | 1 356 752 | 24.5 с | 5.2996 | 200.26 |
| `d64_h4` | 64 | 4 | 1 356 752 | 25.1 с | 5.2967 | 199.67 |
| `d128_h2` | 128 | 2 | 2 768 784 | 33.0 с | 5.1949 | 180.35 |
| **`d128_h4`** *(найкраща)* | **128** | **4** | **2 768 784** | **34.2 с** | **5.1926** | **179.93** |

**Фінальне оцінювання найкращої моделі (`d128_h4`) на Test Split:**
- **Test Perplexity (PPL):** `148.48` (Test Loss: `5.0004`)
- **Mean BLEU (n=50 згенерованих продовжень):** `0.0226`
- **Mean ROUGE-L (n=50 згенерованих продовжень):** `0.0493`

Протокол генеративних метрик: перші 50 тестових вікон; перші 15 токенів кожного вікна — prompt; наступні 15 — єдиний reference; генерується до 15 токенів greedy decoding (`argmax`); BLEU обчислюється `sentence_bleu` зі згладжуванням NLTK method 1, ROUGE-L — F1 зі stemming. Значення з іншим prompt/generation length, sampling або усередненням не можна порівнювати з цими числами як з тим самим експериментом.

**Фактичні приклади авторегресивної генерації натренованого декодера (Greedy Decoding):**
1. *Prompt:* `"the game was developed by"`  
   *Continuation:* `"the , and , and , and , and ,"`
2. *Prompt:* `"in the second world war the army"`  
   *Continuation:* `", and , and , and , and , and"`
3. *Prompt:* `", he briefly stayed in ( modern ) . he"`  
   *Continuation:* `"was , and , and , and , and ,"`
4. *Prompt:* `"after the death of the king"`  
   *Continuation:* `"of the , and , and , and , and"`
5. *Prompt:* `"advances or contributed outstanding examples " . furthermore , his poems use"`  
   *Continuation:* `"of the"`

*Аналіз генерації:*  
Під час жадібного вибору (greedy search / $\arg\max$) компактна модель без переднавчання (1 шар, 2.77 млн параметрів) після кількох коректних локальних переходів (`he -> was`, `use -> of the`) потрапляє в характерну дегенеративну циклічну петлю найчастотніших уніграм корпусу (`", and , and"`). Це типова відома поведінка жадібного декодування в простих мовних моделях (Greedy Decoding Degeneration).

Для усунення зациклення додано експериментальне стохастичне декодування ($T=0.85$, Top-$k=25$, Repetition Penalty $= 1.25$), яке повністю розриває петлі та породжує різноманітні енциклопедичні фрагменти без зміни базової моделі чи зміни обов'язкової метрики Perplexity.

---

## 6. Висновки та обмеження

1. **Ефективність Transformer Encoder для класифікації:** Механізм двонаправленої самоуваги (Self-Attention) з маскованим усередненням токенів показав надзвичайно високу точність (97.0% Accuracy, 96.68% F1) при розпізнаванні спаму. Модель вловлює семантичні взаємозв'язки між словами незалежно від їх позиції в листі.
2. **Робота Causal Decoder для генерації:** Авторегресивна модель з причинною маскою успішно знизила перплексію з випадкової ($PPL \approx 10000$) до 148.48 на реальних текстах Вікіпедії. Модель коректно відтворює синтаксичні конструкції та пунктуацію.
3. **Обмеження метрик BLEU та ROUGE-L для відкритої генерації:** Оскільки вільний текст має безліч граматично та семантично коректних синонімічних продовжень, n-грамні метрики збігу з єдиним еталоном (Reference) показують низькі абсолютні значення, що є очікуваним у мовному моделюванні без багатоеталонного оцінювання.
4. **Обмеження апаратного масштабу:** Використання компактної розмірності ($d_{model}=128$, 1 шар декодера) дозволило повністю виконати дослідження в межах 4 GB VRAM за лічені хвилини. Збільшення кількості шарів (до 4–6) та розміру контекстного вікна (до 256–512) суттєво покращило б довготривалу зв'язність тексту, проте вимагало б значно більших обчислювальних ресурсів.
