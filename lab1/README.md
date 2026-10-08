# Лабораторна робота №1. Побудова простої трансформерної моделі

**Дисципліна:** Нейромережеві технології і системи  
**Студент:** Max  
**Спеціальність:** 122 «Комп'ютерні науки (Обчислювальний інтелект та смарт-системи)»  
**Середовище розробки:** Python 3.11–3.13 (протестовано на Python 3.13.5), PyTorch 2.x

---

## 1. Мета роботи

Засвоїти фундаментальні принципи трансформерної архітектури (механізми самоуваги Self-Attention, позиційне кодування, причинне маскування Causal Mask). Навчитися проєктувати, навчати та оцінювати спрощені варіанти трансформерних моделей на реальних текстових корпусах для задач класифікації тексту та авторегресивної генерації послідовностей.

---

## 2. Реалізовані завдання та датасети

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
- **Обсяг корпусу:** Train — 2 122 137 лексичних токенів, Validation — 221 611, Test — 249 691. Після додавання межі рядка `<eos>` сформовано 33 529 / 3 501 / 3 946 неперекривних вікон довжиною 64 (Train / Validation / Test).
- **Маскування:** Причинна трикутна маска (Causal Mask), що гарантує відсутність доступу до майбутніх токенів під час навчання та генерації.
- **Ключові метрики:** Perplexity (PPL), BLEU, ROUGE-L на реальних авторегресивних генераціях.

---

## 3. Структура пакета

```text
LAB1_Transformer_FOR_MAX/
├── README.md                              # Головний опис проєкту, інструкції та результати
├── requirements.txt                      # Перелік залежностей для швидкого встановлення
├── utils.py                              # Спільний модуль (токенізація, Vocabulary, PE, маски, генерація)
├── execute_and_save_notebooks.py         # Скрипт пакетного запуску 01→04 зі збереженням outputs
├── REPORT_DRAFT.md                       # Розгорнутий академічний драфт звіту з формулами та таблицями
├── DEFENSE_NOTES.md                      # Конспект для захисту та відповіді на контрольні питання
├── requirements/
│   ├── README.md                         # Детальна інструкція з налаштування середовища (venv, PyCharm, kernel)
│   └── requirements.txt                  # Дублікат залежностей
├── notebooks/
│   ├── 01_enron_eda_preprocessing.ipynb  # EDA, очищення від дублікатів, словник Enron
│   ├── 02_enron_transformer_classifier.ipynb # Transformer Encoder: тюнінг, навчання й оцінка
│   ├── 03_wikitext_eda_preprocessing.ipynb # EDA, аналіз токенів, формування вікон WikiText-2
│   └── 04_wikitext_transformer_decoder.ipynb # Causal Decoder: тюнінг, генерація, PPL, BLEU, ROUGE-L
├── data/
│   ├── raw/                              # Локальні сирі файли (enron parquet, wikitext txt)
│   └── processed/                        # Підготовлені PyTorch-тензори та словники (.pt)
├── outputs/
│   ├── checkpoints/                      # Збережені натреновані чекпоінти моделей (.pt)
│   │   ├── best_enron_transformer.pt     # Ваги класифікатора Enron (~6.07 MB)
│   │   └── best_wikitext_decoder.pt      # Ваги декодера WikiText-2 (~11.50 MB)
│   ├── metrics/                          # Результати тюнінгу та тестування у JSON/CSV
│   │   ├── enron_test_metrics.json
│   │   ├── enron_training_history.csv
│   │   ├── enron_tuning_results.csv
│   │   ├── wikitext_test_metrics.json
│   │   ├── wikitext_training_history.csv
│   │   └── wikitext_tuning_results.csv
│   └── figures/                          # 15 збережених графіків та візуалізацій
└── tests/
    └── test_lab1.py                      # 9 unit-тестів математики, масок, архітектури та інференсу
```

---

## 4. Порядок запуску та інструкції

### 4.1. Встановлення середовища
1. Створіть та активуйте віртуальне середовище:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Встановіть залежності:
   ```powershell
   pip install -r requirements.txt
   ```
3. (Опціонально) Зареєструйте Jupyter kernel:
   ```powershell
   python -m ipykernel install --user --name=lab1_env --display-name="Python (lab1_env)"
   ```
   *Детальніше про налаштування у PyCharm та VS Code див. у `requirements/README.md`.*

### 4.2. Послідовність відкриття Notebooks
Усі ноутбуки налаштовані за принципом **notebook-first**. Вони вже містять виконані виходи, графіки та числові таблиці. Відкривати рекомендується у такій послідовності:

1. **`notebooks/01_enron_eda_preprocessing.ipynb`**  
   EDA датасету Enron, усунення витоку даних між train/validation/test, розрахунок вагових коефіцієнтів класів, побудова словника та збереження тензорів у `data/processed/enron_processed.pt`.
2. **`notebooks/02_enron_transformer_classifier.ipynb`**  
   Завантаження `enron_processed.pt`, візуалізація Positional Encoding, порівняльний тюнінг конфігурацій `d_model` та `nhead`, навчання найкращої моделі із збереженням ваг у `outputs/checkpoints/best_enron_transformer.pt`, оцінювання на незалежному тесті.
3. **`notebooks/03_wikitext_eda_preprocessing.ipynb`**  
   EDA корпусу статей Вікіпедії, аналіз частот токенів, формування авторегресивних вікон зі зсувом цілей на 1 токен, збереження у `data/processed/wikitext_processed.pt`.
4. **`notebooks/04_wikitext_transformer_decoder.ipynb`**  
   Завантаження `wikitext_processed.pt`, перевірка причинної маски на відсутність заглядання у майбутнє, навчання Transformer Decoder, збереження `outputs/checkpoints/best_wikitext_decoder.pt`, аналіз жадібної генерації та стохастичного семплювання, розрахунок Perplexity, BLEU та ROUGE-L.

### 4.3. Використання збережених Checkpoints без перенавчання
Вам **не потрібно повторно навчати моделі** протягом багатьох епох.
- Ваги моделей уже натреновані та збережені у папці `outputs/checkpoints/`.
- У ноутбуках 02 та 04 комірки тестування завантажують збережені чекпоінти автоматично та виконують фінальний інференс за кілька секунд.

### 4.4. Запуск Unit-тестів
Перевірити правильність реалізації ключових компонентів (маски, позиційне кодування, відсутність витоку, завантаження моделей):
```powershell
pytest tests/test_lab1.py -v
```
Усі 9 тестів повинні пройти успішно.

### 4.5. Вирішення можливої проблеми `ModuleNotFoundError: utils`
Усі ноутбуки містять динамічне автовизначення кореня проєкту:
```python
current_dir = Path.cwd().resolve()
if (current_dir / "utils.py").exists():
    LAB_ROOT = current_dir
elif (current_dir / ".." / "utils.py").resolve().exists():
    LAB_ROOT = (current_dir / "..").resolve()
else:
    LAB_ROOT = next((p for p in [current_dir, *current_dir.parents] if (p / "utils.py").exists()), current_dir)

if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))
```
Якщо ви запускаєте ноутбук в іншому нетиповому середовищі і отримуєте помилку:
1. Переконайтеся, що робочою директорією (Working Directory) обрано або корінь розпакованої папки (`LAB1_Transformer_FOR_MAX`), або папку `notebooks/`.
2. У PyCharm натисніть правою кнопкою миші на кореневу папку проєкту `LAB1_Transformer_FOR_MAX` -> `Mark Directory as` -> `Sources Root`.

---

## 5. Фактичні результати експериментів

Усі наведені нижче результати отримано в ході реальних експериментів.

### 5.1. Тюнінг гіперпараметрів Transformer Encoder (Enron)
Фіксовані параметри: `vocab_size = 10000`, `max_seq_len = 64`, `num_layers = 1`, `dim_feedforward = 256`, `dropout = 0.1`, `batch_size = 64`, `lr = 1e-3`.

| Конфігурація | $d_{model}$ | $n_{head}$ | Параметри | Час тюнінгу (5 епох) | Val Loss | Val Accuracy | Val Precision | Val Recall | Val F1-score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
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
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `d64_h2` | 64 | 2 | 1 356 752 | 24.5 с | 5.2996 | 200.26 |
| `d64_h4` | 64 | 4 | 1 356 752 | 25.1 с | 5.2967 | 199.67 |
| `d128_h2` | 128 | 2 | 2 768 784 | 33.0 с | 5.1949 | 180.35 |
| **`d128_h4`** *(найкраща)* | **128** | **4** | **2 768 784** | **34.2 с** | **5.1926** | **179.93** |

**Фінальне оцінювання найкращої моделі (`d128_h4`) на Test Split:**
- **Test Perplexity (PPL):** `148.48` (Test Loss: `5.0004`)
- **Mean BLEU (n=50 згенерованих продовжень):** `0.0226`
- **Mean ROUGE-L (n=50 згенерованих продовжень):** `0.0493`

**Приклади авторегресивної генерації натренованого декодера (Greedy Decoding):**
1. *Prompt:* `"the game was developed by"`  
   *Continuation:* `"the , and , and , and , and ,"`
2. *Prompt:* `"in the second world war the army"`  
   *Continuation:* `", and , and , and , and , and"`
3. *Prompt:* `", he briefly stayed in ( modern ) . he"`  
   *Continuation:* `"was , and , and , and , and ,"`
4. *Prompt:* `"after the death of the king"`  
   *Continuation:* `"of the , and , and , and , and"`

*Аналіз генерації:*  
Під час жадібного вибору (greedy search / $\arg\max$) компактна модель без переднавчання (1 шар, 2.77 млн параметрів) після кількох коректних локальних переходів (`he -> was`, `use -> of the`) потрапляє в характерну дегенеративну циклічну петлю найчастотніших уніграм корпусу (`", and , and"`). Це типова відома поведінка жадібного декодування в простих мовних моделях (Greedy Decoding Degeneration).

Для усунення зациклення у Ноутбуці №4 реалізовано стохастичне декодування ($T=0.85$, Top-$k=25$, Repetition Penalty $= 1.25$), яке повністю розриває петлі та породжує граматично різноманітні фрагменти тексту.

---

## 6. Висновки

1. **Transformer Encoder:** Механізм двонаправленої самоуваги (Self-Attention) з маскованим усередненням токенів показав точність 97.00% Accuracy та 96.68% F1-score при розпізнаванні спаму.
2. **Transformer Decoder:** Авторегресивна модель з причинною трикутною маскою знизила перплексію до 148.48 на реальних текстах Вікіпедії.
3. **Готовність до захисту:** Усі необхідні теоретичні обґрунтування, формули та відповіді на типові запитання викладача систематизовано у файлі [`DEFENSE_NOTES.md`](DEFENSE_NOTES.md), а повний текст звіту — у [`REPORT_DRAFT.md`](REPORT_DRAFT.md).
