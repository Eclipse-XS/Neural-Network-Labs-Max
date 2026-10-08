# Інструкція з налаштування робочого середовища (Environment Setup)

Цей документ містить покрокові вказівки для розгортання віртуального середовища Python, встановлення бібліотек та запуску Jupyter Notebook для лабораторної роботи №1 («Побудова простої трансформерної моделі»).

---

## 1. Рекомендована версія Python

- **Рекомендована версія:** **Python 3.11 – 3.13**
- **Підтверджена робоча версія середовища розробки:** **Python 3.13.5 (64-bit)**
- Перевірити встановлену версію у консолі:
  ```powershell
  python --version
  ```

---

## 2. Створення віртуального середовища (`.venv`)

Перейдіть у кореневу директорію розпакованого проєкту (`LAB1_Transformer_FOR_MAX`) та виконайте команду:

```powershell
python -m venv .venv
```

Це створить ізольовану директорію `.venv` з інтерпретатором Python та утилітою `pip`.

---

## 3. Активація середовища на Windows

### PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(Якщо виникає помилка політики виконання скриптів, дозвольте виконання для поточного процесу: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### Command Prompt (cmd.exe):
```cmd
.\.venv\Scripts\activate.bat
```

Після активації у початку командного рядка з'явиться маркер `(.venv)`.

---

## 4. Встановлення залежностей

Оновіть менеджер пакетів `pip` та встановіть бібліотеки зі списку вимог:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Усі залежності ретельно підібрані під фактичні імпорти коду (PyTorch, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, NLTK, PyArrow, Rouge-score, Jupyter/IPyKernel, Pytest).

---

## 5. Реєстрація Jupyter Kernel

Щоб Jupyter Notebook бачив створене середовище як окреме ядро, зареєструйте його:

```powershell
python -m ipykernel install --user --name=lab1_env --display-name="Python (lab1_env)"
```

---

## 6. Вибір Kernel у PyCharm або VS Code

### У PyCharm:
1. Відкрийте папку `LAB1_Transformer_FOR_MAX` як проєкт: `File -> Open...`.
2. Перейдіть до `Settings` (`Ctrl + Alt + S`) -> `Project: LAB1_Transformer_FOR_MAX` -> `Python Interpreter`.
3. Натисніть `Add Interpreter` -> `Add Local Interpreter...` -> оберіть `Existing` -> вкажіть шлях до `.\.venv\Scripts\python.exe`.
4. Відкрийте будь-який файл `.ipynb`. У правому верхньому кутку редактора ноутбука у полі **Kernel** оберіть доданий інтерпретатор або `Python (lab1_env)`.

### У VS Code:
1. Відкрийте папку через `File -> Open Folder...`.
2. Відкрийте будь-який `.ipynb` файл.
3. У верхньому правому кутку натисніть **Select Kernel** -> **Python Environments...** -> оберіть `.venv`.

---

## 7. Перевірка імпорту локального модуля `utils.py`

Виконайте швидку діагностичну команду в терміналі з активованим `.venv`:

```powershell
python -c "import utils; print('[OK] Модуль utils успішно імпортовано!')"
```

Також можна перевірити запуск тестів pytest:

```powershell
pytest tests/test_lab1.py
```

Усі 9 тестів повинні пройти зі статусом `PASSED`.

---

## 8. Відкриття та виконання Notebooks

Для запуску класичного інтерфейсу Jupyter у браузері:

```powershell
jupyter notebook
```

Рекомендований порядок ознайомлення та запуску:
1. `notebooks/01_enron_eda_preprocessing.ipynb` — EDA та підготовка датасету Enron Spam.
2. `notebooks/02_enron_transformer_classifier.ipynb` — Архітектура Transformer Encoder, класифікація спаму, метрики.
3. `notebooks/03_wikitext_eda_preprocessing.ipynb` — EDA та токенізація корпусу WikiText-2.
4. `notebooks/04_wikitext_transformer_decoder.ipynb` — Авторегресивний Transformer Decoder, генерація тексту, BLEU/ROUGE.

---

## 9. Використання збережених Checkpoints без повторного навчання

- Натреновані моделі вже включені до пакета в каталозі `outputs/checkpoints/`:
  - `best_enron_transformer.pt` (~6.07 МБ)
  - `best_wikitext_decoder.pt` (~11.50 МБ)
- Усі чотири ноутбуки вже містять повністю виконані комірки, збережені графіки та таблиці.
- Щоб перевірити роботу моделей на тестових даних, **не потрібно повторно навчати моделі** протягом багатьох епох. Достатньо виконати комірки ініціалізації та секції **«Тестування та оцінка якості на контрольних вибірках»** — вони автоматично завантажать збережені ваги та миттєво виведуть метрики і приклади генерації.
