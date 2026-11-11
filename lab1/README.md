# Лабораторна робота №1. Трансформерні моделі на PyTorch

Реалізація власних спрощених трансформерних архітектур на PyTorch для двох задач обробки тексту: бінарної класифікації електронних листів (Enron Spam) та авторегресивного мовного моделювання (WikiText-2).

---

## Реалізовані варіанти

### Варіант 1. Класифікація повідомлень (Enron Spam)
- **Архітектура:** `Embedding` $\to$ `PositionalEncoding` $\to$ `TransformerEncoderLayer` $\to$ Masked Average Pooling $\to$ `Linear` класифікатор (`ham` / `spam`).
- **Датасет:** `SetFit/enron_spam` (`data/raw/enron_{train,test}.parquet`) — 31 716 сирих записів train (28 811 після дедуплікації: 23 048 train / 5 763 val) та 2 000 сирих записів test (1 666 незалежних після очищення дублікатів і перетину: 930 `ham`, 736 `spam`).
- **Навчання:** Зважена функція втрат `CrossEntropyLoss` ($w_{\text{ham}} \approx 0.981$, $w_{\text{spam}} \approx 1.020$), Early Stopping.

### Варіант 2. Авторегресивне моделювання (WikiText-2)
- **Архітектура:** `Embedding` $\to$ `PositionalEncoding` $\to$ `TransformerDecoderLayer` (з причинною трикутною маскою) $\to$ лінійна проєкція на словник.
- **Датасет:** `Salesforce/wikitext` (`wikitext-2-raw-v1`, `data/raw/wikitext2_*.txt`) — 2 122 137 / 221 611 / 249 691 токенів; після додавання `<eos>` сформовано 33 529 / 3 501 / 3 946 неперекривних вікон довжиною 64 зі зсувом $Y_t = X_{t+1}$.

---

## Структура каталогу

```text
lab1/
├── README.md
├── requirements.txt
├── utils.py                                 # Токенізація, Vocabulary, PE, causal mask
├── notebooks/
│   ├── 01_enron_eda_preprocessing.ipynb     # EDA та побудова enron_processed.pt
│   ├── 02_enron_transformer_classifier.ipynb # Тюнінг і оцінка Transformer Encoder
│   ├── 03_wikitext_eda_preprocessing.ipynb  # EDA та побудова wikitext_processed.pt
│   └── 04_wikitext_transformer_decoder.ipynb # Causal Decoder, PPL, BLEU, ROUGE-L
├── data/
│   └── raw/                                 # Сирі файли датасетів (.parquet, .txt)
└── outputs/
    ├── checkpoints/                         # best_enron_transformer.pt, best_wikitext_decoder.pt
    ├── metrics/                             # JSON/CSV історії навчання та фінальні метрики
    └── figures/                             # 4 підсумкові графіки (loss, CM, фінальні метрики)
```

---

## Запуск і відтворення

```powershell
pip install -r lab1/requirements.txt
```

1. Блокноти `01` та `03` детерміновано формують `data/processed/*.pt` із `data/raw/` (кеш `.pt` виключено з Git).
2. Виконуйте блокноти послідовно: `01 -> 02` (Enron) та `03 -> 04` (WikiText-2).
3. У блокнотах `02` і `04` прапорець `FORCE_RETRAIN = False` дозволяє завантажити збережені чекпоінти з `outputs/checkpoints/` без повторного тренування на GPU.

---

## Спостережувані результати (RTX 3050 Laptop GPU, 4 ГБ VRAM)

| Задача | Найкраща конфігурація | Основні тестові метрики |
|:---|:---|:---|
| **Enron Spam (Encoder)** | `d_model=128`, `nhead=2` | Accuracy `97.00%`, F1 `96.68%`, Loss `0.0964` (`TN=889, FP=41, FN=9, TP=727`) |
| **WikiText-2 (Decoder)** | `d_model=128`, `nhead=4` | Perplexity `148.48`, Loss `5.0004`, BLEU `0.0226`, ROUGE-L `0.0493` |

Обмеження: моделі навчені зі спрощеним словником рівня слів і фіксованим вікном 64 токени для демонстрації в межах 4 ГБ VRAM, що пояснює невисокі значення BLEU/ROUGE-L при жадібному декодуванні відкритого тексту.
