# Фінальний інженерний аудит лабораторної роботи №1

Дата перевірки: 8 жовтня 2026 року.

## 1. Initial State

- Сирий Enron: `31 716` train і `2 000` test.
- Документація містила взаємовиключні або непідтверджені числа: `25 372/6 344`, `2 996/339`, альтернативну confusion matrix `[[900, 26], [24, 716]]`, WikiText vocabulary `20 000`, BLEU/ROUGE-L `0.0520/0.1740`.
- README і REPORT містили застарілі WikiText counts `33 149/3 462/3 901`, хоча notebook outputs і serialized tensors показували інше.
- Processed tensors були записані пізніше за checkpoints, тому їх сумісність не можна було вважати доведеною лише за timestamps.
- Сім одноразових або дубльованих скриптів лежали в корені лабораторної поруч з runtime-файлами.

## 2. Root Causes

### Enron

Початкові `25 372/6 344` — це 80/20 split сирих `31 716` записів до очищення. Реальний pipeline спочатку видаляє `51` порожній текст і `2 854` дублікати серед непорожніх train-повідомлень, отримуючи `28 811`, після чого робить stratified split `23 048/5 763`. Із test видалено внутрішні дублікати та перетин із train, тому залишено `1 666` прикладів (`930 ham`, `736 spam`). Числа `2 996/339` не відповідають жодному serialized split і не мають артефактного підтвердження.

### Enron metrics

Незалежний CPU inference із `best_enron_transformer.pt` на `enron_processed.pt` дав:

| Метрика | Значення |
|---|---:|
| Loss | 0.0964043 |
| Accuracy | 0.9699880 |
| Precision | 0.9466146 |
| Recall | 0.9877717 |
| F1 | 0.9667553 |
| Confusion matrix | `[[889, 41], [9, 727]]` |

Альтернативна matrix `[[900, 26], [24, 716]]` не відтворюється поточним checkpoint, vocabulary і test tensor.

### WikiText-2

Serialized vocabulary і checkpoint мають рівно `10 000` токенів; `token2id` та `id2token` збігаються. Embedding і output projection мають по `10 000` рядків. Значення `20 000` несумісне з checkpoint.

Raw corpus містить `2 122 137/221 611/249 691` лексичних токенів. Pipeline додає `<eos>` після кожного з `23 767/2 461/2 891` рядків, а потім використовує stride 64. Тому фактичні shapes — `33 529/3 501/3 946` вікон, а test tensor містить `252 544` цільові токени. Застарілі counts у документації ігнорували EOS-маркери.

Незалежне оцінювання checkpoint дало loss `5.0004225` і perplexity `148.4758783`. Підтверджений генеративний протокол: перші 50 test windows, prompt 15 токенів, reference 15 токенів, greedy generation до 15 токенів, NLTK BLEU зі smoothing method 1, ROUGE-L F1 зі stemming. Результат: BLEU `0.0226`, ROUGE-L `0.0493`. Числа `0.0520/0.1740` належать іншому або незбереженому протоколу й не можуть замінювати підтверджені метрики.

### Training history і notebook comparison

- Поточні та backup notebooks використовують однакові ключові принципи: split після deduplication, vocabulary лише з train, `|V|=10 000`, `L=64`, masked average pooling, causal mask, stride 64 та EOS між WikiText-рядками.
- Візуальний рефакторинг змінив presentation, таблиці, графіки, naming і контрольований checkpoint-loading path. Змінилася реалізаційна форма окремих блоків, але не shapes, vocabulary або checkpoint architecture.
- CSV histories узгоджуються з округленими notebook tables і checkpoint metadata (`Enron epoch=10`, `best_val_f1=0.9799642`; WikiText epoch=6`, `best_val_ppl=159.0026`). Однак CSV записані пізніше за checkpoints і checkpoints не містять повного optimizer state/log provenance. Тому їх узгодженість підтверджена, але незалежно довести походження кожного CSV-рядка з training run неможливо.

## 3. Fixes

- Виправлено WikiText split/window counts у README, REPORT і висновку notebook 03.
- Додано точний BLEU/ROUGE-L protocol до README, REPORT і DEFENSE_NOTES.
- README розділено на режим перегляду готових outputs і контрольований запуск без retraining.
- Одноразові генератори notebooks, patch script, дубльований runner та standalone generation probe переміщено в `_cleanup_review/`, не видалено.
- Усі notebooks повторно виконано 01→04 з чистими kernels; outputs збережено.
- Моделі, raw data, processed tensors, histories, metrics, figures і backups не видалялися.

Резервна копія до змін: `C:\Users\Eclipse\.codex\backups\neural-network-technologies-and-systems\max-lab1-pre-audit-20261008-1900`.

## 4. Final Structure

```text
max/lab1/
├── README.md
├── REPORT_DRAFT.md
├── DEFENSE_NOTES.md
├── AUDIT_REPORT.md
├── requirements.txt
├── utils.py
├── execute_and_save_notebooks.py
├── notebooks/                 # 4 виконані notebooks із outputs
├── notebooks_backup/          # історичні резервні копії
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── checkpoints/
│   ├── metrics/
│   └── figures/
├── tests/
└── _cleanup_review/           # 7 скриптів і caches, нічого не видалено
```

## 5. Validation

| Перевірка | Результат |
|---|---|
| Pytest | `9 passed` |
| Encoder forward + padding mask | Passed |
| Decoder forward + causal mask | Passed |
| No-future-token leakage | Passed |
| Checkpoint loading | Passed |
| Enron independent inference | Passed; metrics наведено вище |
| WikiText generation | Passed |
| WikiText independent perplexity | `148.4758783` |
| Notebook 01 | Passed, outputs saved, 68.8 s |
| Notebook 02 | Passed, outputs saved, 29.3 s |
| Notebook 03 | Passed, outputs saved, 30.3 s |
| Notebook 04 | Passed, outputs saved, 33.4 s |
| Processed tensor integrity | SHA-256 до/після повного notebook run не змінився |
| Checkpoint integrity | SHA-256 до/після не змінився |

Checkpoint SHA-256:

- `best_enron_transformer.pt`: `64DE77DF8E599AE6D8E066164C4F6A13074AE1354C2CF308A75949E454D97A45`
- `best_wikitext_decoder.pt`: `96C50BFFD58E8C064247B30442D46B77B10A680D1F9F6D51C12F82ABF45F4794`

Processed SHA-256:

- `enron_processed.pt`: `C125116A71465A64A67C0A5C0F6F14652D81FA7FD5F9F214243BC21577C00D60`
- `wikitext_processed.pt`: `39CA61E81089E630C13EFA2417DA820FED0E162178765CBB18F1B11C2AE229B1`

## 6. Final Verdict

Лабораторна технічно працездатна, checkpoint-compatible і відтворювана без retraining у контрольованому режимі. Критичної підміни vocabulary, architecture або processed tensors не виявлено. Основні суперечності були помилками звітності та змішуванням непідтверджених evaluation results, а не пошкодженням моделей.

Проєкт готовий бути джерелом для остаточного DOCX-звіту за умови, що DOCX використовуватиме лише метрики й протоколи з цього аудиту. Твердження про повну provenance training CSV слід формулювати обмежено: артефакти узгоджені, але повний незалежний доказ історії навчання відсутній.
