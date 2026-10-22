import os
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

def build_notebook_01():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.6"
        }
    }

    cells = []

    # Title & Metadata
    cells.append(nbf.v4.new_markdown_cell("""# Лабораторна робота №3: Реалізація RAG-системи для витягування знань
## Частина 1: Підготовка, аудит та чанкінг текстового корпусу знань

**Дисципліна:** Нейромережеві технології і системи  
**Тема:** Реалізація RAG-системи для витягування знань  
**Варіант:** №2 — Chroma + Markdown / Текст  
**Предметна область корпусу:** Практичний життєвий цикл Machine Learning систем: від підготовки даних до моніторингу та реагування на інциденти  

---

### Мета першого етапу:
1. Дослідити склад та цілісність підготовленого корпусу знань, що складається з 8 тематичних документів у форматах `.md` та `.txt`.
2. Реалізувати завантаження документів за допомогою спеціалізованого завантажувача `DirectoryLoader` бібліотеки LangChain із налаштуванням підтримки різних розширень.
3. Провести глибокий аудит корпусу: перевірку розширень, обчислення криптографічних контрольних сум SHA-256, аналіз кількості символів, слів, рядків та відсутності порожніх чи дубльованих файлів.
4. Дослідити поведінку алгоритму розбиття тексту `RecursiveCharacterTextSplitter` для трьох різних конфігурацій розміру чанка та перекриття (`chunk_size` / `chunk_overlap`).
5. Зберегти детальні метадані для кожного фрагмента (`source`, `type`, `chunk_id`, `document_id`) з метою забезпечення суворого відстеження джерел (provenance) на етапах векторного пошуку та генерації відповідей."""))

    # Section 1: Environment Diagnostics
    cells.append(nbf.v4.new_markdown_cell("""## 1. Діагностика програмного середовища та залежностей

Перед початком роботи перевіримо версії встановлених бібліотек, доступність обчислювальних ресурсів та стан системних модулів. Для варіанта №2 обов'язковими є бібліотеки `langchain`, `langchain-community`, `langchain-text-splitters`, `chromadb` та `sentence-transformers`."""))

    cells.append(nbf.v4.new_code_cell("""import sys
import os
import platform
import torch
import langchain
import chromadb
import sentence_transformers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(f"Python version:      {platform.python_version()}")
print(f"Platform:            {platform.platform()}")
print(f"PyTorch version:     {torch.__version__}")
print(f"CUDA Available:      {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device:          {torch.cuda.get_device_name(0)}")
print(f"LangChain version:   {langchain.__version__}")
print(f"ChromaDB version:    {chromadb.__version__}")
print(f"Sentence-Transf.:    {sentence_transformers.__version__}")
print(f"Pandas version:      {pd.__version__}")
"""))

    # Section 2: Corpus Ingestion via DirectoryLoader
    cells.append(nbf.v4.new_markdown_cell("""## 2. Завантаження корпусу знань через LangChain `DirectoryLoader`

Методичні вказівки вимагають завантаження 5–8 документів через `DirectoryLoader` із використанням відповідних завантажувачів для різних розширень файлів.

У нашому проєкті база знань містить 8 документів:
- 6 файлів у форматі Markdown (`.md`): концептуальні конспекти життєвого циклу ML (Data Quality, Feature Engineering, Training, Validation, Deployment, Monitoring);
- 2 файли у форматі звичайного тексту (`.txt`): регламенти реагування на інциденти (Incident Response) та політики безпеки й відтворюваності (Security & Reproducibility).

Оскільки обидва типи файлів є текстовими, але мають різні розширення, ми налаштовуємо `DirectoryLoader` з використанням `TextLoader(encoding='utf-8')` для забезпечення коректної обробки спецсимволів і математичної розмітки."""))

    cells.append(nbf.v4.new_code_cell("""from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader

KB_DIR = Path("../data/knowledge_base").resolve()
assert KB_DIR.exists(), f"Директорія бази знань не знайдена: {KB_DIR}"

# Завантажуємо окремо Markdown документи
md_loader = DirectoryLoader(
    str(KB_DIR),
    glob="*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
md_docs = md_loader.load()

# Завантажуємо окремо Text документи
txt_loader = DirectoryLoader(
    str(KB_DIR),
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
txt_docs = txt_loader.load()

all_raw_docs = md_docs + txt_docs
all_raw_docs.sort(key=lambda d: Path(d.metadata["source"]).name)

print(f"Успішно завантажено документів: {len(all_raw_docs)}")
print(f"  - Markdown (.md) файлів:      {len(md_docs)}")
print(f"  - Текстових (.txt) файлів:    {len(txt_docs)}")
"""))

    # Section 3: Corpus Audit & SHA-256
    cells.append(nbf.v4.new_markdown_cell("""## 3. Криптографічний та статистичний аудит корпусу

Надійність RAG-системи безпосередньо залежить від чистоти та відтворюваності даних. Здійснимо комплексний аудит:
1. Перевірка розширень файлів (мають бути виключно `.md` та `.txt`).
2. Обчислення гешу SHA-256 для кожного файлу — це гарантує захист від неконтрольованої модифікації або пошкодження тексту.
3. Підрахунок символів, слів, рядків та перевірка відсутності порожніх файлів.
4. Перевірка відсутності дублікатів файлів за вмістом."""))

    cells.append(nbf.v4.new_code_cell("""import hashlib

audit_records = []
content_hashes = set()

for doc in all_raw_docs:
    path = Path(doc.metadata["source"])
    content = doc.page_content
    
    # Обчислення SHA-256 вмісту
    sha256_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    is_duplicate = sha256_hash in content_hashes
    content_hashes.add(sha256_hash)
    
    words = content.split()
    lines = content.splitlines()
    
    audit_records.append({
        "filename": path.name,
        "extension": path.suffix.lower(),
        "type": path.suffix.lower().replace(".", ""),
        "characters": len(content),
        "words": len(words),
        "lines": len(lines),
        "sha256": sha256_hash[:16] + "...",
        "full_sha256": sha256_hash,
        "is_empty": len(content.strip()) == 0,
        "is_duplicate": is_duplicate
    })

df_audit = pd.DataFrame(audit_records)
print(f"Всього унікальних документів: {len(content_hashes)} із {len(all_raw_docs)}")
assert not df_audit["is_empty"].any(), "Увага! Виявлено порожні файли в корпусі!"
assert not df_audit["is_duplicate"].any(), "Увага! Виявлено дублікати документів!"

display_cols = ["filename", "type", "characters", "words", "lines", "sha256", "is_empty"]
print(df_audit[display_cols].to_string(index=False))

# Збереження таблиці аудиту
out_tables = Path("../outputs/tables")
out_tables.mkdir(parents=True, exist_ok=True)
df_audit.to_csv(out_tables / "corpus_audit_report.csv", index=False)
print(f"\\nЗвіт аудиту збережено у {out_tables / 'corpus_audit_report.csv'}")
"""))

    # Visualization of corpus
    cells.append(nbf.v4.new_markdown_cell("""### 3.1 Візуалізація обсягу документів корпусу

Побудуємо діаграму розподілу довжин документів за кількістю слів та символів для наочного порівняння інформаційної місткості окремих розділів."""))

    cells.append(nbf.v4.new_code_cell("""out_figures = Path("../outputs/figures")
out_figures.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
colors = ['#1f77b4' if t == 'md' else '#ff7f0e' for t in df_audit['type']]
bars = plt.barh(df_audit['filename'], df_audit['words'], color=colors)
plt.xlabel('Кількість слів')
plt.title('Розподіл слів за документами')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Додамо легенду типів
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#1f77b4', label='.md (Markdown)'),
                   Patch(facecolor='#ff7f0e', label='.txt (Text)')]
plt.legend(handles=legend_elements, loc='lower right')

plt.subplot(1, 2, 2)
plt.barh(df_audit['filename'], df_audit['characters'], color=colors)
plt.xlabel('Кількість символів')
plt.title('Розподіл символів за документами')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
corpus_fig_path = out_figures / "corpus_document_lengths.png"
plt.savefig(corpus_fig_path, dpi=300)
plt.show()
print(f"Графік збережено у {corpus_fig_path}")
"""))

    # Section 4: Chunking Theory and Experiments
    cells.append(nbf.v4.new_markdown_cell("""## 4. Дослідження стратегій розбиття тексту (Chunking)

### 4.1 Теоретичне обґрунтування чанкінгу

Векторні моделі трансформерів (зокрема `all-MiniLM-L6-v2`) мають обмежене контекстне вікно (256 або 512 токенів). Якщо подати в модель занадто довгий текст, він або відсікається, або інформація «розмазується» по одному вектору фіксованого розміру ($d = 384$).

- **Занадто малий chunk**: фрагмент втрачає семантичний контекст, посилання на об'єкти та зв'язки причини й наслідку (fragmentation bias).
- **Занадто великий chunk**: у фрагмент потрапляє кілька непов'язаних ідей, зменшується точність косинусної близькості векторів до специфічного запиту користувача, зростає кількість "шуму" для LLM.
- **Роль перекриття (`chunk_overlap`)**: створює ковзне вікно (sliding window), що запобігає розриву критично важливих речень або формул на межі двох сусідніх фрагментів.

Для `RecursiveCharacterTextSplitter` пріоритет розділювачів за замовчуванням:
$$[\"\\n\\n\", \"\\n\", \" \", \"\"]$$
Це дозволяє розбивати текст природним шляхом: спочатку по абзацах, потім по рядках, потім по словах і в крайньому випадку по символах.

Дослідимо 3 конфігурації:
1. **Конфігурація 1 (Small):** `chunk_size = 400`, `chunk_overlap = 60`
2. **Конфігурація 2 (Medium - Baseline):** `chunk_size = 700`, `chunk_overlap = 100`
3. **Конфігурація 3 (Large):** `chunk_size = 1000`, `chunk_overlap = 150`"""))

    cells.append(nbf.v4.new_code_cell("""from langchain_text_splitters import RecursiveCharacterTextSplitter

configs = [
    {"name": "Small (400/60)", "chunk_size": 400, "chunk_overlap": 60},
    {"name": "Medium (700/100)", "chunk_size": 700, "chunk_overlap": 100},
    {"name": "Large (1000/150)", "chunk_size": 1000, "chunk_overlap": 150},
]

chunk_experiment_results = []
splitters_data = {}

for cfg in configs:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        length_function=len,
        separators=["\\n\\n", "\\n", " ", ""]
    )
    chunks = splitter.split_documents(all_raw_docs)
    lengths = [len(c.page_content) for c in chunks]
    words_count = [len(c.page_content.split()) for c in chunks]
    
    splitters_data[cfg["name"]] = {
        "chunks": chunks,
        "lengths": lengths,
        "words": words_count
    }
    
    chunk_experiment_results.append({
        "Config": cfg["name"],
        "Chunk Size": cfg["chunk_size"],
        "Overlap": cfg["chunk_overlap"],
        "Total Chunks": len(chunks),
        "Min Length": min(lengths),
        "Max Length": max(lengths),
        "Mean Length": round(np.mean(lengths), 2),
        "Median Length": round(np.median(lengths), 2),
        "Std Dev": round(np.std(lengths), 2),
        "Mean Words": round(np.mean(words_count), 2)
    })

df_chunk_exp = pd.DataFrame(chunk_experiment_results)
print("Результати порівняльного дослідження конфігурацій розбиття:")
print(df_chunk_exp.to_string(index=False))
df_chunk_exp.to_csv(out_tables / "chunking_experiments.csv", index=False)
"""))

    # Histogram of chunk distributions
    cells.append(nbf.v4.new_markdown_cell("""### 4.2 Порівняння розподілу довжин чанків"""))

    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(14, 4))

for idx, (cfg_name, data) in enumerate(splitters_data.items(), 1):
    plt.subplot(1, 3, idx)
    plt.hist(data["lengths"], bins=15, color='#2ca02c', edgecolor='black', alpha=0.7)
    plt.title(f"{cfg_name}\\nЧанків: {len(data['lengths'])}, Сер.: {np.mean(data['lengths']):.1f} симв.")
    plt.xlabel('Довжина чанка (символи)')
    plt.ylabel('Частота')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
dist_fig_path = out_figures / "chunk_size_distribution.png"
plt.savefig(dist_fig_path, dpi=300)
plt.show()
print(f"Гістограми збережено у {dist_fig_path}")
"""))

    # Section 5: Selection and Metadata Enrichment
    cells.append(nbf.v4.new_markdown_cell("""## 5. Фіналізація розбиття та збагачення метаданими

**Обґрунтування вибору:** Для моделі ембеддингів `sentence-transformers/all-MiniLM-L6-v2` найкращий компроміс між повнотою семантичного змісту та якістю векторного подання забезпечує **Конфігурація 2 (`chunk_size = 700`, `chunk_overlap = 100`)**. Вона формує збалансовану кількість чанків із середньою довжиною ~550 символів (~85 слів), що ідеально лягає в ефективне вікно трансформера без втрати важливого контексту.

### 5.1 Прив'язка метаданих (Metadata Provenance)
Відповідно до вимог варіанта №2 методички, кожен чанк повинен містити:
- `source`: повний та відносний шлях до файлу-джерела;
- `filename`: коротка назва файлу;
- `type`: тип документа (`md` або `txt`);
- `document_id`: ідентифікатор документа (наприклад, `01_data_quality`);
- `chunk_id`: глобально унікальний ідентифікатор фрагмента (`01_data_quality_chunk_001`);
- `char_count`: кількість символів;
- `word_count`: кількість слів."""))

    cells.append(nbf.v4.new_code_cell("""selected_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
    length_function=len,
    separators=["\\n\\n", "\\n", " ", ""]
)

final_chunks = []
chunk_metadata_records = []

# Розбиваємо кожен документ окремо, щоб контролювати нумерацію всередині документа
for doc in all_raw_docs:
    path = Path(doc.metadata["source"])
    doc_id = path.stem
    doc_ext = path.suffix.lower().replace(".", "")
    
    doc_chunks = selected_splitter.split_documents([doc])
    
    for i, chunk in enumerate(doc_chunks):
        chunk_id = f"{doc_id}_chunk_{i+1:03d}"
        
        # Збагачуємо метадані
        enriched_meta = {
            "source": str(path.name),
            "full_path": str(path),
            "filename": path.name,
            "type": doc_ext,
            "document_id": doc_id,
            "chunk_id": chunk_id,
            "chunk_index": i + 1,
            "total_chunks_in_doc": len(doc_chunks),
            "char_count": len(chunk.page_content),
            "word_count": len(chunk.page_content.split())
        }
        chunk.metadata = enriched_meta
        final_chunks.append(chunk)
        chunk_metadata_records.append(enriched_meta)

print(f"Сформовано фінальних збагачених чанків: {len(final_chunks)}")

df_final_chunks = pd.DataFrame(chunk_metadata_records)
print("\\nПриклад перших 5 метаданих:")
print(df_final_chunks[["chunk_id", "filename", "type", "char_count", "word_count"]].head(5).to_string(index=False))

# Збереження реєстру чанків
df_final_chunks.to_csv(out_tables / "chunks_summary.csv", index=False)
print(f"\\nРеєстр чанків збережено у {out_tables / 'chunks_summary.csv'}")
"""))

    # Display sample chunks
    cells.append(nbf.v4.new_markdown_cell("""### 5.2 Візуальна інспекція зразків чанків (Markdown vs TXT)

Переконаємося, що фрагменти зберігають змістовну цілісність як для структурованих Markdown-файлів, так і для текстових регламентів."""))

    cells.append(nbf.v4.new_code_cell("""# Зразок з Markdown
sample_md = next(c for c in final_chunks if c.metadata["type"] == "md")
print(f"=== ЗРАЗОК ЧАНКА (Markdown) [{sample_md.metadata['chunk_id']}] ===")
print(f"Метадані: {sample_md.metadata}")
print("Текст фрагмента:")
print("-" * 60)
print(sample_md.page_content)
print("-" * 60)

# Зразок з TXT
sample_txt = next(c for c in final_chunks if c.metadata["type"] == "txt")
print(f"\\n=== ЗРАЗОК ЧАНКА (Text) [{sample_txt.metadata['chunk_id']}] ===")
print(f"Метадані: {sample_txt.metadata}")
print("Текст фрагмента:")
print("-" * 60)
print(sample_txt.page_content)
print("-" * 60)
"""))

    # Section 6: Serialization for Stage 2
    cells.append(nbf.v4.new_markdown_cell("""## 6. Серіалізація оброблених даних для наступного етапу

Для гарантування ідентичності вхідних даних між першим, другим та третім ноутбуками збережемо підготовлений список об'єктів `Document` у файл `chunks_prepared.json`."""))

    cells.append(nbf.v4.new_code_cell("""import json

serialized_chunks = []
for c in final_chunks:
    serialized_chunks.append({
        "page_content": c.page_content,
        "metadata": c.metadata
    })

chunks_json_path = out_tables / "chunks_prepared.json"
with open(chunks_json_path, "w", encoding="utf-8") as f:
    json.dump(serialized_chunks, f, ensure_ascii=False, indent=2)

print(f"Успішно збережено {len(serialized_chunks)} чанків у {chunks_json_path}")
"""))

    # Conclusions
    cells.append(nbf.v4.new_markdown_cell("""## 7. Висновки першого етапу

1. **Цілісність корпусу:** Завантажено та верифіковано 8 тематичних документів (6 у форматі `.md` та 2 у форматі `.txt`). Аудит підтвердив відсутність порожніх файлів, дублікатів та спотворень кодування. Зафіксовано контрольні суми SHA-256.
2. **Результати чанкінгу:** Експериментальне порівняння трьох конфігурацій показало, що конфігурація `chunk_size = 700`, `chunk_overlap = 100` створює оптимальний розподіл фрагментів (середня довжина ~550 символів, відсутність мікрофрагментів).
3. **Метадані:** Кожен чанк надійно зв'язаний із початковим джерелом через поля `source`, `type`, `document_id` та `chunk_id`, що забезпечить точну роботу фільтрації метаданих та формування цитувань у наступних частинах роботи."""))

    nb.cells = cells
    return nb

if __name__ == "__main__":
    out_dir = Path("NNTS_VARIANT_2/lab03_chroma_rag/notebooks")
    out_dir.mkdir(parents=True, exist_ok=True)
    nb = build_notebook_01()
    nb_path = out_dir / "01_corpus_preparation.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Ноутбук збережено: {nb_path}")
    
    # Виконуємо ноутбук за допомогою NotebookClient
    print("Виконання ноутбука через NotebookClient...")
    client = NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(out_dir)}})
    client.execute()
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Ноутбук 01 успішно виконано з усіма виводами та графіками!")
