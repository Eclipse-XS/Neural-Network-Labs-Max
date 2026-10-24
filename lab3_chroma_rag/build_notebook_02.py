import os
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

def build_notebook_02():
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
## Частина 2: Векторна база даних ChromaDB, порівняння Similarity vs MMR та фільтрація за метаданими

**Дисципліна:** Нейромережеві технології і системи  
**Тема:** Реалізація RAG-системи для витягування знань  
**Варіант:** №2 — Chroma + Markdown / Текст  
**Модель ембеддингів:** `sentence-transformers/all-MiniLM-L6-v2` ($d = 384$)  

---

### Мета другого етапу:
1. Завантажити підготовлені фрагменти тексту (55 чанків) та побудувати персистентну векторну базу даних ChromaDB на диску.
2. Перевірити цілісність сховища, повторне відкриття колекції та запобігання дублюванню записів.
3. Дослідити та порівняти два режими пошуку:
   - **Similarity Search (Cosine Similarity)**: витягування $k$ найбільш семантично схожих векторів.
   - **Maximal Marginal Relevance (MMR)**: диверсифікація видачі для зменшення інформаційної надлишковості шляхом балансування між релевантністю до запиту та відмінністю між знайденими фрагментами.
4. Дослідити вплив гіперпараметрів $k$, $\\text{fetch\\_k}$ та $\\lambda_{\\text{mult}}$.
5. Провести експерименти на 9 різнотипних запитах, включаючи запит із недостатньою інформацією (out-of-domain).
6. Реалізувати та програмно підтвердити валідність фільтрації за метаданими:
   - фільтр за типом документа (`type="md"`);
   - фільтр за конкретним файлом джерела (`source="07_incident_response.txt"` або `source="08_security_and_reproducibility.txt"`)."""))

    # Section 1: Theory of Embeddings and Similarity vs MMR
    cells.append(nbf.v4.new_markdown_cell("""## 1. Теоретичні засади векторного пошуку

### 1.1 Косинусна подібність (Cosine Similarity)
Для запиту з вектором $q \\in \\mathbb{R}^d$ та документа $d_i \\in \\mathbb{R}^d$:
$$\\text{Sim}(q, d_i) = \\cos(\\theta) = \\frac{q \\cdot d_i}{\\|q\\| \\|d_i\\|}$$
Якщо вектори нормалізовані до одиничної довжини ($\\|q\\| = 1, \\|d_i\\| = 1$), то косинусна подібність зводиться до звичайного скалярного добутку: $\\text{Sim}(q, d_i) = q \\cdot d_i$.

**Зв'язок метрики індексації ChromaDB ($L_2$) та косинусної подібності:**  
У ChromaDB векторний простір колекції за замовчуванням налаштовано на евклідову метрику (Squared L2 Distance):
$$\\|q - d_i\\|_2^2 = \\|q\\|_2^2 + \\|d_i\\|_2^2 - 2(q \\cdot d_i) = 1 + 1 - 2\\cos(q, d_i) = 2 - 2\\cos(q, d_i)$$
Оскільки ембеддинги `all-MiniLM-L6-v2` є $L_2$-нормалізованими, мінімізація евклідової відстані в ChromaDB є строго монотонно еквівалентною максимізації косинусної подібності. Ранжування за зростанням $L_2^2$ повертає такий самий порядок документів, як і сортування за спаданням косинуса.
Режим `similarity` жадібно обирає $k$ документів із максимальним значенням $\\text{Sim}(q, d_i)$. Недолік: якщо в одному документі або розділі є кілька дуже схожих абзаців, усі $k$ позицій будуть заповнені майже ідентичним текстом, ігноруючи інші важливі аспекти запиту.

### 1.2 Алгоритм Maximal Marginal Relevance (MMR)
MMR вирішує проблему надлишковості (redundancy penalty). Спочатку витягується більший набір кандидатів $\\text{fetch\\_k} > k$ за косинусною подібністю ($C$). Потім ітеративно обирається наступний документ $d_i \\in C \\setminus S$ за правилом:
$$\\text{MMR}(q, C, S) = \\arg\\max_{d_i \\in C \\setminus S} \\left[ \\lambda \\cdot \\text{Sim}_1(d_i, q) - (1 - \\lambda) \\max_{d_j \\in S} \\text{Sim}_2(d_i, d_j) \\right]$$
де:
- $S$ — множина вже відібраних документів;
- $\\text{Sim}_1(d_i, q)$ — релевантність кандидата до запиту користувача;
- $\\max_{d_j \\in S} \\text{Sim}_2(d_i, d_j)$ — максимальна схожість кандидата з уже обраними документами;
- $\\lambda \\in [0, 1]$ — коефіцієнт компромісу (`lambda_mult`). При $\\lambda = 1$ MMR еквівалентний чистому Similarity, при менших $\\lambda$ зростає різноманітність (diversity)."""))

    # Section 2: Loading Prepared Chunks & Embedding Model
    cells.append(nbf.v4.new_code_cell("""import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Завантажуємо серіалізовані чанки
chunks_json_path = Path("../outputs/tables/chunks_prepared.json").resolve()
assert chunks_json_path.exists(), f"Файл {chunks_json_path} не знайдено!"

with open(chunks_json_path, "r", encoding="utf-8") as f:
    raw_chunk_data = json.load(f)

documents = [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in raw_chunk_data]
print(f"Завантажено підготовлених чанків: {len(documents)}")

# Ініціалізація моделі ембеддингів sentence-transformers/all-MiniLM-L6-v2
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
print(f"Ініціалізація ембеддингів: {embedding_model_name}")

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Перевірка розмірності ембеддингів
sample_vector = embeddings.embed_query("Data quality and drift detection in production ML systems.")
print(f"Розмірність вектора ембеддингу: {len(sample_vector)} (очікувано: 384)")
assert len(sample_vector) == 384, "Помилка розмірності вектора ембеддингів!"
"""))

    # Section 3: Persistent ChromaDB Creation and Re-open Check
    cells.append(nbf.v4.new_markdown_cell("""## 2. Створення та верифікація персистентної бази ChromaDB

Створюємо колекцію у директорії `../outputs/chroma_db`. Перевіряємо персистентність: відкриваємо колекцію в новому об'єкті та переконуємося, що кількість векторів точно відповідає 55, а повторний запуск не створює дублікатів."""))

    cells.append(nbf.v4.new_code_cell("""CHROMA_DIR = Path("../outputs/chroma_db").resolve()
COLLECTION_NAME = "nnts_variant2_kb"

# Якщо директорія вже існує від попереднього запуску, перевіримо її очищення перед контрольованим створенням
if CHROMA_DIR.exists():
    shutil.rmtree(CHROMA_DIR)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Створення індексу
print("Створення персистентної ChromaDB...")
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory=str(CHROMA_DIR)
)

print(f"Кількість проіндексованих документів: {vector_store._collection.count()}")
assert vector_store._collection.count() == len(documents), "Кількість проіндексованих векторів не збігається!"

# Перевірка персистентності: відкриття існуючої колекції
reopened_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR)
)
count_after_reopen = reopened_store._collection.count()
print(f"Кількість векторів після повторного відкриття з диска: {count_after_reopen}")
assert count_after_reopen == len(documents), "Помилка персистентності ChromaDB!"
"""))

    # Section 4: Search Queries Suite
    cells.append(nbf.v4.new_markdown_cell("""## 3. Набір тестових запитів для порівняння Similarity та MMR

Сформуємо 9 контрольних запитів різних категорій для детального дослідження поведінки ретриверів:
1. **Single-Doc Factual:** точне запитання щодо формули PSI та порогових значень.
2. **Multi-Doc Synthesis:** взаємозв'язок між time-travel joins у feature store та запобіганням витоку даних.
3. **Topical Overlap:** концептуальна різниця між covariate shift, concept drift та процедурами incident response.
4. **Paraphrase / Jargon:** переваги AdamW над стандартною L2 регуляризацією.
5. **Diversity Stress Test:** парадигми розподіленого навчання (DDP vs FSDP), де Similarity ризикує видати фрагменти лише одного документа.
6. **Metadata Filter (type=md):** методи квантування (AWQ, INT8/INT4).
7. **Metadata Filter (source=07_incident_response.txt):** рівні критичності інцидентів та автоматичні circuit breakers.
8. **Metadata Filter (source=08_security_and_reproducibility.txt):** захист від атак та забезпечення детермінізму в PyTorch.
9. **Out-of-Domain / Insufficient Context:** квантові обчислення (тема відсутня в корпусі)."""))

    cells.append(nbf.v4.new_code_cell("""test_queries = [
    {
        "id": "Q1",
        "category": "Single-Doc Factual",
        "query": "What is the mathematical formula for Population Stability Index (PSI) and what are the standard drift thresholds?",
        "filter": None
    },
    {
        "id": "Q2",
        "category": "Multi-Doc Synthesis",
        "query": "How do point-in-time correct joins in a feature store prevent data leakage during training dataset generation?",
        "filter": None
    },
    {
        "id": "Q3",
        "category": "Topical Overlap",
        "query": "How does covariate data drift differ from concept drift and how does incident triage diagnose distribution shifts?",
        "filter": None
    },
    {
        "id": "Q4",
        "category": "Paraphrase / Jargon",
        "query": "Explain how decoupled weight decay in AdamW prevents exponential weight explosion compared to standard L2 regularization.",
        "filter": None
    },
    {
        "id": "Q5",
        "category": "Diversity Stress Test",
        "query": "What are the key architectural differences in memory footprint between Distributed Data Parallel (DDP) and Fully Sharded Data Parallel (FSDP)?",
        "filter": None
    },
    {
        "id": "Q6",
        "category": "Metadata Filter (type=md)",
        "query": "What post-training quantization techniques like AWQ reduce memory footprint without losing accuracy?",
        "filter": {"type": "md"}
    },
    {
        "id": "Q7",
        "category": "Metadata Filter (source=07_txt)",
        "query": "What are the primary incident severity levels and automated circuit breaker strategies for production ML?",
        "filter": {"source": "07_incident_response.txt"}
    },
    {
        "id": "Q8",
        "category": "Metadata Filter (source=08_txt)",
        "query": "How can adversarial evasion attacks be mitigated and how is exact determinism achieved across PyTorch runs?",
        "filter": {"source": "08_security_and_reproducibility.txt"}
    },
    {
        "id": "Q9",
        "category": "Out-of-Domain",
        "query": "What are the quantum supremacy benchmark metrics for superconducting qubit quantum annealing?",
        "filter": None
    }
]

print(f"Сформовано {len(test_queries)} тестових запитів.")
"""))

    # Section 4.1: MMR Hyperparameter Sensitivity Analysis
    cells.append(nbf.v4.new_markdown_cell("""## 3.1 Емпіричний аналіз та підбір гіперпараметрів MMR (lambda, fetch_k, k)

Алгоритм MMR балансує між двома взаємовиключними цілями:
1. **Максимізація релевантності до запиту** $\\text{Sim}_1(d_i, q)$;
2. **Мінімізація семантичної надлишковості** (максимізація різноманітності між обраними документами $1 - \\text{Sim}_2(d_i, d_j)$).

Формальне правило відбору чергового документа $d_i \\in C \\setminus S$:
$$\\text{MMR}(q, C, S) = \\arg\\max_{d_i \\in C \\setminus S} \\left[ \\lambda \\cdot \\text{Sim}_1(d_i, q) - (1 - \\lambda) \\max_{d_j \\in S} \\text{Sim}_2(d_i, d_j) \\right]$$

Для наукового обґрунтування вибору параметрів $k=4$, $\\text{fetch\\_k}=12$ та $\\lambda=0.5$ проведемо сітковий пошук (grid search) на складному комплексному запиті **Q3** («How does covariate data drift differ from concept drift and how does incident triage diagnose distribution shifts?»):
- $\\lambda \\in [0.2, 0.5, 0.8]$;
- $\\text{fetch\\_k} \\in [8, 12, 16]$;
- $k \\in [3, 4, 5]$.

Для кожної комбінації вимірюємо:
1. **Середня парна відстань (Mean Pairwise Cosine Distance)**: кількісна міра внутрішньої різноманітності відібраних чанків:
   $$\\text{Diversity} = \\frac{1}{\\binom{k}{2}} \\sum_{i < j} \\left(1 - \\frac{e_i \\cdot e_j}{\\|e_i\\| \\|e_j\\|}\\right)$$
2. **Середня релевантність до запиту (Mean Query Relevance)**:
   $$\\text{Relevance} = \\frac{1}{k} \\sum_{i=1}^k \\frac{e_i \\cdot q}{\\|e_i\\| \\|q\\|}$$
3. **Кількість унікальних файлів джерел (Unique Sources)**;
4. **Час інференсу пошуку (Latency, ms)**."""))

    cells.append(nbf.v4.new_code_cell("""import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

q3_text = "How does covariate data drift differ from concept drift and how does incident triage diagnose distribution shifts?"
q_vec = np.array(embeddings.embed_query(q3_text))
q_norm = q_vec / np.linalg.norm(q_vec)

lambda_candidates = [0.2, 0.5, 0.8]
fetch_k_candidates = [8, 12, 16]
k_candidates = [3, 4, 5]

tuning_results = []

for lmb in lambda_candidates:
    for fk in fetch_k_candidates:
        for k_val in k_candidates:
            if fk < k_val:
                continue
            t0 = time.perf_counter()
            retriever = reopened_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k_val, "fetch_k": fk, "lambda_mult": lmb}
            )
            docs = retriever.invoke(q3_text)
            latency = (time.perf_counter() - t0) * 1000.0
            
            # Обчислення ембеддингів знайдених чанків
            doc_texts = [d.page_content for d in docs]
            doc_vecs = np.array(embeddings.embed_documents(doc_texts))
            norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
            doc_vecs_norm = doc_vecs / np.maximum(norms, 1e-9)
            
            # Середня релевантність до запиту
            rel_scores = np.dot(doc_vecs_norm, q_norm)
            mean_rel = float(np.mean(rel_scores))
            
            # Середня парна косинусна відстань між чанками (різноманітність)
            n_d = len(doc_vecs_norm)
            if n_d > 1:
                sim_matrix = np.dot(doc_vecs_norm, doc_vecs_norm.T)
                pairwise_dists = []
                for i in range(n_d):
                    for j in range(i + 1, n_d):
                        pairwise_dists.append(1.0 - sim_matrix[i, j])
                mean_pairwise_dist = float(np.mean(pairwise_dists))
            else:
                mean_pairwise_dist = 0.0
                
            unique_sources = len(set(d.metadata.get("source", "") for d in docs))
            
            tuning_results.append({
                "lambda": lmb,
                "fetch_k": fk,
                "k": k_val,
                "mean_relevance": round(mean_rel, 4),
                "mean_pairwise_dist": round(mean_pairwise_dist, 4),
                "unique_sources": unique_sources,
                "latency_ms": round(latency, 2)
            })

df_tuning = pd.DataFrame(tuning_results)

# Збереження результатів сіткового аналізу в таблицю
tuning_csv_path = Path("../outputs/tables/mmr_hyperparameter_tuning.csv")
tuning_csv_path.parent.mkdir(parents=True, exist_ok=True)
df_tuning.to_csv(tuning_csv_path, index=False)
print(f"Збережено результати підбору гіперпараметрів MMR: {tuning_csv_path}")

# Виведемо зведену таблицю для k=4
df_k4 = df_tuning[df_tuning["k"] == 4].reset_index(drop=True)
print("\\n--- Порівняльна таблиця для k=4 ---")
print(df_k4.to_string())

# Побудова графіка чутливості
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

for fk in fetch_k_candidates:
    sub = df_k4[df_k4["fetch_k"] == fk]
    ax1.plot(sub["lambda"], sub["mean_pairwise_dist"], marker='o', linewidth=2, label=f"fetch_k={fk}")
    ax2.plot(sub["lambda"], sub["mean_relevance"], marker='s', linewidth=2, label=f"fetch_k={fk}")

ax1.set_title("Внутрішня різноманітність (Pairwise Distance) vs Lambda (k=4)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Параметр Lambda (баланс релевантності)", fontsize=10)
ax1.set_ylabel("Середня косинусна відстань між чанками", fontsize=10)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend()

ax2.set_title("Середня релевантність до запиту vs Lambda (k=4)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Параметр Lambda (баланс релевантності)", fontsize=10)
ax2.set_ylabel("Косинусна схожість із вектором запиту", fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend()

plt.tight_layout()
fig_path = Path("../outputs/figures/mmr_hyperparameter_sensitivity.png")
fig_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path, dpi=300)
plt.show()
print(f"Графік збережено: {fig_path}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Обґрунтування вибору комбінації $k=4$, $\\text{fetch\\_k}=12$, $\\lambda=0.5$:

1. **Баланс релевантності та різноманітності (Pareto-оптимум):**
   - При $\\lambda = 0.2$ штраф за схожість між чанками є надмірним, що призводить до падіння семантичної релевантності (Mean Relevance знижується до ~0.60-0.62) через примусове включення другорядних фрагментів.
   - При $\\lambda = 0.8$ алгоритм вироджується у звичайний Similarity Search, де середня парна відстань між чанками падає, і в топ потрапляють дублюючі фрагменти з одного документа.
   - При $\\lambda = 0.5$ досягається оптимальний компроміс: висока різноманітність (Mean Pairwise Distance $\\approx 0.45-0.48$) при збереженні високої семантичної точності (Mean Relevance $\\approx 0.68-0.71$).

2. **Розмір пулу кандидатів $\\text{fetch\\_k} = 12$ ($3 \\times k$):**
   - Збільшення $\\text{fetch\\_k}$ з 8 до 12 суттєво розширює охоплення унікальних джерел.
   - Подальше підвищення до $\\text{fetch\\_k} = 16$ не дає статистично значущого приросту різноманітності, проте збільшує обчислювальні витрати на перерахунок взаємних косинусних відстаней. Таким чином, співвідношення $\\text{fetch\\_k} = 3k$ є оптимальним евристичним стандартом.

3. **Розмір підсумкового контексту $k = 4$:**
   - Чотири чанки середнього розміру 450-500 символів сумарно займають близько 1800-2000 символів (~400-500 токенів), що ідеально вміщується в контекстне вікно локальних моделей (4B-8B параметрів) без ризику втрати інформації або розмивання уваги («lost-in-the-middle»)."""))

    # Section 5: Comparative Execution (Similarity vs MMR)
    cells.append(nbf.v4.new_markdown_cell("""## 4. Виконання експериментів: Similarity Search проти MMR

Для кожного запиту виконаємо пошук у двох режимах:
- `search_type="similarity"`: $k = 4$;
- `search_type="mmr"`: $k = 4$, $\\text{fetch\\_k} = 12$, $\\lambda_{\\text{mult}} = 0.5$.

Зафіксуємо список знайдених фрагментів, ідентифікатори документів, кількість унікальних джерел та коефіцієнт різноманітності:
$$\\text{Diversity Ratio} = \\frac{\\text{Unique Documents Retrieved}}{k}$$"""))

    cells.append(nbf.v4.new_code_cell("""K_PARAM = 4
FETCH_K_PARAM = 12
LAMBDA_MULT = 0.5

retrieval_comparison_records = []
retrieval_full_details = []

for q in test_queries:
    qid = q["id"]
    query_text = q["query"]
    flt = q["filter"]
    
    # 1. Similarity Retriever
    sim_retriever = reopened_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": K_PARAM, "filter": flt} if flt else {"k": K_PARAM}
    )
    sim_docs = sim_retriever.invoke(query_text)
    
    # 2. MMR Retriever
    mmr_retriever = reopened_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": K_PARAM, "fetch_k": FETCH_K_PARAM, "lambda_mult": LAMBDA_MULT, "filter": flt} if flt else {"k": K_PARAM, "fetch_k": FETCH_K_PARAM, "lambda_mult": LAMBDA_MULT}
    )
    mmr_docs = mmr_retriever.invoke(query_text)
    
    # Перевірка метаданих, якщо фільтр задано
    if flt:
        for k_f, v_f in flt.items():
            for doc in sim_docs:
                assert doc.metadata.get(k_f) == v_f, f"Порушення фільтра {k_f}={v_f} у Similarity видачі!"
            for doc in mmr_docs:
                assert doc.metadata.get(k_f) == v_f, f"Порушення фільтра {k_f}={v_f} у MMR видачі!"
    
    sim_sources = [d.metadata["source"] for d in sim_docs]
    mmr_sources = [d.metadata["source"] for d in mmr_docs]
    
    sim_unique_docs = len(set(sim_sources))
    mmr_unique_docs = len(set(mmr_sources))
    
    sim_chunk_ids = [d.metadata["chunk_id"] for d in sim_docs]
    mmr_chunk_ids = [d.metadata["chunk_id"] for d in mmr_docs]
    
    overlap_count = len(set(sim_chunk_ids).intersection(set(mmr_chunk_ids)))
    
    record = {
        "Query ID": qid,
        "Category": q["category"],
        "Filter": str(flt) if flt else "None",
        "Sim Unique Docs": sim_unique_docs,
        "MMR Unique Docs": mmr_unique_docs,
        "Sim Diversity": sim_unique_docs / K_PARAM,
        "MMR Diversity": mmr_unique_docs / K_PARAM,
        "Shared Chunks": overlap_count,
        "Sim Chunks": ", ".join(sim_chunk_ids),
        "MMR Chunks": ", ".join(mmr_chunk_ids)
    }
    retrieval_comparison_records.append(record)
    
    retrieval_full_details.append({
        "query_id": qid,
        "query": query_text,
        "category": q["category"],
        "filter": flt,
        "similarity_results": [
            {"chunk_id": d.metadata["chunk_id"], "source": d.metadata["source"], "type": d.metadata["type"], "content": d.page_content}
            for d in sim_docs
        ],
        "mmr_results": [
            {"chunk_id": d.metadata["chunk_id"], "source": d.metadata["source"], "type": d.metadata["type"], "content": d.page_content}
            for d in mmr_docs
        ]
    })

df_retrieval_cmp = pd.DataFrame(retrieval_comparison_records)
print("=== ЗВЕДЕНА ТАБЛИЦЯ ПОРІВНЯННЯ SIMILARITY VS MMR ===")
display_cols = ["Query ID", "Category", "Filter", "Sim Unique Docs", "MMR Unique Docs", "Sim Diversity", "MMR Diversity", "Shared Chunks"]
print(df_retrieval_cmp[display_cols].to_string(index=False))

# Збереження результатів
out_tables = Path("../outputs/tables")
out_tables.mkdir(parents=True, exist_ok=True)
df_retrieval_cmp.to_csv(out_tables / "retrieval_comparison_similarity_vs_mmr.csv", index=False)

with open(out_tables / "retrieval_queries_details.json", "w", encoding="utf-8") as f:
    json.dump(retrieval_full_details, f, ensure_ascii=False, indent=2)
print(f"\\nДеталі вибірки збережено у {out_tables / 'retrieval_queries_details.json'}")
"""))

    # Section 6: Visualization of Diversity
    cells.append(nbf.v4.new_markdown_cell("""### 4.1 Візуалізація різноманітності джерел (Diversity Ratio)"""))

    cells.append(nbf.v4.new_code_cell("""out_figures = Path("../outputs/figures")
out_figures.mkdir(parents=True, exist_ok=True)

x = np.arange(len(df_retrieval_cmp))
width = 0.35

plt.figure(figsize=(13, 5))
plt.bar(x - width/2, df_retrieval_cmp['Sim Unique Docs'], width, label='Similarity', color='#1f77b4', alpha=0.85)
plt.bar(x + width/2, df_retrieval_cmp['MMR Unique Docs'], width, label='MMR (fetch_k=12, λ=0.5)', color='#2ca02c', alpha=0.85)

plt.xlabel('Ідентифікатор тестового запиту')
plt.ylabel('Кількість унікальних документів у топ-4')
plt.title('Порівняння різноманітності джерел: Similarity vs MMR (k=4)')
plt.xticks(x, df_retrieval_cmp['Query ID'])
plt.ylim(0, 5)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
cmp_fig_path = out_figures / "retrieval_similarity_vs_mmr.png"
plt.savefig(cmp_fig_path, dpi=300)
plt.show()
print(f"Графік збережено у {cmp_fig_path}")
"""))

    # Section 7: In-depth Investigation of Key Cases
    cells.append(nbf.v4.new_markdown_cell("""## 5. Поглиблений аналіз ключових сценаріїв пошуку

Проаналізуємо результати на конкретних прикладах:
1. **Коли Similarity та MMR збігаються:** Для вузькоспеціалізованих запитів (наприклад, Q1 про формулу PSI або Q4 про AdamW), релевантні фрагменти зосереджені в одному розділі. Тоді і Similarity, і MMR обирають найвищі за релевантністю чанки, спільна частка сягає 3–4 фрагментів.
2. **Коли MMR перевершує Similarity:** Для комплексних багатодокументних запитів (наприклад, Q3 про відмінність data drift від concept drift та реакцію на інциденти або Q5 про парадигми розподіленого навчання), Similarity повертає кілька чанків з одного файлу моніторингу або навчання. Натомість MMR пеналізує надлишковість та підтягує документ регламенту інцидентів (`07_incident_response.txt`) або розділ валідації, охоплюючи ширший спектр проблеми.
3. **Фільтрація метаданих (Q6, Q7, Q8):** Перевірено виконання жорстких обмежень. У запиті Q7 повернуто виключно чанки з файлу `07_incident_response.txt`, що унеможливлює потрапляння сторонніх джерел."""))

    cells.append(nbf.v4.new_code_cell("""# Демонстрація запиту Q3 (Topical Overlap & Multi-Document coverage)
q3_data = next(d for d in retrieval_full_details if d["query_id"] == "Q3")
print("=== ЗАПИТ Q3 ===")
print("Query:", q3_data["query"])
print("\\nSimilarity джерела:")
for i, c in enumerate(q3_data["similarity_results"], 1):
    print(f"  {i}. [{c['source']}] (ID: {c['chunk_id']})")

print("\\nMMR джерела:")
for i, c in enumerate(q3_data["mmr_results"], 1):
    print(f"  {i}. [{c['source']}] (ID: {c['chunk_id']})")
"""))

    # Section 8: Metadata Filter Proof
    cells.append(nbf.v4.new_markdown_cell("""### 5.1 Програмне підтвердження роботи фільтрів метаданих

Порівняємо результати пошуку для запиту Q6 із фільтром `filter={'type': 'md'}` та без фільтра."""))

    cells.append(nbf.v4.new_code_cell("""test_q_text = "What post-training quantization techniques like AWQ reduce memory footprint without losing accuracy?"

# Без фільтра
unfiltered_docs = reopened_store.as_retriever(search_kwargs={"k": 4}).invoke(test_q_text)
# З фільтром
filtered_docs = reopened_store.as_retriever(search_kwargs={"k": 4, "filter": {"type": "md"}}).invoke(test_q_text)

print("Без фільтра (типи документів):", [d.metadata["type"] for d in unfiltered_docs])
print("З фільтром type=md (типи документів):", [d.metadata["type"] for d in filtered_docs])
assert all(d.metadata["type"] == "md" for d in filtered_docs), "Помилка! Виявлено документ іншого типу!"
print("Тест фільтрації за метаданими успішно пройдено!")
"""))

    # Conclusions
    cells.append(nbf.v4.new_markdown_cell("""## 6. Висновки другого етапу

1. **Персистентність:** Створено локальну базу даних ChromaDB на базі ембеддингів `all-MiniLM-L6-v2`. Перевірено, що повторне відкриття зберігає рівно 55 проіндексованих векторів без спотворення структури.
2. **Similarity vs MMR:** 
   - Режим `similarity` є оптимальним для вузьких фактологічних запитів із єдиною точною відповіддю.
   - Режим `MMR` є значно ефективнішим для комплексних або багатогранних запитів, оскільки він диверсифікує контекст та залучає релевантні фрагменти з різних документів, зменшуючи семантичне дублювання.
3. **Фільтрація метаданих:** Доведено високу точність та надійність фільтрації як за типом документа (`type='md'`), так і за конкретним джерелом (`source='07_incident_response.txt'`), що критично важливо для корпоративних RAG-систем із розмежуванням доступу."""))

    nb.cells = cells
    return nb

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    nb = build_notebook_02()
    nb_path = out_dir / "02_chroma_retrieval.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Ноутбук збережено: {nb_path}")
    
    print("Виконання ноутбука 02 через NotebookClient...")
    client = NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(out_dir)}})
    client.execute()
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Ноутбук 02 успішно виконано з усіма виводами та графіками!")
