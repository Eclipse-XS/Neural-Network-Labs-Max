import os
import json
import pytest
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"
CHROMA_DIR = PROJECT_ROOT / "outputs" / "chroma_db"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

def test_01_knowledge_base_files_count_and_extensions():
    """Перевірка кількості та розширень вхідних файлів бази знань."""
    assert KB_DIR.exists(), f"Директорія {KB_DIR} не існує"
    files = list(KB_DIR.glob("*.*"))
    assert len(files) == 8, f"Очікувалось 8 файлів, знайдено {len(files)}"
    
    md_files = list(KB_DIR.glob("*.md"))
    txt_files = list(KB_DIR.glob("*.txt"))
    assert len(md_files) == 6, f"Очікувалось 6 .md файлів, знайдено {len(md_files)}"
    assert len(txt_files) == 2, f"Очікувалось 2 .txt файлів, знайдено {len(txt_files)}"

def test_02_corpus_audit_and_non_empty():
    """Перевірка відсутності порожніх файлів та коректності контрольних сум."""
    audit_csv = TABLES_DIR / "corpus_audit_report.csv"
    assert audit_csv.exists(), f"Звіт аудиту {audit_csv} не знайдено"
    
    for f in KB_DIR.glob("*.*"):
        content = f.read_text(encoding="utf-8")
        assert len(content.strip()) > 500, f"Файл {f.name} занадто малий або порожній"

def test_03_directory_loader_and_splitting():
    """Перевірка завантаження через DirectoryLoader та збереження метаданих."""
    md_loader = DirectoryLoader(str(KB_DIR), glob="*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    txt_loader = DirectoryLoader(str(KB_DIR), glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    
    docs = md_loader.load() + txt_loader.load()
    assert len(docs) == 8
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    assert len(chunks) == 55, f"Очікувалось 55 чанків, отримано {len(chunks)}"

def test_04_prepared_chunks_metadata_integrity():
    """Перевірка повноти метаданих у серіалізованих чанках."""
    chunks_json = TABLES_DIR / "chunks_prepared.json"
    assert chunks_json.exists(), "Файл chunks_prepared.json відсутній"
    
    with open(chunks_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert len(data) == 55
    required_keys = {"source", "type", "chunk_id", "document_id", "char_count", "word_count"}
    for item in data:
        meta = item["metadata"]
        assert required_keys.issubset(meta.keys()), f"Відсутні метадані у чанку {meta.get('chunk_id')}"
        assert meta["type"] in ["md", "txt"]

def test_05_chromadb_persistence_and_vector_count():
    """Перевірка відкриття персистентного векторного сховища ChromaDB."""
    assert CHROMA_DIR.exists(), f"Директорія {CHROMA_DIR} не існує"
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    store = Chroma(
        collection_name="nnts_variant2_kb",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    
    count = store._collection.count()
    assert count == 55, f"Очікувалось 55 векторів у ChromaDB, знайдено {count}"

def test_06_similarity_retrieval():
    """Перевірка працездатності ретривера у режимі similarity."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    store = Chroma(
        collection_name="nnts_variant2_kb",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    results = retriever.invoke("What is Population Stability Index?")
    assert len(results) == 3
    assert any("06_model_monitoring" in r.metadata["source"] for r in results)

def test_07_mmr_retrieval():
    """Перевірка працездатності ретривера у режимі mmr."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    store = Chroma(
        collection_name="nnts_variant2_kb",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
    )
    results = retriever.invoke("What is Population Stability Index?")
    assert len(results) == 3

def test_08_metadata_filter_by_type():
    """Перевірка жорсткої фільтрації за типом документа (type=md)."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    store = Chroma(
        collection_name="nnts_variant2_kb",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4, "filter": {"type": "md"}}
    )
    results = retriever.invoke("How to triage incidents?")
    assert len(results) > 0
    for doc in results:
        assert doc.metadata["type"] == "md", f"Знайдено документ забороненого типу: {doc.metadata}"

def test_09_metadata_filter_by_specific_file():
    """Перевірка жорсткої фільтрації за конкретним файлом джерела."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    store = Chroma(
        collection_name="nnts_variant2_kb",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    target_file = "07_incident_response.txt"
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3, "filter": {"source": target_file}}
    )
    results = retriever.invoke("What are the incident severity levels?")
    assert len(results) == 3
    for doc in results:
        assert doc.metadata["source"] == target_file, f"Порушення фільтра джерела: {doc.metadata['source']}"

def test_10_evaluation_and_answers_exist():
    """Перевірка наявності згенерованих відповідей та звіту оцінювання RAG."""
    eval_csv = TABLES_DIR / "rag_evaluation_summary.csv"
    assert eval_csv.exists(), "Звіт оцінювання rag_evaluation_summary.csv не знайдено"
    
    answers_dir = PROJECT_ROOT / "outputs" / "answers"
    answer_files = list(answers_dir.glob("*.txt"))
    assert len(answer_files) == 9, f"Очікувалось 9 файлів відповідей, знайдено {len(answer_files)}"
