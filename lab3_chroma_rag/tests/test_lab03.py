import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "knowledge_base"
NOTEBOOKS = ROOT / "notebooks"


def notebook_source(name: str) -> str:
    payload = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def test_text_loading_is_utf8_and_nonempty():
    paths = sorted([*DATA.glob("*.md"), *DATA.glob("*.txt")])
    assert len(paths) == 8
    contents = [path.read_text(encoding="utf-8") for path in paths]
    assert all(text.strip() for text in contents)
    assert len(set(contents)) == len(contents)


def test_chunk_boundaries_and_overlap():
    splitters = pytest.importorskip("langchain_text_splitters")
    text = (DATA / "01_data_quality.md").read_text(encoding="utf-8")
    splitter = splitters.RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 700 for chunk in chunks)
    assert all(chunk in text for chunk in chunks)


def test_persisted_chunk_metadata_matches_content():
    records = json.loads((ROOT / "outputs/tables/chunks_prepared.json").read_text(encoding="utf-8"))
    ids = {item["metadata"]["chunk_id"] for item in records}
    assert len(ids) == len(records)
    for item in records:
        meta = item["metadata"]
        assert meta["source"] in {p.name for p in DATA.iterdir()}
        assert meta["type"] == Path(meta["source"]).suffix.lstrip(".")
        assert meta["char_count"] == len(item["page_content"])


def test_notebooks_contain_real_retrieval_and_prompt_construction():
    retrieval = notebook_source("02_chroma_retrieval.ipynb")
    rag = notebook_source("03_rag_evaluation.ipynb")
    assert "similarity_search_with_score" in retrieval
    assert 'search_type="mmr"' in retrieval
    assert '"filter": {"type": "md"}' in retrieval
    assert "format_docs_with_sources" in rag
    assert "ChatOpenAI" in rag and ".invoke(" in rag


@pytest.mark.integration
def test_embedding_dimension_and_chroma_roundtrip(tmp_path):
    chroma_mod = pytest.importorskip("langchain_chroma")
    hf_mod = pytest.importorskip("langchain_huggingface")
    docs_mod = pytest.importorskip("langchain_core.documents")
    embeddings = hf_mod.HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True}
    )
    assert len(embeddings.embed_query("metadata filtering")) == 384
    docs = [
        docs_mod.Document(page_content="Incident response uses severity levels.", metadata={"type": "txt"}),
        docs_mod.Document(page_content="Feature stores require point-in-time joins.", metadata={"type": "md"}),
    ]
    store = chroma_mod.Chroma.from_documents(docs, embeddings, persist_directory=str(tmp_path))
    assert store._collection.count() == 2
    filtered = store.similarity_search("feature store", k=2, filter={"type": "md"})
    assert filtered and all(doc.metadata["type"] == "md" for doc in filtered)


@pytest.mark.integration
def test_real_llm_endpoint_lists_configured_model():
    requests = pytest.importorskip("requests")
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
    except requests.RequestException as exc:
        pytest.skip(f"Ollama unavailable: {exc}")
    response.raise_for_status()
    assert "qwen3:4b-instruct" in {item["name"] for item in response.json().get("models", [])}

