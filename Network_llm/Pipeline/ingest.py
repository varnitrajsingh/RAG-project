import os
import numpy as np
import faiss
import pickle
import time

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai


os.makedirs("vectorstore", exist_ok=True)
os.makedirs("data/uploads", exist_ok=True)


# ── Lazy client init ──────────────────────────────────────────────────────────
_client = None


def get_client():
    global _client
    if _client is not None:
        return _client

    api_key = None
    try:
        import streamlit as st
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

    if not api_key:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. "
            "Set it in Streamlit Secrets (cloud) or a .env file (local)."
        )

    _client = genai.Client(api_key=api_key)
    return _client


# ── Embeddings (batched with rate-limit handling) ─────────────────────────────
def get_embeddings(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    """
    Embed texts in small batches.
    Smaller default batch_size (50) avoids request-size limits on large PDFs.
    """
    client = get_client()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"  Embedding batch {i // batch_size + 1} / {-(-len(texts) // batch_size)} ({len(batch)} chunks)...")

        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch,
        )
        all_embeddings.extend([e.values for e in result.embeddings])

        # Respect rate limits between batches
        if i + batch_size < len(texts):
            time.sleep(0.6)

    return all_embeddings


# ── Build or extend FAISS index in chunk-batches ──────────────────────────────
def _build_index_from_chunks(
    chunks: list[str],
    index: faiss.IndexFlatL2 | None,
    embed_batch_size: int = 50,
    index_batch_size: int = 200,
) -> faiss.IndexFlatL2:
    """
    Embed `chunks` in `embed_batch_size` groups, then add to FAISS index
    in `index_batch_size` groups. Creates the index on first call.
    """
    for start in range(0, len(chunks), index_batch_size):
        batch_chunks = chunks[start : start + index_batch_size]
        print(f"  Indexing chunks {start + 1}–{start + len(batch_chunks)} of {len(chunks)}...")

        embeddings = get_embeddings(batch_chunks, batch_size=embed_batch_size)
        embeddings_np = np.array(embeddings, dtype=np.float32)

        if index is None:
            dim = embeddings_np.shape[1]
            index = faiss.IndexFlatL2(dim)

        index.add(embeddings_np)

    return index


# ── PDF ingestion ─────────────────────────────────────────────────────────────
def process_pdf(
    file_path: str,
    embed_batch_size: int = 50,
    index_batch_size: int = 200,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> tuple[list[str], faiss.IndexFlatL2]:
    """
    Load a PDF, split into chunks, embed in batches, and store in FAISS.

    Args:
        file_path:        Path to the PDF file.
        embed_batch_size: How many chunks per Gemini embedding call (reduce if hitting limits).
        index_batch_size: How many chunks to embed+index per outer batch (reduce for huge files).
        chunk_size:       Character size for each text chunk.
        chunk_overlap:    Overlap between consecutive chunks.

    Returns:
        (texts, index) — the chunk texts and the populated FAISS index.
    """
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]

    print(f"📄 Loaded {len(docs)} pages → {len(texts)} chunks")
    print(f"   Batching: embed={embed_batch_size} chunks/call, index={index_batch_size} chunks/batch\n")

    index = _build_index_from_chunks(
        texts,
        index=None,
        embed_batch_size=embed_batch_size,
        index_batch_size=index_batch_size,
    )

    # Persist to disk
    faiss.write_index(index, "vectorstore/faiss_index")
    with open("vectorstore/chunks.pkl", "wb") as f:
        pickle.dump(texts, f)

    print(f"\n✅ Indexed {len(texts)} chunks into FAISS")
    return texts, index


# ── Load persisted index ──────────────────────────────────────────────────────
def load_index() -> tuple[list[str], faiss.IndexFlatL2] | tuple[None, None]:
    """Load an existing FAISS index from disk. Returns (None, None) if not found."""
    index_path = "vectorstore/faiss_index"
    chunks_path = "vectorstore/chunks.pkl"

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return None, None

    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        texts = pickle.load(f)

    print(f"📦 Loaded existing index: {len(texts)} chunks")
    return texts, index


# ── Query ─────────────────────────────────────────────────────────────────────
def query_index(
    question: str,
    texts: list[str],
    index: faiss.IndexFlatL2,
    top_k: int = 3,
) -> list[str]:
    """Embed the question and retrieve the top_k most similar chunks."""
    q_embedding = get_embeddings([question], batch_size=1)
    q_np = np.array(q_embedding, dtype=np.float32)

    distances, indices = index.search(q_np, top_k)

    print(f"\n🔍 Query: {question}")
    print("─" * 50)
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
        print(f"\n[{rank}] Score: {dist:.4f}")
        print(f"{texts[idx][:300]}...")

    return [texts[i] for i in indices[0]]


# ── Local test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import fitz  # PyMuPDF

    sample_pdf_path = "sample_test.pdf"
    doc = fitz.open()

    pages_content = [
        (
            "Introduction to Neural Networks",
            "A neural network is a computational model inspired by the human brain. "
            "It consists of layers of interconnected nodes called neurons. "
            "Each connection has a weight that adjusts during training. "
            "Deep learning uses many such layers to learn complex patterns from data.",
        ),
        (
            "Transformer Architecture",
            "The Transformer model was introduced in the paper 'Attention is All You Need'. "
            "It relies entirely on self-attention mechanisms to draw global dependencies. "
            "BERT and GPT are popular models built on the Transformer architecture. "
            "Transformers have revolutionized natural language processing tasks.",
        ),
        (
            "Retrieval Augmented Generation",
            "RAG combines retrieval systems with generative models to improve factual accuracy. "
            "A document is split into chunks and stored in a vector database. "
            "At query time, semantically similar chunks are retrieved and passed to an LLM. "
            "FAISS is a popular library for efficient similarity search in RAG pipelines.",
        ),
    ]

    for title, content in pages_content:
        page = doc.new_page()
        page.insert_text((50, 50), title, fontsize=16, fontname="helv")
        page.insert_text((50, 100), content, fontsize=11, fontname="helv")

    doc.save(sample_pdf_path)
    doc.close()
    print(f"📝 Created sample PDF: {sample_pdf_path}\n")

    texts, index = process_pdf(
        sample_pdf_path,
        embed_batch_size=50,   # lower this (e.g. 20) if you hit quota errors
        index_batch_size=200,  # lower this (e.g. 100) for very large PDFs
    )

    test_queries = [
        "What is a neural network?",
        "How does RAG work with FAISS?",
        "What is the Transformer architecture?",
    ]

    for q in test_queries:
        query_index(q, texts, index, top_k=2)