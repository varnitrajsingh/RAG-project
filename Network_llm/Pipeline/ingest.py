import os
import numpy as np
import faiss
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ── Directory Setup ──────────────────────────────────────────────────────────
os.makedirs("vectorstore", exist_ok=True)
os.makedirs("data/uploads", exist_ok=True)

# ── Lazy Model Initialization ────────────────────────────────────────────────
_embed_model = None

def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print("🔄 Loading embedding model (first run only)...")
        _embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        print("✅ Embedding model loaded.")
    return _embed_model

# ── Embedding Logic ──────────────────────────────────────────────────────────
def get_embeddings(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Embed texts in batches using local BAAI/bge-small-en-v1.5 model.
    No API key, no rate limits, true batching supported.
    """
    model = get_embed_model()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"  [Embed] Batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1} ({len(batch)} chunks)...")
        vecs = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        all_embeddings.extend(vecs.tolist())

    return all_embeddings

# ── FAISS Indexing Logic ─────────────────────────────────────────────────────
def _build_index_from_chunks(
    chunks: list[str],
    embed_batch_size: int = 64,
    index_batch_size: int = 500,
) -> faiss.IndexFlatL2:
    index = None

    for start in range(0, len(chunks), index_batch_size):
        batch_chunks = chunks[start : start + index_batch_size]
        print(f"📦 Indexing chunks {start + 1}–{start + len(batch_chunks)} of {len(chunks)}...")

        embeddings = get_embeddings(batch_chunks, batch_size=embed_batch_size)
        embeddings_np = np.array(embeddings, dtype=np.float32)

        if index is None:
            dim = embeddings_np.shape[1]
            index = faiss.IndexFlatL2(dim)

        index.add(embeddings_np)

    return index

# ── Main PDF Processor ───────────────────────────────────────────────────────
def process_pdf(
    file_path: str,
    embed_batch_size: int = 64,
    index_batch_size: int = 500,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> tuple[list[str], faiss.IndexFlatL2]:
    print(f"📖 Loading PDF: {file_path}")
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]

    print(f"✂️  {len(texts)} chunks created. Starting indexing...")

    index = _build_index_from_chunks(
        texts,
        embed_batch_size=embed_batch_size,
        index_batch_size=index_batch_size,
    )

    print("✅ Indexing complete.")
    return texts, index


# ── Query Embedding (for search/retrieval) ───────────────────────────────────
def embed_query(query: str) -> np.ndarray:
    """Embed a single query string for FAISS similarity search."""
    model = get_embed_model()
    vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec.astype(np.float32)


if __name__ == "__main__":
    # Example usage:
    # texts, index = process_pdf("your_document.pdf")
    # faiss.write_index(index, "vectorstore/my_index.faiss")
    #
    # Search example:
    # query_vec = embed_query("What is this document about?")
    # distances, indices = index.search(query_vec, k=5)
    # results = [texts[i] for i in indices[0]]
    pass