
import os
import sys
import pickle
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────
# Fix import path
# ─────────────────────────────────────────────────────────────
sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__))
)

from llm import generate_answer
from cache import get_cache, set_cache
from hybrid import keyword_search


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
VECTORSTORE_DIR = "vectorstore"

INDEX_PATH = os.path.join(
    VECTORSTORE_DIR,
    "faiss.index"
)

CHUNKS_PATH = os.path.join(
    VECTORSTORE_DIR,
    "chunks.pkl"
)


# ─────────────────────────────────────────────────────────────
# Lazy Model Loading
# ─────────────────────────────────────────────────────────────
_embedder = None


def get_embedder() -> SentenceTransformer:
    """
    Lazy-load embedding model.
    """

    global _embedder

    if _embedder is None:
        print("🔄 Loading embedding model...")

        _embedder = SentenceTransformer(
            "BAAI/bge-base-en-v1.5"
        )

        print("✅ Embedding model loaded.")

    return _embedder


# ─────────────────────────────────────────────────────────────
# Query Embedding
# ─────────────────────────────────────────────────────────────
def get_query_embedding(query: str) -> np.ndarray:
    """
    Generate normalized query embedding.
    """

    model = get_embedder()

    # IMPORTANT:
    # BGE models work better with instruction prefix
    formatted_query = (
        "Represent this sentence for searching relevant passages: "
        + query
    )

    embedding = model.encode(
        [formatted_query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embedding.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Load Vectorstore
# ─────────────────────────────────────────────────────────────
def load_vectorstore():
    """
    Load FAISS index + chunks metadata.
    """

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at: {INDEX_PATH}"
        )

    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"Chunks file not found at: {CHUNKS_PATH}"
        )

    print("📦 Loading vectorstore...")

    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    print(f"✅ Loaded {len(chunks)} chunks.")

    return index, chunks


# ─────────────────────────────────────────────────────────────
# Semantic Search
# ─────────────────────────────────────────────────────────────
def semantic_search(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: List[Dict],
    top_k: int = 10,
) -> List[Dict]:
    """
    Perform semantic vector search.
    """

    query_embedding = get_query_embedding(query)

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        if idx >= len(chunks):
            continue

        chunk = chunks[idx]

        results.append(
            {
                "score": float(score),
                "text": chunk["text"],
                "page": chunk.get("page"),
                "source": chunk.get("source"),
                "retrieval_type": "semantic",
            }
        )

    return results


# ─────────────────────────────────────────────────────────────
# Keyword Search
# ─────────────────────────────────────────────────────────────
def keyword_retrieval(
    query: str,
    chunks: List[Dict],
) -> List[Dict]:
    """
    Perform keyword/BM25 retrieval.
    """

    texts = [chunk["text"] for chunk in chunks]

    indices = keyword_search(query, texts)

    results = []

    for idx in indices:

        if idx >= len(chunks):
            continue

        chunk = chunks[idx]

        results.append(
            {
                "score": None,
                "text": chunk["text"],
                "page": chunk.get("page"),
                "source": chunk.get("source"),
                "retrieval_type": "keyword",
            }
        )

    return results


# ─────────────────────────────────────────────────────────────
# Merge Results
# ─────────────────────────────────────────────────────────────
def merge_results(
    semantic_results: List[Dict],
    keyword_results: List[Dict],
    max_chunks: int = 8,
) -> List[Dict]:
    """
    Merge semantic + keyword results.
    Deduplicate by text.
    """

    merged = []
    seen = set()

    for result in semantic_results + keyword_results:

        text = result["text"]

        if text in seen:
            continue

        seen.add(text)
        merged.append(result)

        if len(merged) >= max_chunks:
            break

    return merged


# ─────────────────────────────────────────────────────────────
# Context Builder
# ─────────────────────────────────────────────────────────────
def build_context(results: List[Dict]) -> str:
    """
    Build final LLM context.
    """

    context_parts = []

    for idx, result in enumerate(results, start=1):

        page = result.get("page")

        chunk_text = (
            f"[Chunk {idx} | Page {page}]\n"
            f"{result['text']}"
        )

        context_parts.append(chunk_text)

    return "\n\n".join(context_parts)


# ─────────────────────────────────────────────────────────────
# Main Query Pipeline
# ─────────────────────────────────────────────────────────────
def query_pipeline(
    query: str,
    pdf_name: str,
) -> str:
    """
    Full RAG query pipeline.
    """

    # ─────────────────────────────────────────
    # 1. Cache Check
    # ─────────────────────────────────────────
    cached_answer = get_cache(
        query=query,
        pdf_name=pdf_name,
    )

    if cached_answer:
        print("⚡ Cache hit.")
        return cached_answer

    # ─────────────────────────────────────────
    # 2. Load Vectorstore
    # ─────────────────────────────────────────
    index, chunks = load_vectorstore()

    # ─────────────────────────────────────────
    # 3. Semantic Retrieval
    # ─────────────────────────────────────────
    semantic_results = semantic_search(
        query=query,
        index=index,
        chunks=chunks,
        top_k=10,
    )

    # ─────────────────────────────────────────
    # 4. Keyword Retrieval
    # ─────────────────────────────────────────
    keyword_results = keyword_retrieval(
        query=query,
        chunks=chunks,
    )

    # ─────────────────────────────────────────
    # 5. Merge Results
    # ─────────────────────────────────────────
    final_results = merge_results(
        semantic_results=semantic_results,
        keyword_results=keyword_results,
        max_chunks=8,
    )

    # ─────────────────────────────────────────
    # DEBUG RETRIEVAL
    # ─────────────────────────────────────────
    print("\n🔍 Retrieval Results")
    print("=" * 80)

    for idx, result in enumerate(final_results, start=1):

        print(f"\nResult {idx}")
        print(f"Type  : {result['retrieval_type']}")
        print(f"Page  : {result.get('page')}")
        print(f"Score : {result.get('score')}")

        print("-" * 80)
        print(result["text"][:500])

    # ─────────────────────────────────────────
    # 6. Build Context
    # ─────────────────────────────────────────
    context = build_context(final_results)

    if not context.strip():
        return (
            "❌ No relevant information found "
            "in the document."
        )

    # ─────────────────────────────────────────
    # 7. Generate Final Answer
    # ─────────────────────────────────────────
    answer = generate_answer(
        query=query,
        context=context,
    )

    # ─────────────────────────────────────────
    # 8. Cache Answer
    # ─────────────────────────────────────────
    set_cache(
        query=query,
        pdf_name=pdf_name,
        answer=answer,
    )

    return answer


# ─────────────────────────────────────────────────────────────
# Local Testing
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("🔍 Testing RAG Pipeline")
    print("=" * 80)

    test_queries = [
        "What is BGP and what port does it use?",
        "Explain OSPF shortest path algorithm",
        "What are routing protocols?",
    ]

    for query in test_queries:

        print("\n" + "=" * 80)
        print(f"❓ Query: {query}")

        try:

            answer = query_pipeline(
                query=query,
                pdf_name="sample.pdf",
            )

            print("\n💬 Final Answer")
            print("-" * 80)
            print(answer)

        except Exception as e:

            print(f"❌ Error: {e}")

        print("=" * 80)