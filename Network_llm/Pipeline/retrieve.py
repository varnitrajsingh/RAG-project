import os
import sys
import faiss
import pickle
import numpy as np

# Fix import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from llm import generate_answer
from cache import get_cache, set_cache
from hybrid import keyword_search


# ── Lazy embedder init ────────────────────────────────────────────────────────
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder

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

    _embedder = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
        task_type="retrieval_query",
    )
    return _embedder


# ── Embedding helper ──────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        print("🔄 Loading embedding model...")
        _embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        print("✅ Embedding model loaded.")
    return _embedder

def get_query_embedding(query: str):
    model = get_embedder()
    vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec.astype(np.float32)

# ── Main pipeline ─────────────────────────────────────────────────────────────
def query_pipeline(query: str, pdf_name: str) -> str:
    # 1. Check cache
    cached = get_cache(query, pdf_name)
    if cached:
        print("Cache hit!")
        return cached

    # 2. Load FAISS index and chunks
    index_path = os.path.join("vectorstore", "faiss_index")
    chunks_path = os.path.join("vectorstore", "chunks.pkl")

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return "❌ Vectorstore not found. Please run the ingestion pipeline first."

    index = faiss.read_index(index_path)

    with open(chunks_path, "rb") as f:
        texts = pickle.load(f)

    # 3. Semantic search
    query_embedding = get_query_embedding(query)
    D, I = index.search(query_embedding, k=5)
    semantic_chunks = [texts[i] for i in I[0] if i < len(texts)]

    # 4. Keyword search
    keyword_indices = keyword_search(query, texts)
    keyword_chunks = [texts[i] for i in keyword_indices if i < len(texts)]

    # 5. Deduplicate while preserving order
    seen = set()
    final_chunks = []
    for chunk in semantic_chunks + keyword_chunks:
        if chunk not in seen:
            seen.add(chunk)
            final_chunks.append(chunk)

    context = "\n\n".join(final_chunks)

    # 6. Generate answer
    answer = generate_answer(query, context)

    # 7. Cache result
    set_cache(query, pdf_name, answer)
    return answer


# ── Local test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔍 Testing Query Pipeline")
    print("=" * 50)

    test_queries = [
        "What is BGP and what port does it use?",
        "Explain OSPF shortest path algorithm",
    ]

    for query in test_queries:
        print(f"\n❓ Query: {query}")
        try:
            answer = query_pipeline(query)
            print(f"💬 Answer: {answer.strip()}")
        except Exception as e:
            print(f"❌ Error: {e}")
        print("-" * 50)
