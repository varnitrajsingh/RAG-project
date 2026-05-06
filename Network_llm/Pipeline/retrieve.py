import os
import sys
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv

# Fix import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Use LangChain's embedding wrapper — handles SDK versioning automatically
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from llm import generate_answer
from cache import get_cache, set_cache
from hybrid import keyword_search

load_dotenv()

# Initialize embeddings via LangChain wrapper
embedder = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    task_type="retrieval_query"
)


def get_query_embedding(query: str):
    """Returns embedding as a 2D numpy array for FAISS."""
    vector = embedder.embed_query(query)
    return np.array([vector], dtype="float32")


def query_pipeline(query: str) -> str:
    # 1. Check cache
    cached = get_cache(query)
    if cached:
        print("✅ Cache hit!")
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
    set_cache(query, answer)
    return answer


# ── TEST ──────────────────────────────────────────────────────────────────────
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