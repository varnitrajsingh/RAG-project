from llm import generate_answer
import os
import time
import pickle
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# ─────────────────────────────────────────────────────────────
# CONFIG
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

TOP_K_SEMANTIC = 10
TOP_K_KEYWORD = 10
FINAL_TOP_K = 8


# ─────────────────────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────
_embed_model = None


def get_embed_model() -> SentenceTransformer:

    global _embed_model

    if _embed_model is None:

        print("🔄 Loading embedding model...")

        _embed_model = SentenceTransformer(
            "BAAI/bge-base-en-v1.5"
        )

        print("✅ Embedding model loaded.")

    return _embed_model


# ─────────────────────────────────────────────────────────────
# QUERY EMBEDDING
# ─────────────────────────────────────────────────────────────
def embed_query(query: str) -> np.ndarray:
    """
    Generate normalized query embedding.
    """

    print("\n🧠 Embedding Query")
    print("=" * 80)

    print(f"\n📝 Original Query:\n{query}")

    model = get_embed_model()

    formatted_query = (
        "Represent this sentence for searching relevant passages: "
        + query
    )

    print("\n📝 Formatted Query")
    print("-" * 80)
    print(formatted_query)

    embedding = model.encode(
        [formatted_query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embedding = embedding.astype(np.float32)

    print("\n📊 Embedding Stats")
    print("-" * 80)

    print("Shape :", embedding.shape)
    print("Min   :", np.min(embedding))
    print("Max   :", np.max(embedding))
    print("Mean  :", np.mean(embedding))

    return embedding


# ─────────────────────────────────────────────────────────────
# LOAD VECTORSTORE
# ─────────────────────────────────────────────────────────────
def load_vectorstore():
    """
    Load FAISS index + chunks.
    """

    print("\n📦 Loading Vectorstore")
    print("=" * 80)

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"❌ FAISS index not found: {INDEX_PATH}"
        )

    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"❌ Chunks file not found: {CHUNKS_PATH}"
        )

    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    print(f"✅ Loaded {len(chunks)} chunks.")
    print(f"✅ FAISS vectors: {index.ntotal}")

    return index, chunks


# ─────────────────────────────────────────────────────────────
# EXACT MATCH DEBUG
# ─────────────────────────────────────────────────────────────
def exact_match_debug(
    query: str,
    chunks: List[Dict],
):
    """
    Check whether exact query text exists.
    """

    print("\n🔎 EXACT MATCH DEBUG")
    print("=" * 80)

    lower_query = query.lower()

    found = False

    for idx, chunk in enumerate(chunks):

        text = chunk["text"].lower()

        if lower_query in text:

            found = True

            print(f"\n✅ Exact match found in chunk #{idx}")

            print("-" * 80)
            print(chunk["text"][:1000])

            break

    if not found:
        print("❌ No exact query match found.")


# ─────────────────────────────────────────────────────────────
# SEMANTIC SEARCH
# ─────────────────────────────────────────────────────────────
def semantic_search(
    query: str,
    index,
    chunks,
    top_k: int = TOP_K_SEMANTIC,
):
    """
    FAISS semantic retrieval.
    """

    print("\n" + "=" * 80)
    print("🔍 SEMANTIC SEARCH")
    print("=" * 80)

    query_embedding = embed_query(query)

    start = time.time()

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    elapsed = time.time() - start

    print(f"\n⏱ Retrieval Time: {elapsed:.4f}s")

    print("\n📊 Raw Scores")
    print("-" * 80)
    print(scores[0])

    print("\n📊 Raw Indices")
    print("-" * 80)
    print(indices[0])

    results = []

    print("\n📦 Retrieved Chunks")
    print("=" * 80)

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1,
    ):

        print(f"\n🔹 Rank #{rank}")

        print(f"Index : {idx}")
        print(f"Score : {score:.4f}")

        if score >= 0.80:
            relevance = "VERY STRONG"

        elif score >= 0.70:
            relevance = "STRONG"

        elif score >= 0.60:
            relevance = "MEDIUM"

        else:
            relevance = "WEAK"

        print(f"Relevance : {relevance}")

        if idx == -1:
            print("❌ Invalid index")
            continue

        if idx >= len(chunks):
            print("❌ Index out of bounds")
            continue

        chunk = chunks[idx]

        print(f"Page : {chunk.get('page')}")

        print("\n📄 Chunk Preview")
        print("-" * 80)

        print(chunk["text"][:1000])

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
# BM25 KEYWORD SEARCH
# ─────────────────────────────────────────────────────────────
def keyword_search(
    query: str,
    chunks: List[Dict],
    top_k: int = TOP_K_KEYWORD,
):
    """
    BM25 keyword retrieval.
    """

    print("\n" + "=" * 80)
    print("🔍 KEYWORD SEARCH")
    print("=" * 80)

    texts = [chunk["text"] for chunk in chunks]

    tokenized_corpus = [
        text.lower().split()
        for text in texts
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()

    print("\n📝 Tokenized Query")
    print("-" * 80)

    print(tokenized_query)

    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    print("\n📊 BM25 Top Scores")
    print("-" * 80)

    print(scores[top_indices])

    results = []

    print("\n📦 BM25 Retrieved Chunks")
    print("=" * 80)

    for rank, idx in enumerate(top_indices, start=1):

        chunk = chunks[idx]

        print(f"\n🔹 Rank #{rank}")
        print(f"Index : {idx}")
        print(f"Score : {scores[idx]:.4f}")
        print(f"Page  : {chunk.get('page')}")

        print("\n📄 Chunk Preview")
        print("-" * 80)

        print(chunk["text"][:1000])

        results.append(
            {
                "score": float(scores[idx]),
                "text": chunk["text"],
                "page": chunk.get("page"),
                "source": chunk.get("source"),
                "retrieval_type": "keyword",
            }
        )

    return results


# ─────────────────────────────────────────────────────────────
# MERGE RESULTS
# ─────────────────────────────────────────────────────────────
def merge_results(
    semantic_results,
    keyword_results,
):
    """
    Merge + deduplicate results.
    """

    print("\n" + "=" * 80)
    print("🧩 MERGING RESULTS")
    print("=" * 80)

    merged = []
    seen = set()

    for result in semantic_results + keyword_results:

        text = result["text"]

        if text in seen:
            continue

        seen.add(text)

        merged.append(result)

        print(
            f"✅ Added "
            f"({result['retrieval_type']}) "
            f"score={result['score']}"
        )

        if len(merged) >= FINAL_TOP_K:
            break

    print(f"\n✅ Final merged results: {len(merged)}")

    return merged


# ─────────────────────────────────────────────────────────────
# BUILD CONTEXT
# ─────────────────────────────────────────────────────────────
def build_context(results):
    """
    Build final context for LLM.
    """

    print("\n" + "=" * 80)
    print("🧱 BUILDING CONTEXT")
    print("=" * 80)

    context_parts = []

    for idx, result in enumerate(results, start=1):

        chunk_text = (
            f"[Chunk {idx} | "
            f"Page {result.get('page')}]\n\n"
            f"{result['text']}"
        )

        context_parts.append(chunk_text)

    context = "\n\n".join(context_parts)

    print(f"\n📏 Context Length: {len(context)} chars")

    print("\n📄 Context Preview")
    print("-" * 80)

    print(context[:4000])

    return context


# ─────────────────────────────────────────────────────────────
# MAIN RETRIEVER
# ─────────────────────────────────────────────────────────────
def retrieve(query: str):
    print("\n" + "=" * 80)
    print("🚀 STARTING RETRIEVAL PIPELINE")
    print("=" * 80)

    index, chunks = load_vectorstore()
    exact_match_debug(query, chunks)
    semantic_results = semantic_search(query=query, index=index, chunks=chunks)
    keyword_results = keyword_search(query=query, chunks=chunks)
    final_results = merge_results(semantic_results, keyword_results)

    if not final_results:
        return "Not in document."

    context = build_context(final_results)
    return generate_answer(query, context)

def query_pipeline(query, pdfname=None):
    return retrieve(query)

# ─────────────────────────────────────────────────────────────
# TESTING
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    query = "how to change AS number in transport"
    answer = retrieve(query)
    print(answer)