import os
import time
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from google.api_core import exceptions

# ── Directory Setup ──────────────────────────────────────────────────────────
os.makedirs("vectorstore", exist_ok=True)
os.makedirs("data/uploads", exist_ok=True)

# ── Lazy Client Initialization ───────────────────────────────────────────────
_client = None

def get_client():
    global _client
    if _client is not None:
        return _client

    api_key = None
    # 1. Try Streamlit Secrets
    try:
        import streamlit as st
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

    # 2. Try .env file
    if not api_key:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. "
            "Set it in Streamlit Secrets (cloud) or a .env file (local)."
        )

    _client = genai.Client(api_key=api_key)
    return _client

# ── Robust Embedding Logic (Free Tier Friendly) ──────────────────────────────
@retry(
    retry=retry_if_exception_type((exceptions.ResourceExhausted, exceptions.ServiceUnavailable)),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(7)
)
def _embed_with_retry(client, model, contents):
    """Internal helper to call Gemini with automatic retries for 429 errors."""
    return client.models.embed_content(
        model=model,
        contents=contents,
    )

def get_embeddings(texts: list[str], batch_size: int = 90) -> list[list[float]]:
    """
    Embed texts in batches. 
    Batch size 90 is used to stay under the 100-item-per-call limit.
    """
    client = get_client()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"  [API] Embedding batch {i // batch_size + 1} / {-(-len(texts) // batch_size)}...")

        # text-embedding-004 is the latest, most stable model for free tier
        result = _embed_with_retry(client, "text-embedding-001", batch)
        
        all_embeddings.extend([e.values for e in result.embeddings])

        # 1.5s sleep ensures we stay well below the 100 Requests Per Minute (RPM) limit
        if i + batch_size < len(texts):
            time.sleep(1.5)

    return all_embeddings

# ── FAISS Indexing Logic ─────────────────────────────────────────────────────
def _build_index_from_chunks(
    chunks: list[str],
    index: faiss.IndexFlatL2 | None,
    embed_batch_size: int = 90,
    index_batch_size: int = 450,
) -> faiss.IndexFlatL2:
    """
    Groups chunks, fetches embeddings, and adds them to the FAISS index.
    """
    for start in range(0, len(chunks), index_batch_size):
        batch_chunks = chunks[start : start + index_batch_size]
        print(f"📦 Indexing group: {start + 1} to {start + len(batch_chunks)} of {len(chunks)}...")

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
    embed_batch_size: int = 90,
    index_batch_size: int = 450,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> tuple[list[str], faiss.IndexFlatL2]:
    """
    Full pipeline: Load PDF -> Split -> Embed -> Index.
    """
    print(f"📖 Loading PDF: {file_path}")
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]

    print(f"✂️ Created {len(texts)} chunks. Starting the embedding process...")
    
    index = _build_index_from_chunks(
        texts, 
        None, 
        embed_batch_size=embed_batch_size,
        index_batch_size=index_batch_size
    )

    print("✅ Indexing complete.")
    return texts, index

# ── Execution Block ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example usage:
    # pdf_path = "data/uploads/your_large_file.pdf"
    # texts, index = process_pdf(pdf_path)
    # faiss.write_index(index, "vectorstore/index.faiss")
    pass