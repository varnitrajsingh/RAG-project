import os
import re
import time
import numpy as np
import faiss
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from google.api_core import exceptions

os.makedirs("vectorstore", exist_ok=True)
os.makedirs("data/uploads", exist_ok=True)

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
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found.")

    _client = genai.Client(api_key=api_key)
    return _client


def _safe_truncate(text: str, max_bytes: int = 9900) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _parse_retry_delay(error: Exception, default: float = 60.0) -> float:
    """Extract the suggested retry delay (in seconds) from a 429 error message."""
    match = re.search(r"retry[^\d]*(\d+(?:\.\d+)?)\s*s", str(error), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 2.0  # small buffer on top
    return default


def _embed_one(client, text: str) -> list[float]:
    """Single embed call — no tenacity here, we handle 429 manually below."""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return result.embeddings[0].values


def get_embeddings(
    texts: list[str],
    rpm_limit: int = 90,       # stay ~10% under the 100 RPM cap
    max_retries: int = 5,
) -> list[list[float]]:
    """
    Embeds texts one-by-one with:
    - Proactive throttling (stay under RPM cap)
    - Reactive backoff (respect the exact retry-after delay from 429 errors)
    """
    client = get_client()
    min_delay = 60.0 / rpm_limit  # ~0.67s between requests at 90 RPM
    all_embeddings = []

    for i, text in enumerate(texts):
        safe_text = _safe_truncate(text)
        attempt = 0

        while True:
            try:
                print(f"  [API] Embedding {i + 1}/{len(texts)}...")
                embedding = _embed_one(client, safe_text)
                all_embeddings.append(embedding)

                # Proactive throttle after every successful call
                if i < len(texts) - 1:
                    time.sleep(min_delay)
                break

            except exceptions.ResourceExhausted as e:
                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError(f"Exceeded {max_retries} retries on chunk {i + 1}.") from e

                wait = _parse_retry_delay(e)
                print(f"  [429] Rate limit hit on chunk {i + 1}. Waiting {wait:.1f}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait)

            except exceptions.ServiceUnavailable as e:
                attempt += 1
                if attempt > max_retries:
                    raise
                wait = min(5 * (2 ** attempt), 60)
                print(f"  [503] Service unavailable. Retrying in {wait}s...")
                time.sleep(wait)

    return all_embeddings


def _build_index_from_chunks(
    chunks: list[str],
    index_batch_size: int = 200,
    rpm_limit: int = 90,
) -> faiss.IndexFlatL2:
    index = None

    for start in range(0, len(chunks), index_batch_size):
        batch = chunks[start : start + index_batch_size]
        print(f"\n📦 Indexing chunks {start + 1}–{start + len(batch)} of {len(chunks)}...")

        embeddings = get_embeddings(batch, rpm_limit=rpm_limit)
        embeddings_np = np.array(embeddings, dtype=np.float32)

        if index is None:
            index = faiss.IndexFlatL2(embeddings_np.shape[1])

        index.add(embeddings_np)

    return index


def process_pdf(
    file_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    index_batch_size: int = 200,
    rpm_limit: int = 90,
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

    est_min = len(texts) / rpm_limit
    print(f"✂️  {len(texts)} chunks created. Estimated time: ~{est_min:.1f} min")

    index = _build_index_from_chunks(texts, index_batch_size=index_batch_size, rpm_limit=rpm_limit)

    print("✅ Indexing complete.")
    return texts, index