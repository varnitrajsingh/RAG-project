import os
import pickle
from typing import List, Dict, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─────────────────────────────────────────────────────────────
# Directory Setup
# ─────────────────────────────────────────────────────────────
VECTORSTORE_DIR = "vectorstore"
UPLOAD_DIR = "data/uploads"

os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Lazy Embedding Model
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
# Text Cleaning
# ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Basic PDF cleanup.
    """

    text = text.replace("\x00", " ")
    text = text.replace("\t", " ")

    # remove excessive newlines
    text = "\n".join(
        line.strip()
        for line in text.splitlines()
        if line.strip()
    )

    # remove excessive spaces
    text = " ".join(text.split())

    return text.strip()


# ─────────────────────────────────────────────────────────────
# Embedding Generation
# ─────────────────────────────────────────────────────────────
def generate_embeddings(
    texts: List[str],
    batch_size: int = 64,
) -> np.ndarray:
    """
    Generate normalized embeddings.
    """

    model = get_embed_model()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    return embeddings.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# PDF Loading
# ─────────────────────────────────────────────────────────────
def load_pdf(file_path: str):
    """
    Load PDF documents.
    """

    print(f"📖 Loading PDF: {file_path}")

    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    print(f"✅ Loaded {len(docs)} pages.")

    return docs


# ─────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────
def create_chunks(
    docs,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict]:
    """
    Create semantic chunks with metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    split_docs = splitter.split_documents(docs)

    chunks = []

    for idx, doc in enumerate(split_docs):

        cleaned_text = clean_text(doc.page_content)

        if not cleaned_text:
            continue

        chunks.append(
            {
                "chunk_id": idx,
                "text": cleaned_text,
                "page": doc.metadata.get("page"),
                "source": doc.metadata.get("source"),
            }
        )

    print(f"✂️ Created {len(chunks)} chunks.")

    return chunks


# ─────────────────────────────────────────────────────────────
# FAISS Index Creation
# ─────────────────────────────────────────────────────────────
def build_faiss_index(
    embeddings: np.ndarray,
) -> faiss.IndexFlatIP:
    """
    Build cosine similarity FAISS index.
    """

    dimension = embeddings.shape[1]

    # IMPORTANT:
    # Using IP because embeddings are normalized.
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(f"✅ Indexed {index.ntotal} vectors.")

    return index


# ─────────────────────────────────────────────────────────────
# Save Vectorstore
# ─────────────────────────────────────────────────────────────
def save_vectorstore(
    index: faiss.IndexFlatIP,
    chunks: List[Dict],
):
    """
    Save FAISS index + metadata.
    """

    index_path = os.path.join(
        VECTORSTORE_DIR,
        "faiss.index"
    )

    metadata_path = os.path.join(
        VECTORSTORE_DIR,
        "chunks.pkl"
    )

    faiss.write_index(index, index_path)

    with open(metadata_path, "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Vectorstore saved.")


# ─────────────────────────────────────────────────────────────
# Main Ingestion Pipeline
# ─────────────────────────────────────────────────────────────
def process_pdf(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_batch_size: int = 64,
) -> Tuple[List[Dict], faiss.IndexFlatIP]:
    """
    Complete ingestion pipeline.
    """

    # 1. Load PDF
    docs = load_pdf(file_path)

    # 2. Chunking
    chunks = create_chunks(
        docs=docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    texts = [chunk["text"] for chunk in chunks]

    # 3. Generate embeddings
    print("🧠 Generating embeddings...")

    embeddings = generate_embeddings(
        texts=texts,
        batch_size=embedding_batch_size,
    )

    # 4. Build FAISS index
    print("📦 Building FAISS index...")

    index = build_faiss_index(embeddings)

    # 5. Save vectorstore
    save_vectorstore(index, chunks)

    print("🎉 PDF ingestion completed.")

    return chunks, index


# ─────────────────────────────────────────────────────────────
# Query Embedding
# ─────────────────────────────────────────────────────────────
def embed_query(query: str) -> np.ndarray:
    """
    Embed query for retrieval.
    """

    model = get_embed_model()

    # IMPORTANT FOR BGE MODELS
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
# Retrieval
# ─────────────────────────────────────────────────────────────
def search(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: List[Dict],
    top_k: int = 10,
):
    """
    Search relevant chunks.
    """

    query_embedding = embed_query(query)

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        chunk = chunks[idx]

        results.append(
            {
                "score": float(score),
                "text": chunk["text"],
                "page": chunk["page"],
                "source": chunk["source"],
            }
        )

    return results


# ─────────────────────────────────────────────────────────────
# Example Usage
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    chunks, index = process_pdf(
        file_path="sample.pdf",
        chunk_size=1000,
        chunk_overlap=200,
    )

    results = search(
        query="What is this document about?",
        index=index,
        chunks=chunks,
        top_k=5,
    )

    for i, result in enumerate(results, 1):

        print("\n" + "=" * 80)
        print(f"Result {i}")
        print(f"Score: {result['score']:.4f}")
        print(f"Page: {result['page']}")
        print("-" * 80)
        print(result["text"][:1000])