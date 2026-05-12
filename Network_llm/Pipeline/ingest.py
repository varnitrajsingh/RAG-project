"""
ingest.py

Changes:
- create_chunks() now tags each chunk with protocol keywords found in its text
- Tags stored as chunk["tags"] — used by hybrid.py for protocol-aware boosting
- Full print logging for tag detection
"""

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
# Protocol Keywords for Tagging
# ─────────────────────────────────────────────────────────────

PROTOCOL_TAGS = [
    "BGP", "OSPF", "MPLS", "VLAN", "DHCP", "NAT", "ACL",
    "STP", "AAA", "LDP", "ISIS", "RSVP", "BFD", "VPN",
    "TCP", "UDP", "IP", "ICMP", "IGMP", "DNS", "HTTP",
    "HTTPS", "SSH", "TELNET", "SNMP", "NTP", "RADIUS",
    "TACACS", "LACP", "LLDP", "QOS", "VXLAN", "EVPN",
]

# ─────────────────────────────────────────────────────────────
# Lazy Embedding Model
# ─────────────────────────────────────────────────────────────

_embed_model = None

def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print("🔄 Loading embedding model...")
        _embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        print("✅ Embedding model loaded.")
    return _embed_model


# ─────────────────────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\t", " ")
    text = "\n".join(
        line.strip()
        for line in text.splitlines()
        if line.strip()
    )
    text = " ".join(text.split())
    return text.strip()


# ─────────────────────────────────────────────────────────────
# Protocol Tag Detection
# ─────────────────────────────────────────────────────────────

def detect_tags(text: str) -> List[str]:
    """
    Scan chunk text for known protocol keywords.
    Returns list of matched protocol names (uppercase).
    Used by hybrid.py for protocol-aware BM25 boosting.
    """
    text_upper = text.upper()
    found = [tag for tag in PROTOCOL_TAGS if tag in text_upper]
    return found


# ─────────────────────────────────────────────────────────────
# Embedding Generation
# ─────────────────────────────────────────────────────────────

def generate_embeddings(
    texts: List[str],
    batch_size: int = 64,
) -> np.ndarray:
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
    print(f"📖 Loading PDF: {file_path}")
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} pages.")
    return docs


# ─────────────────────────────────────────────────────────────
# Chunking  (UPDATED: adds tags per chunk)
# ─────────────────────────────────────────────────────────────

def create_chunks(
    docs,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict]:
    """
    Create chunks with metadata.
    Each chunk now includes a 'tags' field listing detected protocol keywords.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    split_docs = splitter.split_documents(docs)
    chunks = []
    tag_counts: Dict[str, int] = {}

    print("\n" + "=" * 60)
    print("🏷️  TAGGING CHUNKS WITH PROTOCOL KEYWORDS")
    print("=" * 60)

    for idx, doc in enumerate(split_docs):
        cleaned_text = clean_text(doc.page_content)
        if not cleaned_text:
            continue

        # Detect protocol tags
        tags = detect_tags(cleaned_text)

        # Accumulate tag stats for summary log
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        chunks.append({
            "chunk_id": idx,
            "text":     cleaned_text,
            "page":     doc.metadata.get("page"),
            "source":   doc.metadata.get("source"),
            "tags":     tags,      # ← NEW
        })

    print(f"\n✂️  Created {len(chunks)} chunks.")
    print(f"\n📊 Protocol Tag Distribution across all chunks:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"   {tag:<10} : {count} chunks")

    if not tag_counts:
        print("   ⚠️  No protocol tags detected — check PROTOCOL_TAGS list.")

    return chunks


# ─────────────────────────────────────────────────────────────
# FAISS Index
# ─────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"✅ Indexed {index.ntotal} vectors.")
    return index


# ─────────────────────────────────────────────────────────────
# Save Vectorstore
# ─────────────────────────────────────────────────────────────

def save_vectorstore(index: faiss.IndexFlatIP, chunks: List[Dict]):
    index_path    = os.path.join(VECTORSTORE_DIR, "faiss.index")
    metadata_path = os.path.join(VECTORSTORE_DIR, "chunks.pkl")
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

    docs = load_pdf(file_path)

    chunks = create_chunks(
        docs=docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    texts = [chunk["text"] for chunk in chunks]

    print("🧠 Generating embeddings...")
    embeddings = generate_embeddings(texts=texts, batch_size=embedding_batch_size)

    print("📦 Building FAISS index...")
    index = build_faiss_index(embeddings)

    save_vectorstore(index, chunks)

    print("🎉 PDF ingestion completed.")
    return chunks, index


# ─────────────────────────────────────────────────────────────
# Query Embedding
# ─────────────────────────────────────────────────────────────

def embed_query(query: str) -> np.ndarray:
    model = get_embed_model()
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
    query_embedding = embed_query(query)
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append({
            "score":  float(score),
            "text":   chunk["text"],
            "page":   chunk["page"],
            "source": chunk["source"],
            "tags":   chunk.get("tags", []),    # ← NEW: pass tags forward
        })

    return results