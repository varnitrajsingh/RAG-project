"""
full_pipeline_test.py

Purpose:
- Test COMPLETE RAG pipeline
- PDF ingestion
- Chunk creation
- Embedding generation
- FAISS indexing
- Retrieval debugging
- Context generation

Run:
python full_pipeline_test.py
"""

import os
import time

from Pipeline.ingest import process_pdf
from Pipeline.retrieve import retrieve


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
PDF_PATH = r"C:\Users\varni\excitel\Final Project\data\uploads\01-05 Configuration.pdf"

TEST_QUERIES = [

    # Exact retrieval
    "What port does BGP use?",

    # Semantic retrieval
    "How to change AS number in transport",

    # Config style
    "neighbor remote-as",

    # Networking
    "Explain OSPF",

    # Broad semantic
    "How to configure peer session",

    # Impossible query
    "How to cook pasta",
]


# ─────────────────────────────────────────────────────────────
# INGESTION TEST
# ─────────────────────────────────────────────────────────────
def test_ingestion():

    print("\n" + "=" * 100)
    print("📥 TESTING INGESTION PIPELINE")
    print("=" * 100)

    start = time.time()

    chunks, index = process_pdf(
        file_path=PDF_PATH,
        chunk_size=1000,
        chunk_overlap=200,
        embedding_batch_size=64,
    )

    elapsed = time.time() - start

    print("\n" + "=" * 100)
    print("✅ INGESTION COMPLETED")
    print("=" * 100)

    print(f"\n📄 Total Chunks: {len(chunks)}")
    print(f"📦 Total Vectors: {index.ntotal}")
    print(f"⏱ Total Time: {elapsed:.2f}s")

    print("\n📄 SAMPLE CHUNKS")
    print("=" * 100)

    for idx, chunk in enumerate(chunks[:3], start=1):

        print(f"\n🔹 Chunk #{idx}")

        print("-" * 80)

        print(chunk["text"][:1000])

    return chunks, index


# ─────────────────────────────────────────────────────────────
# RETRIEVAL TEST
# ─────────────────────────────────────────────────────────────
def test_retrieval():

    print("\n" + "=" * 100)
    print("🔍 TESTING RETRIEVAL PIPELINE")
    print("=" * 100)

    for idx, query in enumerate(TEST_QUERIES, start=1):

        print("\n\n" + "#" * 100)
        print(f"🧪 QUERY TEST #{idx}")
        print("#" * 100)

        print(f"\n❓ Query:\n{query}")

        try:

            start = time.time()

            context = retrieve(query)

            elapsed = time.time() - start

            print("\n" + "=" * 100)
            print("✅ FINAL CONTEXT")
            print("=" * 100)

            print(context[:5000])

            print(f"\n⏱ Retrieval Time: {elapsed:.2f}s")

        except Exception as e:

            print("\n❌ RETRIEVAL FAILED")
            print("=" * 100)

            print(str(e))


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "=" * 100)
    print("🚀 FULL RAG PIPELINE TEST")
    print("=" * 100)

    # 1. Ingestion
    test_ingestion()

    # 2. Retrieval
    test_retrieval()

    print("\n" + "=" * 100)
    print("🎉 FULL PIPELINE TEST COMPLETE")
    print("=" * 100)