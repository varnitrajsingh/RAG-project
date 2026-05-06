import os

# Ensure these directories exist when app starts fresh on cloud
os.makedirs("vectorstore", exist_ok=True)
os.makedirs("data/uploads", exist_ok=True)


from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
import numpy as np
import faiss, pickle, os, time
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_embeddings(texts, batch_size=100):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch,
        )
        all_embeddings.extend([e.values for e in result.embeddings])
        if i + batch_size < len(texts):
            time.sleep(0.6)
    return all_embeddings


def process_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]

    print(f"📄 Loaded {len(docs)} pages → {len(texts)} chunks")

    embeddings = get_embeddings(texts)
    embeddings_np = np.array(embeddings, dtype=np.float32)

    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)

    os.makedirs("vectorstore", exist_ok=True)
    faiss.write_index(index, "vectorstore/faiss_index")
    with open("vectorstore/chunks.pkl", "wb") as f:
        pickle.dump(texts, f)

    print(f"✅ Indexed {len(texts)} chunks into FAISS")
    return texts, index


def query_index(question, texts, index, top_k=3):
    q_embedding = get_embeddings([question])
    q_np = np.array(q_embedding, dtype=np.float32)

    distances, indices = index.search(q_np, top_k)

    print(f"\n🔍 Query: {question}")
    print(f"{'─'*50}")
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
        print(f"\n[{rank}] Score: {dist:.4f}")
        print(f"{texts[idx][:300]}...")
    return [texts[i] for i in indices[0]]



# ──────────────────────────────────────────────
# SAMPLE TEST — creates a PDF on-the-fly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import fitz  # PyMuPDF

    # 1. Create a sample PDF
    sample_pdf_path = "sample_test.pdf"
    doc = fitz.open()

    pages_content = [
        ("Introduction to Neural Networks",
         "A neural network is a computational model inspired by the human brain. "
         "It consists of layers of interconnected nodes called neurons. "
         "Each connection has a weight that adjusts during training. "
         "Deep learning uses many such layers to learn complex patterns from data."),

        ("Transformer Architecture",
         "The Transformer model was introduced in the paper 'Attention is All You Need'. "
         "It relies entirely on self-attention mechanisms to draw global dependencies. "
         "BERT and GPT are popular models built on the Transformer architecture. "
         "Transformers have revolutionized natural language processing tasks."),

        ("Retrieval Augmented Generation",
         "RAG combines retrieval systems with generative models to improve factual accuracy. "
         "A document is split into chunks and stored in a vector database. "
         "At query time, semantically similar chunks are retrieved and passed to an LLM. "
         "FAISS is a popular library for efficient similarity search in RAG pipelines."),
    ]

    for title, content in pages_content:
        page = doc.new_page()
        page.insert_text((50, 50), title, fontsize=16, fontname="helv")
        page.insert_text((50, 100), content, fontsize=11, fontname="helv")

    doc.save(sample_pdf_path)
    doc.close()
    print(f"📝 Created sample PDF: {sample_pdf_path}")

    # 2. Process the PDF
    texts, index = process_pdf(sample_pdf_path)

    # 3. Run test queries
    test_queries = [
        "What is a neural network?",
        "How does RAG work with FAISS?",
        "What is the Transformer architecture?",
    ]

    for query in test_queries:
        query_index(query, texts, index, top_k=2)