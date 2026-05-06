import re
from typing import List


def keyword_search(query: str, texts: List[str], top_k: int = 5) -> List[int]:
    """
    Simple keyword-based search over a list of text chunks.
    Returns indices of chunks that contain any query keyword.
    """
    # Tokenize query into individual keywords (lowercase, strip punctuation)
    keywords = [
        re.sub(r'[^\w]', '', word).lower()
        for word in query.split()
        if len(word) > 2  # skip very short words like "is", "a", "of"
    ]

    scores = []
    for idx, chunk in enumerate(texts):
        chunk_lower = chunk.lower()
        # Score = number of keywords found in the chunk
        score = sum(1 for kw in keywords if kw in chunk_lower)
        if score > 0:
            scores.append((idx, score))

    # Sort by score descending, return top_k indices
    scores.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scores[:top_k]]


# ── TEST ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_texts = [
        "BGP uses TCP port 179 for establishing connections between peers.",
        "OSPF uses Dijkstra's shortest path first algorithm.",
        "DNS resolves domain names to IP addresses.",
        "MPLS is used for fast packet forwarding in networks.",
        "BGP is the protocol used between autonomous systems on the internet.",
    ]

    query = "What port does BGP use?"
    results = keyword_search(query, sample_texts)

    print("🧪 Testing keyword_search...")
    print(f"Query: {query}")
    print(f"Matched chunk indices: {results}")
    for i in results:
        print(f"  [{i}] {sample_texts[i]}")
    print("✅ hybrid.py working!")