"""
hybrid.py

Changes:
- Added DOMAIN_STOPWORDS to filter generic config-guide words
- Added PROTOCOL_SYNONYMS for query expansion
- Added SEMANTIC_THRESHOLD (0.65) to drop weak chunks
- Weighted merge of semantic + BM25 scores
- Full print logging throughout
"""

import re
from typing import List, Dict, Tuple

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

SEMANTIC_THRESHOLD = 0.65

DOMAIN_STOPWORDS = {
    "configure", "configuration", "click", "step", "page", "shown",
    "figure", "table", "select", "displayed", "window", "list",
    "create", "delete", "modify", "setting", "settings", "status",
    "enable", "disable", "button", "menu", "tab", "parameter",
    "parameters", "description", "procedure", "note", "end",
    "interface", "value", "values", "specify", "specifies",
    "required", "default", "following", "complete", "open",
    "navigation", "tree", "apply", "displays", "pane", "dialog",
    "box", "item", "items", "area", "section", "information",
    "method", "options", "option", "check", "enter", "set",
    "how", "the", "and", "for", "from", "with", "this", "that",
    "are", "can", "not", "all", "use", "used", "each", "has",
}

PROTOCOL_SYNONYMS = {
    "bgp":  ["bgp", "border gateway protocol"],
    "ospf": ["ospf", "open shortest path first"],
    "mpls": ["mpls", "multiprotocol label switching"],
    "vlan": ["vlan", "virtual lan", "virtual local area network"],
    "dhcp": ["dhcp", "dynamic host configuration protocol"],
    "nat":  ["nat", "network address translation"],
    "acl":  ["acl", "access control list"],
    "stp":  ["stp", "spanning tree protocol"],
    "as":   ["as", "autonomous system"],
    "aaa":  ["aaa", "authentication authorization accounting"],
    "ldp":  ["ldp", "label distribution protocol"],
    "isis": ["isis", "intermediate system to intermediate system"],
    "bfd":  ["bfd", "bidirectional forwarding detection"],
    "vpn":  ["vpn", "virtual private network"],
}


# ─────────────────────────────────────────────────────────────
# QUERY EXPANSION
# ─────────────────────────────────────────────────────────────

def expand_query(query: str) -> List[str]:
    """
    Expand abbreviations in query to full protocol names.
    Returns all keyword variants (original + expanded).
    """
    print("\n" + "=" * 60)
    print("🔀 QUERY EXPANSION")
    print("=" * 60)
    print(f"📝 Original Query : {query}")

    tokens = re.sub(r'[^\w\s]', '', query.lower()).split()
    expanded_terms = set(tokens)

    for token in tokens:
        if token in PROTOCOL_SYNONYMS:
            for synonym in PROTOCOL_SYNONYMS[token]:
                expanded_terms.update(synonym.split())
            print(f"   ↳ Expanded '{token}' → {PROTOCOL_SYNONYMS[token]}")

    result = list(expanded_terms)
    print(f"📦 Expanded Terms  : {result}")
    return result


# ─────────────────────────────────────────────────────────────
# KEYWORD SEARCH
# ─────────────────────────────────────────────────────────────

def keyword_search(
    query: str,
    texts: List[str],
    top_k: int = 5,
) -> List[int]:
    """
    Domain-aware BM25-style keyword search.
    Filters DOMAIN_STOPWORDS and expands protocol abbreviations.
    """
    print("\n" + "=" * 60)
    print("🔍 KEYWORD SEARCH (BM25-style)")
    print("=" * 60)

    expanded_terms = expand_query(query)

    raw_keywords = [
        re.sub(r'[^\w]', '', term).lower()
        for term in expanded_terms
        if len(term) > 2
    ]
    keywords = [kw for kw in raw_keywords if kw not in DOMAIN_STOPWORDS]

    print(f"\n📝 Raw tokens       : {raw_keywords}")
    print(f"✂️  After stopwords  : {keywords}")

    if not keywords:
        print("⚠️  All keywords were stopwords — skipping BM25 search.")
        return []

    scores = []
    for idx, chunk in enumerate(texts):
        chunk_lower = chunk.lower()
        score = sum(1 for kw in keywords if kw in chunk_lower)
        if score > 0:
            scores.append((idx, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in scores[:top_k]]

    print(f"\n📊 Top BM25 Results :")
    for rank, (idx, sc) in enumerate(scores[:top_k], 1):
        print(f"   Rank #{rank} | Index {idx:>4} | Score {sc} | Preview: {texts[idx][:80]}...")

    print(f"\n✅ BM25 returned {len(top_indices)} results.")
    return top_indices


# ─────────────────────────────────────────────────────────────
# SCORE THRESHOLD FILTER
# ─────────────────────────────────────────────────────────────

def filter_by_threshold(
    merged: List[Dict],
    threshold: float = SEMANTIC_THRESHOLD,
) -> Tuple[List[Dict], bool]:
    """
    Drop chunks whose semantic score is below threshold.
    Falls back to top-1 with low_confidence=True if nothing passes.
    """
    print("\n" + "=" * 60)
    print(f"🚦 THRESHOLD FILTER  (min score = {threshold})")
    print("=" * 60)

    above = [c for c in merged if c.get("score", 0) >= threshold]
    below = [c for c in merged if c.get("score", 0) < threshold]

    print(f"   ✅ Above threshold : {len(above)} chunks")
    print(f"   ❌ Below threshold : {len(below)} chunks (dropped)")

    for c in below:
        print(
            f"      Dropped → score={c.get('score', 0):.4f} "
            f"| page={c.get('page')} "
            f"| {c.get('text', '')[:60]}..."
        )

    if above:
        return above, False

    # Fallback — nothing passed threshold
    print("\n⚠️  FALLBACK: No chunks above threshold. Returning top-1 with low_confidence=True.")
    if merged:
        fallback = [dict(merged[0])]
        fallback[0]["low_confidence"] = True
        return fallback, True

    print("❌ Merged list is empty — returning empty.")
    return [], True


# ─────────────────────────────────────────────────────────────
# WEIGHTED MERGE
# ─────────────────────────────────────────────────────────────

def merge_results(
    semantic_results: List[Dict],
    bm25_indices: List[int],
    all_chunks: List[Dict],
    semantic_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> List[Dict]:
    """
    Merge semantic + BM25 results using weighted final_score.
    Chunks in both lists get a boosted score.
    """
    print("\n" + "=" * 60)
    print("🧩 MERGING SEMANTIC + BM25 RESULTS")
    print("=" * 60)
    print(f"   Semantic weight : {semantic_weight}")
    print(f"   BM25 weight     : {bm25_weight}")

    merged_map: Dict[int, Dict] = {}

    # Add semantic hits
    for chunk in semantic_results:
        chunk_id = chunk.get("chunk_id", id(chunk))
        merged_map[chunk_id] = {
            **chunk,
            "final_score": round(chunk["score"] * semantic_weight, 6),
            "source_tags": ["semantic"],
        }

    # Add / boost BM25 hits
    for rank, idx in enumerate(bm25_indices, 1):
        if idx >= len(all_chunks):
            continue
        bm25_chunk = all_chunks[idx]
        bm25_score_norm = (1.0 / rank) * bm25_weight
        chunk_id = bm25_chunk.get("chunk_id", idx)

        if chunk_id in merged_map:
            merged_map[chunk_id]["final_score"] += bm25_score_norm
            merged_map[chunk_id]["source_tags"].append("bm25_boost")
            print(
                f"   🔁 Boosted  chunk_id={chunk_id} "
                f"| new final_score={merged_map[chunk_id]['final_score']:.4f}"
            )
        else:
            merged_map[chunk_id] = {
                **bm25_chunk,
                "score": bm25_chunk.get("score", 0.0),
                "final_score": round(bm25_score_norm, 6),
                "source_tags": ["bm25_only"],
            }
            print(
                f"   ➕ BM25-only chunk_id={chunk_id} "
                f"| final_score={bm25_score_norm:.4f}"
            )

    merged = sorted(merged_map.values(), key=lambda x: x["final_score"], reverse=True)

    print(f"\n✅ Total merged chunks : {len(merged)}")
    for rank, c in enumerate(merged[:5], 1):
        print(
            f"   Rank #{rank} | final_score={c['final_score']:.4f} "
            f"| semantic={c.get('score', 0):.4f} "
            f"| tags={c.get('source_tags')} "
            f"| page={c.get('page')}"
        )

    return merged


# ─────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    sample_texts = [
        "BGP uses TCP port 179 for establishing connections between peers.",
        "OSPF uses Dijkstra's shortest path first algorithm.",
        "DNS resolves domain names to IP addresses.",
        "MPLS is used for fast packet forwarding in networks.",
        "BGP is the protocol used between autonomous systems on the internet.",
        "Configure DHCP relay agent. Click Create under DHCP Relay List.",
        "DTLS session using the initial certificate — enable or disable.",
    ]

    print("\n" + "=" * 80)
    print("🧪 TESTING hybrid.py")
    print("=" * 80)

    query = "What port does BGP use?"
    print(f"\n❓ Query: {query}")

    indices = keyword_search(query, sample_texts, top_k=3)

    print(f"\n✅ Matched indices : {indices}")
    for i in indices:
        print(f"   [{i}] {sample_texts[i]}")

    print("\n✅ hybrid.py working!")