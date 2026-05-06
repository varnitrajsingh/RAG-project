import os
import json
import hashlib

CACHE_FILE = os.path.join(os.path.dirname(__file__), "query_cache.json")


def _load_cache() -> dict:
    """Load cache from disk."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    """Persist cache to disk."""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _hash_query(query: str) -> str:
    """Normalize and hash the query as cache key."""
    return hashlib.md5(query.strip().lower().encode()).hexdigest()


def get_cache(query: str):
    """Return cached answer for query, or None if not found."""
    cache = _load_cache()
    key = _hash_query(query)
    return cache.get(key, None)


def set_cache(query: str, answer: str):
    """Store query-answer pair in cache."""
    cache = _load_cache()
    key = _hash_query(query)
    cache[key] = answer
    _save_cache(cache)


# ── TEST ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🧪 Testing cache...")

    set_cache("What is BGP?", "BGP uses TCP port 179.")
    result = get_cache("What is BGP?")
    print(f"✅ Cache hit: {result}")

    result2 = get_cache("Some unknown query")
    print(f"❌ Cache miss: {result2}")