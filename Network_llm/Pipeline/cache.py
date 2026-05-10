import os
import json
import hashlib

CACHE_FILE = os.path.join(os.path.dirname(__file__), "query_cache.json")

def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE) and os.path.getsize(CACHE_FILE) > 0:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def _save_cache(cache: dict):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def _hash_key(query: str, pdf_name: str) -> str:
    """Hash (query + pdf_name) together so the same question
    for different PDFs always produces a different cache key."""
    combined = f"{pdf_name.strip().lower()}::{query.strip().lower()}"
    return hashlib.md5(combined.encode()).hexdigest()

def get_cache(query: str, pdf_name: str):
    """Return cached answer for (query, pdf_name), or None if not found."""
    cache = _load_cache()
    key = _hash_key(query, pdf_name)
    return cache.get(key, None)

def set_cache(query: str, pdf_name: str, answer: str):
    """Store (query, pdf_name) → answer in cache."""
    cache = _load_cache()
    key = _hash_key(query, pdf_name)
    cache[key] = answer
    _save_cache(cache)