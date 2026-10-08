"""Semantic cache for ResearchLens AI — Step 10.

Caches RAG query results when the query is semantically similar to a
previously answered one.  Only verified / supported answers are cached.

Design goals:
- Avoid redundant LLM calls for repeat or near-duplicate questions.
- Only store answers that passed evidence verification.
- Configurable TTL, max cache size, and similarity threshold.
- Thread-safe: uses a simple in-process dictionary with a read-write lock.
- Serialisable: can optionally be persisted to a JSON file.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
_DEFAULT_SIMILARITY_THRESHOLD: float = 0.92
_DEFAULT_TTL_SECONDS: int = 3600          # 1 hour
_DEFAULT_MAX_CACHE_SIZE: int = 200        # entries
_UNCACHEABLE_STATUSES = {
    "contradicted",
    "unsupported",
    "insufficient_evidence",
    "error",
}


class SemanticCache:
    """In-process semantic cache backed by sentence-transformer embeddings.

    Parameters
    ----------
    embedding_model:
        Name of the sentence-transformers model used to embed queries.
        Defaults to ``all-MiniLM-L6-v2`` (the same model as the pipeline).
    similarity_threshold:
        Cosine-similarity threshold above which two queries are considered
        equivalent.  Defaults to 0.92 (configurable via ``CACHE_SIM_THRESHOLD``
        env var).
    ttl_seconds:
        Seconds before a cache entry expires.  0 means never expires.
        Defaults to 3 600 s (1 h).  Configurable via ``CACHE_TTL_SECONDS``.
    max_size:
        Maximum number of entries to hold in memory.  When the limit is
        reached, the oldest entry is evicted (LRU-style).  Defaults to 200.
        Configurable via ``CACHE_MAX_SIZE``.
    persist_path:
        Optional path to a JSON file for persistence across restarts.
    enabled:
        Master switch.  When ``False``, ``get`` always returns ``None`` and
        ``put`` is a no-op.  Configurable via ``SEMANTIC_CACHE_ENABLED``.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: Optional[float] = None,
        ttl_seconds: Optional[int] = None,
        max_size: Optional[int] = None,
        persist_path: Optional[Path] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        # Resolve config from env vars (with argument overrides taking priority)
        _env_enabled = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower()
        self.enabled: bool = (
            enabled
            if enabled is not None
            else _env_enabled in ("true", "1", "yes")
        )
        self.threshold: float = (
            similarity_threshold
            if similarity_threshold is not None
            else float(os.getenv("CACHE_SIM_THRESHOLD", str(_DEFAULT_SIMILARITY_THRESHOLD)))
        )
        self.ttl: int = (
            ttl_seconds
            if ttl_seconds is not None
            else int(os.getenv("CACHE_TTL_SECONDS", str(_DEFAULT_TTL_SECONDS)))
        )
        self.max_size: int = (
            max_size
            if max_size is not None
            else int(os.getenv("CACHE_MAX_SIZE", str(_DEFAULT_MAX_CACHE_SIZE)))
        )
        self.persist_path: Optional[Path] = persist_path

        self._lock = threading.RLock()
        # { query_text: { "embedding": List[float], "result": dict, "ts": float } }
        self._store: Dict[str, Dict[str, Any]] = {}
        self._model = None  # lazy-loaded

        # Stats
        self._hits = 0
        self._misses = 0
        self._skipped = 0  # non-cacheable (unverified) answers

        if self.enabled:
            self._load_from_disk()
            logger.info(
                "SemanticCache initialised | threshold=%.2f | ttl=%ds | max_size=%d",
                self.threshold,
                self.ttl,
                self.max_size,
            )
        else:
            logger.info("SemanticCache disabled via configuration.")

    # ── Embedding helper ──────────────────────────────────────────────────────

    def _get_model(self):
        """Lazy-load the sentence-transformer model (thread-safe)."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                        self._model = SentenceTransformer("all-MiniLM-L6-v2")
                        logger.debug("SemanticCache: loaded embedding model.")
                    except Exception as exc:
                        logger.warning("SemanticCache: could not load model — %s", exc)
        return self._model

    def _embed(self, text: str) -> Optional[List[float]]:
        """Return a unit-normalised embedding vector for *text*, or ``None``."""
        model = self._get_model()
        if model is None:
            return None
        try:
            vec = model.encode(text, normalize_embeddings=True)
            return vec.tolist()
        except Exception as exc:
            logger.warning("SemanticCache: embedding failed — %s", exc)
            return None

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        """Fast cosine similarity for pre-normalised vectors (dot product)."""
        dot = sum(x * y for x, y in zip(a, b))
        return max(-1.0, min(1.0, dot))

    # ── Cache operations ──────────────────────────────────────────────────────

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached result for *query*, or ``None`` on a cache miss.

        Args:
            query: User query string.

        Returns:
            Previously cached result dict, or ``None``.
        """
        if not self.enabled:
            return None

        query_emb = self._embed(query)
        if query_emb is None:
            self._misses += 1
            return None

        now = time.time()
        with self._lock:
            best_key: Optional[str] = None
            best_sim: float = -1.0
            expired_keys: List[str] = []

            for key, entry in self._store.items():
                # Evict expired entries on the fly
                if self.ttl > 0 and (now - entry["ts"]) > self.ttl:
                    expired_keys.append(key)
                    continue
                sim = self._cosine(query_emb, entry["embedding"])
                if sim > best_sim:
                    best_sim = sim
                    best_key = key

            for k in expired_keys:
                del self._store[k]
                logger.debug("SemanticCache: expired entry removed.")

            if best_key is not None and best_sim >= self.threshold:
                result = self._store[best_key]["result"]
                self._hits += 1
                logger.info(
                    "SemanticCache HIT | sim=%.4f | query='%.60s'",
                    best_sim,
                    query,
                )
                # Return a shallow copy so callers cannot mutate the store
                return dict(result, _cache_hit=True, _cache_similarity=round(best_sim, 4))

        self._misses += 1
        logger.debug("SemanticCache MISS | query='%.60s'", query)
        return None

    def put(self, query: str, result: Dict[str, Any]) -> bool:
        """Store *result* for *query*.

        Entries are only stored when the answer is "supported" or
        "partially_supported" by evidence (i.e. the pipeline did not flag it
        as unverified, contradicted, or insufficient).

        Args:
            query: User query string.
            result: Full RAG pipeline result dict.

        Returns:
            ``True`` if the entry was stored, ``False`` otherwise.
        """
        if not self.enabled:
            return False

        # Only cache verified answers
        verification_status = (result.get("verification_status") or "").lower()
        if verification_status in _UNCACHEABLE_STATUSES:
            self._skipped += 1
            logger.debug(
                "SemanticCache: skipping cache storage for status='%s'",
                verification_status,
            )
            return False

        query_emb = self._embed(query)
        if query_emb is None:
            return False

        with self._lock:
            # Evict oldest if at capacity
            if len(self._store) >= self.max_size:
                oldest_key = min(self._store, key=lambda k: self._store[k]["ts"])
                del self._store[oldest_key]
                logger.debug("SemanticCache: evicted oldest entry.")

            self._store[query] = {
                "embedding": query_emb,
                "result": result,
                "ts": time.time(),
            }
            logger.debug("SemanticCache: stored entry for query='%.60s'", query)

        self._save_to_disk()
        return True

    def invalidate(self, query: Optional[str] = None) -> int:
        """Invalidate one or all cache entries.

        Args:
            query: Specific query to invalidate, or ``None`` to clear everything.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            if query is None:
                count = len(self._store)
                self._store.clear()
                logger.info("SemanticCache: cleared all %d entries.", count)
            else:
                count = 0
                keys_to_del = [k for k in self._store if k == query]
                for k in keys_to_del:
                    del self._store[k]
                    count += 1
        if count:
            self._save_to_disk()
        return count

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return cache hit/miss statistics.

        Returns:
            Dictionary with ``hits``, ``misses``, ``skipped``, ``size``,
            and ``hit_rate`` keys.
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        with self._lock:
            size = len(self._store)
        return {
            "hits": self._hits,
            "misses": self._misses,
            "skipped": self._skipped,
            "size": size,
            "max_size": self.max_size,
            "hit_rate": round(hit_rate, 4),
            "ttl_seconds": self.ttl,
            "similarity_threshold": self.threshold,
            "enabled": self.enabled,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_to_disk(self) -> None:
        """Persist the cache to a JSON file (if ``persist_path`` is set)."""
        if self.persist_path is None:
            return
        try:
            with self._lock:
                data = {
                    k: {"embedding": v["embedding"], "ts": v["ts"], "result": v["result"]}
                    for k, v in self._store.items()
                }
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            logger.debug("SemanticCache: persisted %d entries to %s.", len(data), self.persist_path)
        except Exception as exc:
            logger.warning("SemanticCache: could not persist to disk — %s", exc)

    def _load_from_disk(self) -> None:
        """Load persisted cache from disk (if ``persist_path`` is set)."""
        if self.persist_path is None or not Path(self.persist_path).exists():
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as fh:
                data: Dict[str, Any] = json.load(fh)
            now = time.time()
            loaded = 0
            with self._lock:
                for key, entry in data.items():
                    if self.ttl > 0 and (now - entry.get("ts", 0)) > self.ttl:
                        continue  # skip expired entries
                    self._store[key] = entry
                    loaded += 1
            logger.info("SemanticCache: loaded %d entries from %s.", loaded, self.persist_path)
        except Exception as exc:
            logger.warning("SemanticCache: could not load from disk — %s", exc)
