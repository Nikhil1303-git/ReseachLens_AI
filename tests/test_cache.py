"""Step 10 — Tests for app/cache/semantic_cache.py.

Covers:
- Cache hit / miss
- Similarity threshold enforcement
- TTL expiry
- Uncacheable statuses not stored
- Max-size eviction
- Cache invalidation
- Stats reporting
- Disk persistence round-trip
- Disabled cache pass-through
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from app.cache.semantic_cache import SemanticCache, _UNCACHEABLE_STATUSES


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def cache_no_persist():
    """Return an enabled SemanticCache with no disk persistence."""
    return SemanticCache(
        enabled=True,
        similarity_threshold=0.90,
        ttl_seconds=3600,
        max_size=50,
        persist_path=None,
    )


@pytest.fixture()
def cache_disabled():
    """Return a disabled SemanticCache."""
    return SemanticCache(enabled=False)


def _fake_embed(text: str):
    """Return a fake embedding based on text length (deterministic for testing)."""
    import math
    n = len(text) % 100
    angle = n * math.pi / 100.0
    # 3-d unit vector so cosine = dot product works correctly
    vec = [math.cos(angle), math.sin(angle), 0.0]
    norm = sum(x ** 2 for x in vec) ** 0.5
    return [x / norm for x in vec]


def _patch_embed(cache: SemanticCache):
    """Monkey-patch _embed to use the fake deterministic embedder."""
    cache._embed = _fake_embed


# ── Helper ────────────────────────────────────────────────────────────────────


def _make_result(status: str = "supported") -> dict:
    return {
        "query": "test",
        "response": "Test answer.",
        "verification_status": status,
        "citations": [],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1. Basic enabled/disabled behaviour
# ══════════════════════════════════════════════════════════════════════════════


class TestCacheEnabledDisabled:
    def test_disabled_get_returns_none(self, cache_disabled):
        result = cache_disabled.get("any query")
        assert result is None

    def test_disabled_put_returns_false(self, cache_disabled):
        assert cache_disabled.put("any query", _make_result()) is False

    def test_disabled_stats_shows_enabled_false(self, cache_disabled):
        stats = cache_disabled.stats()
        assert stats["enabled"] is False

    def test_enabled_cache_is_enabled(self, cache_no_persist):
        assert cache_no_persist.enabled is True


# ══════════════════════════════════════════════════════════════════════════════
# 2. Cache miss (nothing stored yet)
# ══════════════════════════════════════════════════════════════════════════════


class TestCacheMiss:
    def test_empty_cache_returns_none(self, cache_no_persist):
        _patch_embed(cache_no_persist)
        result = cache_no_persist.get("What is deep learning?")
        assert result is None

    def test_miss_increments_miss_counter(self, cache_no_persist):
        _patch_embed(cache_no_persist)
        cache_no_persist.get("unknown query")
        assert cache_no_persist._misses == 1


# ══════════════════════════════════════════════════════════════════════════════
# 3. Cache hit (exact same query)
# ══════════════════════════════════════════════════════════════════════════════


class TestCacheHit:
    def test_exact_same_query_returns_cached(self, cache_no_persist):
        _patch_embed(cache_no_persist)
        q = "What is gradient descent?"
        r = _make_result()
        cache_no_persist.put(q, r)
        cached = cache_no_persist.get(q)
        assert cached is not None
        assert cached["response"] == r["response"]

    def test_hit_has_cache_hit_flag(self, cache_no_persist):
        _patch_embed(cache_no_persist)
        q = "Explain backpropagation."
        cache_no_persist.put(q, _make_result())
        cached = cache_no_persist.get(q)
        assert cached.get("_cache_hit") is True

    def test_hit_includes_similarity_score(self, cache_no_persist):
        _patch_embed(cache_no_persist)
        q = "What is overfitting?"
        cache_no_persist.put(q, _make_result())
        cached = cache_no_persist.get(q)
        assert "_cache_similarity" in cached

    def test_hit_increments_hit_counter(self, cache_no_persist):
        _patch_embed(cache_no_persist)
        q = "Describe regularization."
        cache_no_persist.put(q, _make_result())
        cache_no_persist.get(q)
        assert cache_no_persist._hits == 1


# ══════════════════════════════════════════════════════════════════════════════
# 4. Uncacheable statuses
# ══════════════════════════════════════════════════════════════════════════════


class TestUncacheableStatuses:
    @pytest.mark.parametrize("status", sorted(_UNCACHEABLE_STATUSES))
    def test_uncacheable_status_not_stored(self, cache_no_persist, status):
        _patch_embed(cache_no_persist)
        q = f"Query with status {status}"
        stored = cache_no_persist.put(q, _make_result(status))
        assert stored is False
        result = cache_no_persist.get(q)
        assert result is None

    def test_supported_status_is_cached(self, cache_no_persist):
        _patch_embed(cache_no_persist)
        stored = cache_no_persist.put("supported query", _make_result("supported"))
        assert stored is True

    def test_partially_supported_is_cached(self, cache_no_persist):
        _patch_embed(cache_no_persist)
        stored = cache_no_persist.put("partial query", _make_result("partially_supported"))
        assert stored is True

    def test_skipped_counter_incremented(self, cache_no_persist):
        _patch_embed(cache_no_persist)
        cache_no_persist.put("bad query", _make_result("contradicted"))
        assert cache_no_persist._skipped == 1


# ══════════════════════════════════════════════════════════════════════════════
# 5. TTL expiry
# ══════════════════════════════════════════════════════════════════════════════


class TestTTLExpiry:
    def test_expired_entry_returns_none(self):
        cache = SemanticCache(
            enabled=True,
            similarity_threshold=0.90,
            ttl_seconds=1,  # 1 second TTL
            max_size=10,
            persist_path=None,
        )
        _patch_embed(cache)
        q = "Will this expire?"
        cache.put(q, _make_result())

        # Manually backdate the timestamp so it's expired
        with cache._lock:
            for key in list(cache._store.keys()):
                cache._store[key]["ts"] -= 5  # 5 seconds in the past

        result = cache.get(q)
        assert result is None

    def test_non_expired_entry_returned(self):
        cache = SemanticCache(
            enabled=True,
            similarity_threshold=0.90,
            ttl_seconds=3600,
            max_size=10,
            persist_path=None,
        )
        _patch_embed(cache)
        q = "Should still be valid."
        cache.put(q, _make_result())
        result = cache.get(q)
        assert result is not None


# ══════════════════════════════════════════════════════════════════════════════
# 6. Max size eviction
# ══════════════════════════════════════════════════════════════════════════════


class TestMaxSizeEviction:
    def test_cache_does_not_exceed_max_size(self):
        cache = SemanticCache(
            enabled=True,
            similarity_threshold=0.99,
            ttl_seconds=3600,
            max_size=3,
            persist_path=None,
        )
        _patch_embed(cache)
        for i in range(5):
            cache.put(f"unique query number {i * 7}", _make_result())
        with cache._lock:
            assert len(cache._store) <= 3


# ══════════════════════════════════════════════════════════════════════════════
# 7. Invalidation
# ══════════════════════════════════════════════════════════════════════════════


class TestInvalidation:
    def test_invalidate_all_clears_cache(self, cache_no_persist):
        _patch_embed(cache_no_persist)
        for i in range(3):
            cache_no_persist.put(f"query {i * 13}", _make_result())
        removed = cache_no_persist.invalidate()
        assert removed >= 3
        assert cache_no_persist.get("query 0") is None

    def test_invalidate_specific_query(self, cache_no_persist):
        _patch_embed(cache_no_persist)
        q = "specific query to remove"
        cache_no_persist.put(q, _make_result())
        removed = cache_no_persist.invalidate(q)
        assert removed == 1


# ══════════════════════════════════════════════════════════════════════════════
# 8. Stats
# ══════════════════════════════════════════════════════════════════════════════


class TestStats:
    def test_stats_returns_correct_keys(self, cache_no_persist):
        stats = cache_no_persist.stats()
        expected_keys = {"hits", "misses", "skipped", "size", "max_size", "hit_rate", "ttl_seconds", "similarity_threshold", "enabled"}
        assert expected_keys.issubset(set(stats.keys()))

    def test_hit_rate_correct(self):
        cache = SemanticCache(
            enabled=True,
            similarity_threshold=0.90,
            ttl_seconds=3600,
            max_size=50,
            persist_path=None,
        )
        _patch_embed(cache)
        # len 10: angle = 10*pi/100 = 0.1*pi
        # len 60: angle = 60*pi/100 = 0.6*pi -> delta = 0.5*pi -> cos(pi/2) = 0.0 -> MISS!
        q1 = "0123456789"
        q2 = "x" * 60
        cache.put(q1, _make_result())
        cache.get(q1)  # hit (sim = 1.0)
        cache.get(q2)  # miss (sim = 0.0)
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


# ══════════════════════════════════════════════════════════════════════════════
# 9. Disk persistence
# ══════════════════════════════════════════════════════════════════════════════


class TestDiskPersistence:
    def test_cache_round_trip(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            cache1 = SemanticCache(
                enabled=True,
                similarity_threshold=0.90,
                ttl_seconds=3600,
                max_size=20,
                persist_path=Path(path),
            )
            _patch_embed(cache1)
            q = "What is transfer learning?"
            cache1.put(q, _make_result())

            # New cache instance — should reload from disk
            cache2 = SemanticCache(
                enabled=True,
                similarity_threshold=0.90,
                ttl_seconds=3600,
                max_size=20,
                persist_path=Path(path),
            )
            _patch_embed(cache2)
            with cache2._lock:
                size = len(cache2._store)
            assert size >= 1
        finally:
            Path(path).unlink(missing_ok=True)

    def test_persisted_file_is_valid_json(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            cache = SemanticCache(
                enabled=True,
                similarity_threshold=0.90,
                ttl_seconds=3600,
                max_size=20,
                persist_path=Path(path),
            )
            _patch_embed(cache)
            cache.put("persisted query", _make_result())
            with open(path, "r") as fh:
                data = json.load(fh)
            assert isinstance(data, dict)
        finally:
            Path(path).unlink(missing_ok=True)
