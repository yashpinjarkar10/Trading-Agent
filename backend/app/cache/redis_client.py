"""
Cache abstraction with Redis backend and in-process fallback.

Designed so future upgrades (rate-limiting, Celery broker, pub/sub) can reuse
the same Redis connection by calling `get_redis()`.
"""
from __future__ import annotations

import json
import pickle
import threading
import time
from typing import Any, Callable, Optional

from app.config.settings import settings

try:
    import redis  # type: ignore
    _REDIS_AVAILABLE = True
except ImportError:  # pragma: no cover
    redis = None  # type: ignore
    _REDIS_AVAILABLE = False


_redis_client: Optional["redis.Redis"] = None
_redis_lock = threading.Lock()


def get_redis() -> Optional["redis.Redis"]:
    """
    Return a singleton Redis client, or None if Redis is disabled / unreachable.
    Reusable for cache, rate limiting, pub/sub, Celery broker, etc.
    """
    global _redis_client
    if not settings.REDIS_ENABLED or not _REDIS_AVAILABLE:
        return None
    if _redis_client is not None:
        return _redis_client
    with _redis_lock:
        if _redis_client is not None:
            return _redis_client
        try:
            client = redis.Redis.from_url(
                settings.REDIS_URL,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=False,  # we store bytes (pickle/json)
            )
            client.ping()
            _redis_client = client
            print(f"✅ Redis connected: {settings.REDIS_URL}")
        except Exception as e:
            print(f"⚠️  Redis unavailable ({e}); falling back to in-process cache")
            _redis_client = None
    return _redis_client


class _InProcessCache:
    """Tiny TTL cache for when Redis is unavailable. Not multi-process safe."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, bytes]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at and expires_at < time.time():
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: bytes, ttl: int) -> None:
        with self._lock:
            expires_at = time.time() + ttl if ttl else 0
            self._store[key] = (expires_at, value)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)


_fallback = _InProcessCache()


class Cache:
    """
    Unified cache interface. Supports JSON-serializable and pickle-able values.

    Usage:
        cache.set("k", {"a": 1}, ttl=60)
        cache.get("k")
        cache.get_or_set("k", lambda: expensive(), ttl=60)
    """

    def _client(self):
        return get_redis() or _fallback

    @staticmethod
    def _serialize(value: Any) -> bytes:
        try:
            return b"j:" + json.dumps(value, default=str).encode("utf-8")
        except (TypeError, ValueError):
            return b"p:" + pickle.dumps(value)

    @staticmethod
    def _deserialize(blob: bytes) -> Any:
        if blob is None:
            return None
        if blob.startswith(b"j:"):
            return json.loads(blob[2:].decode("utf-8"))
        if blob.startswith(b"p:"):
            return pickle.loads(blob[2:])
        # Legacy / unknown — try json then pickle
        try:
            return json.loads(blob.decode("utf-8"))
        except Exception:
            return pickle.loads(blob)

    def get(self, key: str) -> Any:
        try:
            client = self._client()
            blob = client.get(key)
            if blob is None:
                return None
            return self._deserialize(blob)
        except Exception as e:
            print(f"⚠️  cache.get error for {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        try:
            client = self._client()
            blob = self._serialize(value)
            if isinstance(client, _InProcessCache):
                client.set(key, blob, ttl)
            else:
                client.set(key, blob, ex=ttl)
            return True
        except Exception as e:
            print(f"⚠️  cache.set error for {key}: {e}")
            return False

    def delete(self, key: str) -> None:
        try:
            self._client().delete(key)
        except Exception as e:
            print(f"⚠️  cache.delete error for {key}: {e}")

    def get_or_set(self, key: str, producer: Callable[[], Any], ttl: int = 300) -> Any:
        cached_value = self.get(key)
        if cached_value is not None:
            return cached_value
        value = producer()
        if value is not None:
            self.set(key, value, ttl=ttl)
        return value


cache = Cache()
