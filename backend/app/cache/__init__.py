"""
Reusable cache layer.

Public API:
    from app.cache import cache, cached

`cache` is a singleton with `.get`, `.set`, `.delete`, `.get_or_set`.
`cached(ttl=..., key_prefix=...)` is a decorator for functions.

Backend: Redis if REDIS_ENABLED and reachable, otherwise an in-process
fallback so dev / CI works without Redis running.
"""
from app.cache.redis_client import cache, get_redis
from app.cache.decorators import cached

__all__ = ["cache", "cached", "get_redis"]
