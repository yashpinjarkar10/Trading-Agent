"""
Function-level caching decorator.

Usage:
    from app.cache import cached

    @cached(ttl=900, key_prefix="ohlcv")
    def load_history(ticker: str, period: str) -> pd.DataFrame:
        ...
"""
from __future__ import annotations

import functools
import hashlib
from typing import Any, Callable

from app.cache.redis_client import cache


def _make_key(prefix: str, args: tuple, kwargs: dict) -> str:
    raw = repr(args) + repr(sorted(kwargs.items()))
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def cached(ttl: int = 300, key_prefix: str = "fn") -> Callable:
    """Cache the return value of a sync function keyed by args."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        prefix = f"{key_prefix}:{fn.__module__}.{fn.__name__}"

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = _make_key(prefix, args, kwargs)
            hit = cache.get(key)
            if hit is not None:
                return hit
            result = fn(*args, **kwargs)
            if result is not None:
                cache.set(key, result, ttl=ttl)
            return result

        wrapper.cache_key = lambda *a, **kw: _make_key(prefix, a, kw)  # type: ignore[attr-defined]
        return wrapper

    return decorator
