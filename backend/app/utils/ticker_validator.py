"""
Ticker validation utility.

Bug #5: avoid an LLM round-trip on every request via:
  1. Regex short-circuit for inputs that already look like a ticker.
  2. Redis-backed cache (TTL = settings.CACHE_TTL_TICKER_VALIDATION) for resolved names.

Bug #10: accept international ticker patterns (BRK.B, BF-B, 9988.HK, 7203.T,
RDS-A, ^GSPC, BTC-USD). The previous `isalpha()` filter rejected all of them.
"""
import re

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config.settings import settings
from app.cache import cache

# Bug #10: dot, hyphen, caret, digits all valid in real-world tickers.
#   AAPL, BRK.B, BF-B, 9988.HK, 7203.T, RDS-A, ^GSPC (S&P500 index), BTC-USD
TICKER_PATTERN = re.compile(r"^\^?[A-Z0-9]{1,6}([.\-][A-Z0-9]{1,4})*$")

# Lightweight known-aliases map — short-circuits a Gemini call for the
# overwhelmingly common case. Extend as needed (or load from a JSON file).
_KNOWN_ALIASES: dict[str, str] = {
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "AMAZON": "AMZN",
    "TESLA": "TSLA",
    "META": "META",
    "FACEBOOK": "META",
    "NVIDIA": "NVDA",
    "NETFLIX": "NFLX",
    "AMD": "AMD",
    "PAYPAL": "PYPL",
    "BERKSHIRE": "BRK.B",
    "BITCOIN": "BTC-USD",
    "ETHEREUM": "ETH-USD",
}


_llm = None


def _get_llm():
    """Lazy-init LLM so importing this module never crashes on missing key."""
    global _llm
    if _llm is None and settings.GEMINI_API_KEY:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            api_key=settings.GEMINI_API_KEY,
            temperature=0,
        )
    return _llm


def _looks_like_ticker(s: str) -> bool:
    return bool(TICKER_PATTERN.match(s))


def _resolve_with_llm(input_ticker: str) -> str:
    llm = _get_llm()
    if llm is None:
        return input_ticker.upper()
    prompt = (
        f'Given the input "{input_ticker}", return ONLY the valid stock ticker '
        f"symbol in uppercase. Use Yahoo Finance conventions (e.g. BRK.B, "
        f"9988.HK, 7203.T, BTC-USD). If already a ticker, return as-is. "
        f"No explanation, just the symbol.\nInput: {input_ticker}\nOutput:"
    )
    try:
        response = llm.invoke(prompt)
        ticker = response.content.strip().upper().split()[0].rstrip(".")
        # Bug #10: accept international patterns instead of isalpha()
        if _looks_like_ticker(ticker):
            return ticker
        return input_ticker.upper()
    except Exception as e:
        print(f"⚠️ Ticker LLM resolution error: {e}")
        return input_ticker.upper()


def get_valid_ticker(input_ticker: str) -> str:
    """
    Validate or resolve a ticker.

    Order of operations (cheapest first):
      1. Strip + uppercase.
      2. If empty → return as-is.
      3. If it already looks like a ticker → return immediately.
      4. If it's in the alias map → return mapped value.
      5. Cache lookup (Redis with in-process fallback).
      6. LLM fallback, then cache the result.
    """
    if not input_ticker:
        return ""
    raw = input_ticker.strip()
    upper = raw.upper()

    if _looks_like_ticker(upper):
        return upper

    alias = _KNOWN_ALIASES.get(upper)
    if alias:
        return alias

    cache_key = f"ticker_validation:{upper}"
    cached_value = cache.get(cache_key)
    if cached_value:
        return cached_value

    resolved = _resolve_with_llm(raw)
    if resolved and resolved != upper:
        cache.set(cache_key, resolved, ttl=settings.CACHE_TTL_TICKER_VALIDATION)
    return resolved
