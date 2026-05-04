"""
Finance-tuned sentiment analyzer (no LLM, no GPU).

Approach (industry-standard, per the Loughran-McDonald finance dictionary +
VADER lexicon-rule hybrid that quants and news APIs typically use):

  1. Base lexicon: VADER's ~7.5k-token sentiment lexicon, which already
     handles negation, intensifiers, capitalization, punctuation, and
     contractions out of the box.
  2. Domain overlay: a curated finance-specific lexicon (Loughran-McDonald
     style) — words that mean BAD in finance even if VADER scores them
     neutral ("downgrade", "miss", "covenant breach"), and words that mean
     GOOD in finance ("beat", "raised guidance", "buyback").
  3. Multi-word phrase boosts: regex matches for high-signal phrases
     ("earnings miss", "guidance raised", "stock plunges").
  4. Source-quality weighting (Bloomberg / Reuters / FT > random blogs).
  5. Time-decay weighting: recent articles count more, older fade.

Output is a `SentimentResult` with `score` in [-1, 1] and a label.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
except ImportError:  # pragma: no cover
    SentimentIntensityAnalyzer = None  # type: ignore
    _VADER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Finance-specific lexicon overlay (Loughran-McDonald inspired, condensed).
# Values are added to VADER's compound score before clipping to [-1, 1].
# ---------------------------------------------------------------------------
FINANCE_LEXICON: dict[str, float] = {
    # Negative finance terms VADER often misses or under-weights
    "downgrade": -2.5, "downgraded": -2.5, "downgrading": -2.0,
    "miss": -1.5, "misses": -1.5, "missed": -1.5, "shortfall": -2.0,
    "guidance cut": -3.0, "lowered guidance": -2.8, "reduced outlook": -2.5,
    "warning": -2.0, "profit warning": -3.0, "earnings warning": -3.0,
    "bankruptcy": -3.5, "bankrupt": -3.5, "insolvency": -3.5, "default": -3.0,
    "lawsuit": -1.8, "subpoena": -2.0, "investigation": -1.8, "probe": -1.5,
    "fraud": -3.5, "scandal": -3.0, "restatement": -2.5, "delisted": -3.0,
    "covenant breach": -3.0, "going concern": -3.5,
    "layoffs": -1.8, "layoff": -1.8, "restructuring": -1.0,
    "underperform": -2.0, "underperforms": -2.0, "underperforming": -2.0,
    "sell rating": -2.0, "downside": -1.5, "headwind": -1.5, "headwinds": -1.5,
    "plunge": -2.5, "plunges": -2.5, "plunged": -2.5,
    "tumble": -2.0, "tumbles": -2.0, "tumbled": -2.0,
    "slump": -2.0, "slumps": -2.0, "slumped": -2.0,
    "halted": -2.5, "suspended": -2.0, "recall": -2.0, "recalled": -2.0,
    "sec probe": -2.5, "doj investigation": -2.5,
    "dilution": -1.5, "dilutive": -1.5,
    "impairment": -2.0, "writedown": -2.0, "write-down": -2.0,
    "trial halted": -3.0, "fda rejection": -3.5,

    # Positive finance terms
    "beat": 2.0, "beats": 2.0, "beaten": 1.5, "topped": 2.0, "exceeded": 2.0,
    "raised guidance": 3.0, "guidance raised": 3.0, "upgraded": 2.5, "upgrade": 2.5,
    "outperform": 2.0, "outperforms": 2.0, "outperforming": 2.0, "outperformed": 2.0,
    "buy rating": 2.0, "strong buy": 2.5, "overweight": 1.8,
    "buyback": 2.0, "buybacks": 2.0, "share repurchase": 2.0, "repurchase": 1.8,
    "dividend hike": 2.0, "dividend increase": 2.0, "raised dividend": 2.0,
    "record revenue": 2.5, "record profit": 2.5, "record earnings": 2.5,
    "approval": 1.8, "fda approval": 3.0, "patent granted": 2.0,
    "breakthrough": 2.0, "milestone": 1.5, "all-time high": 2.5, "all time high": 2.5,
    "rally": 1.8, "rallies": 1.8, "rallied": 1.8,
    "surge": 2.0, "surges": 2.0, "surged": 2.0, "soared": 2.0, "soars": 2.0,
    "tailwind": 1.5, "tailwinds": 1.5, "momentum": 1.0,
    "acquisition": 1.0, "merger": 0.8, "strategic partnership": 1.5,
}


# Multi-word phrase patterns are checked separately so word-boundary regex catches them.
_PHRASE_PATTERNS = [(re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE), score)
                    for phrase, score in FINANCE_LEXICON.items() if " " in phrase]
_SINGLE_WORD_LEXICON = {k: v for k, v in FINANCE_LEXICON.items() if " " not in k}


# Source quality weights (multiplier applied to per-article score).
SOURCE_WEIGHTS: dict[str, float] = {
    "bloomberg": 1.0,
    "reuters": 1.0,
    "wall street journal": 1.0,
    "wsj": 1.0,
    "financial times": 1.0,
    "ft.com": 1.0,
    "barron's": 0.95,
    "barrons": 0.95,
    "cnbc": 0.85,
    "marketwatch": 0.8,
    "yahoo finance": 0.75,
    "yahoo": 0.75,
    "google news": 0.7,
    "seeking alpha": 0.6,
    "motley fool": 0.55,
    "investorplace": 0.5,
    "benzinga": 0.7,
    "the verge": 0.6,
    "techcrunch": 0.6,
    # Default for unknown sources is 0.5
}
DEFAULT_SOURCE_WEIGHT = 0.5


@dataclass
class SentimentResult:
    score: float            # [-1, +1] after weighting
    raw_score: float        # [-1, +1] before source/time weights
    label: str              # 'Positive' | 'Neutral' | 'Negative'
    confidence: float       # [0, 1] — distance from neutral
    source_weight: float
    time_weight: float
    matched_terms: list[str]


# Lazy global so library import doesn't crash if vader missing.
_vader: Optional[SentimentIntensityAnalyzer] = None


def _get_vader() -> Optional[SentimentIntensityAnalyzer]:
    global _vader
    if _vader is None and _VADER_AVAILABLE:
        _vader = SentimentIntensityAnalyzer()
        # Inject finance lexicon — VADER will then handle negation/intensifiers
        # for these terms automatically.
        for word, score in _SINGLE_WORD_LEXICON.items():
            _vader.lexicon[word] = score
    return _vader


def _phrase_adjustment(text: str) -> tuple[float, list[str]]:
    """Sum lexicon adjustments for any multi-word phrases present in the text."""
    total = 0.0
    matches: list[str] = []
    for pattern, score in _PHRASE_PATTERNS:
        if pattern.search(text):
            total += score
            matches.append(pattern.pattern)
    return total, matches


def _source_weight(publisher: Optional[str], source: Optional[str]) -> float:
    blob = f"{publisher or ''} {source or ''}".lower()
    if not blob.strip():
        return DEFAULT_SOURCE_WEIGHT
    for key, weight in SOURCE_WEIGHTS.items():
        if key in blob:
            return weight
    return DEFAULT_SOURCE_WEIGHT


def _time_weight(published: Optional[datetime], half_life_hours: float = 48.0) -> float:
    """Exponential decay: weight = 0.5 ** (age_hours / half_life)."""
    if published is None:
        return 0.5
    try:
        age_hours = max(0.0, (datetime.now() - published).total_seconds() / 3600.0)
    except Exception:
        return 0.5
    return float(math.pow(0.5, age_hours / half_life_hours))


def _label_from_score(score: float) -> str:
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"


def analyze_sentiment(
    text: str,
    publisher: Optional[str] = None,
    source: Optional[str] = None,
    published: Optional[datetime] = None,
) -> SentimentResult:
    """
    Score the sentiment of a single article / snippet.

    The compound score is computed by VADER (with our finance-lexicon overlay
    already injected), then nudged by multi-word finance phrases, then scaled
    by source-quality and time-decay weights.
    """
    if not text or not text.strip():
        return SentimentResult(
            score=0.0, raw_score=0.0, label="Neutral", confidence=0.0,
            source_weight=0.0, time_weight=0.0, matched_terms=[]
        )

    vader = _get_vader()
    if vader is not None:
        scores = vader.polarity_scores(text)
        compound = scores["compound"]  # already in [-1, 1]
    else:
        # Fallback: very rough lexicon sum if VADER unavailable
        compound = 0.0
        lower = text.lower()
        for word, score in _SINGLE_WORD_LEXICON.items():
            if re.search(rf"\b{re.escape(word)}\b", lower):
                compound += score / 4.0  # rough scaling
        compound = max(-1.0, min(1.0, compound))

    # Phrase-level adjustment (clipped contribution)
    phrase_adj, matches = _phrase_adjustment(text)
    if phrase_adj:
        # VADER's compound is in [-1,1]; phrase_adj is unbounded; squash via tanh.
        compound = max(-1.0, min(1.0, compound + math.tanh(phrase_adj / 4.0)))

    sw = _source_weight(publisher, source)
    tw = _time_weight(published)

    # Multiplicative weighting preserves sign and keeps |score| <= 1.
    weighted = max(-1.0, min(1.0, compound * sw * tw))
    confidence = min(1.0, abs(compound))

    return SentimentResult(
        score=weighted,
        raw_score=compound,
        label=_label_from_score(weighted),
        confidence=confidence,
        source_weight=sw,
        time_weight=tw,
        matched_terms=matches,
    )


def aggregate_sentiments(results: Iterable[SentimentResult]) -> dict:
    """
    Combine many per-article sentiments into an overall picture.

    Uses each article's source*time weight as the contribution weight.
    """
    items = list(results)
    if not items:
        return {
            "overall_score": 0.0,
            "overall_label": "Neutral",
            "article_count": 0,
            "positive": 0,
            "neutral": 0,
            "negative": 0,
        }

    total_w = 0.0
    weighted_sum = 0.0
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for r in items:
        w = max(1e-3, r.source_weight * r.time_weight)
        weighted_sum += r.raw_score * w
        total_w += w
        counts[r.label] = counts.get(r.label, 0) + 1

    overall = weighted_sum / total_w if total_w else 0.0
    return {
        "overall_score": round(overall, 4),
        "overall_label": _label_from_score(overall),
        "article_count": len(items),
        "positive": counts["Positive"],
        "neutral": counts["Neutral"],
        "negative": counts["Negative"],
    }
