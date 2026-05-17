"""
CAMEO event-code mappings for the GDELT scraper.

CAMEO (Conflict and Mediation Event Observations) is a fixed ontology of 20
top-level "root codes" and ~300 sub-codes used by GDELT. Mapping is purely
table-driven — NO LLM CALLS — which keeps GDELT ingestion at zero LLM cost.

References:
  - https://www.gdeltproject.org/data.html#documentation
  - https://parusanalytics.com/eventdata/cameo.dir/CAMEO.Manual.1.1b3.pdf
"""
from __future__ import annotations

from .models import EventCategory

# ── Top-level (Root) CAMEO event codes ────────────────────────────────────────
# Two-digit string codes, "01"–"20".
CAMEO_ROOT_DESCRIPTIONS: dict[str, str] = {
    "01": "Public statement",
    "02": "Appeal",
    "03": "Intent to cooperate",
    "04": "Consultation",
    "05": "Diplomatic cooperation",
    "06": "Material cooperation",
    "07": "Aid provision",
    "08": "Yielding",
    "09": "Investigation",
    "10": "Demand",
    "11": "Disapproval",
    "12": "Rejection",
    "13": "Threat",
    "14": "Protest",
    "15": "Force posture",
    "16": "Reduction in relations",
    "17": "Coercion",
    "18": "Assault",
    "19": "Armed conflict",
    "20": "Unconventional mass violence",
}

# Map each root code to one of our 9 EventMap categories.
CAMEO_ROOT_TO_CATEGORY: dict[str, EventCategory] = {
    "01": EventCategory.other,
    "02": EventCategory.politics,
    "03": EventCategory.politics,
    "04": EventCategory.politics,
    "05": EventCategory.politics,
    "06": EventCategory.politics,
    "07": EventCategory.disaster_hum,
    "08": EventCategory.politics,
    "09": EventCategory.politics,
    "10": EventCategory.politics,
    "11": EventCategory.politics,
    "12": EventCategory.politics,
    "13": EventCategory.politics,
    "14": EventCategory.protest,
    "15": EventCategory.politics,
    "16": EventCategory.politics,
    "17": EventCategory.conflict,
    "18": EventCategory.conflict,
    "19": EventCategory.conflict,
    "20": EventCategory.conflict,
}

# Codes we actually ingest. Everything else is too low-signal for the
# Event Map (statements, appeals, consultations, …). The LLM enricher
# never sees the dropped categories — savings compound.
INGESTED_ROOT_CODES: frozenset[str] = frozenset(
    {"07", "13", "14", "17", "18", "19", "20"}
)


def root_description(root_code: str) -> str:
    """Return a short human-readable description of a CAMEO root code."""
    return CAMEO_ROOT_DESCRIPTIONS.get(root_code, "Unspecified event")


def root_to_category(root_code: str) -> EventCategory:
    """Return the EventMap category for a CAMEO root code."""
    return CAMEO_ROOT_TO_CATEGORY.get(root_code, EventCategory.other)
