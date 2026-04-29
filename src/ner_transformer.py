"""
ner_transformer.py
──────────────────
Transformer-based Named Entity Recognition for legal contracts.

Replaces the tiny spaCy custom model with:
  • dslim/bert-base-NER   — BERT fine-tuned on CoNLL-2003 (PER / ORG / LOC)
  • Regex fallback        — DATE and MONEY patterns (BERT doesn't tag these)

Each extracted entity carries a confidence score (0.0 – 1.0).
"""

import re
from collections import defaultdict
from transformers import pipeline

# ─────────────────────────────────────────────
# 1.  Load BERT NER Pipeline (lazy singleton)
# ─────────────────────────────────────────────
_ner_pipeline = None


def _get_ner_pipeline():
    """Load the HuggingFace NER pipeline once and reuse it."""
    global _ner_pipeline
    if _ner_pipeline is None:
        print("[INFO] Loading dslim/bert-base-NER transformer model …")
        _ner_pipeline = pipeline(
            "token-classification",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",   # merges sub-word tokens
        )
        print("[INFO] Transformer model loaded successfully.")
    return _ner_pipeline


# ─────────────────────────────────────────────
# 2.  HuggingFace → Legal label mapping
# ─────────────────────────────────────────────
#   CoNLL label   →  Our legal entity label
LABEL_MAP = {
    "PER":  "PERSON",       # signatories / individuals
    "ORG":  "PARTY",        # companies / organisations
    "LOC":  "JURISDICTION", # locations / governing law
    "MISC": None,           # skip — not relevant for legal NER
}


# ─────────────────────────────────────────────
# 3.  Regex patterns for DATE and MONEY
#     (BERT-base-NER doesn't tag these natively)
# ─────────────────────────────────────────────
DATE_PATTERNS = [
    # "January 15, 2025"  /  "15 January 2025"
    r'\b(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',

    r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+\d{4}\b',

    # ISO: 2025-01-15
    r'\b\d{4}-\d{2}-\d{2}\b',

    # US short: 01/15/2025
    r'\b\d{1,2}/\d{1,2}/\d{4}\b',

    # "15th day of January, 2025"
    r'\b\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+'
    r'(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December),?\s+\d{4}\b',
]

MONEY_PATTERNS = [
    # $2,500,000  /  €1.5M  /  £500,000.00
    r'[\$\€\£]\s*[\d,]+(?:\.\d{1,2})?(?:\s*(?:million|billion|thousand|M|B|K))?',

    # 1,200,000 USD / 500000 EUR
    r'\b[\d,]+(?:\.\d{1,2})?\s*(?:USD|EUR|GBP|INR|AUD|CAD)\b',

    # Two Million Five Hundred Thousand US Dollars (written form)
    r'\b(?:[A-Z][a-z]+\s+){1,6}(?:US\s+)?Dollars\b',
]


# ─────────────────────────────────────────────
# 4.  Utility helpers
# ─────────────────────────────────────────────
def _add_entity(entities: dict, label: str, text: str,
                confidence: float, source: str) -> None:
    """Deduplicate and append an entity to the result dict."""
    text = text.strip()
    if not text:
        return
    existing_texts = [e["text"] for e in entities[label]]
    if text not in existing_texts:
        entities[label].append({
            "text":       text,
            "confidence": round(confidence, 4),
            "source":     source,           # "transformer" or "regex"
        })


def _run_regex_dates(text: str, entities: dict) -> None:
    for pattern in DATE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            _add_entity(entities, "DATE", match.group(), 0.95, "regex")


def _run_regex_money(text: str, entities: dict) -> None:
    for pattern in MONEY_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            _add_entity(entities, "MONEY", match.group(), 0.95, "regex")


# ─────────────────────────────────────────────
# 5.  Main extraction function
# ─────────────────────────────────────────────
def extract_with_transformer(text: str) -> dict:
    """
    Extract legal entities from raw contract text.

    Returns
    -------
    dict  {label: [{"text": str, "confidence": float, "source": str}, …]}

    Labels: PERSON | PARTY | JURISDICTION | DATE | MONEY
    """
    entities: dict = defaultdict(list)

    # ── Transformer NER  (PERSON, PARTY, JURISDICTION) ───────────────────
    try:
        ner = _get_ner_pipeline()
        results = ner(text)

        for item in results:
            hf_label    = item.get("entity_group", "")   # "PER" / "ORG" / "LOC"
            legal_label = LABEL_MAP.get(hf_label)
            if legal_label is None:
                continue

            _add_entity(
                entities,
                legal_label,
                item["word"],
                float(item["score"]),
                "transformer",
            )

    except Exception as exc:
        print(f"[WARNING] Transformer NER failed: {exc}")
        print("[INFO] Falling back to regex-only extraction for this document.")

    # ── Regex fallback  (DATE, MONEY) ─────────────────────────────────────
    _run_regex_dates(text, entities)
    _run_regex_money(text, entities)

    return dict(entities)
