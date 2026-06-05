"""
extract_entities.py
───────────────────
Public interface for the entity extraction pipeline.
Now powered by a BERT transformer model instead of the tiny spaCy custom model.
"""

import os
from src.ocr import extract_text_from_pdf
from src.ner_transformer import extract_with_transformer


ENTITY_ORDER = ["DATE", "PARTY", "JURISDICTION", "MONEY", "PERSON"]


def _flatten_entities(entities: dict) -> dict:
  """Convert internal entity records into a simple label -> [text] payload."""
  flattened = {}

  for label in ENTITY_ORDER:
    if label in entities:
      flattened[label] = [item["text"] for item in entities[label]]

  for label, items in entities.items():
    if label not in flattened:
      flattened[label] = [item["text"] for item in items]

  return flattened


def extract_entities_from_pdf(pdf_path: str) -> dict:
    """
    Full pipeline:
      PDF → OCR text → Transformer NER → structured entities dict

  Returns:
    {
      "filename": "sample.pdf",
      "entities": {
        "DATE": ["2025-01-15"],
        "PARTY": ["GlobalTech Innovations Pvt Ltd"],
        ...
      }
    }
    """
    # Resolve to absolute path (required by pytesseract / tests)
    pdf_path = os.path.abspath(pdf_path)

    # Step 1 — OCR: convert PDF pages to raw text
    text = extract_text_from_pdf(pdf_path)

    # Step 2 — Transformer NER + regex DATE/MONEY extraction
    entities = extract_with_transformer(text)

    return {
      "filename": os.path.basename(pdf_path),
      "entities": _flatten_entities(entities),
    }