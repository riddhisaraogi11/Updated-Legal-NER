"""
extract_entities.py
───────────────────
Public interface for the entity extraction pipeline.
Now powered by a BERT transformer model instead of the tiny spaCy custom model.
"""

import os
from src.ocr import extract_text_from_pdf
from src.ner_transformer import extract_with_transformer


def extract_entities_from_pdf(pdf_path: str) -> dict:
    """
    Full pipeline:
      PDF → OCR text → Transformer NER → structured entities dict

    Each entity value is a list of:
        {"text": str, "confidence": float, "source": "transformer"|"regex"}
    """
    # Resolve to absolute path (required by pytesseract / tests)
    pdf_path = os.path.abspath(pdf_path)

    # Step 1 — OCR: convert PDF pages to raw text
    text = extract_text_from_pdf(pdf_path)

    # Step 2 — Transformer NER + regex DATE/MONEY extraction
    entities = extract_with_transformer(text)

    return entities