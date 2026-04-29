"""
main.py
───────
Standalone CLI script for LexiScan Auto.

Usage:
    python -m src.main
    python -m src.main --pdf path/to/contract.pdf

Runs the full pipeline:
  PDF  →  OCR  →  BERT NER + Regex  →  Validation  →  JSON output
"""

import argparse
import json
import re
from datetime import datetime

from ocr import extract_text_from_pdf
from ner_transformer import extract_with_transformer


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def parse_date(date_string: str):
    for fmt in ("%B %d, %Y", "%B %d %Y", "%d %B %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    return None


def validate_entities(entities: dict) -> dict:
    """
    Sanity checks on extracted entities.
    Returns a validation report dict.
    """
    validation = {}

    # ── Date ordering ──────────────────────────────────────────────────
    if "DATE" in entities and len(entities["DATE"]) >= 2:
        dates_text   = [e["text"] for e in entities["DATE"]]
        parsed_dates = [d for d in (parse_date(t) for t in dates_text) if d]
        parsed_dates.sort()

        if len(parsed_dates) >= 2:
            validation["date_sequence_valid"] = parsed_dates[0] < parsed_dates[-1]

    # ── Money — must contain a currency symbol ─────────────────────────
    if "MONEY" in entities:
        valid_money = [
            e["text"] for e in entities["MONEY"]
            if re.search(r"[\$\€\£]", e["text"])
        ]
        validation["valid_monetary_values"] = valid_money

    # ── Parties found ──────────────────────────────────────────────────
    validation["parties_found"] = len(entities.get("PARTY", []))

    return validation


def pretty_print(entities: dict, validation: dict) -> None:
    """Human-readable terminal output with confidence scores."""
    LABEL_ICONS = {
        "PARTY":        "🏢",
        "PERSON":       "👤",
        "DATE":         "📅",
        "MONEY":        "💰",
        "JURISDICTION": "🌍",
    }
    print("\n" + "═" * 58)
    print("   🔬  LexiScan Auto — Extraction Results")
    print("   Model: dslim/bert-base-NER (BERT Transformer)")
    print("═" * 58)

    for label, items in entities.items():
        icon = LABEL_ICONS.get(label, "🔹")
        print(f"\n{icon}  {label}  ({len(items)} found)")
        print("─" * 40)
        for ent in items:
            bar_len  = int(ent["confidence"] * 20)
            conf_bar = "█" * bar_len + "░" * (20 - bar_len)
            src_tag  = f"[{ent['source']}]"
            print(f"  {ent['text']}")
            print(f"  {conf_bar}  {ent['confidence']*100:.1f}%  {src_tag}")

    print("\n" + "─" * 58)
    print("  ✅  Validation Checks")
    print("─" * 58)
    for key, val in validation.items():
        print(f"  {key}: {val}")
    print("═" * 58 + "\n")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LexiScan Auto — Legal NER")
    parser.add_argument(
        "--pdf",
        default="data/raw_pdfs/sample.pdf",
        help="Path to the PDF contract (default: data/raw_pdfs/sample.pdf)",
    )
    args = parser.parse_args()

    print(f"\n[INFO] Processing: {args.pdf}")

    # ── Pipeline ─────────────────────────────────────────────────────
    raw_text = extract_text_from_pdf(args.pdf)
    entities = extract_with_transformer(raw_text)
    validation = validate_entities(entities)

    # ── Terminal output ───────────────────────────────────────────────
    pretty_print(entities, validation)

    # ── JSON output ───────────────────────────────────────────────────
    final_output = {
        "model":      "dslim/bert-base-NER",
        "source_pdf": args.pdf,
        "entities":   entities,
        "validation": validation,
    }

    print("\n------ FULL JSON OUTPUT ------\n")
    print(json.dumps(final_output, indent=4))