"""
smoke_test.py  —  Quick verification that the BERT transformer NER works.
Run from project root: python smoke_test.py
"""
from transformers import pipeline
import re

# ── 1. Load model ─────────────────────────────────────────────────────────────
print("\n[INFO] Loading dslim/bert-base-NER …")
ner = pipeline(
    "token-classification",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple",
)
print("[INFO] Model loaded ✅\n")

# ── 2. Sample legal text ───────────────────────────────────────────────────────
TEXT = (
    "This Agreement is entered into between GlobalTech Innovations Pvt Ltd "
    "and Apex Financial Holdings Inc. The total contract value is $2,500,000, "
    "payable by January 15, 2025. This Agreement is governed by the laws of "
    "the State of California, USA. Signed by Arjun Mehta and Laura Thompson."
)

# ── 3. Run NER ────────────────────────────────────────────────────────────────
LABEL_MAP = {"PER": "PERSON", "ORG": "PARTY", "LOC": "JURISDICTION"}
results    = ner(TEXT)

print("=" * 60)
print("  🔬  BERT NER — Extraction Results")
print("=" * 60)

for ent in results:
    hf_label    = ent["entity_group"]
    legal_label = LABEL_MAP.get(hf_label, hf_label)
    print(f"  {legal_label:14s} | {ent['word']:35s} | {ent['score']*100:.1f}%")

# ── 4. Regex DATE / MONEY ─────────────────────────────────────────────────────
print("\n--- Regex DATE / MONEY ---")
DATE_RE  = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
MONEY_RE = r'[\$\€\£]\s*[\d,]+(?:\.\d{1,2})?'

for m in re.finditer(DATE_RE, TEXT, re.IGNORECASE):
    print(f"  DATE           | {m.group()}")
for m in re.finditer(MONEY_RE, TEXT):
    print(f"  MONEY          | {m.group()}")

print("\n✅  Smoke test passed — transformer integration is working!\n")
