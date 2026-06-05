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

# Merge adjacent spans with the same label to produce full entities
normalized = []
for e in results:
    start = e.get("start")
    end = e.get("end")
    label = LABEL_MAP.get(e["entity_group"], e["entity_group"])
    score = e.get("score", 0.0)
    normalized.append({"label": label, "start": start, "end": end, "score": score})

normalized = sorted(normalized, key=lambda x: (x["start"] if x["start"] is not None else 0))

merged = []
for ent in normalized:
    if not merged:
        merged.append(ent)
        continue
    prev = merged[-1]
    # merge if same label and spans touch or are very close (allow comma/space)
    if ent["label"] == prev["label"] and ent["start"] is not None and prev["end"] is not None and ent["start"] <= prev["end"] + 2:
        prev["end"] = ent["end"]
        prev["score"] = max(prev["score"], ent["score"])
    else:
        merged.append(ent)

for ent in merged:
    display = TEXT[ent["start"]:ent["end"]].strip() if ent.get("start") is not None and ent.get("end") is not None else ""
    print(f"  {ent['label']:14s} | {display:35s} | {ent['score']*100:.1f}%")

# ── 4. Regex DATE / MONEY ─────────────────────────────────────────────────────
print("\n--- Regex DATE / MONEY ---")
DATE_RE  = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
MONEY_RE = r'[\$\€\£]\s*[\d,]+(?:\.\d{1,2})?'

for m in re.finditer(DATE_RE, TEXT, re.IGNORECASE):
    print(f"  DATE           | {m.group()}")
for m in re.finditer(MONEY_RE, TEXT):
    print(f"  MONEY          | {m.group()}")

print("\n✅  Smoke test passed — transformer integration is working!\n")
