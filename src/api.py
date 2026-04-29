"""
api.py
──────
FastAPI REST API for LexiScan Auto.

POST /extract   — Upload a PDF contract; returns structured legal entities
                  with confidence scores from the BERT transformer model.
GET  /health    — Quick health check / model info endpoint.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from src.extract_entities import extract_entities_from_pdf

app = FastAPI(
    title="LexiScan Auto — Legal NER API",
    description=(
        "AI-powered legal document analysis. "
        "Extracts PARTY, PERSON, DATE, MONEY, and JURISDICTION entities "
        "from PDF contracts using a BERT transformer model."
    ),
    version="2.0.0",
)

# Allow browser / frontend access during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ────────────────────────────────────────────────────
#  Health check
# ────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status":  "ok",
        "model":   "dslim/bert-base-NER",
        "version": "2.0.0",
        "entities_supported": ["PARTY", "PERSON", "JURISDICTION", "DATE", "MONEY"],
    }


# ────────────────────────────────────────────────────
#  Main extraction endpoint
# ────────────────────────────────────────────────────
@app.post("/extract")
async def extract_entities(file: UploadFile = File(...)):
    """
    Upload a PDF contract (digital or scanned).

    Returns structured legal entities, each with:
    - `text`       — extracted string
    - `confidence` — model confidence score (0.0 – 1.0)
    - `source`     — "transformer" (BERT) or "regex" (DATE/MONEY)
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        entities = extract_entities_from_pdf(file_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        # Clean up temp file after processing
        if os.path.exists(file_path):
            os.remove(file_path)

    return {
        "filename":   file.filename,
        "model":      "dslim/bert-base-NER (BERT transformer)",
        "entities":   entities,
        "entity_count": {label: len(items) for label, items in entities.items()},
    }