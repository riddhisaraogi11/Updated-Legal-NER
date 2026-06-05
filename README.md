# Automated Legal Entity Extractor (LexiScan Auto)

## Overview

LexiScan Auto is an **AI-powered legal document analysis system** that extracts structured entities from legal contracts.
The system processes PDF contracts and automatically identifies key legal information such as **dates, parties, jurisdictions, monetary values, and signatories**.

This project combines **OCR, Natural Language Processing (NLP), and rule-based post-processing** to convert unstructured legal documents into structured JSON data.

---

## Features

* Extract entities from legal contracts
* Handles scanned or image-based PDFs using OCR
* Custom **spaCy Named Entity Recognition (NER) model**
* Rule-based layer to improve precision
* REST API using **FastAPI**
* End-to-end automated pipeline
* Unit tests using **pytest**

---

## Extracted Entities

The system extracts the following entities:

* **DATE** – contract dates and deadlines
* **PARTY** – organizations involved in the contract
* **PERSON** – signatories or individuals
* **JURISDICTION** – locations and governing law
* **MONEY** – contract values and payments

---

## Project Architecture

```
PDF Contract
     │
     ▼
OCR (pdf2image + Tesseract)
     │
     ▼
spaCy Custom NER Model
     │
     ▼
Rule-based Post Processing
     │
     ▼
Structured JSON Output
```

---

## Project Structure

```
legal_ner_project
│
├── data_raw_pdfs
│   └── sample.pdf
│
├── models
│   └── legal_ner_model
│
├── src
│   ├── api.py
│   ├── extract_entities.py
│   ├── ocr.py
│   ├── post_processing.py
│   ├── train_ner.py
│   └── evaluate.py
│
├── tests
│   ├── test_entities.py
│   └── test_pipeline.py
│
├── README.md
└── .gitignore
```

---

## Installation

### 1. Clone the Repository

```
git clone https://github.com/your-username/legal-ner-lexiscan.git
cd legal-ner-lexiscan
```

### 2. Create Virtual Environment

```
python -m venv venv
```

Activate:

Windows

```
venv\Scripts\activate
```

Mac/Linux

```
source venv/bin/activate
```

---

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

### 4. Install OCR Dependencies

Install **Tesseract OCR**

https://github.com/tesseract-ocr/tesseract

Install **Poppler**

https://github.com/oschwartz10612/poppler-windows

---

## Running the API

Start the FastAPI server:

```
uvicorn src.api:app --reload
```

Open API documentation:

```
http://127.0.0.1:8000/docs
```

---

## API Endpoint

### Upload Contract for Entity Extraction

POST `/extract`

Upload a **PDF contract** and the system will return extracted entities.

---

## Example Output

```
{
 "filename": "sample.pdf",
 "entities": {
   "DATE": [
      "2025-01-15",
      "2027-01-14"
   ],
   "PARTY": [
      "GlobalTech Innovations Pvt Ltd",
      "Apex Financial Holdings Inc"
   ],
   "JURISDICTION": [
      "State of California, USA",
      "San Francisco, California"
   ],
   "MONEY": [
      "$2,500,000",
      "$500,000"
   ],
   "PERSON": [
      "Arjun Mehta",
      "Laura Thompson"
   ]
 }
}
```

---

## Running Tests

```
pytest
```

Expected output:

```
2 passed
```

---

## Technologies Used

* Python
* spaCy
* FastAPI
* pdf2image
* Tesseract OCR
* pytest

---

## Applications

* Legal contract analysis
* Compliance automation
* Document intelligence systems
* Enterprise document processing

---

## Future Improvements

* Support multiple document formats
* Improve NER model accuracy with larger datasets
* Deploy using Docker
* Integrate with cloud storage APIs

---

## Author

**Riddhi Saraogi**

B.Tech Computer Science
Artificial Intelligence & Data Science Enthusiast

---
