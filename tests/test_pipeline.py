import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.extract_entities import extract_entities_from_pdf

def test_pipeline():
    pdf_path = os.path.join("data", "raw_pdfs", "sample.pdf")
    result = extract_entities_from_pdf(pdf_path)

    assert isinstance(result, dict)