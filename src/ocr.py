from pdf2image import convert_from_path
import pytesseract

# your poppler path
POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"


def extract_text_from_pdf(pdf_path):

    pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)

    text = ""

    for page in pages:
        text += pytesseract.image_to_string(page)

    return text