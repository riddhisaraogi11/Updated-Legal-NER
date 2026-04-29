import re
from collections import defaultdict
from datetime import datetime

def standardize_date(date_string):
    formats = [
        "%B %d, %Y",
        "%d %B %Y",
        "%Y-%m-%d"
    ]

    for f in formats:
        try:
            return datetime.strptime(date_string, f).strftime("%Y-%m-%d")
        except:
            continue

    return date_string


def clean_entities(doc):

    entities = defaultdict(list)

    for ent in doc.ents:

        text = ent.text.strip()

        # remove extra spaces
        text = re.sub(r"\s+", " ", text)

        if ent.label_ == "DATE":
            text = standardize_date(text)

        if text not in entities[ent.label_]:
            entities[ent.label_].append(text)

    return dict(entities)

def fix_ocr_noise(text):

    text = text.replace("0", "O")
    text = text.replace("1", "I")

    return text