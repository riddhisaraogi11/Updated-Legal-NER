import spacy
from spacy.training import Example
import random

# 1. Training Data (Safe Span Calculation)

def make_example(text, entities_list):
    entities = []
    for phrase, label in entities_list:
        if phrase not in text:
            print("ERROR: phrase not found ->", phrase)
        start = text.index(phrase)
        end = start + len(phrase)
        entities.append((start, end, label))
    return (text, {"entities": entities})

TRAIN_DATA = [

    make_example(
        "This Agreement shall commence on January 15, 2025 and shall remain in effect until January 14, 2027.",
        [
            ("January 15, 2025", "DATE"),
            ("January 14, 2027", "DATE"),
        ],
    ),

    make_example(
        "The total contract value for the services under this Agreement shall be $2,500,000 (Two Million Five Hundred Thousand US Dollars), payable in quarterly installments.",
        [
            ("$2,500,000", "MONEY"),
        ],
    ),

    make_example(
        "An advance payment of $500,000 shall be paid within thirty (30) days of the Effective Date.",
        [
            ("$500,000", "MONEY"),
        ],
    ),

    make_example(
        "This Agreement shall be governed by and construed in accordance with the laws of the State of California, USA.",
        [
            ("State of California, USA", "JURISDICTION"),
        ],
    ),

    make_example(
        "Any disputes arising out of or relating to this Agreement shall be subject to the exclusive jurisdiction of the courts located in San Francisco, California.",
        [
            ("San Francisco, California", "JURISDICTION"),
        ],
    ),

    
    make_example(
        "Signed on January 15, 2025.",
        [
            ("January 15, 2025", "DATE"),
        ],
    ),

    make_example(
        "Total Estimated Project Budget: $2,500,000. Secondary Support Contract Effective Date: February 1, 2025. Secondary Support Contract Termination Date: January 31, 2026. Jurisdiction: State of California, USA.",
        [
            ("$2,500,000", "MONEY"),
            ("February 1, 2025", "DATE"),
            ("January 31, 2026", "DATE"),
            ("State of California, USA", "JURISDICTION"),
        ],
    ),

    make_example(
        "For GlobalTech Innovations Pvt Ltd:",
        [
            ("GlobalTech Innovations Pvt Ltd", "PARTY"),
        ],
    ),

    make_example(
        "For Apex Financial Holdings Inc:",
        [
            ("Apex Financial Holdings Inc", "PARTY"),
        ],
    ),

    make_example(
        "This Agreement is entered into between Horizon Tech Systems Ltd and Sigma Financial Services Pvt Ltd.",
        [
            ("Horizon Tech Systems Ltd", "PARTY"),
            ("Sigma Financial Services Pvt Ltd", "PARTY"),
        ],
    ),
]

# 2. Create Blank Model

nlp = spacy.load("en_core_web_sm")
ner = nlp.get_pipe("ner")

ner.add_label("DATE")
ner.add_label("PARTY")
ner.add_label("MONEY")
ner.add_label("JURISDICTION")


nlp = spacy.load("en_core_web_sm")

# Disable other components (train ONLY NER)
pipe_exceptions = ["ner"]
other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

with nlp.disable_pipes(*other_pipes):

    ner = nlp.get_pipe("ner")

    ner.add_label("DATE")
    ner.add_label("PARTY")
    ner.add_label("MONEY")
    ner.add_label("JURISDICTION")

    optimizer = nlp.resume_training()

    for epoch in range(40):
        random.shuffle(TRAIN_DATA)
        losses = {}

        for text, annotations in TRAIN_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.2, losses=losses)

        print(f"Epoch {epoch+1}, Loss: {losses}")

# 3. Save Model

nlp.to_disk("models/legal_ner_model")
print("\nModel training complete and saved.")