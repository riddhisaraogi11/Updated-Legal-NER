import spacy

nlp = spacy.load("models/legal_ner_model")

# Realistic held-out contract style text
EVAL_TEXT = """
This Master Agreement is signed on March 10, 2026 between Alpha Tech Solutions Pvt Ltd 
and Omega Financial Services Inc. The total agreement value is $1,200,000 payable 
until March 9, 2028. This Agreement is governed by the laws of Texas, USA.
"""

doc = nlp(EVAL_TEXT)

print("\n------ PREDICTED ENTITIES ON HELD-OUT TEXT ------\n")

for ent in doc.ents:
    print(ent.text, "|", ent.label_)