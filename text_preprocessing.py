import re
import spacy

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer")

boilerplate_patterns   = [
    re.compile(r"^students? (will|can|are) (be able to|understand|apply|describe|identify|use|prepared to|recognize|explain|differentiate|evaluate|outline|conduct|analyze|construct)\b", re.IGNORECASE),
    re.compile(r"^this course (is|was|presents|provides|covers|explores|focuses on|is designed to)\b", re.IGNORECASE),
    re.compile(r"^the course (explores|covers|focuses on|utilizes|is designed to|presents)\b", re.IGNORECASE),
    re.compile(r"^a graduate course\b", re.IGNORECASE),
    re.compile(r"^topics (include|will include)\b", re.IGNORECASE),
    re.compile(r"^an undergraduate course\b", re.IGNORECASE),
    re.compile(r"^students (are|will be) prepared\b", re.IGNORECASE),
    re.compile(r"^the class (uses|utilizes|provides|focuses on)\b", re.IGNORECASE),
    re.compile(r"^the ability to\b", re.IGNORECASE),
    re.compile(r"^often we are involved\b", re.IGNORECASE),
    re.compile(r"^as such, (they|this)\b", re.IGNORECASE),
]

def remove_boilerplate(text: str) -> str:
    doc = nlp(text)
    kept_sents = []

    for sent in doc.sents:
        original = sent.text.strip()
        cleaned = original

        for pattern in boilerplate_patterns:
            cleaned = pattern.sub("", cleaned).strip()  # Remove the matched phrase if found
        if cleaned: 
            kept_sents.append(cleaned)

    return " ".join(kept_sents)
