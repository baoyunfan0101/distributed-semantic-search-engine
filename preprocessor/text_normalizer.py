# preprocessor/text_normalizer.py

import re

URL_PATTERN = re.compile(r"http\S+|www\S+")
NUM_PATTERN = re.compile(r"\d+")
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z]+")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = NUM_PATTERN.sub(" ", text)
    text = NON_ALPHA_PATTERN.sub(" ", text)
    text = " ".join(text.split())
    return text
