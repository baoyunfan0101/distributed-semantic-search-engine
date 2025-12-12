# tokenizer/filters.py

import os

STOPWORDS = set()

# Load stopwords from file
STOPWORD_PATH = os.path.join(os.path.dirname(__file__), "stopwords.txt")

with open(STOPWORD_PATH, "r") as f:
    for line in f:
        word = line.strip().lower()
        if word:
            STOPWORDS.add(word)


def is_stopword(tok: str) -> bool:
    return tok in STOPWORDS


def is_too_short(tok: str, min_len: int = 2) -> bool:
    return len(tok) < min_len


def filter_tokens(tokens: list[str]) -> list[str]:
    filtered = []
    for tok in tokens:
        if is_stopword(tok):
            continue
        if is_too_short(tok):
            continue
        filtered.append(tok)
    return filtered
