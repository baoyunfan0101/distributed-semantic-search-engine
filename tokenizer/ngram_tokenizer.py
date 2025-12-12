# tokenizer/ngram_tokenizer.py

import re


class NGramTokenizer:

    def __init__(self, n: int = 3, normalize_whitespace: bool = True):
        assert n >= 2, "Character n-gram size should be >= 2"
        self.n = n
        self.normalize_whitespace = normalize_whitespace

        # Keep only alphanumeric characters and spaces
        self._cleanup_pattern = re.compile(r"[^a-zA-Z0-9 ]+")

    def __call__(self, text: str) -> list[str]:
        # Lowercase
        text = text.lower()

        # Remove special characters
        text = self._cleanup_pattern.sub("", text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = " ".join(text.split())

        # Pad text to preserve boundary information
        padded = f" {text} "

        # Generate character n-grams
        ngrams = [
            padded[i: i + self.n]
            for i in range(len(padded) - self.n + 1)
        ]

        return ngrams
