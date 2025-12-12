# tokenizer/regex_tokenizer.py

import re
from tokenizer.filters import filter_tokens


class RegexTokenizer:

    def __init__(self, pattern: str = r"[a-zA-Z0-9]+"):
        # Compile the regular expression
        self.pattern = re.compile(pattern)

    def __call__(self, text: str) -> list[str]:
        # Convert to lowercase
        text = text.lower()

        # Extract tokens
        tokens = self.pattern.findall(text)

        # Filter tokens
        tokens = filter_tokens(tokens)

        return tokens
