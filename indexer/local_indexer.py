# indexer/local_indexer.py

from typing import Callable, Iterable

from collections import Counter

from tokenizer.regex_tokenizer import RegexTokenizer


# Local Global TF Indexer
class LocalGlobalTFIndexer:

    def __init__(self, tokenizer: Callable[[str], Iterable[str]] = RegexTokenizer()):
        self.tokenizer = tokenizer
        self.global_tf = Counter()

    def update_from_text(self, text: str):
        tokens = self.tokenizer(text)
        self.global_tf.update(tokens)

    def build_tf(self, documents: Iterable[str]) -> Counter:
        for text in documents:
            self.update_from_text(text)

        return self.global_tf

    def get_tf(self, term: str) -> int:
        return self.global_tf.get(term.lower(), 0)

    def top_k(self, k: int):
        return self.global_tf.most_common(k)

    def vocabulary_size(self) -> int:
        return len(self.global_tf)

    def total_tokens(self) -> int:
        return sum(self.global_tf.values())
