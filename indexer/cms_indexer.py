# indexer/cms_indexer.py

from typing import Iterable, Callable, List

import json
import xxhash

from tokenizer.regex_tokenizer import RegexTokenizer


# Count-Min Sketch
class CountMinSketch:

    def __init__(self, width: int = 20_000, depth: int = 7):
        # Size of hash tables
        self.width = width

        # Number of hash functions (number of hash tables)
        self.depth = depth

        # Hash tables
        self.tables = [[0] * width for _ in range(depth)]

        # Total updates
        self.total_updates = 0

    # Internal: hash function for (term, i)
    def _hash(self, term: str, i: int) -> int:
        h = xxhash.xxh64((str(i) + term).encode()).intdigest() % self.width
        return h

    # Update frequency
    def update(self, term: str, count: int = 1):
        # Do hashing for each row
        for i in range(self.depth):
            idx = self._hash(term, i)
            self.tables[i][idx] += count

        self.total_updates += count

    # Query estimated frequency
    def query(self, term: str) -> int:
        estimates = []
        for i in range(self.depth):
            idx = self._hash(term, i)
            estimates.append(self.tables[i][idx])
        return min(estimates)

    # Save CMS
    def save(self, path: str):
        data = {
            "width": self.width,
            "depth": self.depth,
            "tables": self.tables,
            "total_updates": self.total_updates,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    # Load CMS
    @staticmethod
    def load(path: str) -> "CountMinSketch":
        with open(path, "r") as f:
            data = json.load(f)

        cms = CountMinSketch(width=data["width"], depth=data["depth"])
        cms.tables = data["tables"]
        cms.total_updates = data["total_updates"]
        return cms


# CMS Global TF Indexer
class CMSGlobalTFIndexer:

    def __init__(
            self,
            width: int = 20_000,
            depth: int = 7,
            tokenizer: Callable[[str], List[str]] = RegexTokenizer(),
    ):
        self.cms = CountMinSketch(width=width, depth=depth)
        self.tokenizer = tokenizer

    # Update single document
    def update_from_text(self, text: str):
        tokens = self.tokenizer(text)
        for tok in tokens:
            self.cms.update(tok)

    # Build CMS TF from multiple documents
    def build_tf(self, documents: Iterable[str]):
        for text in documents:
            self.update_from_text(text)

        return self.cms

    # Query CMS TF
    def get_tf(self, term: str) -> int:
        return self.cms.query(term.lower())

    # Convenience helpers
    def total_tokens(self):
        return self.cms.total_updates

    def save(self, path: str):
        self.cms.save(path)

    @staticmethod
    def load(path: str) -> "CMSGlobalTFIndexer":
        cms = CountMinSketch.load(path)
        idx = CMSGlobalTFIndexer()
        idx.cms = cms
        return idx
