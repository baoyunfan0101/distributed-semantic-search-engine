# vectorizer/minhasher.py

from typing import Iterable

import random
import hashlib


class MinHasher:

    def __init__(self, num_hashes: int, seed: int = 47):
        self.num_hashes = num_hashes
        self.seed = seed

        random.seed(seed)
        self._salts = [random.randint(0, 2**32 - 1) for _ in range(num_hashes)]

    def _hash(self, value: str, salt: int) -> int:
        h = hashlib.sha1()
        h.update(value.encode("utf-8"))
        h.update(salt.to_bytes(4, byteorder="little"))
        return int(h.hexdigest(), 16)

    def signature(self, items: Iterable[str]) -> list[int]:
        items = set(items)
        if not items:
            return [0] * self.num_hashes

        sig = []
        for salt in self._salts:
            min_hash = None
            for item in items:
                hv = self._hash(item, salt)
                if min_hash is None or hv < min_hash:
                    min_hash = hv
            sig.append(min_hash)

        return sig
