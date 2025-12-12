# searcher/lsh.py

from typing import Any

from collections import defaultdict


class LocalitySensitiveHashing:

    def __init__(self, num_bands: int, rows_per_band: int):
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.signature_size = num_bands * rows_per_band

        # One hash table per band
        self.tables: list[dict] = [
            defaultdict(set) for _ in range(num_bands)
        ]

        self._size = 0

    def _get_band(self, signature: list[int], band_idx: int) -> tuple[int, ...]:
        start = band_idx * self.rows_per_band
        end = start + self.rows_per_band
        return tuple(signature[start:end])

    def add(self, signature: list[int], doc_id: Any) -> None:
        if len(signature) != self.signature_size:
            raise ValueError(
                f"Signature length {len(signature)} does not match "
                f"expected size {self.signature_size}"
            )

        for band_idx in range(self.num_bands):
            band_key = self._get_band(signature, band_idx)
            self.tables[band_idx][band_key].add(doc_id)

        self._size += 1

    def query(self, signature: list[int]) -> set[Any]:
        if len(signature) != self.signature_size:
            raise ValueError(
                f"Signature length {len(signature)} does not match "
                f"expected size {self.signature_size}"
            )

        candidates: set[Any] = set()

        for band_idx in range(self.num_bands):
            band_key = self._get_band(signature, band_idx)
            candidates |= self.tables[band_idx].get(band_key, set())

        return candidates

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        for table in self.tables:
            table.clear()
        self._size = 0
