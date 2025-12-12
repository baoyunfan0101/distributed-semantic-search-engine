# vectorizer/hyperplane_hasher.py

from typing import Optional, Sequence, Union

import numpy as np

ArrayLike1D = Union[np.ndarray, Sequence[float]]


class HyperplaneHasher:

    def __init__(
            self,
            num_hashes: int,
            dim: int,
            seed: int = 42,
            normalize: bool = True,
            dtype: np.dtype = np.float32,
    ):
        if num_hashes <= 0:
            raise ValueError("num_hashes must be positive")
        if dim <= 0:
            raise ValueError("dim must be positive")

        self.num_hashes = int(num_hashes)
        self.dim = int(dim)
        self.seed = int(seed)
        self.normalize = bool(normalize)
        self.dtype = dtype

        rng = np.random.default_rng(self.seed)

        # Random hyperplanes: shape (num_hashes, dim)
        self.hyperplanes = rng.standard_normal(size=(self.num_hashes, self.dim)).astype(self.dtype)

    def _to_vector(self, x: ArrayLike1D) -> np.ndarray:
        v = np.asarray(x, dtype=self.dtype)
        if v.ndim != 1:
            raise ValueError(f"Expected 1D vector, got shape {v.shape}")
        if v.shape[0] != self.dim:
            raise ValueError(f"Vector dim {v.shape[0]} does not match expected dim {self.dim}")
        return v

    def _maybe_normalize(self, v: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return v
        norm = np.linalg.norm(v)
        if norm == 0.0:
            return v
        return v / norm

    def signature(self, x: ArrayLike1D) -> list[int]:
        v = self._to_vector(x)
        v = self._maybe_normalize(v)

        # Projections: (num_hashes,)
        proj = self.hyperplanes @ v

        # Bits: 1 if projection >= 0 else 0
        bits = (proj >= 0).astype(np.int8)

        return bits.tolist()

    def signatures(self, X: np.ndarray, batch_size: Optional[int] = None) -> list[list[int]]:
        X = np.asarray(X, dtype=self.dtype)
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"Expected X shape (N, {self.dim}), got {X.shape}")

        if batch_size is None or batch_size <= 0:
            batch_size = X.shape[0]

        out: list[list[int]] = []
        for i in range(0, X.shape[0], batch_size):
            chunk = X[i: i + batch_size]

            if self.normalize:
                norms = np.linalg.norm(chunk, axis=1, keepdims=True)
                norms = np.where(norms == 0.0, 1.0, norms)
                chunk = chunk / norms

            # Projections: (B, num_hashes)
            proj = chunk @ self.hyperplanes.T
            bits = (proj >= 0).astype(np.int8)

            out.extend(bits.tolist())

        return out
