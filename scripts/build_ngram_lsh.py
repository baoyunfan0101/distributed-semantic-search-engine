# scripts/build_ngram_lsh.py

from tokenizer.ngram_tokenizer import NGramTokenizer
from vectorizer.minhasher import MinHasher
from searcher.lsh import LocalitySensitiveHashing


def build_ngram_lsh(
        documents: list[str],
        ngram_n: int = 3,
        num_hashes: int = 128,
        num_bands: int = 32,
        rows_per_band: int = 4,
        seed: int = 47,
):
    assert num_bands * rows_per_band == num_hashes

    # Tokenize into n-grams
    tokenizer = NGramTokenizer(n=ngram_n)

    ngram_sets = [
        set(tokenizer(doc))
        for doc in documents
    ]

    # Vectorize into signatures
    vectorizer = MinHasher(num_hashes=num_hashes, seed=seed)

    signatures = [
        vectorizer.signature(grams)
        for grams in ngram_sets
    ]

    # Build LSH index
    lsh = LocalitySensitiveHashing(
        num_bands=num_bands,
        rows_per_band=rows_per_band,
    )

    for doc_id, sig in enumerate(signatures):
        lsh.add(sig, doc_id)

    return lsh, signatures, ngram_sets
