# scripts/build_bert_lsh.py

from vectorizer.bert_embedder import BertEmbedder
from vectorizer.hyperplane_hasher import HyperplaneHasher
from searcher.lsh import LocalitySensitiveHashing


def build_bert_lsh(
        documents: list[str],
        num_hashes: int = 512,
        num_bands: int = 32,
        rows_per_band: int = 16,
        seed: int = 47,
        batch_size: int = 8,
):
    assert num_bands * rows_per_band == num_hashes

    # Vectorize into embeddings
    vectorizer = BertEmbedder()

    embeddings = vectorizer.encode(documents, batch_size=batch_size)

    # Hash into signatures
    dim = embeddings.shape[1]

    hasher = HyperplaneHasher(
        num_hashes=num_hashes,
        dim=dim,
        seed=seed,
    )

    signatures = [
        hasher.signature(vec)
        for vec in embeddings
    ]

    # Build LSH index
    lsh = LocalitySensitiveHashing(
        num_bands=num_bands,
        rows_per_band=rows_per_band,
    )

    for doc_id, sig in enumerate(signatures):
        lsh.add(sig, doc_id)

    return lsh, signatures, embeddings
