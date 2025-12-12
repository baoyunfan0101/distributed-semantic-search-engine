import time
import numpy as np
import streamlit as st

from scripts.load_dataset import load_warc
from scripts.build_ngram_lsh import build_ngram_lsh
from scripts.build_bert_lsh import build_bert_lsh

from tokenizer.ngram_tokenizer import NGramTokenizer
from vectorizer.minhasher import MinHasher
from vectorizer.bert_embedder import BertEmbedder
from vectorizer.hyperplane_hasher import HyperplaneHasher


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@st.cache_resource
def build_indices(max_docs: int = 2000):
    # Load dataset
    docs = load_warc(max_docs=max_docs)

    # Build n-gram LSH (+ keep ngram_sets for Jaccard rerank)
    lsh_ng, sig_ng, ngram_sets = build_ngram_lsh(documents=docs)

    # Build BERT LSH (+ keep embeddings for cosine rerank)
    lsh_bt, sig_bt, emb_bt = build_bert_lsh(documents=docs)

    # Build query-side helpers
    # n-gram query signature generator should match doc-side settings
    ngram_tokenizer = NGramTokenizer(n=3)
    num_hashes = len(sig_ng[0])
    minhasher = MinHasher(num_hashes=num_hashes, seed=42)

    # BERT query vectorizer + hyperplane hasher should match doc-side settings
    bert_embedder = BertEmbedder()
    dim = emb_bt.shape[1]
    hyperplane_hasher = HyperplaneHasher(num_hashes=len(sig_bt[0]), dim=dim, seed=42)

    return {
        "docs": docs,
        "lsh_ng": lsh_ng,
        "sig_ng": sig_ng,
        "ngram_sets": ngram_sets,
        "ngram_tokenizer": ngram_tokenizer,
        "minhasher": minhasher,
        "lsh_bt": lsh_bt,
        "sig_bt": sig_bt,
        "emb_bt": emb_bt,
        "bert_embedder": bert_embedder,
        "hyperplane_hasher": hyperplane_hasher,
    }


def topk_ngram(query: str, state, top_k: int = 5, rerank_k: int = 50):
    # Get signature
    q_grams = state["ngram_tokenizer"](query)
    q_sig = state["minhasher"].signature(q_grams)

    t0 = time.perf_counter()
    candidates = state["lsh_ng"].query(q_sig)
    t1 = time.perf_counter()

    # Rerank by exact Jaccard within candidates
    scored = []
    for doc_id in list(candidates)[:rerank_k] if rerank_k else candidates:
        score = jaccard(q_grams, state["ngram_sets"][doc_id])
        scored.append((doc_id, score))
    scored.sort(key=lambda x: x[1], reverse=True)

    return {
        "query_time_ms": (t1 - t0) * 1e3,
        "candidate_size": len(candidates),
        "results": scored[:top_k],
    }


def topk_bert(query: str, state, top_k: int = 5, rerank_k: int = 50):
    # Get signature
    q_emb = state["bert"].encode([query], batch_size=1)[0]
    q_sig = state["hyper_hasher"].signature(q_emb)

    t0 = time.perf_counter()
    candidates = state["lsh_bt"].query(q_sig)
    t1 = time.perf_counter()

    # Rerank by exact cosine within candidates
    scored = []
    for doc_id in list(candidates)[:rerank_k] if rerank_k else candidates:
        score = cosine(q_emb, state["emb_bt"][doc_id])
        scored.append((doc_id, score))
    scored.sort(key=lambda x: x[1], reverse=True)

    return {
        "query_time_ms": (t1 - t0) * 1e3,
        "candidate_size": len(candidates),
        "results": scored[:top_k],
    }


def snippet(text: str, n: int = 350) -> str:
    t = " ".join(text.split())
    return t[:n] + ("..." if len(t) > n else "")


def main():
    st.set_page_config(page_title="LSH Search Demo", layout="wide")
    st.title("LSH Search Demo: 3-gram (MinHash) vs BERT (Hyperplane)")

    with st.sidebar:
        max_docs = st.slider("Max docs to index", 200, 3000, 2000, 200)
        top_k = st.slider("Show top-K", 1, 10, 5, 1)
        rerank_k = st.slider("Max candidates to rerank", 20, 500, 100, 10)

    state = build_indices(max_docs=max_docs)

    query = st.text_input("Type a search query", value="diabetes risk prediction from wearable data")

    if st.button("Search"):
        col1, col2 = st.columns(2)

        ng = topk_ngram(query, state, top_k=top_k, rerank_k=rerank_k)
        bt = topk_bert(query, state, top_k=top_k, rerank_k=rerank_k)

        with col1:
            st.subheader("3-gram LSH (MinHash + Jaccard rerank)")
            st.caption(f"Candidates: {ng['candidate_size']} | LSH query time: {ng['query_time_ms']:.3f} ms")
            for rank, (doc_id, score) in enumerate(ng["results"], 1):
                st.markdown(f"**#{rank}  doc_id={doc_id}  score={score:.4f}**")
                st.write(snippet(state["docs"][doc_id]))

        with col2:
            st.subheader("BERT LSH (Hyperplane + Cosine rerank)")
            st.caption(f"Candidates: {bt['candidate_size']} | LSH query time: {bt['query_time_ms']:.3f} ms")
            for rank, (doc_id, score) in enumerate(bt["results"], 1):
                st.markdown(f"**#{rank}  doc_id={doc_id}  score={score:.4f}**")
                st.write(snippet(state["docs"][doc_id]))


if __name__ == "__main__":
    main()
