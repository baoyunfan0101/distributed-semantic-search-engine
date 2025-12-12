# scripts/lsh_experiments.py

from typing import Optional

import time
import random
import numpy as np
import matplotlib.pyplot as plt

from scripts.load_dataset import load_warc
from scripts.build_ngram_lsh import build_ngram_lsh
from scripts.build_bert_lsh import build_bert_lsh
from utils.logger import log, log_section


# Jaccard similarity
def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# Cosine similarity
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# Exact top-k using cosine similarity on BERT embedding
def exact_topk_bert(
        query_emb: np.ndarray,
        doc_embs: np.ndarray,
        k: int,
) -> set[int]:
    scores = [
        (i, cosine_similarity(query_emb, doc_embs[i]))
        for i in range(len(doc_embs))
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    return {i for i, _ in scores[:k]}


# Exact top-k using Jaccard similarity on 3-grams
def exact_topk_jaccard(
        query_ngrams: set[str],
        doc_ngrams: list[set[str]],
        k: int,
) -> set[int]:
    scores = [
        (i, jaccard_similarity(query_ngrams, doc_ngrams[i]))
        for i in range(len(doc_ngrams))
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    return {i for i, _ in scores[:k]}


# Evaluation
def evaluate_lsh(
        lsh,
        query_ids: list[int],
        signatures: list[list[int]],
        *,
        ngram_sets: Optional[list[set[str]]] = None,
        embeddings: Optional[np.ndarray] = None,
        top_k: int,
) -> dict:
    recall_jaccard = []
    recall_bert = []
    candidate_sizes = []
    latencies = []

    for qid in query_ids:
        q_sig = signatures[qid]

        # Ground truths
        gt_jaccard = set()
        if ngram_sets is not None:
            q_grams = ngram_sets[qid]
            gt_jaccard = exact_topk_jaccard(q_grams, ngram_sets, top_k)

        gt_bert = set()
        if embeddings is not None:
            q_emb = embeddings[qid]
            gt_bert = exact_topk_bert(q_emb, embeddings, top_k)

        # LSH query
        start = time.perf_counter()
        candidates = lsh.query(q_sig)
        latency = time.perf_counter() - start

        # Metrics
        hit_jac = len(candidates & gt_jaccard)
        hit_bert = len(candidates & gt_bert)

        recall_jaccard.append(hit_jac / max(len(gt_jaccard), 1))
        recall_bert.append(hit_bert / max(len(gt_bert), 1))
        candidate_sizes.append(len(candidates))
        latencies.append(latency)

    return {
        "recall_jaccard": recall_jaccard,
        "recall_bert": recall_bert,
        "candidate_size": candidate_sizes,
        "latency": latencies,
    }


# Main entry
def main():
    log_section("EXPERIMENT 3.2")

    # Load dataset
    max_docs = 2000
    documents = load_warc(max_docs=max_docs)

    # Build two LSH indices
    log("Building 3-gram LSH...", "INFO")
    lsh_ngram, sig_ngram, ngram_sets = build_ngram_lsh(
        documents=documents,
    )

    log("Building BERT LSH...", "INFO")
    lsh_bert, sig_bert, emb_bert = build_bert_lsh(
        documents=documents,
    )

    # Select queries
    num_queries = 50
    random.seed(47)
    query_ids = random.sample(range(max_docs), num_queries)

    top_k = 20

    # Evaluate
    log("Evaluating 3-gram LSH...", "INFO")
    res_ngram = evaluate_lsh(
        lsh_ngram,
        query_ids,
        sig_ngram,
        embeddings=emb_bert,
        ngram_sets=ngram_sets,
        top_k=top_k,
    )

    log("Evaluating BERT LSH...", "INFO")
    res_bert = evaluate_lsh(
        lsh_bert,
        query_ids,
        sig_bert,
        embeddings=emb_bert,
        ngram_sets=ngram_sets,
        top_k=top_k,
    )

    # Plot
    plt.figure()
    plt.scatter(
        res_ngram["candidate_size"],
        res_ngram["recall_jaccard"],
        color="#1f77b4",
        alpha=0.6,
        label="3-gram LSH (Jaccard GT)",
    )
    plt.scatter(
        res_bert["candidate_size"],
        res_bert["recall_jaccard"],
        color="#ff7f0e",
        alpha=0.6,
        label="BERT LSH (Jaccard GT)",
    )
    plt.xlabel("Candidate Size")
    plt.ylabel("Recall@50")
    plt.title("Experiment 3.2.1: Lexical Relevance (3-gram Jaccard)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("3.2.1.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure()
    plt.scatter(
        res_ngram["candidate_size"],
        res_ngram["recall_bert"],
        color="#1f77b4",
        alpha=0.6,
        label="3-gram LSH (BERT GT)",
    )
    plt.scatter(
        res_bert["candidate_size"],
        res_bert["recall_bert"],
        color="#ff7f0e",
        alpha=0.6,
        label="BERT LSH (BERT GT)",
    )
    plt.xlabel("Candidate Size")
    plt.ylabel("Recall@50")
    plt.title("Experiment 3.2.2: Semantic Relevance (BERT Cosine)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("3.2.2.png", dpi=300, bbox_inches="tight")
    plt.show()

    log(
        f"3-gram LSH:\n"
        f"\tRecall@50(Jaccard): {np.mean(res_ngram['recall_jaccard']):.3f}\n"
        f"\tRecall@50(BERT): {np.mean(res_ngram['recall_bert']):.3f}\n"
        f"\tAvg candidate size: {np.mean(res_ngram['candidate_size']):.1f}\n"
        f"\tAvg query time: {np.mean(res_ngram['latency']) * 1e3:.3f} ms\n",
        "RESULT"
    )

    log(
        f"BERT LSH:\n"
        f"\tRecall@50(Jaccard): {np.mean(res_bert['recall_jaccard']):.3f}\n"
        f"\tRecall@50(BERT): {np.mean(res_bert['recall_bert']):.3f}\n"
        f"\tAvg candidate size: {np.mean(res_bert['candidate_size']):.1f}\n"
        f"\tAvg query time: {np.mean(res_bert['latency']) * 1e3:.3f} ms\n",
        "RESULT"
    )


if __name__ == "__main__":
    main()
