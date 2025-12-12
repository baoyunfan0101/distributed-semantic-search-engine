# scripts/bert_lsh_experiments.py

import random
import numpy as np
import matplotlib.pyplot as plt

from scripts.load_dataset import load_warc
from scripts.build_bert_lsh import build_bert_lsh
from scripts.lsh_contrast_experiments import evaluate_lsh
from utils.logger import log, log_section


def run_bert_lsh_sweep(
        documents: list[str],
        *,
        num_hashes: int,
        band_configs: list[tuple[int, int]],
        num_queries: int = 50,
        top_k: int = 50,
        seed: int = 47,
):
    random.seed(seed)

    results = []

    for num_bands, rows_per_band in band_configs:
        log(
            f"Building BERT LSH: bands={num_bands}, rows={rows_per_band}",
            "INFO",
        )

        lsh, signatures, embeddings = build_bert_lsh(
            documents=documents,
            num_hashes=num_hashes,
            num_bands=num_bands,
            rows_per_band=rows_per_band,
            seed=seed,
        )

        # Randomly select query documents
        query_ids = random.sample(range(len(documents)), num_queries)

        res = evaluate_lsh(
            lsh=lsh,
            query_ids=query_ids,
            signatures=signatures,
            embeddings=embeddings,
            ngram_sets=None,
            top_k=top_k,
        )

        results.append({
            "num_bands": num_bands,
            "rows_per_band": rows_per_band,
            "avg_recall_bert": float(np.mean(res["recall_bert"])),
            "avg_recall_jaccard": float(np.mean(res["recall_jaccard"])),
            "avg_candidate_size": float(np.mean(res["candidate_size"])),
            "avg_latency_ms": float(np.mean(res["latency"]) * 1e3),
        })

    return results


def plot_bert_lsh_tradeoff(results: list[dict]):
    plt.figure()

    for r in results:
        label = f"b={r['num_bands']}, r={r['rows_per_band']}"
        plt.scatter(
            r["avg_candidate_size"],
            r["avg_recall_bert"],
            s=80,
            label=label,
        )

    plt.xlabel("Average Candidate Size")
    plt.ylabel("Recall@50 (Cosine GT)")
    plt.title("BERT LSH: Recall–Candidate Trade-off")
    plt.legend(fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("3.1.png", dpi=300)
    plt.show()


def main():
    log_section("EXPERIMENT 3.1")

    # Load dataset
    max_docs = 2000
    documents = load_warc(max_docs=max_docs)

    # Define sweep parameters
    num_hashes = 512

    # (num_bands, rows_per_band)
    band_configs = [
        (64, 8),
        (32, 16),
        (16, 32),
        (8, 64),
    ]

    # Run experiments
    results = run_bert_lsh_sweep(
        documents,
        num_hashes=num_hashes,
        band_configs=band_configs,
        num_queries=50,
        top_k=50,
    )

    # Plot results
    plot_bert_lsh_tradeoff(results)

    # Log summary
    for r in results:
        log(
            f"BERT LSH (bands={r['num_bands']}, rows={r['rows_per_band']}):\n"
            f"\tRecall@50 (BERT): {r['avg_recall_bert']:.3f}\n"
            f"\tRecall@50 (Jaccard): {r['avg_recall_jaccard']:.3f}\n"
            f"\tAvg candidate size: {r['avg_candidate_size']:.1f}\n"
            f"\tAvg query time: {r['avg_latency_ms']:.3f} ms\n",
            "RESULT",
        )


if __name__ == "__main__":
    main()
