# scripts/cms_experiments.py

import math
import matplotlib.pyplot as plt

from scripts.load_dataset import load_warc
from indexer.local_indexer import LocalGlobalTFIndexer
from indexer.cms_indexer import CMSGlobalTFIndexer
from utils.logger import log, log_section

TOP_K = 5000  # Evaluation terms


def run_cms_experiment(
        documents: list[str],
        top_terms: list[tuple[str, int]],
        width: int,
        depth: int):
    # Calculate CMS TF
    cms_indexer = CMSGlobalTFIndexer(
        width=width,
        depth=depth,
    )
    cms_indexer.build_tf(documents)

    # Calculate errors
    true_vals = []
    cms_vals = []
    errors = []

    for term, true_count in top_terms:
        cms_count = cms_indexer.get_tf(term)

        true_vals.append(true_count)
        cms_vals.append(cms_count)
        errors.append(cms_count - true_count)

    return true_vals, cms_vals, errors


# Experiment 2.1: Vary width
def vary_width_experiment(
        documents: list[str],
        top_terms: list[tuple[str, int]],
        width_list: list[int],
        depth: int,
):
    ncols = 3
    nrows = math.ceil(len(width_list) / ncols)

    fig_hist, axes_hist = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4 * nrows),
        squeeze=False,
    )

    fig_scatter, axes_scatter = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4 * nrows),
        squeeze=False,
    )

    fig_hist.suptitle(
        f"Experiment 2.1: Error Distribution (depth={depth})",
        fontsize=16
    )
    fig_hist.supxlabel("Error", fontsize=12)
    fig_hist.supylabel("Count", fontsize=12)

    fig_scatter.suptitle(
        f"Experiment 2.1: Non-Zero Error vs True TF (depth={depth})",
        fontsize=16
    )
    fig_scatter.supxlabel("True TF", fontsize=12)
    fig_scatter.supylabel("Error", fontsize=12)

    for idx, width in enumerate(width_list):
        log(f"Experiment 2.1: width={width}, depth={depth}", "INFO")

        r = idx // ncols
        c = idx % ncols

        true_vals, cms_vals, errors = run_cms_experiment(documents, top_terms, width, depth)

        ax_h = axes_hist[r][c]
        ax_s = axes_scatter[r][c]

        ax_h.hist(errors, bins=60, color="#1f77b4", edgecolor="black")
        ax_h.set_title(f"width={width}")
        ax_h.grid(True, linestyle="--", alpha=0.3)

        # Skip zero errors
        pairs = [(t, e) for t, e in zip(true_vals, errors) if e > 0]
        if pairs:
            xs = [t for t, _ in pairs]
            ys = [e for _, e in pairs]
            ax_s.scatter(xs, ys, s=4, color="#1f77b4", alpha=0.6)

        ax_s.set_xscale("log")
        ax_s.set_yscale("log")
        ax_s.set_title(f"width={width}")
        ax_s.grid(True, linestyle="--", alpha=0.3)

    fig_hist.tight_layout()
    fig_hist.savefig(
        "2.1.1.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig_scatter.tight_layout()
    fig_scatter.savefig(
        "2.1.2.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# Experiment 2.2: Vary depth
def vary_depth_experiment(
        documents: list[str],
        top_terms: list[tuple[str, int]],
        width: int,
        depth_list: list[int],
):
    ncols = 3
    nrows = math.ceil(len(depth_list) / ncols)

    fig_hist, axes_hist = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4 * nrows),
        squeeze=False,
    )

    fig_scatter, axes_scatter = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4 * nrows),
        squeeze=False,
    )

    fig_hist.suptitle(
        f"Experiment 2.2: Error Distribution (width={width})",
        fontsize=16
    )
    fig_hist.supxlabel("Error", fontsize=12)
    fig_hist.supylabel("Count", fontsize=12)

    fig_scatter.suptitle(
        f"Experiment 2.2: Non-Zero Error vs True TF (width={width})",
        fontsize=16
    )
    fig_scatter.supxlabel("True TF", fontsize=12)
    fig_scatter.supylabel("Error", fontsize=12)

    for idx, depth in enumerate(depth_list):
        log(f"Experiment 2.2: width={width}, depth={depth}", "INFO")

        r = idx // ncols
        c = idx % ncols

        true_vals, cms_vals, errors = run_cms_experiment(documents, top_terms, width, depth)

        ax_h = axes_hist[r][c]
        ax_s = axes_scatter[r][c]

        ax_h.hist(errors, bins=60, color="#1f77b4", edgecolor="black")
        ax_h.set_title(f"depth={depth}")
        ax_h.grid(True, linestyle="--", alpha=0.3)

        # Skip zero errors
        pairs = [(t, e) for t, e in zip(true_vals, errors) if e > 0]
        if pairs:
            xs = [t for t, _ in pairs]
            ys = [e for _, e in pairs]
            ax_s.scatter(xs, ys, s=4, color="#1f77b4", alpha=0.6)

        ax_s.set_xscale("log")
        ax_s.set_yscale("log")
        ax_s.set_title(f"depth={depth}")
        ax_s.grid(True, linestyle="--", alpha=0.3)

    fig_hist.tight_layout()
    fig_hist.savefig(
        "2.2.1.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig_scatter.tight_layout()
    fig_scatter.savefig(
        "2.2.2.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# Main entry
def main():
    log_section("EXPERIMENT 2")

    documents = load_warc()

    # Calculate true TF
    true_indexer = LocalGlobalTFIndexer()
    true_indexer.build_tf(documents)
    top_terms = true_indexer.top_k(TOP_K)

    vary_width_experiment(
        documents=documents,
        top_terms=top_terms,
        width_list=[5_000, 10_000, 15_000, 20_000, 25_000, 30_000],
        depth=7,
    )

    vary_depth_experiment(
        documents=documents,
        top_terms=top_terms,
        width=20000,
        depth_list=[5, 6, 7, 8, 9, 10],
    )


if __name__ == "__main__":
    main()
