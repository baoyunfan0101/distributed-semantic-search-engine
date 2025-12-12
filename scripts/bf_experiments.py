# scripts/bf_experiments.py

import asyncio
import random
import string
import time
from typing import Any

import redis.asyncio as redis
import matplotlib.pyplot as plt

from crawler.redis_bloom import RedisBloomFilter
from utils.logger import log, log_section


# Random URL Generation
def generate_random_urls(
        n: int,
        dup_ratio: float = 0.3,
        domain: str = "https://example.com"
) -> list[str]:
    base_paths: list[str] = []
    urls: list[str] = []

    for _ in range(n):
        if base_paths and random.random() < dup_ratio:
            # Reuse a previous path
            path = random.choice(base_paths)
        else:
            # Create a new unique path
            path = "/" + "".join(
                random.choices(string.ascii_lowercase + string.digits, k=12)
            )
            base_paths.append(path)

        urls.append(domain + path)

    return urls


def generate_non_existing_urls(
        n: int,
        domain: str = "https://nonexistent.com"
) -> list[str]:
    urls = []
    for _ in range(n):
        path = "/" + "".join(
            random.choices(string.ascii_lowercase + string.digits, k=12)
        )
        urls.append(domain + path)
    return urls


# Helper: BF.INFO to dict
async def get_bf_info(
        client: redis.Redis,
        key: str
) -> dict[str, Any]:
    try:
        info = await client.execute_command("BF.INFO", key)
    except Exception:
        return {}

    if isinstance(info, list) and len(info) % 2 == 0:
        it = iter(info)
        return {str(k): v for k, v in zip(it, it)}
    else:
        # Fallback for unexpected formats
        return {"raw": info}


async def get_memory_usage(
        client: redis.Redis,
        key: str
) -> int:
    try:
        usage = await client.execute_command("MEMORY", "USAGE", key)
        if usage is None:
            return 0
        return int(usage)
    except Exception:
        return 0


# Build Bloom Filter and measure metrics
async def build_bf_and_measure(
        client: redis.Redis,
        key: str,
        capacity: int,
        error_rate: float,
        expansion: int,
        n_insert: int,
        n_query: int,
) -> dict[str, Any]:
    # Clear key if exists
    await client.delete(key)

    bf = RedisBloomFilter(
        client=client,
        key=key,
        capacity=capacity,
        error_rate=error_rate,
        expansion=expansion,
    )

    # Insert random URLs
    urls_insert = generate_random_urls(n_insert)
    for url in urls_insert:
        await bf.add(url)

    # Generate missing URLs for FPR test
    urls_query = generate_non_existing_urls(n_query)

    # Measure FPR and query latency
    false_positives = 0
    t0 = time.perf_counter()

    for url in urls_query:
        exists = await bf.exists(url)
        if exists:
            false_positives += 1

    t1 = time.perf_counter()

    fpr = false_positives / n_query
    avg_query_ms = (t1 - t0) * 1000.0 / n_query

    # Collect bloom filter metadata
    info = await get_bf_info(client, key)
    memory_bytes = await get_memory_usage(client, key)

    # Extract number of filters (layers)
    num_filters = None
    for k in info.keys():
        if "Number of filters" in k or "Num filters" in k or "numFilters" in k:
            num_filters = int(info[k])
            break

    return {
        "capacity": capacity,
        "error_rate": error_rate,
        "expansion": expansion,
        "layers": num_filters,
        "memory_bytes": memory_bytes,
        "fpr": fpr,
        "avg_query_ms": avg_query_ms,
        "info": info,
    }


# Experiment 1.1: Vary error_rate
async def vary_error_rate_experiment(
        redis_url: str = "redis://localhost:6379/0",
        capacity: int = 1_000_000,
        expansion: int = 2,
        error_rates: list[float] = None,
        n_insert: int = 600_000,
        n_query: int = 50_000,
):
    if error_rates is None:
        error_rates = [1e-1, 9e-2, 8e-2, 7e-2, 6e-2, 5e-2, 4e-2, 3e-2, 2e-2, 1e-2]

    client = redis.from_url(redis_url, decode_responses=False)

    results = []

    for p in error_rates:
        log(
            f"Experiment 1.1: capacity={capacity}, expansion={expansion}, error_rate={p}",
            "INFO"
        )
        res = await build_bf_and_measure(
            client=client,
            key=f"bf:error_rate:{p}",
            capacity=capacity,
            error_rate=p,
            expansion=expansion,
            n_insert=n_insert,
            n_query=n_query,
        )
        results.append(res)

    await client.close()

    # Prepare data for plotting
    xs = [r["error_rate"] for r in results]
    ys = [r["fpr"] for r in results]

    plt.figure()
    # theoretical line
    plt.plot(xs, xs, "--", color="#1f77b4", alpha=0.6, label="Ideal y = x")
    plt.scatter(xs, ys, color="#1f77b4", alpha=0.6, label="Measured FPR")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Configured error_rate (log scale)")
    plt.ylabel("Measured FPR (log scale)")
    plt.title("Experiment 1.1: error_rate vs actual FPR")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("1.1.png", dpi=300, bbox_inches="tight")
    plt.show()


# Experiment 1.2: Vary capacity
async def vary_capacity_experiment(
        redis_url: str = "redis://localhost:6379/0",
        error_rate: float = 0.001,
        expansion: int = 2,
        capacities: list[int] = None,
        n_insert: int = 50_000,
        n_query: int = 50_000,
):
    if capacities is None:
        capacities = [25_000, 50_000, 75_000, 100_000, 125_000, 150_000, 175_000, 200_000]

    client = redis.from_url(redis_url, decode_responses=False)

    results = []
    for cap in capacities:
        log(
            f"Experiment 1.2: capacity={cap}, expansion={expansion}, error_rate={error_rate}",
            "INFO"
        )
        res = await build_bf_and_measure(
            client=client,
            key=f"bf:capacity:{cap}",
            capacity=cap,
            error_rate=error_rate,
            expansion=expansion,
            n_insert=n_insert,
            n_query=n_query,
        )
        results.append(res)

    await client.close()

    # Extract data
    caps = [r["capacity"] for r in results]
    layers = [r["layers"] or 0 for r in results]
    mem_mb = [r["memory_bytes"] / (1024 * 1024) for r in results]
    fprs = [r["fpr"] for r in results]
    avg_ms = [r["avg_query_ms"] for r in results]

    # Plot 1: capacity vs layers
    plt.figure()
    plt.plot(caps, layers, marker="o", color="#1f77b4")
    plt.xlabel("Capacity")
    plt.ylabel("Number of filters (layers)")
    plt.title("Experiment 1.2.1: capacity vs number of filters")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("1.2.1.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot 2: capacity vs memory usage
    plt.figure()
    plt.plot(caps, mem_mb, marker="o", color="#1f77b4")
    plt.xlabel("Capacity")
    plt.ylabel("Memory usage (MB)")
    plt.title("Experiment 1.2.2: capacity vs memory usage")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("1.2.2.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot 3: capacity vs FPR
    plt.figure()
    plt.plot(caps, fprs, marker="o", color="#1f77b4")
    plt.xlabel("Capacity")
    plt.ylabel("Measured FPR")
    plt.yscale("log")
    plt.title("Experiment 1.2.3: capacity vs measured FPR")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("1.2.3.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot 4: capacity vs query latency
    plt.figure()
    plt.plot(caps, avg_ms, marker="o", color="#1f77b4")
    plt.xlabel("Capacity")
    plt.ylabel("Avg query time (ms)")
    plt.title("Experiment 1.2.4: capacity vs query time")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("1.2.4.png", dpi=300, bbox_inches="tight")
    plt.show()


# Experiment 1.3: Vary expansion
async def vary_expansion_experiment(
        redis_url: str = "redis://localhost:6379/0",
        error_rate: float = 0.001,
        capacity: int = 50_000,
        expansions: list[int] = None,
        n_insert: int = 150_000,
        n_query: int = 50_000,
):
    if expansions is None:
        expansions = [1, 2, 3, 4]

    client = redis.from_url(redis_url, decode_responses=False)

    results = []
    for g in expansions:
        log(
            f"Experiment 1.3: capacity={capacity}, expansion={g}, error_rate={error_rate}",
            "INFO"
        )
        res = await build_bf_and_measure(
            client=client,
            key=f"bf:expansion:{g}",
            capacity=capacity,
            error_rate=error_rate,
            expansion=g,
            n_insert=n_insert,
            n_query=n_query,
        )
        results.append(res)

    await client.close()

    # Actual expansions used (integer-rounded)
    g_used = [r["expansion"] for r in results]
    layers = [r["layers"] or 0 for r in results]
    mem_mb = [r["memory_bytes"] / (1024 * 1024) for r in results]
    fprs = [r["fpr"] for r in results]
    avg_ms = [r["avg_query_ms"] for r in results]

    # Plot 1: expansion vs layers
    plt.figure()
    plt.plot(g_used, layers, marker="o", color="#1f77b4")
    plt.xlabel("Expansion factor (integer)")
    plt.ylabel("Number of filters (layers)")
    plt.title("Experiment 1.3.1: expansion vs number of filters")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("1.3.1.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot 2: expansion vs memory usage
    plt.figure()
    plt.plot(g_used, mem_mb, marker="o", color="#1f77b4")
    plt.xlabel("Expansion factor (integer)")
    plt.ylabel("Memory usage (MB)")
    plt.title("Experiment 1.3.2: expansion vs memory usage")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("1.3.2.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot 3: expansion vs FPR
    plt.figure()
    plt.plot(g_used, fprs, marker="o", color="#1f77b4")
    plt.xlabel("Expansion factor (integer)")
    plt.ylabel("Measured FPR")
    plt.yscale("log")
    plt.title("Experiment 1.3.3: expansion vs measured FPR")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("1.3.3.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot 4: expansion vs query latency
    plt.figure()
    plt.plot(g_used, avg_ms, marker="o", color="#1f77b4")
    plt.xlabel("Expansion factor (integer)")
    plt.ylabel("Avg query time (ms)")
    plt.title("Experiment 1.3.4: expansion vs query time")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("1.3.4.png", dpi=300, bbox_inches="tight")
    plt.show()


# Main entry
async def main():
    log_section("EXPERIMENT 1")

    # error_rate vs actual FPR
    await vary_error_rate_experiment(
        capacity=100_000,
        expansion=2,
        n_insert=60_000,
        n_query=50_000,
        error_rates=[1e-1, 9e-2, 8e-2, 7e-2, 6e-2, 5e-2, 4e-2, 3e-2, 2e-2, 1e-2],
    )

    # capacity vs layers/memory/FPR/time
    await vary_capacity_experiment(
        error_rate=0.001,
        expansion=2,
        capacities=[25_000, 50_000, 75_000, 100_000, 125_000, 150_000, 175_000, 200_000],
        n_insert=50_000,
        n_query=50_000,
    )

    # expansion vs layers/memory/FPR/time
    await vary_expansion_experiment(
        error_rate=0.001,
        capacity=50_000,
        expansions=[1, 2, 3, 4, 5],
        n_insert=150_000,
        n_query=50_000,
    )


if __name__ == "__main__":
    asyncio.run(main())
