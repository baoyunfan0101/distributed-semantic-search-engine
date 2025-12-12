# crawler/distributed_worker.py

from typing import Awaitable, Callable, Optional

import time
import aiohttp
import asyncio
import redis.asyncio as redis
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urldefrag, urlparse

from crawler.redis_frontier import RedisFrontier
from crawler.redis_bloom import RedisBloomFilter
from utils.logger import log


class DistributedCrawlerWorker:

    def __init__(
            self,
            redis_url: str,
            base_seed_url: Optional[str] = None,
            allowed_domain: Optional[str] = None,
            frontier_key: str = "crawler:url_frontier",
            bloom_key: str = "crawler:url_bloom",
            max_concurrent_tasks: int = 50,
            request_timeout: int = 10,
            page_handler: Optional[Callable[[str, dict], Awaitable[None]]] = None
    ):
        self.redis_client = redis.from_url(redis_url, decode_responses=False)

        self.frontier = RedisFrontier(
            client=self.redis_client,
            key=frontier_key,
            pop_timeout=5,
        )

        self.bloom = RedisBloomFilter(
            client=self.redis_client,
            key=bloom_key,
            capacity=10_000_000,
            error_rate=0.001,
            expansion=2,
        )

        # Base seed URLs
        self.base_seed_url = base_seed_url

        # Allowed host name
        self.allowed_domain = allowed_domain

        # Max number of concurrent tasks
        self.max_concurrent_tasks = max_concurrent_tasks

        # Request timeout
        self.request_timeout = request_timeout

        # Method to handle HTMLs
        self.page_handler = page_handler

        if self.allowed_domain is None and self.base_seed_url:
            self.allowed_domain = urlparse(self.base_seed_url).netloc

    # Resolve relative URLs
    def _normalize_url(self, url: str, parent_url: str) -> Optional[str]:
        # If url is an absolute URL, parent_url will be ignored
        url = urljoin(parent_url, url)

        # Remove fragment
        # e.g., "#section1" in "https://example.com/page#section1"
        url, _ = urldefrag(url)

        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            return None

        if self.allowed_domain and parsed.netloc != self.allowed_domain:
            return None

        return url.rstrip("/")

    # Extract URLs from HTML
    def _extract_links(self, html: str, base_url: str) -> set[str]:
        soup = BeautifulSoup(html, "html.parser")

        links: set[str] = set()

        for tag in soup.find_all("a", href=True):
            normalized = self._normalize_url(tag["href"], base_url)
            if normalized:
                links.add(normalized)

        return links

    # Fetch HTML from URL
    async def _fetch(
            self,
            session: aiohttp.ClientSession,
            url: str
    ) -> tuple[str, Optional[str], int, str, Optional[int], int]:
        start = time.time()

        try:
            async with session.get(
                    url,
                    timeout=self.request_timeout,
                    allow_redirects=True  # ensure we get the final URL
            ) as resp:

                final_url = str(resp.url)

                status = resp.status
                content_type = resp.headers.get("content-type", "")
                content_length_raw = resp.headers.get("content-length")
                content_length = int(content_length_raw) if content_length_raw else None

                # Reject non-HTML early
                if "text/html" not in content_type.lower():
                    elapsed = int((time.time() - start) * 1000)
                    log(f"SKIP non-HTML: {url} ({content_type})", "INFO")
                    return final_url, None, status, content_type, content_length, elapsed

                # Read body safely
                try:
                    text = await resp.text()
                except UnicodeDecodeError:
                    # fallback decoding
                    raw = await resp.read()
                    text = raw.decode("utf-8", errors="ignore")

                elapsed = int((time.time() - start) * 1000)
                log(
                    f"Fetched OK: {url} -> {final_url} "
                    f"(status={status}, {elapsed}ms)",
                    "INFO"
                )

                return final_url, text, status, content_type, content_length, elapsed

        except Exception as e:
            elapsed = int((time.time() - start) * 1000)
            log(f"ERROR fetching {url}: {e}", "WARNING")
            return url, None, -1, "", None, elapsed

    # Worker loop
    async def _worker_loop(self, session: aiohttp.ClientSession, worker_id: int) -> None:
        log(f"Worker-{worker_id} started", "INFO")

        while True:
            # Pop next item
            item = await self.frontier.pop()

            # Frontier is empty within timeout period
            if item is None:
                # Small sleep to avoid busy loop
                await asyncio.sleep(0.1)
                continue

            # get next URL
            url = item["url"]
            depth = item.get("depth", 0)

            # RedisBloom dedupe using url
            already_seen = await self.bloom.exists(url)
            if already_seen:
                continue
            await self.bloom.add(url)

            # Fetch HTML
            final_url, html, status, content_type, content_length, elapsed = await self._fetch(session, url)

            # RedisBloom dedupe using final_url
            already_seen = await self.bloom.exists(final_url)
            if already_seen:
                continue
            await self.bloom.add(final_url)

            # If fetch failed, skip
            if html is None:
                continue

            # Extract relative URLs
            new_links = self._extract_links(html, final_url)

            # Build fetch metadata
            fetch_info = {
                "requested_url": url,
                "url": final_url,
                "parent_url": item.get("parent_url"),
                "depth": depth,
                "discover_timestamp": item.get("discover_timestamp"),
                "fetch_timestamp": time.time(),
                "http_status": status,
                "content_type": content_type,
                "content_length": content_length,
                "response_time_ms": elapsed,
                "out_links_count": len(new_links),
                "out_links": list(new_links)[:50],
            }

            # Build metadata
            new_items = []
            for link in new_links:
                new_items.append({
                    "url": link,
                    "parent_url": final_url,
                    "depth": depth + 1,
                    "discover_timestamp": time.time(),
                })

            # Push new items
            if new_items:
                await self.frontier.push(new_items)

            # TODO: Send (html, fetch_info) to indexing pipeline
            if self.page_handler is not None:
                try:
                    await self.page_handler(html, fetch_info)
                except Exception as e:
                    log(
                        f"ERROR handling {url} -> {final_url}: {e}",
                        "WARNING"
                    )

    # Start worker
    async def run(self) -> None:
        # Seed initial URLs.
        if self.base_seed_url:
            # Seed only if frontier is currently empty
            frontier_len = await self.redis_client.llen(self.frontier.key)
            if frontier_len == 0:
                log(f"Seeding frontier with: {self.base_seed_url}", "INFO")
                await self.frontier.seed([{
                    "url": self._normalize_url(self.base_seed_url, self.base_seed_url),
                    "parent_url": None,
                    "depth": 0,
                    "discover_timestamp": time.time(),
                }])

        # Async TCP connection pool
        connector = aiohttp.TCPConnector(limit=None)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                asyncio.create_task(self._worker_loop(session, worker_id=i))
                for i in range(self.max_concurrent_tasks)
            ]

            # Run until cancelled
            try:
                # Run tasks concurrently and wait for all tasks to end
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                # Cancelled
                log("Worker cancelled, shutting down...", "INFO")
            finally:
                # Cancel all tasks
                for t in tasks:
                    t.cancel()


# CLI entry
async def main():
    from crawler.handlers.save_html import SaveHTMLHandler
    handler = SaveHTMLHandler()
    worker = DistributedCrawlerWorker(
        redis_url="redis://localhost:6379/0",
        base_seed_url="https://quotes.toscrape.com/",
        allowed_domain="quotes.toscrape.com",
        frontier_key="crawler:url_frontier",
        bloom_key="crawler:url_bloom",
        max_concurrent_tasks=20,
        request_timeout=10,
        page_handler=handler,
    )
    await worker.run()


if __name__ == "__main__":
    '''Setup RedisStack through docker
    docker run -it -p 6379:6379 redis/redis-stack-server:latest
    '''
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("KeyboardInterrupt received, exiting...", "INFO")
