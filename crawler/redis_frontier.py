# crawler/redis_frontier.py

from typing import Any, Iterable, Optional, Union

import json
import time
import redis.asyncio as redis


class RedisFrontier:

    def __init__(
            self,
            client: redis.Redis,
            key: str = "crawler:url_frontier",
            pop_timeout: int = 5,
    ):
        # Redis URL
        self.client = client

        # Frontier ID
        self.key = key

        # Timeout
        self.pop_timeout = pop_timeout

    def _ensure_item(self, item: Union[str, dict[str, Any]]) -> dict[str, Any]:
        if isinstance(item, str):
            return {
                "url": item,
                "parent_url": None,
                "depth": 0,
                "discover_timestamp": time.time(),
            }
        elif isinstance(item, dict):
            # Fill missing fields with defaults
            item.setdefault("parent_url", None)
            item.setdefault("depth", 0)
            item.setdefault("discover_timestamp", time.time())
            return item
        else:
            raise TypeError(f"Invalid frontier item: {item}")

    async def _push_items(self, items: Iterable[Union[str, dict[str, Any]]]):
        serialized = []
        for it in items:
            normalized = self._ensure_item(it)
            serialized.append(json.dumps(normalized))

        if serialized:
            # LPUSH stores list in reverse order
            await self.client.lpush(self.key, *serialized)

    # LPUSH initial seed items
    async def seed(self, items: Iterable[Union[str, dict[str, Any]]]) -> None:
        items = list(items)
        if not items:
            return
        await self._push_items(items)

    # LPUSH items
    async def push(self, items: Iterable[Union[str, dict[str, Any]]]) -> None:
        items = list(items)
        if not items:
            return
        await self._push_items(items)

    # BRPOP
    async def pop(self) -> Optional[dict[str, Any]]:
        result = await self.client.brpop(self.key, timeout=self.pop_timeout)
        if result is None:
            return None

        _, raw = result
        raw_str = raw.decode("utf-8")

        try:
            item = json.loads(raw_str)
        except Exception:
            item = self._ensure_item(raw_str)

        return item
